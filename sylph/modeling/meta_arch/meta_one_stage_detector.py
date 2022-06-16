"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

#!/usr/bin/env python3
import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Union  # noqa

import torch
from adet.modeling.one_stage_detector import detector_postprocess
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList, Boxes
from sylph.modeling.code_generator.build import build_code_generator
from torch import nn
from detectron2.config import configurable

from detectron2.layers.batch_norm import FrozenBatchNorm2d


__all__ = ["MetaOneStageDetector", "MetaProposalNetwork"]

debug = False
logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class MetaProposalNetwork(nn.Module):
    """
    Integrates all components in the main architecture.
    Example:
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg). It calls from_config first to get arguments for __init__.
    Forward functions include:
        forward_base_detector: training and testing for base detector
        forward_few_shot_detector_training: training stage for few-shot detection
        forward: calls forward_base_detector and forward_few_shot_detector_training
        forward_class_code: inference for generating class codes
        forward_instances: inference for generating instances
    """
    @configurable
    def __init__(self, cfg, episodic_learning, backbone, proposal_generator, code_generator):
        """
        Support two training stages: meta_learning while episodic_learning = True and normal detector pretraining while episodic_learning=False

        """
        super().__init__()
        self.episodic_learning = episodic_learning
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.code_generator = code_generator
        if self.episodic_learning:
            assert self.code_generator is not None

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )

        self._freeze_parameters(cfg)

        # Add useful parameters from cfg
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES

    @classmethod
    def from_config(cls, cfg):
        episodic_learning = cfg.MODEL.META_LEARN.EPISODIC_LEARNING
        backbone = build_backbone(cfg)
        proposal_generator = build_proposal_generator(
            cfg, backbone.output_shape()
        )

        input_shape = backbone.output_shape()
        in_features = cfg.MODEL.FCOS.IN_FEATURES
        input_shapes = [input_shape[f] for f in in_features]
        code_generator = (
            build_code_generator(cfg, feature_channels=input_shapes[0].channels, feature_levels=len(in_features), strides=cfg.MODEL.FCOS.FPN_STRIDES)
            if episodic_learning
            else None
        )
        return {"cfg": cfg, "episodic_learning": episodic_learning, "backbone": backbone, "proposal_generator": proposal_generator, "code_generator": code_generator}

    @property
    def device(self):
        return self.pixel_mean.device


    def _check_contain(self, name: str, key_words: List[str]) -> bool:
        for kw in key_words:
            if kw in name:
                return True
        return False

    def _freeze_backbone_parameters(self, cfg):
        if cfg.MODEL.BACKBONE.FREEZE:
            freeze_exclude_backbone_param_names = []
            for name, p in self.backbone.named_parameters():
                if len(cfg.MODEL.BACKBONE.FREEZE_EXCLUDE) > 0 and self._check_contain(
                    name, cfg.MODEL.BACKBONE.FREEZE_EXCLUDE
                ):
                    freeze_exclude_backbone_param_names.append(name)
                    continue
                p.requires_grad = False
            # convert the batch norm to frozen batch norm
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.backbone)
            logger.info(
                f"froze backbone parameters, but exclude {freeze_exclude_backbone_param_names}"
            )

    def _freeze_detector_head(self, cfg):
        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:  # freeze the whole network
            logger.info("froze proposal_generator parameters")
            for (name, p) in self.proposal_generator.named_parameters():
                p.requires_grad = False
                logger.info(f"freeze {name}")
        else:
            # class branch
            if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE_CLS_TOWER or cfg.MODEL.PROPOSAL_GENERATOR.OWD:
                for (
                    name,
                    p,
                ) in self.proposal_generator.fcos_head.cls_tower.named_parameters():
                    p.requires_grad = False
                    logger.info(f"froze {name}")

            if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE_CLS_LOGITS or cfg.MODEL.PROPOSAL_GENERATOR.OWD:
                for (
                    name,
                    p,
                ) in self.proposal_generator.fcos_head.cls_logits.named_parameters():
                    p.requires_grad = False
                    logger.info(f"froze {name}")

            # box branch TODO: freeze iou
            freeze_box_branch, freeze_box_tower = cfg.MODEL.PROPOSAL_GENERATOR.FREEZE_BBOX_BRANCH, cfg.MODEL.PROPOSAL_GENERATOR.FREEZE_BBOX_TOWER
            if freeze_box_branch or freeze_box_tower:
                branch_modules = {"box_tower": self.proposal_generator.fcos_head.bbox_tower.named_parameters()}
                if freeze_box_branch:
                    branch_modules.update({"bbox_pred": self.proposal_generator.fcos_head.bbox_pred.named_parameters(),
                                "ctrness": self.proposal_generator.fcos_head.ctrness.named_parameters(),
                                "iou_overlap":self.proposal_generator.fcos_head.iou_overlap.named_parameters(),})
                for key, branch_module in branch_modules.items():
                    for (
                        name,
                        p,
                    ) in branch_module:
                        p.requires_grad = False
                        logger.info(f"froze {name} in branch {key}")

    def _freeze_code_generator(self, cfg):
         if (
            self.code_generator is not None
            and cfg.MODEL.META_LEARN.CODE_GENERATOR.FREEZE
        ):
            for p in self.code_generator.parameters():
                p.requires_grad = False
            logger.info("froze code generator parameters")

    def _freeze_parameters(self, cfg):
        """
        Freeze parameters from BACKBONE, PROPOSAL_GENERATOR, and META_LEARN.CODE_GENERATOR
        """
        self._freeze_backbone_parameters(cfg)
        self._freeze_detector_head(cfg)
        self._freeze_code_generator(cfg)

    def convert_batched_inputs_to_image_list(self, batched_inputs: List[Dict])-> ImageList:
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def _extract_backbone_features(self, images: torch.Tensor):
        features = self.backbone(images)
        return features

    def _get_gt(self, batched_inputs, support_set_targets=None):
        """
        Filter out instances based on support_set_targets
        """
        if "instances" in batched_inputs[0]:
            if support_set_targets is None:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                assert isinstance(
                    support_set_targets, list
                ), "support_set_targets is not list"
                gt_instances = []
                support_set_targets = [
                    class_id.item() for class_id in support_set_targets
                ]
                for x in batched_inputs:
                    gt_boxes = x["instances"].gt_boxes
                    gt_classes = x["instances"].gt_classes
                    filtered_gt_boxes, filtered_gt_classes = [], []
                    filtered_x_instance = copy.deepcopy(x["instances"])
                    filtered_x_instance.remove("gt_boxes")
                    filtered_x_instance.remove("gt_classes")
                    for class_id, box in zip(gt_classes, gt_boxes):
                        class_id = class_id.item()
                        if class_id in support_set_targets:
                            filtered_gt_boxes.append(
                                box.numpy()
                            )  # convert a tensor to numpy array
                            filtered_gt_classes.append(class_id)
                    filtered_x_instance.gt_boxes = Boxes(
                        torch.tensor(filtered_gt_boxes)
                    )
                    filtered_x_instance.gt_classes = torch.tensor(filtered_gt_classes)
                    gt_instances.append(filtered_x_instance.to(self.device))

        elif "targets" in batched_inputs[0]:
            raise NotImplementedError("targets are not supported")

        else:
            gt_instances = None
        return gt_instances

    def _put_to_device(self, batched_support_set_target: List[Any]):
        return [x.to(self.device) for x in batched_support_set_target]

    def forward_class_code(self, batched_inputs: List[Dict[str, Any]]):
        """
        Inputs is support set of length 1. Used only for inference. Generate class codes for all given support sets in batched_inputs
        Args:
            batched_inputs: a list of dict with "support_set" (a list).
        Returns:
            support_set_class_codes: a list of dict output from code generator
        """
        assert not self.training
        assert len(batched_inputs) == 1, f"batched_inputs has length: {len(batched_inputs)}"
        # 1. extend to the real batch size
        batched_inputs_support_set = [
            record for x in batched_inputs for record in x["support_set"]
        ]  # 1* (c*s)

        batched_support_set_gt_instances = [x["instances"].to(self.device) for x in batched_inputs_support_set]
        support_set_image_tensor = self.convert_batched_inputs_to_image_list(batched_inputs_support_set).tensor
        # 2. Extract backbone features
        support_set_image_features = self._extract_backbone_features(support_set_image_tensor)

        # 3. Generate class codes
        features = [support_set_image_features[f] for f in self.in_features]
        support_set_class_codes = self.code_generator(
            features, batched_support_set_gt_instances
        )
        return support_set_class_codes

    def normalize_class_code(self, codes: List[Dict]):
        assert self.episodic_learning
        assert not self.training
        return self.code_generator(features=None, target_instances=None, cls_norm=True, class_codes=codes)

    def forward_instances(self, batched_inputs: List[Dict[str, Any]], class_codes: Dict[str, torch.Tensor]):
        """
        Inputs is support set of length 1. Used only for inference. Generate class codes for all given support sets in batched_inputs
        Args:
            batched_inputs: a list of images as dict from a normal data loader.
            class_codes: a dict with "cls_conv" and "cls_bias" as torch tensor
        Returns:
            support_set_class_codes: a list of dict output from code generator
        """
        assert self.episodic_learning
        assert not self.training
        images_lst = self.convert_batched_inputs_to_image_list(batched_inputs)
        images_features = self.backbone(images_lst.tensor)
        print_test_loss = True
        gt_instances = None
        if print_test_loss:
            gt_instances = self._get_gt(batched_inputs)

        proposals, _ = self.proposal_generator(
            images_lst,
            images_features,
            gt_instances=gt_instances,
            top_module=None,
            support_set_per_class_code=class_codes,
            support_set_targets=None,
        )

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images_lst.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def forward_base_detector(self, batched_inputs: List[Dict]):
        """
        """
        assert not self.episodic_learning
        # Process input and get features
        images_lst = self.convert_batched_inputs_to_image_list(batched_inputs)
        images_features = self._extract_backbone_features(images_lst.tensor)
        gt_instances = self._get_gt(batched_inputs)
        # Run predictor head to get prediction and loss
        proposals, proposal_losses = self.proposal_generator(
            images_lst, images_features, gt_instances
        )
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images_lst.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

    def forward_few_shot_detector_training(self, batched_inputs: List[Dict[str, Any]]):
        """
        Args:
            batched_inputs: each item represents the query + support set of a class
        """
        assert self.training
        assert "support_set" in batched_inputs[0]
        assert "query_set" in batched_inputs[0]
        # 1. Separate batched inputs to batched support set and query set

        # bs * SHOT
        batched_inputs_support_set = [
            record for x in batched_inputs for record in x["support_set"]
        ]
        # bs
        batched_inputs_support_set_targets = [
            x["support_set_target"] for x in batched_inputs
        ]
        batched_inputs_support_set_targets = self._put_to_device(batched_inputs_support_set_targets)

        support_set_gt_instances =[x["instances"].to(self.device) for x in batched_inputs_support_set]
        # check the batch norm info, need to use all modules
        for name, module in self.backbone.named_children():
            if isinstance(module, FrozenBatchNorm2d):
                print(f"module {name}, statistics: {module.running_mean,module.running_var,module.weight,module.bias}")
        # bs * QUERY_SHOT
        batched_inputs_query_set = [
            record for x in batched_inputs for record in x["query_set"]
        ]

        # 2. Extract features

        query_set_images_lst = self.convert_batched_inputs_to_image_list(batched_inputs_query_set)
        query_set_images_feature = self._extract_backbone_features(query_set_images_lst.tensor)

        # extract support set features
        support_set_images_lst = self.convert_batched_inputs_to_image_list(batched_inputs_support_set)
        support_set_images_feature = self._extract_backbone_features(support_set_images_lst.tensor)



        # filter gt_instances
        query_set_gt_instances = self._get_gt(
            batched_inputs_query_set, support_set_targets=batched_inputs_support_set_targets
        )

        # Get support set class codes
        support_set_class_codes = self.code_generator([support_set_images_feature[f] for f in self.in_features], support_set_gt_instances)

        # Run predictor head
        # TODO: add condConv as a separate module
        proposals, proposal_losses = self.proposal_generator(
            query_set_images_lst.tensor,
            query_set_images_feature,
            query_set_gt_instances,
            None,
            support_set_class_codes,
            batched_inputs_support_set_targets,
        )
        if "snnl" in support_set_class_codes:
            proposal_losses.update(
                {"loss_snnl": support_set_class_codes["snnl"]}
            )
        return proposal_losses


    def forward(self, batched_inputs: List[Dict[str, Any]]):
        """
        Forward for base detector's training and inference and meta-learning's training stage.
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """

        if not self.episodic_learning:
            return self.forward_base_detector(batched_inputs)
        # episodic learning
        if self.training:
            return self.forward_few_shot_detector_training(batched_inputs)
        else:
            raise NotImplementedError(
                "Episodic learning inferrence for image and features is not supported in forward."
            )


@META_ARCH_REGISTRY.register()
class MetaOneStageDetector(MetaProposalNetwork):
    """
    Meta Arach for few-shot detection, four forward types:
    1. training model, includes both pretraining and meta-learning training (TODO: pretraining can be done in other code base)
    2. run_tupe = "meta_learn_test_support": takes batched input from support set data loader, return class codes as dict
    3. run_tupe = "meta_learn_test_instance": takes batched inputs from query set data loader and class codes, return processed result
    4. run_type = None: a normal base detector inference
    """

    def forward(
        self, batched_inputs, class_code=None, run_type=None
    ):
        # logics for pretraining and meta-learning training
        if self.training:
            return super().forward(batched_inputs)
        # logics for testing
        if run_type is None:
            processed_results = super().forward(batched_inputs)
            temp_processed_results = [
                {"instances": r["proposals"]} for r in processed_results
            ]
            return processed_results if len(temp_processed_results) == 0 else temp_processed_results
        if run_type == "meta_learn_test_support":
            return self.forward_class_code(batched_inputs)
        if run_type == "meta_learn_normalize_code":
            return self.normalize_class_code(class_code)
        if run_type == "meta_learn_test_instance":
            # class code could be None here if eval with pretrained class code
            return self.forward_instances(batched_inputs, class_code)
        raise NotImplementedError(f"not support this forward type: {run_type}, class_code: {class_code}")

    def forward_class_code(self, batched_inputs):
        assert not self.training, "Not for training"
        return super().forward_class_code(batched_inputs)

    def forward_instances(self, batched_inputs, class_codes):
        assert not self.training, "Not for training"
        return super().forward_instances(
            batched_inputs, class_codes
        )  # proposal and losses
