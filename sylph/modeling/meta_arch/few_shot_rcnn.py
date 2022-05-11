# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage

from detectron2.modeling import META_ARCH_REGISTRY
from typing import Any
from .meta_one_stage_detector import MetaProposalNetwork
from sylph.modeling.code_generator.build import build_code_generator
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.layers.batch_norm import FrozenBatchNorm2d
logger = logging.getLogger(__name__)

__all__ = ["FewShotGeneralizedRCNN", "FewShotDetector"]


@META_ARCH_REGISTRY.register()
class FewShotGeneralizedRCNN(MetaProposalNetwork, GeneralizedRCNN):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction

    Inherits both Faster RCNN and a MetaProposalNetwork
    """
    @configurable
    def __init__(
        self,
        *,
        cfg,
        episodic_learning: bool,
        code_generator: nn.Module, # add code generator, optional
        **kwargs,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        GeneralizedRCNN.__init__(self, **kwargs)
        self.cfg = cfg
        self.episodic_learning = episodic_learning
        self.code_generator = code_generator
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        if self.episodic_learning:
            assert self.code_generator is not None

        self._freeze_parameters(cfg)

    def _freeze_proposal_heads(self, cfg):
        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for _, p in self.proposal_generator.named_parameters():
                p.requires_grad = False
            # convert the batch norm to frozen batch norm
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.proposal_generator)
            logger.info("froze proposal_generator heads parameters")

    def _freeze_roi_heads(self, cfg):
        if cfg.MODEL.ROI_HEADS.FREEZE:
            for _, p in self.roi_heads.named_parameters():
                p.requires_grad = False
            # convert the batch norm to frozen batch norm
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.roi_heads)
            logger.info("froze roi heads parameters")

    def _freeze_parameters(self, cfg):
        """
        Freeze parameters from BACKBONE, PROPOSAL_GENERATOR, and META_LEARN.CODE_GENERATOR
        """
        self._freeze_backbone_parameters(cfg)
        self._freeze_detector_head(cfg) # proposal_generator
        self._freeze_code_generator(cfg)
        self._freeze_roi_heads(cfg)
        self._freeze_proposal_heads(cfg)

    @classmethod
    def from_config(cls, cfg):
        ret = GeneralizedRCNN.from_config(cfg)
        # backbone = build_backbone(cfg)
        episodic_learning = cfg.MODEL.META_LEARN.EPISODIC_LEARNING
        input_shape = ret["backbone"].output_shape()
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        input_shapes = [input_shape[f] for f in in_features]
        strides = [input_shape[f].stride for f in in_features]
        code_generator = ( #TODO: add strides for
            build_code_generator(cfg, feature_channels=input_shapes[0].channels, feature_levels=len(in_features), strides=strides)
            if episodic_learning
            else None
        )
        ret["cfg"] = cfg
        ret["episodic_learning"] = episodic_learning
        ret["code_generator"] = code_generator
        return ret

    def forward_base_detector(self, batched_inputs: List[Dict[str, torch.Tensor]]): # rename the forward to forward_base_detector
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        assert not self.episodic_learning
        if not self.training:
            return GeneralizedRCNN.inference(self, batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def forward_few_shot_detector_training(self, batched_inputs: List[Dict[str, Any]]):
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
        support_set_images_lst = self.convert_batched_inputs_to_image_list(batched_inputs_support_set)
        support_set_images_feature = self._extract_backbone_features(support_set_images_lst.tensor)

        query_set_images_lst = self.convert_batched_inputs_to_image_list(batched_inputs_query_set)
        query_set_images_feature = self._extract_backbone_features(query_set_images_lst.tensor)

        # filter gt_instances
        query_set_gt_instances = self._get_gt(
            batched_inputs_query_set, support_set_targets=batched_inputs_support_set_targets
        )
        # CHANGE START
        if self.proposal_generator is not None:
            # TODO: use gt_instances before filtering in meta-learning if retraining
            proposals, proposal_losses = self.proposal_generator(query_set_images_lst, query_set_images_feature, query_set_gt_instances)
            # print(f"proposal here: {proposal_losses}")
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # Get support set class codes
        support_set_class_codes = self.code_generator([support_set_images_feature[f] for f in self.in_features], support_set_gt_instances)

        # print(f"support_set_class_codes: {support_set_class_codes}")
        # Generate detection
        _, detector_losses = self.roi_heads(query_set_images_lst, query_set_images_feature, proposals, query_set_gt_instances, support_set_class_codes, batched_inputs_support_set_targets)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if "snnl" in support_set_class_codes:
            losses.update(
                {"loss_snnl": support_set_class_codes["snnl"]}
            )
        return losses


    def forward_instances(self, batched_inputs: List[Dict[str, Any]], class_codes: Dict[str, torch.Tensor], do_postprocess: bool =True):
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
        assert class_codes is not None


        images_lst = self.convert_batched_inputs_to_image_list(batched_inputs)
        images_features = self.backbone(images_lst.tensor)

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images_lst, images_features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        results, _ = self.roi_heads(images=images_lst, features=images_features, proposals=proposals, targets=None, class_codes=class_codes, class_codes_target=None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images_lst.image_sizes)
        else:
            return results

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
class FewShotDetector(FewShotGeneralizedRCNN):
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
            return processed_results
        if run_type == "meta_learn_test_support":
            return self.forward_class_code(batched_inputs)
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
