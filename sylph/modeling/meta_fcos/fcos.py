#!/usr/bin/env python3
import logging
import math
from typing import List, Dict, Optional

import torch
from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from sylph.modeling.meta_fcos.fcos_outputs import FCOSOutputs
from sylph.modeling.utils import init_norm
from torch import nn
from torch.nn import functional as F
from detectron2.utils.file_io import PathManager
from detectron2.data import MetadataCatalog
from .head_utils import CondConv, CondConvBasic




from .head_utils import CondConvBlock, Scale

logger = logging.getLogger(__name__)


__all__ = ["MetaFCOS"]

INF = 100000000

def load_pretrained_class_conv_kernel_weights(weight_path: str):
    if "manifold" != weight_path.split("://")[0]:
        logger.info("Path is not manifold")
        return None, None
    if not PathManager.exists(weight_path):
        logger.info(f"Weight path: {weight_path} does not exist")
        return None, None

    local_weight_path = PathManager.get_local_path(weight_path)
    pretrained_model = torch.load(
        local_weight_path, map_location=torch.device("cpu")
    )
    # find : proposal_generator.fcos_head.cls_logits.weight
    weight, bias = None, None
    for key, value in pretrained_model["model"].items():
        # print(f"key {key}, value: {value}")
        if "proposal_generator.fcos_head.cls_logits" in key:
            if "weight" in key:
                weight = value
            elif "bias" in key:
                bias = value
    return weight, bias

class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


def _build_tower_module(
    head_configs: Dict,
    in_channels: int,
    norm: str,
    num_feature_levels: int,
):
    head_tower_sequential = {}
    for head, head_config in head_configs.items():
        num_convs, use_deformable = head_config
        tower = []
        for i in range(num_convs):
            if use_deformable and i == num_convs - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
            )
            if norm == "GN":
                tower.append(nn.GroupNorm(32, in_channels))
            elif norm == "NaiveGN":
                tower.append(NaiveGroupNorm(32, in_channels))
            elif norm == "BN":
                tower.append(
                    ModuleListDial(
                        [nn.BatchNorm2d(in_channels) for _ in range(num_feature_levels)]
                    )
                )
            elif norm == "SyncBN":
                tower.append(
                    ModuleListDial(
                        [
                            NaiveSyncBatchNorm(in_channels)
                            for _ in range(num_feature_levels)
                        ]
                    )
                )
            tower.append(nn.ReLU())
        # pop out the last layer's ReLU in cls_tower
        # if use_l2_norm_conv2d and head == "cls":
        #     tower.pop()
        head_tower_sequential[head] = nn.Sequential(*tower)
    return head_tower_sequential


class BaseFewShotDetectorHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        self.cfg = cfg
        self.input_shape = input_shape
        # get submodule configs
        self._init_base_detector_config()
        # get meta_learning
        self.episodic_learning = cfg.MODEL.META_LEARN.EPISODIC_LEARNING
        if self.episodic_learning:
            self._init_code_generator_config(
                name=cfg.MODEL.META_LEARN.CODE_GENERATOR.NAME
            )
            self._init_merge_module_config()

    def _init_base_detector_config(self, name: Optional[str] = None):
        pass

    def _init_code_generator_config(self, name: Optional[str] = None):
        pass

    def _init_merge_module_config(self, name: Optional[str] = None):
        pass

    #  forward
    # def forward_base_train(self, x, top_module=None, yield_bbox_towers=False):
    #     raise NotImplementedError()

    # def forward(self, x: List[Any],top_module=None,yield_bbox_towers: bool =False, support_set_class_codes: Dict[torch.Tensor]=None):
    #     raise NotImplementedError()


# Head outputs calls few-shot predictor head
@PROPOSAL_GENERATOR_REGISTRY.register()
class MetaFCOS(nn.Module):  # TODO: convert it to Base Few Shot Detector Head
    """
    Implement Meta FCOS.
    FCOS: (https://arxiv.org/abs/1904.01355).
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        DetectorHead (forward)
        Detector outputs (losses or proposals)
        """
        super().__init__()
        self.cfg = cfg
        self.episodic_learning = cfg.MODEL.META_LEARN.EPISODIC_LEARNING
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS.YIELD_PROPOSAL

        # TODO: test more modualized head
        self.fcos_head = MetaFCOSHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module

        self.fcos_outputs = FCOSOutputs(cfg)

    def forward(
        self,
        images,
        features,
        gt_instances=None,
        top_module=None,
        support_set_per_class_code=None,
        support_set_targets=None,
    ):
        """
        Arguments:
            images (list[Tensor] or ImageList): query set images to be processed
            features: query set features.
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, iou_pred, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal, support_set_per_class_code
        )

        results = {}
        if self.yield_proposal:
            results["features"] = {f: b for f, b in zip(self.in_features, bbox_towers)}

        if self.training:
            # TODO(neogong): This need to be optimized. Find out a way to copy params after loading the checkpoint.
            if (
                self.episodic_learning
                and self.cfg.MODEL.META_LEARN.CODE_GENERATOR.DISTILLATION_LOSS_WEIGHT
            ):
                cls_logits_kernel = (
                    self.fcos_head.cls_logits.weight.data.clone().detach(),
                    self.fcos_head.cls_logits.bias.data.clone().detach(),
                )
            else:
                cls_logits_kernel = None
            # If support_set_targets is not None, it goes to meta_loss, else, pretraining loss

            results, losses = self.fcos_outputs.losses(
                logits_pred,
                reg_pred,
                ctrness_pred,
                iou_pred,
                locations,
                gt_instances,
                top_feats,
                support_set_targets=support_set_targets,
                support_set_per_class_code=support_set_per_class_code,
                cls_logits_kernel=cls_logits_kernel,
            )

            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.fcos_outputs.predict_proposals(
                        logits_pred,
                        reg_pred,
                        ctrness_pred,
                        iou_pred,
                        locations,
                        images.image_sizes,
                        top_feats,
                    )
            return results, losses
        else:
            results = self.fcos_outputs.predict_proposals(
                logits_pred,
                reg_pred,
                ctrness_pred,
                iou_pred,
                locations,
                images.image_sizes,
                top_feats,
            )

            return results, {}

    def compute_locations(self, features):
        """
        For each location(x,y)on the feature map Fi, we can map it back onto the input image as(⌊s/2⌋+xs,⌊s/2⌋+ys)
        locations: it has dimension [h*w, 2]
        """
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level], feature.device
            )
            locations.append(locations_per_level)
        return locations


class MetaFCOSHead(BaseFewShotDetectorHead):
    """
    Define Meta-FCOS heads,  it initalize the base detector for training on base detector
    and it also prepare "class-conditional conv" for the meta-learning stage.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)

    def _evaluate_with_base_class_codes(self, eval_dataset_name: str):
        if "all" not in eval_dataset_name:
            print(f"tried to eval with base class codes, but the dataset split is not 'all'")
            return None, None
        if self.base_weight is not None and self.base_bias is not None:
            print(f"evaluating with base weights.")
            base_dataset_name = self.cfg.DATASETS.BASE_CLASSES_SPLIT
            novel_dataset_name = self.cfg.DATASETS.NOVEL_CLASSES_SPLIT
            if base_dataset_name == '' or novel_dataset_name == '':
                print(f"evaluating with base weights.")
                return None, None
            assert base_dataset_name != '', "base data set name is not provided"
            assert novel_dataset_name != '', "novel data set name is not provided"
            # base_classes = MetadataCatalog.get(base_dataset_name).thing_classes
            base_dataset_id_to_continous_id = MetadataCatalog.get(base_dataset_name).thing_dataset_id_to_contiguous_id
            # training data classes
            novel_dataset_id_to_continous_id = MetadataCatalog.get(novel_dataset_name).thing_dataset_id_to_contiguous_id
            # combine base class codes with current class_codes

            current_dataset_id_to_continous_id=MetadataCatalog.get(eval_dataset_name).thing_dataset_id_to_contiguous_id
            N = len(current_dataset_id_to_continous_id)
            assert self.base_weight.size(1) == self.cls_logits.weight.size(1)
            assert self.base_weight.size(2) == self.cls_logits.weight.size(2)
            assert self.base_weight.size(3) == self.cls_logits.weight.size(3)
            new_weights = torch.zeros(N,self.base_weight.size(1), self.base_weight.size(2), self.base_weight.size(3))
            new_bias = torch.zeros(N)
            # new_weights = self.cls_logits.weight.data
            # new_bias = self.cls_logits.bias.data
            # copy base classes
            for base_dataset_id, base_index in base_dataset_id_to_continous_id.items():
                if base_dataset_id in current_dataset_id_to_continous_id.keys():
                    current_index = current_dataset_id_to_continous_id[base_dataset_id]
                    new_weights[current_index] = self.base_weight[base_index]
                    new_bias[current_index] = self.base_bias[base_index]
            # copy novel classes, save the novels,
            for novel_dataset_id, novel_index in novel_dataset_id_to_continous_id.items():
                if novel_dataset_id in current_dataset_id_to_continous_id.keys():
                    current_index = current_dataset_id_to_continous_id[novel_dataset_id]
                    new_weights[current_index] = self.cls_logits.weight[novel_index]
                    new_bias[current_index] = self.cls_logits.bias[novel_index]
            new_weights = new_weights.to(self.cls_logits.weight.device)
            new_bias = new_bias.to(self.cls_logits.weight.device)
            print(f"final weights: {new_weights.size()}")
        return new_weights, new_bias


    def _preload_cls_logits_weights(self):
        use_pretrained_base_cls_logits = self.cfg.MODEL.TFA.USE_PRETRAINED_BASE_CLS_LOGITS
        # if not self.cfg.MODEL.TFA.FINETINE:
        #     return
        if not use_pretrained_base_cls_logits:
            return
        # weight_path = self.cfg.MODEL.WEIGHTS
        # print(f"load class logits from weights path: {weight_path}")
        # base_weight, base_bias = load_pretrained_class_conv_kernel_weights(weight_path)
        if self.base_weight is not None and self.base_bias is not None:
            base_dataset_name = self.cfg.DATASETS.BASE_CLASSES_SPLIT
            novel_dataset_name = self.cfg.DATASETS.NOVEL_CLASSES_SPLIT
            print(f"Loading pretrained weights on cls_logits")
            if base_dataset_name == '' or novel_dataset_name == '':
                return
            assert base_dataset_name != '', "base data set name is not provided"
            assert novel_dataset_name != '', "novel data set name is not provided"
            # base_classes = MetadataCatalog.get(base_dataset_name).thing_classes
            base_dataset_id_to_continous_id = MetadataCatalog.get(base_dataset_name).thing_dataset_id_to_contiguous_id
            # replace
            new_weights = self.cls_logits.weight.data
            new_bias = self.cls_logits.bias.data
            current_dataset_id_to_continous_id=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
            # replace base dataset id to current weights
            for base_dataset_id, base_index in base_dataset_id_to_continous_id.items():
                if base_dataset_id in current_dataset_id_to_continous_id.keys():
                    current_index = current_dataset_id_to_continous_id[base_dataset_id]
                    new_weights[current_index] = self.base_weight[base_index]
                    new_bias[current_index] = self.base_bias[base_index]
            with torch.no_grad():
                self.cls_logits.weight = nn.Parameter(new_weights)
                self.cls_logits.bias = nn.Parameter(new_bias)
                print(f"done Loading pretrained weights on cls_logits, number of classes: {self.cls_logits.weight.data.size(0)}")
        return

    def _init_base_detector_config(self, name: Optional[str] = None):
        """
        Extract useful FCOS config
        """
        self.num_classes = self.cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = self.cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {
            "cls": (
                self.cfg.MODEL.FCOS.NUM_CLS_CONVS,
                self.cfg.MODEL.FCOS.USE_DEFORMABLE,
            ),
            "bbox": (
                self.cfg.MODEL.FCOS.NUM_BOX_CONVS,
                self.cfg.MODEL.FCOS.USE_DEFORMABLE,
            ),
            "share": (self.cfg.MODEL.FCOS.NUM_SHARE_CONVS, False),
        }
        norm = None if self.cfg.MODEL.FCOS.NORM == "none" else self.cfg.MODEL.FCOS.NORM
        self.num_levels = len(self.input_shape)
        in_channels = [s.channels for s in self.input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels
        # Build all layers befor the predictor layer
        head_tower_sequential = _build_tower_module(
            head_configs, in_channels, norm, self.num_levels,
        )
        for head in head_configs.keys():
            self.add_module(f"{head}_tower", head_tower_sequential[head])
        # The original clssification head
        self.use_l2_norm_conv2d = self.cfg.MODEL.FCOS.L2_NORM_CLS_WEIGHT
        if self.use_l2_norm_conv2d: # will not be used
            from .head_utils import CosineSimilarityConv2d

            self.cls_logits = CosineSimilarityConv2d(
                in_channels, out_channel=self.num_classes
            )
        else:
            self.cls_logits = nn.Conv2d(
                in_channels,
                self.num_classes,
                kernel_size=self.cfg.MODEL.FCOS.CLS_LOGITS_KERNEL_SIZE,
                stride=1,
                padding=self.cfg.MODEL.FCOS.CLS_LOGITS_KERNEL_SIZE // 2,
            )  # default is 3 x 3

        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.ctrness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.iou_overlap = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        if self.cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)]
            )
        else:
            self.scales = None

        # Module initialization
        for modules in [
            self.cls_tower,
            self.bbox_tower,
            self.share_tower,
            self.cls_logits,
            self.bbox_pred,
            self.ctrness,
            self.iou_overlap
        ]:
            if modules is None:
                continue
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    print(f"init {layer} in {modules.modules()}")
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

        # initialize the bias for focal loss
        prior_prob = self.cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if not self.use_l2_norm_conv2d:
            torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        else:
            torch.nn.init.constant_(self.cls_logits.cls_layer.bias, bias_value)

        # TFA: load pretrained weights
               # print(f"load class logits from weights path: {weight_path}")
        # Always save it
        weight_path = self.cfg.MODEL.WEIGHTS
        print(f"weight path: {weight_path}")
        self.base_weight, self.base_bias = load_pretrained_class_conv_kernel_weights(weight_path)
        self._preload_cls_logits_weights()
        # save
        self.evaluate_with_base_class_codes = self.cfg.MODEL.TFA.EVAL_WITH_PRETRAINED_BASE_CLS_LOGITS
        print(f"eval with clss codes path: {self.evaluate_with_base_class_codes}")
        if self.evaluate_with_base_class_codes:
            self.eval_conv = CondConvBasic()


    def _init_code_generator_config(self, name: Optional[str]):
        """
        Will only be called while episodic learning is set to True
        """
        # all code generator has some basic configs

        if name == "CodeGenerator":
            # self.cls_logits will not be updated in episodic_learning.
            self.class_code_channel = (
                self.cfg.MODEL.META_LEARN.CODE_GENERATOR.OUT_CHANNEL
            )
            for param in self.cls_logits.parameters():
                param.requires_grad = False
            k_s = self.cfg.MODEL.META_LEARN.CODE_GENERATOR.CLS_LAYER[2]
            padding_size = k_s // 2
            # conditional classification head (takes in weight and bias)

            # self.cond_cls_logits = CondConvBlock(
            #     padding=padding_size, weight_len=self.class_code_channel
            # )

            self.cond_cls_logits = CondConvBasic(
                padding=padding_size,
                use_bias= self.cfg.MODEL.META_LEARN.CODE_GENERATOR.USE_BIAS
            )
            # self.score_scale = Scale(init_value=1.0)

            # CondConv(
            #     padding=padding_size,
            #     scale_score=True,
            #     l2_norm_weight=self.use_l2_norm_conv2d,
            #     scale_value=1.0,
            # )

        elif name == "ROIEncoder":
            self.class_code_channel = (
                self.cfg.MODEL.META_LEARN.CODE_GENERATOR.HEAD.OUTPUT_DIM
            )
            k_s = 1
            padding_size = 0
            # conditional classification head (takes in weight and bias)
            self.cond_cls_logits = CondConvBlock(
                padding=padding_size, weight_len=self.class_code_channel
            )
        else:
            raise NotImplementedError(f"{name} is not implemented")

    # def _build_base_detector(self, *args, **kwargs):
    #     # Build all layers befor the predictor layer
    #     self._build_tower_module(head_configs, in_channels, norm)

    # def _build_code_generator(self, *args, **kwargs):
    #     raise NotImplementedError()

    # def _build_merge_module(self, *args, **kwargs):
    #     raise NotImplementedError()

    #  forward
    def forward_base_train(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        ious = []
        top_feats = []
        bbox_towers = []
        for level, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)
            # TODO: only support the first evaluation dataset and it has to be split "all"
            # slow now because every single time, it has to redo the loading.
            if not self.training and self.evaluate_with_base_class_codes:
                new_weight, new_bias = self._evaluate_with_base_class_codes(self.cfg.DATASETS.TEST[0])
                if new_weight is not None:
                    logger.info(f"evaluating using new weight and new bias, in total {new_weight.size(0)}")
                    logits.append(self.eval_conv(feature=cls_tower, weight=new_weight, bias=new_bias))
                else:
                    logits.append(self.cls_logits(cls_tower))
            else:
                logits.append(self.cls_logits(cls_tower))
            reg = self.bbox_pred(bbox_tower)
            ctrness.append(self.ctrness(bbox_tower))
            ious.append(self.iou_overlap(bbox_tower))
            if self.scales is not None:
                reg = self.scales[level](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, ctrness, ious, top_feats, bbox_towers

    def forward(
        self,
        x,
        top_module=None,
        yield_bbox_towers=False,
        support_set_per_class_code=None,
    ):
        """
        support_set_per_class_code a tensor of dimension t*c, 256, 1, 1
        """
        if support_set_per_class_code is None:
            return self.forward_base_train(x, top_module, yield_bbox_towers)
        logits = []
        bbox_reg = []
        ctrness = []
        ious = []

        top_feats = []
        bbox_towers = []

        support_set_per_class_conv = support_set_per_class_code["cls_conv"]
        S = support_set_per_class_conv.size(0)
        bias = support_set_per_class_code["cls_bias"]
        box_weight = None
        # replace the base class codes
        #print(f"training stage: {self.training}, self.evaluate_with_base_class_codes: {self.evaluate_with_base_class_codes}")
        if (not self.training) and self.evaluate_with_base_class_codes:
            #print(f"replacing {self.training}")
            new_weights, new_bias = self._evaluate_with_base_class_codes(self.cfg.DATASETS.TEST[0])
            if new_weights is not None:
                print(f"evaluating using new weight and new bias, in total {new_weights.size(0)}")
                support_set_per_class_conv = new_weights
                bias = new_bias

        if "loc_conv" in support_set_per_class_code:
            box_weight = support_set_per_class_code["loc_conv"]
            S, C, H, W = box_weight.size()
            # assert H * W == 4
            box_weight = box_weight.view(S, int(C // 4), 4, 1, 1)
            box_weight = box_weight.permute(0, 2, 1, 3, 4)  # S, 4, C, 1, 1

        for level, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            # logit for pretraining and meta-finetuning differs only at dimension 1, it has a size t*c
            # support_set_per_class_code CONV cls_tower
            if self.episodic_learning and support_set_per_class_code is not None:
                logit = self.cond_cls_logits(
                    cls_tower, support_set_per_class_conv, bias
                )
                # logit = self.score_scale(logit)
            else:
                assert self.cls_logits is not None, "cls_logits is None"
                logit = self.cls_logits(cls_tower)

            # location reg
            # if self.box_on and box_weight is not None:
            #     reg_all_class = []
            #     for i in range(S):
            #         reg = self.bbox_pred(bbox_tower, box_weight[i])
            #         if self.scales is not None:
            #             reg = self.scales[level](reg)
            #         reg = F.relu(reg)
            #         reg_all_class.append(reg)
            #     reg_all_class = torch.cat(reg_all_class, dim=1)
            #     # Note that we use relu, as in the improved FCOS, instead of exp.
            #     bbox_reg.append(reg_all_class)  # a list of dim b, c X4, h, w
            # else:
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[level](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))  # a list of dim b, 4, h, w

            logits.append(logit)
            ctrness.append(self.ctrness(bbox_tower))
            ious.append(self.iou_overlap(bbox_tower))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, ctrness, ious, top_feats, bbox_towers


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {
            "cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS, cfg.MODEL.FCOS.USE_DEFORMABLE),
            "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS, cfg.MODEL.FCOS.USE_DEFORMABLE),
            "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS, False),
        }
        self.box_on = cfg.MODEL.META_LEARN.CODE_GENERATOR.BOX_ON
        logger.info(f"box_on: {self.box_on}")

        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        # Build all layers befor the predictor layer
        self._build_tower_module(head_configs, in_channels, norm)
        # Join predictor head with support set, Only replace class right now, no bias right now
        self.episodic_learning = cfg.MODEL.META_LEARN.EPISODIC_LEARNING

        # The original classification head will be loaded in meta-learning stage
        self.cls_logits = nn.Conv2d(
            in_channels,
            self.num_classes,
            kernel_size=cfg.MODEL.FCOS.CLS_LOGITS_KERNEL_SIZE,
            stride=1,
            padding=cfg.MODEL.FCOS.CLS_LOGITS_KERNEL_SIZE // 2,
        )  # default is 3 x 3
        if self.episodic_learning:
            self.class_code_channel = cfg.MODEL.META_LEARN.CODE_GENERATOR.OUT_CHANNEL
            # self.cls_logits will not be updated in episodic_learning.
            for param in self.cls_logits.parameters():
                param.requires_grad = False
            k_s = cfg.MODEL.META_LEARN.CODE_GENERATOR.CLS_LAYER[2]
            padding_size = k_s // 2
            # conditional classification head
            self.cond_cls_logits = CondConvBlock(
                padding=padding_size, weight_len=self.class_code_channel
            )

        if self.episodic_learning and self.box_on:
            self.bbox_pred = CondConvBlock()
        else:
            self.bbox_pred = nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1, padding=1
            )

        self.ctrness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)]
            )
        else:
            self.scales = None

        # Module initialization
        for modules in [
            self.cls_tower,
            self.bbox_tower,
            self.share_tower,
            self.cls_logits,
            self.bbox_pred,
            self.ctrness,
        ]:
            if modules is None:
                continue
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

                if cfg.MODEL.META_LEARN.CODE_GENERATOR.INIT_NORM_LAYER:
                    init_norm(layer)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def _build_tower_module(self, head_configs: Dict, in_channels: int, norm: str):
        for head, head_config in head_configs.items():
            num_convs, use_deformable = head_config
            tower = []
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                tower.append(
                    conv_func(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(
                        ModuleListDial(
                            [
                                nn.BatchNorm2d(in_channels)
                                for _ in range(self.num_levels)
                            ]
                        )
                    )
                elif norm == "SyncBN":
                    tower.append(
                        ModuleListDial(
                            [
                                NaiveSyncBatchNorm(in_channels)
                                for _ in range(self.num_levels)
                            ]
                        )
                    )
                tower.append(nn.ReLU())

            self.add_module(f"{head}_tower", nn.Sequential(*tower))

    def forward_base_train(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for level, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[level](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, ctrness, top_feats, bbox_towers

    def forward(
        self,
        x,
        top_module=None,
        yield_bbox_towers=False,
        support_set_per_class_code=None,
    ):
        """
        support_set_per_class_code a tensor of dimension t*c, 256, 1, 1
        """
        if support_set_per_class_code is None:
            return self.forward_base_train(x, top_module, yield_bbox_towers)
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []

        support_set_per_class_conv = support_set_per_class_code["cls_conv"]
        S = support_set_per_class_conv.size(0)
        bias = support_set_per_class_code["cls_bias"]
        box_weight = None

        if "loc_conv" in support_set_per_class_code:
            box_weight = support_set_per_class_code["loc_conv"]
            S, C, H, W = box_weight.size()
            # assert H * W == 4
            box_weight = box_weight.view(S, int(C // 4), 4, 1, 1)
            box_weight = box_weight.permute(0, 2, 1, 3, 4)  # S, 4, C, 1, 1

        for level, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            # logit for pretraining and meta-finetuning differs only at dimension 1, it has a size t*c
            # support_set_per_class_code CONV cls_tower
            if self.episodic_learning and support_set_per_class_code is not None:
                logit = self.cond_cls_logits(
                    cls_tower, support_set_per_class_conv, bias
                )
                # logit = self.score_scale
            else:
                assert self.cls_logits is not None, "cls_logits is None"
                logit = self.cls_logits(cls_tower)

            # location reg
            if self.box_on and box_weight is not None:
                reg_all_class = []
                for i in range(S):
                    reg = self.bbox_pred(bbox_tower, box_weight[i])
                    if self.scales is not None:
                        reg = self.scales[level](reg)
                    reg = F.relu(reg)
                    reg_all_class.append(reg)
                reg_all_class = torch.cat(reg_all_class, dim=1)
                # Note that we use relu, as in the improved FCOS, instead of exp.
                bbox_reg.append(reg_all_class)  # a list of dim b, c X4, h, w
            else:
                reg = self.bbox_pred(bbox_tower)
                if self.scales is not None:
                    reg = self.scales[level](reg)
                # Note that we use relu, as in the improved FCOS, instead of exp.
                bbox_reg.append(F.relu(reg))  # a list of dim b, 4, h, w

            logits.append(logit)
            ctrness.append(self.ctrness(bbox_tower))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, ctrness, top_feats, bbox_towers
