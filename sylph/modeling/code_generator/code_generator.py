"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

#!/usr/bin/env python3
import logging
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import Instances
from sylph.modeling.code_generator.build import CODE_GENERATOR_REGISTRY
from sylph.modeling.meta_fcos.head_utils import Scale
from sylph.modeling.utils import init_norm, build_fpn_norm

from .utils import (
    select_a_mask,
    GlobalAdaptiveAvgPool2d)
from detectron2.modeling.poolers import ROIPooler

logger = logging.getLogger(__name__)


def convert_box_to_mask(box: torch.tensor, fh: int, fw: int, stride: int):
    """
    Args:
        box: a box tensor
    Returns:
        mask: a tensor of dimension (1, 1, fh, fw)
    """
    mask = torch.zeros(1, 1, fh, fw)
    converted_box = box.cpu().numpy()
    converted_box = (converted_box - stride // 2) // stride
    x0, y0, x1, y1 = np.array(
        [
            np.clip(converted_box[0], 0, fw - 1),
            np.clip(converted_box[1], 0, fh - 1),
            np.clip(converted_box[2], 0, fw - 1),
            np.clip(converted_box[3], 0, fh - 1),
        ],
        dtype=np.int,
    )
    mask[:, :, y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


def convert_boxes_to_mask(boxes: List[torch.tensor], fh: int, fw: int, stride: int):
    """
    Args:
        boxes: a list of box tensor
    Returns:
        mask: a tensor of dimension (1, 1, fh, fw)
    """
    mask = torch.zeros(1, 1, fh, fw)
    for box in boxes:
        converted_box = box.cpu().numpy()
        converted_box = (converted_box - stride // 2) // stride
        x0, y0, x1, y1 = np.array(
            [
                np.clip(converted_box[0], 0, fw - 1),
                np.clip(converted_box[1], 0, fh - 1),
                np.clip(converted_box[2], 0, fw - 1),
                np.clip(converted_box[3], 0, fh - 1),
            ],
            dtype=np.int,
        )
        mask[:, :, y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


class GlobalAdaptiveMaxPool2d(nn.Module):
    """
    Used to synthesize bias
    """

    def __init__(self, k):
        super(GlobalAdaptiveMaxPool2d, self).__init__()
        self.k = k

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        x = F.adaptive_avg_pool2d(x, (self.k, self.k))
        x = F.max_pool2d(x, (self.k, self.k))
        x = x.view(N, C, 1, 1)
        return x


class GlobalPoolBlock(nn.Module):
    def __init__(self, k, c):
        super(GlobalPoolBlock, self).__init__()
        self.k = k  # pool kernel size
        self.c = c  # channel
        self.fc = nn.Linear(c * k * k, c)

    def forward(self, x):  # eventually get 3 x 3
        N = x.data.size(0)
        C = x.data.size(1)
        assert C == self.c
        x = F.adaptive_avg_pool2d(x, (self.k, self.k))
        x = x.flatten(start_dim=1)  # N, c*k*k
        x = self.fc(x)  # N, C
        x = x.view(N, C, 1, 1)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):  # eventually get 3 x 3
        N = x.data.size(0)
        C = x.data.size(1)
        H, W = x.data.size(2), x.data.size(3)
        # N, C
        x = F.avg_pool2d(x, (H, W)).view(N, C)
        return x


class SELayer(nn.Module):
    def __init__(self, channels: int, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# class FeatureFusionModule(nn.Module):
#     """
#     foreground binary mask and background binary mask will be passed to conv layer separately.
#     """

#     def __init__(self, in_channel, use_bkg_mask=True):
#         super(FeatureFusionModule, self).__init__()
#         self.use_bkg_mask = use_bkg_mask
#         self.in_channel = in_channel
#         # self_attention_fun = MS_CAM
#         # build foreground feature fusion
#         # self.foreground_attention = self_attention_fun(channels=in_channel)
#         self._add_module(
#             module_name="foreground_mask_module",
#             in_channel=in_channel,
#             out_channel=in_channel,
#             num_blocks=1,
#         )
#         num_scales = 1
#         if self.use_bkg_mask:
#             self._add_module(
#                 module_name="background_mask_module",
#                 in_channel=in_channel,
#                 out_channel=in_channel,
#                 num_blocks=1,
#             )
#             num_scales = 2
#             # self.background_attention = self_attention_fun(channels=in_channel)

#         # combine
#         self.merge_operator = "prod"
#         if self.merge_operator == "cat":  # cat does not work as well as prod
#             self._add_module(
#                 module_name="merge_module",
#                 in_channel=2 * in_channel,
#                 out_channel=in_channel,
#                 num_blocks=1,
#             )
#         elif self.merge_operator in ["prod", "add"]:
#             self._add_module(
#                 module_name="merge_module",
#                 in_channel=in_channel,
#                 out_channel=in_channel,
#                 num_blocks=1,
#             )
#             self.scales = [Scale(init_value=1.0) for i in range(num_scales)]
#             self.scales = nn.ModuleList(self.scales)
#         else:
#             raise NotImplementedError(
#                 f"operator {self.merge_operator} is not supported!"
#             )

#     def _add_module(
#         self, module_name: str, in_channel: int, out_channel: int, num_blocks: int
#     ):
#         conv_func = nn.Conv2d
#         tower = []
#         intermediate_channels = 128
#         for i in range(num_blocks):
#             if (
#                 i == 0 and i != num_blocks - 1
#             ):  # first layer of more than 1 layer structure
#                 input_channel = in_channel
#                 output_channel = intermediate_channels
#             elif i == num_blocks - 1 and i != 0:  # last layer
#                 input_channel = intermediate_channels
#                 output_channel = out_channel
#             elif i == 0 and i == num_blocks - 1:  # only one layer
#                 input_channel = in_channel
#                 output_channel = out_channel
#             else:
#                 input_channel = intermediate_channels
#                 output_channel = intermediate_channels
#             tower.append(
#                 conv_func(
#                     input_channel,
#                     output_channel,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                     bias=True,
#                 )
#             )
#             tower.append(build_fpn_norm(name="GN", num_channels=output_channel))
#         self.add_module(module_name, nn.Sequential(*tower))

#     def forward(self, x, feature: torch.Tensor = None):
#         assert feature is not None
#         assert feature.size(2) == x.size(2) and feature.size(3) == x.size(3)
#         if self.use_bkg_mask:
#             assert x.size(1) == 2
#         else:
#             assert x.size(1) == 1
#         foreground_mask = x[:, 0, :, :].unsqueeze(dim=1)
#         # foreground feature branch
#         if feature is not None:
#             foreground = foreground_mask.expand_as(feature) * feature
#             # foreground = self.foreground_attention(foreground)
#             foreground = self.foreground_mask_module(foreground)
#         # background feature branch
#         if self.use_bkg_mask:
#             bkg_mask = x[:, 1, :, :].unsqueeze(dim=1)
#             if feature is not None:
#                 bgkground = bkg_mask.expand_as(feature) * feature
#                 # bgkground = self.background_attention(bgkground)
#                 bgkground = self.background_mask_module(bgkground)
#         # combine
#         if self.merge_operator == "cat":
#             x = torch.cat((foreground, bgkground), dim=1)
#             return self.merge_module(x)
#         elif self.merge_operator == "add":
#             return (
#                 self.scales[0](foreground) + self.scales[1](bgkground)
#                 if self.use_bkg_mask
#                 else self.scales[0](foreground)
#             )
#         elif self.merge_operator == "prod":
#             return (
#                 self.merge_module(
#                     torch.mul(self.scales[0](foreground), self.scales[0](bgkground))
#                 )
#                 if self.use_bkg_mask
#                 else self.merge_module(self.scales[0](foreground))
#             )
#         else:
#             raise NotImplementedError(
#                 f"operator {self.merge_operator} is not supported!"
#             )


@CODE_GENERATOR_REGISTRY.register()
class CodeGeneratorHead(nn.Module):
    def __init__(
        self,
        cfg,
        feature_channels: int,  # input_shape: List[ShapeSpec],
        feature_levels: int,
        strides: Tuple[int],
    ):
        super().__init__()
        self.use_mask = cfg.MODEL.META_LEARN.CODE_GENERATOR.USE_MASK
        logger.info(f"use_mask: {self.use_mask}")
        self.in_channel, self.out_channel = (
            cfg.MODEL.META_LEARN.CODE_GENERATOR.IN_CHANNEL,
            cfg.MODEL.META_LEARN.CODE_GENERATOR.OUT_CHANNEL,
        )
        self.strides = strides
        self.all_mask = cfg.MODEL.META_LEARN.CODE_GENERATOR.ALL_MASK
        # self.use_bkg = cfg.MODEL.META_LEARN.CODE_GENERATOR.USE_BKG
        self.conv_l2_norm = cfg.MODEL.META_LEARN.CODE_GENERATOR.CONV_L2_NORM
        self.use_bias = cfg.MODEL.META_LEARN.CODE_GENERATOR.USE_BIAS
        self.use_deformable = cfg.MODEL.META_LEARN.CODE_GENERATOR.USE_DEFORMABLE
        self.cls_reweight = cfg.MODEL.META_LEARN.CODE_GENERATOR.CLS_REWEIGHT
        self.use_weight_scale = cfg.MODEL.META_LEARN.CODE_GENERATOR.USE_WEIGHT_SCALE
        # logger.info(f"use background mask: {self.use_bkg}")
        logger.info(f"CONV_L2_NORM: {self.conv_l2_norm}")
        logger.info(f"use_deformable: {self.use_deformable}")
        self.bias_l2_norm = cfg.MODEL.META_LEARN.CODE_GENERATOR.BIAS_L2_NORM

        self.mask_norm = cfg.MODEL.META_LEARN.CODE_GENERATOR.MASK_NORM
        self.mask_norm = None if self.mask_norm == "none" else self.mask_norm

        logger.info(f"mask_norm: {self.mask_norm}")

        tower_configs = {"shared": cfg.MODEL.META_LEARN.CODE_GENERATOR.TOWER_LAYERS}
        cls_head_configs = {
            "conv": cfg.MODEL.META_LEARN.CODE_GENERATOR.CLS_LAYER,
            "bias": cfg.MODEL.META_LEARN.CODE_GENERATOR.BIAS_LAYER,
            "weight": cfg.MODEL.META_LEARN.CODE_GENERATOR.WEIGHT_LAYER,
            "scale": cfg.MODEL.META_LEARN.CODE_GENERATOR.SCALE_LAYER,
        }
        logger.info(f"weight: {cls_head_configs['weight']}")
        self.num_levels = feature_levels

        self.feature_channels = feature_channels

        self.in_channels_to_top_module = feature_channels
        self.shot = cfg.MODEL.META_LEARN.SHOT
        self.eval_shot = cfg.MODEL.META_LEARN.EVAL_SHOT
        self.num_class = cfg.MODEL.META_LEARN.CLASS
        logger.info(f"Shot: {self.shot}, eval shot: {self.eval_shot}")

        # Build all layers befor the predictor layer
        self.init_norm = nn.ModuleList(
            [
                build_fpn_norm(name="GN", num_channels=feature_channels)
                for _ in range(feature_levels)
            ]
        )
        self._build_tower_module(tower_configs)

        # ROI Pooler
        self.multilevel_feature = (
            cfg.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.FPN_MULTILEVEL_FEATURE
        )

        roi_pool_output_resolution = cfg.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.POOLER_RESOLUTION
        roi_pool_type = cfg.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.POOLER_TYPE
        self.box_pooler = ROIPooler(
            output_size=roi_pool_output_resolution,
            scales=[1.0 / s for s in self.strides],
            sampling_ratio=0,
            pooler_type=roi_pool_type,
        )

        # # Mask module
        # self.mask_attention = None
        # if self.use_mask:
        #     # self._build_mask_module()
        #     self.mask_attention_module = None  # FeatureFusionModule(in_channel=self.feature_channels, use_bkg_mask=self.use_bkg)
        # else:
        #     self.mask_attention_module = None

        # Add a post norm layer after the mean on weights
        self.post_norm = None
        if cfg.MODEL.META_LEARN.CODE_GENERATOR.POST_NORM != "":
            self.post_norm = self._add_norm_layer(
                cfg.MODEL.META_LEARN.CODE_GENERATOR.POST_NORM, self.out_channel
            )
        # Final conv layer to predict dynamic filter
        self._build_predictor_head_conv(cls_head_configs["conv"])
        self._build_predictor_head_bias(cls_head_configs["bias"])
        self._build_predictor_head_weight(cls_head_configs["weight"])
        self._build_predictor_head_per_class_scale(cls_head_configs["scale"])
        # Scale will be used to compensate for the norm, it can be used om
        # 1: conv_weight * scale
        # 2: if weight_norm is on, it became conv_scale * weight_norm * conv_weight
        self.conv_scale = None
        if self.use_weight_scale and (self.conv_l2_norm or (self.post_norm is not None)):  # when scale is on, use predicted scale
            self.conv_scale = Scale(init_value=1.0)
        self.use_max = cfg.MODEL.META_LEARN.CODE_GENERATOR.COMPRESS_CODE_W_MAX

        if self.use_max:
            self.cls_mean_scale = Scale(init_value=0.5)
            self.cls_max_scale = Scale(init_value=0.5)

        # self.bias_scale = Scale(init_value=0.5)
        self.box_on = cfg.MODEL.META_LEARN.CODE_GENERATOR.BOX_ON
        self.contrastive_loss = cfg.MODEL.META_LEARN.CODE_GENERATOR.CONTRASTIVE_LOSS
        self.contrastive_loss_criterion = nn.CosineEmbeddingLoss()
        if self.box_on:
            loc_head_configs = {
                "conv": cfg.MODEL.META_LEARN.CODE_GENERATOR.BOX_CLS_LAYER,
            }
            self._build_predictor_head_conv(loc_head_configs["conv"], branch_type="loc")
        else:
            self.support_set_loc_conv = None

        # CLS reweight layer
        if self.cls_reweight:
            self._build_cls_reweight_layer()
        else:
            self.cls_reweight_module = None

        # Add initialization
        for modules in [
            # self.mask_attention_module,
            self.support_set_shared_tower,
            self.support_set_cls_conv,
            self.support_set_cls_bias,
            self.support_set_cls_weight,
            self.support_set_cls_scale,
            self.support_set_loc_conv,
            self.cls_reweight_module,
        ]:
            if modules is None:
                continue
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.normal_(m.weight, std=0.01)
                    torch.nn.init.constant_(m.bias, 0)
                    print(f"init {m} in module {modules}")

                if cfg.MODEL.META_LEARN.CODE_GENERATOR.INIT_NORM_LAYER:
                    init_norm(m)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        self.bias_value = torch.tensor(-math.log((1 - prior_prob) / prior_prob))
        if cfg.MODEL.META_LEARN.CODE_GENERATOR.META_BIAS:
            self.bias_value = nn.Parameter(self.bias_value)

        # use meta weight
        self.meta_weight = None
        if cfg.MODEL.META_LEARN.CODE_GENERATOR.META_WEIGHT:
            kernel_size = cls_head_configs["conv"][2]
            self.meta_weight = nn.Parameter(
                torch.Tensor(1, self.out_channel, kernel_size, kernel_size)
            )
            # follow the init of nn.Conv2d
            # will this normalization being stable during training?
            nn.init.kaiming_uniform_(self.meta_weight, a=math.sqrt(5))

        # add device
        self.device = torch.device(cfg.MODEL.DEVICE)


    def _add_norm_layer(self, norm: Optional[str], in_c=None):
        return build_fpn_norm(norm, self.num_levels, in_c)

    # def _build_mask_module(self):
    #     """
    #     It uses self.mask_norm
    #     """
    #     # expand mask into BX256X1X1
    #     mask_in_channel = 1
    #     if self.use_bkg:
    #         mask_in_channel = 2

    #     # for head in head_configs:
    #     conv_func = nn.Conv2d
    #     tower = []
    #     tower.append(
    #         conv_func(
    #             mask_in_channel,
    #             self.in_channel,
    #             kernel_size=3,
    #             stride=1,
    #             padding=1,
    #             bias=True,
    #         )
    #     )
    #     tower.append(self._add_norm_layer(self.mask_norm, self.in_channel))
    #     tower.append(nn.ReLU())
    #     self.add_module("mask_attention_module", nn.Sequential(*tower))

    # def _build_mask_module_v2(self):
    #     """
    #     foreground binary mask and background binary mask will be passed to conv layer separately.
    #     """
    #     # expand mask into BX256X1X1
    #     mask_in_channel = 1
    #     conv_func = nn.Conv2d
    #     tower = []
    #     tower.append(
    #         conv_func(
    #             mask_in_channel,
    #             self.in_channel,
    #             kernel_size=3,
    #             stride=1,
    #             padding=1,
    #             bias=True,
    #         )
    #     )
    #     if self.use_bkg:
    #         mask_in_channel = 2

    #     # for head in head_configs:
    #     conv_func = nn.Conv2d
    #     tower = []
    #     tower.append(
    #         conv_func(
    #             mask_in_channel,
    #             self.in_channel,
    #             kernel_size=3,
    #             stride=1,
    #             padding=1,
    #             bias=True,
    #         )
    #     )
    #     tower.append(self._add_norm_layer(self.mask_norm, self.in_channel))
    #     tower.append(nn.ReLU())
    #     self.add_module("mask_attention_module", nn.Sequential(*tower))

    def _build_predictor_head_conv(self, conv_config: List[str], branch_type="cls"):
        conv_func = nn.Conv2d
        tower = []
        if len(conv_config) == 0:
            if branch_type == "cls":
                self.support_set_cls_conv = None
            else:
                self.support_set_loc_conv = None
            return
        assert len(conv_config) == 3
        norm_type, relu, kernel_size = conv_config
        in_c, out_c = 256, self.out_channel
        tower.append(
            conv_func(
                in_c,
                out_c,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        norm_layer = self._add_norm_layer(norm_type, out_c)
        if norm_layer is not None:
            tower.append(norm_layer)
        if relu == "ReLU":
            tower.append(nn.ReLU())
        elif relu == "Tanh":
            tower.append(nn.Tanh())
        tower.append(
            GlobalAdaptiveAvgPool2d(k_s=kernel_size)
        )  # only support kernel size 1
        # add scale
        self.add_module(f"support_set_{branch_type}_conv", nn.Sequential(*tower))



    def _build_predictor_head_bias(self, bias_config: List[str]):
        """
        Produce the final bias
        """
        conv_func = nn.Conv2d
        tower = []
        if len(bias_config) == 0:
            self.support_set_cls_bias = None
            self.bias_scale = None
            return

        assert len(bias_config) == 3
        norm_type, relu, adaptive_avg_pool_size = bias_config
        in_c, out_c = 256, 1
        # now only set for the last layer
        tower.append(
            conv_func(
                in_c,
                out_c,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        norm_layer = self._add_norm_layer(norm_type, out_c)
        if norm_layer is not None:
            tower.append(norm_layer)

        if not self.bias_l2_norm:
            # Add pooling layers
            tower.append(GlobalAdaptiveAvgPool2d())
        else:
            self.independent_global_pool = GlobalAdaptiveAvgPool2d()
        self.add_module("support_set_cls_bias", nn.Sequential(*tower))
        self.bias_scale = Scale(init_value=1.0)

    def _build_predictor_head_weight(self, weight_config: List[str]):
        """
        Produce the final bias
        """
        conv_func = nn.Conv2d
        tower = []
        if len(weight_config) == 0:
            self.support_set_cls_weight= None
            return

        assert len(weight_config) == 3
        norm_type, relu, adaptive_avg_pool_size = weight_config
        in_c, out_c = 256, 1
        # now only set for the last layer
        tower.append(
            conv_func(
                in_c,
                out_c,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        norm_layer = self._add_norm_layer(norm_type, out_c)
        if norm_layer is not None:
            tower.append(norm_layer)

        # Add pooling layers
        tower.append(GlobalAdaptiveAvgPool2d())
        self.add_module("support_set_cls_weight", nn.Sequential(*tower))

    def _build_predictor_head_per_class_scale(self,scale_config: List[str]):
        """
        Produce the final bias
        """
        conv_func = nn.Conv2d
        tower = []
        if len(scale_config) == 0:
            self.support_set_cls_scale= None
            return

        assert len(scale_config) == 3
        norm_type, relu, adaptive_avg_pool_size = scale_config
        in_c, out_c = 256, 1
        # now only set for the last layer
        tower.append(
            conv_func(
                in_c,
                out_c,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        norm_layer = self._add_norm_layer(norm_type, out_c)
        if norm_layer is not None:
            tower.append(norm_layer)

        # Add pooling layers
        tower.append(GlobalAdaptiveAvgPool2d())
        self.add_module("support_set_cls_scale", nn.Sequential(*tower))


    def _build_tower_module(self, head_configs: Dict):
        """
        Build support set shared tower module, use channel 256
        """
        support_types = ["shared"]
        conv_func = nn.Conv2d
        for head in head_configs:
            assert head in support_types, f"head {head} is not supported"
            tower = []
            head_config = head_configs[head]
            num_layers = len(head_config)
            if num_layers == 0:
                if head == "shared":
                    self.support_set_shared_tower = None
                continue
            # set middle layer dimension
            in_c, out_c = 256, 256
            # Add conv layers
            # # add a 1X1 compress layer, input is feature+ mask
            for i in range(num_layers):
                norm_type, relu = head_config[i]
                tower.append(
                    conv_func(
                        in_c,
                        out_c,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
                # tower.append(nn.Dropout(p=0.3))
                norm_layer = self._add_norm_layer(norm_type, out_c)
                if norm_layer is not None:
                    tower.append(norm_layer)
                if relu == "ReLU":
                    tower.append(nn.ReLU())
                elif relu == "Tanh":
                    tower.append(nn.Tanh())

            self.add_module(f"support_set_{head}_tower", nn.Sequential(*tower))

    # def get_batched_bit_masks(
    #     self, gt_instances: List[Instances], stride: int, fh: int, fw: int
    # ):
    #     # gt instances for support set
    #     masks = []
    #     backgrounds = []
    #     for per_gt_instances in gt_instances:
    #         # Position in the initial image
    #         boxes = copy.deepcopy(
    #             per_gt_instances.gt_boxes
    #         )  # this could be in transform
    #         mask = torch.zeros(1, 1, fh, fw)
    #         for box in boxes.tensor:
    #             box = box.cpu().numpy()
    #             box = (box - stride // 2) // stride
    #             x0, y0, x1, y1 = np.array(
    #                 [
    #                     np.clip(box[0], 0, fw - 1),
    #                     np.clip(box[1], 0, fh - 1),
    #                     np.clip(box[2], 0, fw - 1),
    #                     np.clip(box[3], 0, fh - 1),
    #                 ],
    #                 dtype=np.int,
    #             )
    #             mask[:, :, y0 : y1 + 1, x0 : x1 + 1] = 1
    #             if not self.all_mask:
    #                 break
    #         if len(boxes.tensor) == 0:
    #             logger.info("Run into empty box, use zero masks")
    #             logger.info(boxes.tensor)
    #         masks.append(mask)
    #         if self.use_bkg:
    #             background = torch.ones(1, 1, fh, fw) - mask
    #             backgrounds.append(background)
    #     # shape (B, 1, fh, fw)
    #     masks = torch.cat(masks, dim=0)
    #     if self.use_bkg:
    #         # shape (B, 1, fh, fw)
    #         backgrounds = torch.cat(backgrounds, dim=0)
    #         return torch.cat([masks, backgrounds], dim=1)  # B, 2, fh, fw
    #     else:
    #         return torch.cat([masks], dim=1)  # B, 1, fh, fw

    # def get_batched_bit_masks(
    #     self, gt_instances: List[Instances], stride: int, fh: int, fw: int
    # ):
    #     masks = []
    #     backgrounds = []
    #     for per_gt_instances in gt_instances:  # for each image
    #         boxes_tensor = per_gt_instances.gt_boxes.tensor
    #         # check empty boxes
    #         if len(boxes_tensor) == 0:
    #             logger.info("Run into empty box, use zero masks")
    #             logger.info(boxes_tensor)
    #         else:
    #             if not self.all_mask:
    #                 # select a single box randomly
    #                 rand_indexes = np.random.choice(range(len(boxes_tensor)), 1)
    #                 box = boxes_tensor[rand_indexes[0]]
    #                 mask = convert_box_to_mask(box, fh, fw, stride)
    #             else:
    #                 # put all boxes into a single binary map
    #                 mask = convert_boxes_to_mask(boxes_tensor, fh, fw, stride)
    #             masks.append(mask)
    #         if self.use_bkg:
    #             background = torch.ones(1, 1, fh, fw) - mask
    #             backgrounds.append(background)
    #     # shape (B, 1, fh, fw)
    #     masks = torch.cat(masks, dim=0)
    #     if self.use_bkg:
    #         # shape (B, 1, fh, fw)
    #         backgrounds = torch.cat(backgrounds, dim=0)
    #         return torch.cat([masks, backgrounds], dim=1)  # B, 2, fh, fw
    #     else:
    #         return torch.cat([masks], dim=1)  # B, 1, fh, fw

    def process_weight(self, weight_logits: torch.tensor=None):
        if weight_logits is None:
            return None
        if self.training:
            num_shot = self.shot
        else:
            num_shot = weight_logits.size(0)
        weight_logits = weight_logits.view(-1, num_shot, 1, 1, 1) # 2 classes
        m = torch.nn.Softmax(dim=1)
        weight = m(weight_logits)
        return weight

    def compute_code(self, code_feature: List[torch.Tensor], weight: torch.tensor = None, is_bias: bool = False):
        """
        Given a list of features from a single layer, compute the per-layer class code from a support set
        features:  a list of features, one for each level
        Return:
            code_feature: a torch tensor of dimension (N, C, k_s, k_s)  (N way)
        """
        if self.training:
            num_shot = self.shot
            assert (
                code_feature.data.size(0) % num_shot == 0
            ), f"Total size {code_feature.data.size(0)} must be divisible by number of shot {self.shot}"
        else:
            # num_shot = self.eval_shot
            # ensure the test input batch size is 1
            num_shot = code_feature.size(0)
            # assert code_feature.size(0) == 1, "test batch size is not 1"

        # code feature: (bs*num_shot, C, 1, 1)
        code_feature = code_feature.view(
            -1,
            num_shot,
            code_feature.size(1),
            code_feature.size(2),
            code_feature.size(3),
        )

        if weight is None:
            weight = torch.empty(code_feature.size(0), num_shot, 1, 1, 1).fill_(1.0/num_shot).to(code_feature.device)
        else:
            assert code_feature.size(0) == weight.size(0)
            assert code_feature.size(1) == weight.size(1)

        # (bs, C, 1, 1)
        if self.use_max:
            code_feature = self.cls_mean_scale(
                code_feature.mean(1)
            ) + self.cls_max_scale(code_feature.max(1))
        else:
            # code_feature = code_feature.mean(1)
            code_feature = (weight * code_feature).sum(dim=1)
            # logger.info(f"code_feature dim: {code_feature.size()}")
        # Normalize the kernel again to get better performance
        # TODO: train new versions
        # if not is_bias and self.training:
        # if self.training:
        #     code_feature = self.normalize_code(code=code_feature, is_bias=is_bias)
            # if is_bias:
            #     code_feature = self.process_bias(code_feature)
            #     logger.info(f"training bias: {code_feature}")

        return code_feature


    def normalize_code(self, code: torch.tensor, is_bias: bool = False):
        if is_bias:
            return code
        # TODO: assert the dimension, and change is bias
        # (B, C, 1, 1)
        assert code.ndim == 4
        if (self.post_norm is not None) and (code.size(1) % 32 == 0):
            code = self.post_norm(code)
        # normalize them to length one vector
        if self.conv_l2_norm:
            code = torch.nn.functional.normalize(code, p=2, dim=1)
        return code

    def process_bias(self, predicted_bias: torch.tensor=None):
        """
        predicted_bias: either all zeros or output from bias layer, and then add to the initial bias
        """
        # assert predicted_bias.ndim == 4
        if not self.training:
            assert predicted_bias.size(0) == 1, "predicted bias should only have batch size 1"
        # Scale
        bias = predicted_bias.view(predicted_bias.numel())
        if self.bias_scale is not None:
            bias = self.bias_scale(bias)
        initial_bias = self.bias_value.expand_as(bias).to(
                    predicted_bias.device
                )
        # Add initial value
        bias = bias + initial_bias
        # logger.info(f"final bias: {bias}")
        return bias

    def code_process_module(self, conv_weight: torch.Tensor, conv_bias: torch.Tensor, conv_weight_norm: torch.Tensor = None):
        conv_weight = conv_weight.to(self.device)
        conv_bias = conv_bias.to(self.device)

        weight = self.normalize_code(conv_weight, is_bias=False)
        if conv_weight_norm is not None:
            conv_weight_norm = conv_weight_norm.to(self.device)
            # conv_weight_norm = self.conv_scale(conv_weight_norm) if self.conv_scale is not None else conv_weight_norm
            weight = weight * conv_weight_norm
        weight = self.conv_scale(weight) if self.conv_scale is not None else weight
        bias = self.process_bias(predicted_bias=conv_bias)
        return weight, bias

    def forward_normalize_code(self, codes: List[Dict]):
        assert not self.training, "Only support testing mode here"
        assert codes is not None
        if len(codes) == 0:
            return codes
        # conv_size = codes[0]["class_code"]["cls_conv"].size()
        # bias_size = codes[0]["class_code"]["cls_bias"].size()
        # logger.info(f"before norm dimension: {conv_size}, {bias_size}")
        for code in codes:
            assert "class_code" in code, "class_code is not in code"
            assert "cls_conv" in code["class_code"], "class_conv is not in class_code"
            if "cls_weight_norm" in code["class_code"]:
                conv_weight_norm=code["class_code"]["cls_weight_norm"]
            else:
                conv_weight_norm = None

            code["class_code"]["cls_conv"], code["class_code"]["cls_bias"] = self.code_process_module(conv_weight=code["class_code"]["cls_conv"], conv_bias=code["class_code"]["cls_bias"], \
                conv_weight_norm=conv_weight_norm)
            # logger.info(f"normalize and scaled code: {code}")
        # print(f"normalized code: {codes}")
        return codes


    def compute_soft_nearest_neighbor_loss(self, code_feature: torch.Tensor):
        """
        For a code feature of size (N, C)
        """
        N = code_feature.size(0)
        code_feature = code_feature.view(N, -1)
        if self.training:
            num_shot = self.shot
        else:
            num_shot = self.eval_shot
        # randomly sample target
        # samples = np.random.choice(N, N, replace=True)
        # targets = torch.empty(N).fill_(-1.0).to(code_feature.device)
        # pairs = []
        # for i, j in enumerate(samples):
        #     if i // num_shot == j // num_shot:
        #         targets[i] = 1.0
        #     pairs.append(code_feature[j, :])
        # pairs = torch.stack(pairs, dim=0).to(code_feature.device)
        # return self.contrastive_loss_criterion(code_feature, pairs, targets)
        from .utils import SoftNearestNeighborLoss

        return SoftNearestNeighborLoss(code_feature, num_shot)

    def forward_roi_align(
        self, features: List[torch.tensor], gt_instances: List[Instances] = None
    ):
        total_shots = features[0].size(0)
        box_ls = select_a_mask(gt_instances, use_all_masks=self.all_mask)
        # Step 1: roi align to extract (N, C, 7, 7) features
        features = self.box_pooler(features, box_ls)
        if isinstance(features, list):
            tmp_feature = features[0]
        else:
            tmp_feature = features
        assert (
            tmp_feature.shape[0] == total_shots
        ), f"pooled_features.shape[0] {tmp_feature.shape[0]} Vs batch_size * num_shots {total_shots}"

        # Step 2: pass to feature refinement
        # Feature has size b*c*s
        if self.support_set_shared_tower is not None:
            features = (
                [self.support_set_shared_tower(x) for x in features]
                if self.multilevel_feature
                else self.support_set_shared_tower(features)
            )

        # if multi-level features
        if self.multilevel_feature:
            features = torch.mean(torch.stack(features, dim=0), dim=0).squeeze(dim=0)

        # Step 3: obtain per-sample predicted conv weight and bias
        # (N, C, 1, 1)
        conv_feature = self.support_set_cls_conv(features)
        # (N, 1, 1, 1)
        bias_feature = (
            self.support_set_cls_bias(features)
            if self.support_set_cls_bias is not None
            else None
        )
        # apply L2
        if self.bias_l2_norm and bias_feature is not None:
            bf_size = bias_feature.size()
            bias_feature = bias_feature.view(bf_size[0], bf_size[1], -1) # normalize along the whole feature
            bias_feature = torch.nn.functional.normalize(bias_feature, p=2, dim=2)
            bias_feature = bias_feature.view(bf_size)
            bias_feature = self.independent_global_pool(bias_feature)
        # weight, this is not conv weight but to weight on different examples in the support set
        weight = (
            self.support_set_cls_weight(features)
            if self.support_set_cls_weight is not None
            else None
        )
        # weight_norm
        weight_norm = (
            self.support_set_cls_scale(features) if self.support_set_cls_scale is not None else None
        )
        weight = self.process_weight(weight)
        # Step 4: obtain per-class conv weigh
        conv_weights = self.compute_code(conv_feature, weight=weight)

        nearest_neighbor_loss = self.compute_soft_nearest_neighbor_loss(conv_feature)
        n_class = conv_weights.size(0)
        conv_bias = torch.zeros(n_class, 1, 1, 1).to(conv_weights.device)
        # Obtain predicted per-class bias
        if bias_feature is not None:
            conv_bias = self.compute_code(bias_feature, weight=weight)
        conv_weight_norm = None
        if weight_norm is not None:
            conv_weight_norm = self.compute_code(weight_norm, weight=weight)

        # predicted class code process
        if self.training:
            conv_weights, conv_bias = self.code_process_module(conv_weight=conv_weights, conv_bias=conv_bias, conv_weight_norm=conv_weight_norm)
            # logger.info(f"training bias: {conv_bias}, training conv weights : {conv_weights}")

        outputs = {"cls_conv": conv_weights, "cls_bias": conv_bias}
        if conv_weight_norm is not None:
            outputs.update({"cls_weight_norm": conv_weight_norm })
        if self.contrastive_loss == "snnl":
            outputs.update({"snnl": nearest_neighbor_loss})
        return outputs

    def forward(
        self, features: List[torch.tensor], gt_instances: List[Instances] = None, cls_norm: bool = False, class_codes: List[Dict] = None
    ):
        """
        features:  a list of features, one for each level
        Return:
            kernels: a list of kernels
        """
        # normalize the code after the code generation
        if cls_norm and class_codes is not None and not self.training:
            return self.forward_normalize_code(class_codes)
        # generate the code
        return self.forward_roi_align(features, gt_instances)


@CODE_GENERATOR_REGISTRY.register()
class CodeGenerator(nn.Module):
    def __init__(
        self, cfg, feature_channels: int, feature_levels: int, strides: Tuple[int]
    ):
        """
        Args:
            feature_channels: a fixed channel from the feature list
            feature_levels: the level of features to use
            strides: feature strides from FPN
        """
        super().__init__()
        # self.cfg = cfg
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.code_generator_head = CodeGeneratorHead(
            cfg, feature_channels, feature_levels, strides
        )

    def forward(
        self,
        features: List[torch.tensor],
        target_instances=None,
        cls_norm=False,
        class_codes=None,
    ):
        """
        Ensure the features are normalized, if not, add a GN layer first
        """
        if not self.training and cls_norm and class_codes is not None:
            print("code generator class code_norm in testing stage")
            return self.code_generator_head(
                features=None, cls_norm=cls_norm, class_codes=class_codes
            )
        codes = self.code_generator_head(features, target_instances)
        return codes
