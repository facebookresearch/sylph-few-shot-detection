"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

#!/usr/bin/env python3
# import logging
import math
from typing import List, Tuple, Optional

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.structures import Instances
from sylph.modeling.code_generator.build import CODE_GENERATOR_REGISTRY
from torch import nn

from .utils import select_a_mask, FeatureFusionModuleV2


class Tokenizer(nn.Sequential):
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        conv_dims: List[int],
        fc_dims: List[int],
        conv_norm="",
    ):
        super().__init__()

        assert len(fc_dims) > 0, "Empty fc dimensions! Expect at least one FC layer"

        output_size = (input_shape.channels, input_shape.height, input_shape.width)

        conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            conv_norm_relus.append(conv)
            output_size = (conv_dim, output_size[1], output_size[2])

        fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(output_size)), fc_dim)
            self.add_module(f"fc{k + 1}", fc)
            self.add_module(f"fc_relu{k + 1}", nn.ReLU())
            fcs.append(fc)
            output_size = fc_dim

        for layer in conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor with shape (N, C, H, W)
        """
        for layer in self:
            x = layer(x)
        # shape (N, C_out)
        return x


class HyperNetworkHead(nn.Sequential):
    def __init__(
        self,
        num_fc: int,
        fc_dim: int,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        assert num_fc > 0

        fcs = []
        dim_in = input_dim
        for i in range(num_fc):
            dim_out = output_dim if i == num_fc - 1 else fc_dim
            fc = nn.Linear(dim_in, dim_out)
            self.add_module(f"fc{i+1}", fc)
            if i < num_fc - 1:
                self.add_module(f"fc_relu{i+1}", nn.ReLU())
            fcs.append(fc)
            dim_in = dim_out

        for layer in fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor with shape (N, C)
        """
        for layer in self:
            x = layer(x)
        # shape (N, C_out)
        return x


@CODE_GENERATOR_REGISTRY.register()
class ROIEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        cfg,
        box_pooler: nn.Module,
        tokenizer: nn.Module,
        transformer_encoder: nn.Module,
        weight_head: nn.Module,
        bias_head: nn.Module,
    ):
        super().__init__()
        self.cfg = cfg
        self.box_pooler = box_pooler
        self.tokenizer = tokenizer
        self.transformer_encoder = transformer_encoder
        self.weight_head = weight_head
        self.bias_head = bias_head
        # initialize bias in the classification layer as in Focal Loss paper
        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)

        # configs for forward
        self.shot = cfg.MODEL.META_LEARN.SHOT
        self.eval_shot = cfg.MODEL.META_LEARN.EVAL_SHOT

    def forward(
        self,
        support_set_image_features: List[torch.Tensor],
        gt_instances: List[Instances],
    ):
        """
        Args:
            support_set_image_features: shape (bs * num_shots, C, H, W)
            support_imgs_ls: a list of lists. Each list denotes a support set image.
        """
        num_shots = self.shot if self.training else self.eval_shot
        # convert Instances to Boxes, one image select one single mask
        gt_instances = select_a_mask(gt_instances)

        assert (
            support_set_image_features[0].shape[0] % num_shots == 0
        ), f"{support_set_image_features[0].shape[0]} % {num_shots}"
        batch_size = support_set_image_features[0].shape[0] // num_shots
        total_shots = batch_size * num_shots
        # features: a list of num_levels tensors. Each has shape (bs * num_shots, C, H_l, W_l)
        for lvl, feature in enumerate(support_set_image_features):
            assert (
                feature.shape[0] == total_shots
            ), f"lvl {lvl}, {feature.shape[0]} vs {total_shots}"

        assert (
            len(gt_instances) == total_shots
        ), f"boxes_ls {len(gt_instances)} vs {total_shots}"

        # shape (bs * num_shots, C, pooler_resolution, pooler_resolution)
        box_features = self.box_pooler(support_set_image_features, gt_instances)

        assert (
            box_features.shape[0] == total_shots
        ), f"box_features.shape[0] {box_features.shape[0]} Vs batch_size * num_shots {batch_size * num_shots}"

        # shape (bs * num_shots, C_out)
        box_tokens = self.tokenizer(box_features)
        assert (
            box_tokens.shape[0] == total_shots
        ), f"box_tokens shape[0] {box_tokens.shape[0]} vs {total_shots}"

        # shape (bs, num_shots, C_out)
        box_tokens = box_tokens.view(-1, num_shots, box_tokens.shape[-1])

        box_tokens = self.transformer_encoder(box_tokens)
        # shape (bs, C_token)
        class_tokens = box_tokens.mean(1)
        # shape (bs, C_weight)
        class_weights = self.weight_head(class_tokens)
        class_weights = class_weights.view(
            class_weights.size(0), class_weights.size(1), 1, 1
        )
        # shape (bs, 1)
        class_delta_bias = self.bias_head(class_tokens)
        class_bias = self.bias_value + class_delta_bias
        class_bias = class_bias.view(-1)
        outputs = {"cls_conv": class_weights, "cls_bias": class_bias}
        return outputs

    @classmethod
    def from_config(
        cls,
        cfg,
        feature_channels: int,
        feature_levels: Optional[int],
        strides: Tuple[int],
    ):  # input_shape: List[ShapeSpec]
        """
        code_generator = ROIEncoder(cfg,feature_channels, feature_levels, strides)
        """
        # num_levels = len(input_shape) # multi-level for features

        # in_channels = [s.channels for s in input_shape]
        # assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        # channels = in_channels[0]

        box_pooler = FeatureFusionModuleV2(
            in_channel=feature_channels, strides=strides, context_attention=True
        )

        """ROIPooler(
            output_size=cfg.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.POOLER_RESOLUTION,
            scales=tuple(1.0 / s for s in strides),
            sampling_ratio=0,
            pooler_type=cfg.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.POOLER_TYPE,
        )"""

        tokenizer = Tokenizer(
            ShapeSpec(
                channels=feature_channels,
                height=cfg.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.POOLER_RESOLUTION,
                width=cfg.MODEL.META_LEARN.CODE_GENERATOR.ROI_BOX.POOLER_RESOLUTION,
            ),
            conv_dims=[cfg.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.CONV_DIM]
            * cfg.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.NUM_CONV,
            fc_dims=[cfg.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.FC_DIM]
            * cfg.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.NUM_FC,
            conv_norm=cfg.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.NORM,
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.FC_DIM,
            nhead=cfg.MODEL.META_LEARN.CODE_GENERATOR.TRANSFORMER_ENCODER.HEADS,
            dim_feedforward=cfg.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.FC_DIM * 4,
            dropout=cfg.MODEL.META_LEARN.CODE_GENERATOR.TRANSFORMER_ENCODER.DROPOUT,
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=cfg.MODEL.META_LEARN.CODE_GENERATOR.TRANSFORMER_ENCODER.LAYERS,
        )
        for p in transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        weight_head = HyperNetworkHead(
            cfg.MODEL.META_LEARN.CODE_GENERATOR.HEAD.NUM_FC,
            cfg.MODEL.META_LEARN.CODE_GENERATOR.HEAD.FC_DIM,
            cfg.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.FC_DIM,
            cfg.MODEL.META_LEARN.CODE_GENERATOR.HEAD.OUTPUT_DIM,
        )
        bias_head = HyperNetworkHead(
            cfg.MODEL.META_LEARN.CODE_GENERATOR.HEAD.NUM_FC,
            cfg.MODEL.META_LEARN.CODE_GENERATOR.HEAD.FC_DIM,
            cfg.MODEL.META_LEARN.CODE_GENERATOR.TOKENIZER.FC_DIM,
            1,
        )

        return {
            "cfg": cfg,
            "box_pooler": box_pooler,
            "tokenizer": tokenizer,
            "transformer_encoder": transformer_encoder,
            "weight_head": weight_head,
            "bias_head": bias_head,
        }
