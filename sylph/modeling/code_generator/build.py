# Copyright (c) Facebook, Inc. and its affiliates.
import logging

from detectron2.utils.registry import Registry
from typing import List, Tuple, Optional
from detectron2.layers import ShapeSpec


logger = logging.getLogger(__name__)

CODE_GENERATOR_REGISTRY = Registry("CODE_GENERATOR")
CODE_GENERATOR_REGISTRY.__doc__ = """
Registry for code generator, which produces per-class code from support set image feature maps and bounding box ground truth.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""
# from detectron2.modeling.proposal_generator import rpn, rrpn
# from . import rpn, rrpn  # noqa F401 isort:skip


def build_code_generator(cfg, feature_channels: int, feature_levels: Optional[int], strides: Tuple[int]):
    """
    Build a code generator from `cfg.MODEL.META_LEARN.CODE_GENERATOR.NAME`.
    Args:
        feature_channels: pyramid feature channel
        feature_levels: feature levels ouput from any feature extractor
        strides: the stride from input to feature shape
    """
    name = cfg.MODEL.META_LEARN.CODE_GENERATOR.NAME
    logger.info(f"Get code generator: {name}")
    return CODE_GENERATOR_REGISTRY.get(name)(cfg, feature_channels, feature_levels, strides)
