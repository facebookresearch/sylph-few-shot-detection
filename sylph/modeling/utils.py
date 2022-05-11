#!/usr/bin/env python3

import logging
from typing import Optional

import torch.nn as nn
from adet.layers import NaiveGroupNorm
from detectron2.layers import NaiveSyncBatchNorm
from sylph.modeling.modules import ModuleListDial

logger = logging.getLogger(__name__)


def init_norm(m: nn.Module) -> None:
    if isinstance(
        m,
        (
            nn.GroupNorm,
            NaiveGroupNorm,
            nn.modules.batchnorm._NormBase,
            NaiveSyncBatchNorm,
        ),
    ):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)


def build_fpn_norm(
    name: Optional[str],
    num_levels: Optional[int] = None,
    num_channels: Optional[int] = None,
):
    if name is None or name == "none" or name == "":
        return None
    elif name == "GN":
        assert num_channels is not None
        return nn.GroupNorm(32, num_channels)
    elif name == "NaiveGN":
        assert num_channels is not None
        return NaiveGroupNorm(32, num_channels)
    elif name == "LN":
        assert num_channels is not None
        return nn.GroupNorm(1, num_channels)
    elif name == "BN":
        return ModuleListDial([nn.BatchNorm2d(num_channels) for _ in range(num_levels)])
    elif name == "SyncBN":
        return ModuleListDial(
            [NaiveSyncBatchNorm(num_channels) for _ in range(num_levels)]
        )
    elif name == "IN":
        # num_channels is the batch_size
        return nn.InstanceNorm2d(num_channels, affine=True)
    else:
        raise ValueError(f"unknown norm {name}")


def build_activation(name: Optional[str] = None, num_channels=None, **kwargs):
    inplace = kwargs.pop("inplace", True)
    if name is None or name == "none" or name == "":
        return None
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "relu6":
        return nn.ReLU6(inplace=inplace)
    if name == "leakyrelu":
        return nn.LeakyReLU(inplace=inplace, **kwargs)
    if name == "prelu":
        return nn.PReLU(num_parameters=num_channels, **kwargs)
    if name == "Tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"unknown activation {name}")
