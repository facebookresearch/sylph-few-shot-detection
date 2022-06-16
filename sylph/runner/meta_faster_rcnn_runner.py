"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

#!/usr/bin/env python3
import logging

from sylph.config.config import CfgNode as CN
from sylph.runner.default_configs import (
    add_roi_encoder_config,
    add_base_config,
    add_default_meta_learn_config,
    add_code_genertor_config,
    add_customized_mask_rcnn_config,
)
from sylph.runner.meta_fcos_runner import MetaFCOSRunner

logger = logging.getLogger(__name__)


class MetaFasterRCNNRunner(MetaFCOSRunner):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.info("MetaFasterRCNNRunner initialized. ")

    def get_default_cfg(self):
        _C = super().get_default_cfg()  # includes faster rcnn
        # Convert the config node to sylph library type
        _C = CN(_C)
        _C = add_base_config(_C)
        # _C = add_fcos_config(_C)
        _C = add_customized_mask_rcnn_config(_C)
        _C = add_default_meta_learn_config(_C)
        # Config code generators to support
        _C = add_roi_encoder_config(_C)
        _C = add_code_genertor_config(_C)
        return _C
