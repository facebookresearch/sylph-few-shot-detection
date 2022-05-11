#!/usr/bin/env python3
import logging

from sylph.config.config import CfgNode as CN
from sylph.runner.default_configs import (
    add_roi_encoder_config,
    add_base_config,
    add_fcos_config,
    add_default_meta_learn_config,
)
from sylph.runner.meta_fcos_runner import MetaFCOSRunner

logger = logging.getLogger(__name__)


class MetaFCOSROIEncoderRunner(MetaFCOSRunner):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.info("MetaFCOSROIEncoderRunner initialized. ")

    def get_default_cfg(self):
        _C = super().get_default_cfg()
        # Convert the config node to sylph library type
        _C = CN(_C)
        _C = add_base_config(_C)
        _C = add_fcos_config(_C)
        _C = add_default_meta_learn_config(_C)
        _C = add_roi_encoder_config(_C)
        return _C
