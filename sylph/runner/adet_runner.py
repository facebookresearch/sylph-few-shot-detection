import logging

import adet.config.defaults as adet_config_defaults
# from adet.config.defaults import add_adet_config
from sylph.runner.adet_configs import add_adet_config
from d2go.runner.default_runner import GeneralizedRCNNRunner


logger = logging.getLogger(__name__)


class AdelaiDetRunner(GeneralizedRCNNRunner):
    def __init__(self):
        logger.info("AdelaiDetRunner initialized. ")

    def get_default_cfg(self):
        _C = super().get_default_cfg()
        # Overwrite CfgNode in adet so that we can successfully merge config from config files.
        adet_config_defaults.CN = type(_C)

        add_adet_config(_C)
        return _C
