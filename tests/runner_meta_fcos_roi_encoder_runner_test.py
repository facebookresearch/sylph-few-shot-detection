#!/usr/bin/env python3
import logging
import os
import unittest

import pkg_resources
import torch
from d2go.runner import create_runner
from detectron2.utils.logger import setup_logger
from sylph.runner.meta_fcos_roi_encoder_runner import MetaFCOSROIEncoderRunner  # noqa
from sylph.utils import create_cfg

logger = logging.getLogger(__name__)


def once_setup(config_file: str):
    config_file = pkg_resources.resource_filename(
        "sylph", os.path.join("configs", config_file)
    )

    logger.info(f"config_file {config_file}")

    runner = create_runner("sylph.runner.MetaFCOSROIEncoderRunner")
    default_cfg = runner.get_default_cfg()
    lvis_cfg = create_cfg(default_cfg, config_file, None)
    # register dataset
    runner.register(lvis_cfg)

    logger.info(f"cfg {lvis_cfg}")
    return runner, default_cfg


class TestMetaFCOS(unittest.TestCase):
    def setUp(self):
        setup_logger()

    def _do_pretrain(self):
        # Reset train iterations
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        # self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0

        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        # # # setup data loader
        if not self.default_cfg.MODEL.META_LEARN.EPISODIC_LEARNING:  # for pretraining
            data_loader = MetaFCOSROIEncoderRunner.build_detection_train_loader(
                self.default_cfg
            )
        else:
            data_loader = (
                MetaFCOSROIEncoderRunner.build_episodic_learning_detection_train_loader(
                    self.default_cfg
                )
            )
        # Remove the output directory
        if os.path.exists("./output"):
            import shutil

            shutil.rmtree("./output")
        self._data_loader_iter = iter(data_loader)

        # test run_step
        with torch.enable_grad():
            data = next(self._data_loader_iter)
            loss_dict = model(data)
            self.assertTrue(isinstance(loss_dict, dict))

    def _do_meta_learn_train(self):
        # Reset train iterations
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0

        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        # # # setup data loader
        if not self.default_cfg.MODEL.META_LEARN.EPISODIC_LEARNING:  # for pretraining
            data_loader = MetaFCOSROIEncoderRunner.build_detection_train_loader(
                self.default_cfg
            )
        else:
            data_loader = (
                MetaFCOSROIEncoderRunner.build_episodic_learning_detection_train_loader(
                    self.default_cfg
                )
            )
        # Remove the output directory
        if os.path.exists("./output"):
            import shutil

            shutil.rmtree("./output")
        self._data_loader_iter = iter(data_loader)

        # test run_step
        with torch.enable_grad():
            data = next(self._data_loader_iter)
            loss_dict = model(data)
            self.assertTrue(isinstance(loss_dict, dict))

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_lvis_pretrain(self):
        # TODO: replace this to default configs
        config_file = "LVISv1-Detection/Meta-FCOS/Meta-FCOS-pretrain.yaml"
        self.runner, self.default_cfg = once_setup(config_file)
        self._do_pretrain()

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_lvis_meta_learn(self):
        config_file = "LVISv1-Detection/Meta-FCOS/Meta-FCOS-ROI-Encoder-finetune.yaml"
        self.runner, self.default_cfg = once_setup(config_file)
        self._do_meta_learn_train()
