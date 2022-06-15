#!/usr/bin/env python3
import logging
import os
import unittest
from typing import List

import pkg_resources
import torch
from d2go.runner import create_runner
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
# from libfb.py import parutil
from sylph.runner.tfa_runner import TFAFewShotDetectionRunner  # noqa
from sylph.utils import create_cfg
from detectron2.config import set_global_cfg, global_cfg

logger = logging.getLogger(__name__)


def once_setup(config_file: str):
    # config_file = "LVIS-Meta-FCOS-Detection/Meta_FCOS_MS_R_50_1x.yaml"
    config_file = pkg_resources.resource_filename(
        "sylph", os.path.join("configs", config_file)
    )

    logger.info(f"config_file {config_file}")

    runner = create_runner("sylph.runner.TFAFewShotDetectionRunner")
    default_cfg = runner.get_default_cfg()
    lvis_cfg = create_cfg(default_cfg, config_file, None)

    logger.info(f"cfg {lvis_cfg}")
    return runner, default_cfg


class TestTFAFewShotDetectionRunner(unittest.TestCase):
    def setUp(self):
        setup_logger()

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_coco_pretrain_test(self):
        # TODO: replace this to default configs
        config_file = "COCO-Detection/TFA/faster_rcnn_R_50_FPN_3x_base.yaml"
        self.runner, self.default_cfg = once_setup(config_file)
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0  # avoids broken pipe error

        model = self.runner.build_model(self.default_cfg)
        model.train(False)
        # # # setup data loader
        # if not self.default_cfg.MODEL.META_LEARN.EPISODIC_LEARNING:  # for pretraining
        dataset_name = self.default_cfg.DATASETS.TEST[0]
        data_loader = TFAFewShotDetectionRunner.build_detection_test_loader(
            self.default_cfg, dataset_name
        )
        # Remove the output directory
        if os.path.exists("./output"):
            import shutil

            shutil.rmtree("./output")
        self._data_loader_iter = iter(data_loader)

        # test run_step
        with EventStorage(start_iter=0):
            with torch.enable_grad():
                data = next(self._data_loader_iter)
                instances = model(data)
                logger.info(f"instances: {instances}")
                self.assertTrue(isinstance(instances, List))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_coco_pretrain(self):
        # TODO: replace this to default configs
        config_file = "COCO-Detection/TFAFCOS_pretrain.yaml"
        self.runner, self.default_cfg = once_setup(config_file)
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0  # avoids broken pipe error
        set_global_cfg(self.default_cfg)
        logger.info(f"global_cfg: {global_cfg}")

        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        # # # setup data loader
        data_loader = TFAFewShotDetectionRunner.build_detection_train_loader(
            self.default_cfg
        )
        # Remove the output directory
        if os.path.exists("./output"):
            import shutil

            shutil.rmtree("./output")
        self._data_loader_iter = iter(data_loader)

        # test run_step
        with EventStorage(start_iter=0):
            with torch.enable_grad():
                data = next(self._data_loader_iter)
                loss_dict = model(data)
                logger.info(f"loss_dict: {loss_dict}")
                self.assertTrue(isinstance(loss_dict, dict))

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_coco_finetune(self):
        # TODO: replace this to default configs
        config_file = "COCO-Detection/TFA/FCOS_finetune.yaml"
        self.runner, self.default_cfg = once_setup(config_file)
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.TRAIN_SHOT = 10
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0  # avoids broken pipe error
        set_global_cfg(self.default_cfg)

        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        # # # setup data loader
        data_loader = TFAFewShotDetectionRunner.build_detection_train_loader(
            self.default_cfg
        )
        # Remove the output directory
        if os.path.exists("./test_output"):
            import shutil

            shutil.rmtree("./test_output")
        self._data_loader_iter = iter(data_loader)

        # test run_step
        with EventStorage(start_iter=0):
            with torch.enable_grad():
                data = next(self._data_loader_iter)
                loss_dict = model(data)
                logger.info(f"loss_dict: {loss_dict}")
                self.assertTrue(isinstance(loss_dict, dict))
