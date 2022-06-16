"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

#!/usr/bin/env python3
import logging
import os
import unittest
from typing import List, Dict

import pkg_resources
import torch
from d2go.runner import create_runner
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from sylph.runner.meta_faster_rcnn_runner import MetaFasterRCNNRunner  # noqa
from sylph.utils import create_cfg
from detectron2.config import set_global_cfg


logger = logging.getLogger(__name__)


def once_setup(config_file: str):
    # config_file = "LVIS-Meta-FCOS-Detection/Meta_FCOS_MS_R_50_1x.yaml"
    config_file = pkg_resources.resource_filename(
        "sylph", os.path.join("configs", config_file)
    )

    logger.info(f"config_file {config_file}")

    runner = create_runner("sylph.runner.MetaFasterRCNNRunner")
    default_cfg = runner.get_default_cfg()
    lvis_cfg = create_cfg(default_cfg, config_file, None)
    # register dataset
    runner.register(lvis_cfg)

    logger.info(f"cfg {lvis_cfg}")
    return runner, default_cfg


class TestMetaFasterRCNNRunner(unittest.TestCase):
    def setUp(self):
        setup_logger()

    def _do_meta_learn_train(self):
        # Reset train iterations
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0
        self.default_cfg.MODEL.META_LEARN.CODE_GENERATOR.OUT_CHANNEL = 1024

        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        # # # setup data loader
        if not self.default_cfg.MODEL.META_LEARN.EPISODIC_LEARNING:  # for pretraining
            data_loader = MetaFasterRCNNRunner.build_detection_train_loader(
                self.default_cfg
            )
        else:
            data_loader = (
                MetaFasterRCNNRunner.build_episodic_learning_detection_train_loader(
                    self.default_cfg
                )
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
                losses = model(data)
                logger.info(f"Loss: {losses} ")
                self.assertTrue(isinstance(losses, dict))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_lvis_pretrain_test(self):
        # TODO: replace this to default configs
        config_file = "LVISv1-Detection/Meta-RCNN/Meta-RCNN-FPN-pretrain.yaml"
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
        if not self.default_cfg.MODEL.META_LEARN.EPISODIC_LEARNING:  # for pretraining
            dataset_name = self.default_cfg.DATASETS.TEST[0]
            data_loader = MetaFasterRCNNRunner.build_detection_test_loader(
                self.default_cfg, dataset_name
            )
        else:
            data_loader = (
                MetaFasterRCNNRunner.build_episodic_learning_detection_train_loader(
                    self.default_cfg
                )
            )
        # Remove the output directory
        if os.path.exists("./output"):
            import shutil

            shutil.rmtree("./output")
        self._data_loader_iter = iter(data_loader)

        # test run_step
        from detectron2.utils.events import EventStorage

        with EventStorage(start_iter=0):
            with torch.enable_grad():
                data = next(self._data_loader_iter)
                instances = model(data)
                logger.info(f"instances: {instances}")
                self.assertTrue(isinstance(instances, List))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_lvis_pretrain(self):
        # TODO: replace this to default configs
        config_file = "LVISv1-Detection/Meta-RCNN/Meta-RCNN-FPN-pretrain.yaml"
        self.runner, self.default_cfg = once_setup(config_file)
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0  # avoids broken pipe error

        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        # # # setup data loader
        if not self.default_cfg.MODEL.META_LEARN.EPISODIC_LEARNING:  # for pretraining
            data_loader = MetaFasterRCNNRunner.build_detection_train_loader(
                self.default_cfg
            )
        else:
            data_loader = (
                MetaFasterRCNNRunner.build_episodic_learning_detection_train_loader(
                    self.default_cfg
                )
            )
        # Remove the output directory
        if os.path.exists("./output"):
            import shutil

            shutil.rmtree("./output")
        self._data_loader_iter = iter(data_loader)

        # test run_step
        from detectron2.utils.events import EventStorage

        with EventStorage(start_iter=0):
            with torch.enable_grad():
                data = next(self._data_loader_iter)
                loss_dict = model(data)
                logger.info(f"loss_dict: {loss_dict}")
                self.assertTrue(isinstance(loss_dict, dict))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_lvis_meta_learn(self):
        config_file = "LVISv1-Detection/Meta-RCNN/Meta-RCNN-FPN.yaml"
        self.runner, self.default_cfg = once_setup(config_file)
        set_global_cfg(self.default_cfg)
        self._do_meta_learn_train()

    def _setup_few_shot_configs(self):
        config_file = "LVISv1-Detection/Meta-RCNN/Meta-RCNN-FPN-finetune.yaml"
        self.runner, self.default_cfg = once_setup(config_file)
        # Reset train iterations
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0  # avoids broken pipe error
        self.default_cfg.MODEL.META_LEARN.CODE_GENERATOR.OUT_CHANNEL = 1024

    def _few_shot_test(
        self,
        data_loader_fun,
        *,
        class_code: Dict[str, torch.Tensor] = None,
        run_type: str = "meta_learn_test_support",
        **kwargs,
    ):
        model = self.runner.build_model(self.default_cfg)
        model.train(False)
        # # # setup data loader
        data_loader = data_loader_fun(**kwargs)
        # Remove the output directory
        if os.path.exists("./output"):
            import shutil

            shutil.rmtree("./output")
        self._data_loader_iter = iter(data_loader)
        # test run_step
        with EventStorage(start_iter=0):
            with torch.enable_grad():
                data = next(self._data_loader_iter)
                if run_type == "meta_learn_test_instance":
                    instances = model(
                        data, class_code=class_code, run_type=run_type)
                else:
                    instances = model(data, run_type=run_type)
                logger.info(f"instances: {instances} ")
                return instances

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_lvis_meta_learn_test_class_codes(self):
        self._setup_few_shot_configs()
        set_global_cfg(self.default_cfg)
        kwargs = {
            "cfg": self.default_cfg,
            "dataset_name": self.default_cfg.DATASETS.TEST[0],
            "meta_test_seed": 0,
        }
        dataloader_fun = (
            MetaFasterRCNNRunner.build_episodic_learning_detection_test_support_set_loader
        )
        instances = self._few_shot_test(
            data_loader_fun=dataloader_fun, run_type="meta_learn_test_support", **kwargs
        )
        self.assertTrue("cls_conv" in instances)
        self.assertTrue("cls_bias" in instances)

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_lvis_meta_learn_test_instances(self):
        self._setup_few_shot_configs()
        set_global_cfg(self.default_cfg)

        toy_class_codes = {
            "cls_conv": torch.rand(10, 1024, 1, 1),
            "cls_bias": torch.rand(10),
        }
        kwargs = {
            "cfg": self.default_cfg,
            "dataset_name": self.default_cfg.DATASETS.TEST[0],
        }
        dataloader_fun = (
            MetaFasterRCNNRunner.build_episodic_learning_detection_test_query_loader
        )
        instances = self._few_shot_test(
            data_loader_fun=dataloader_fun,
            class_code=toy_class_codes,
            run_type="meta_learn_test_instance",
            **kwargs,
        )
        self.assertTrue(isinstance(instances, List))
