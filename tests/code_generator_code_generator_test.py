#!/usr/bin/env python3
import logging
import os
import unittest

import pkg_resources
import torch
from d2go.runner import create_runner
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from sylph.runner.meta_fcos_runner import MetaFCOSRunner  # noqa
from sylph.utils import create_cfg
from detectron2.config import set_global_cfg


logger = logging.getLogger(__name__)


def once_setup():
    config_file = "LVISv1-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml"
    config_file = pkg_resources.resource_filename(
        "sylph", os.path.join("configs", config_file)
    )

    logger.info(f"config_file {config_file}")

    runner = create_runner("sylph.runner.MetaFCOSRunner")
    default_cfg = runner.get_default_cfg()
    cfg = create_cfg(default_cfg, config_file, None)
    # Reset train iterations
    cfg.SOLVER.MAX_ITER = 1
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    # cfg.MODEL.FPN.MULTILEVEL_ROIS = False
    cfg.TEST.EVAL_PERIOD = 0  # do not test
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.META_LEARN.SHOT = 2
    cfg.MODEL.META_LEARN.EVAL_SHOT = 2
    # cfg.MODEL.META_LEARN.ALL_MASK = True
    cfg.DATALOADER.NUM_WORKERS = 4
    # test kernel_size
    cfg.MODEL.META_LEARN.CODE_GENERATOR.CLS_LAYER = ["GN", "", 1]
    cfg.MODEL.META_LEARN.CODE_GENERATOR.OUT_CHANNEL = 512

    logger.info(f"cfg {cfg}")
    return runner, cfg


class TestCodeGenerator(unittest.TestCase):
    def setUp(self):
        setup_logger()
        self.runner, self.cfg = once_setup()
        set_global_cfg(self.cfg)

    def test_code_generator_inference(self):
        dataset_name = "lvis_meta_val_novelv1"
        data_loader = (
            MetaFCOSRunner.build_episodic_learning_detection_test_support_set_loader(
                self.cfg, dataset_name, 0
            )
        )
        # Remove the output directory
        if os.path.exists("./output"):
            import shutil

            shutil.rmtree("./output")

        data_loader_iter = iter(data_loader)

        arch_model = build_model(self.cfg)
        arch_model.eval()

        # forward class code
        num_iterators = 1
        kernel_size = self.cfg.MODEL.META_LEARN.CODE_GENERATOR.CLS_LAYER[-1]
        class_kernel_size = (
            1,
            self.cfg.MODEL.META_LEARN.CODE_GENERATOR.OUT_CHANNEL,
            kernel_size,
            kernel_size,
        )
        for _ in range(num_iterators):
            batched_input = next(data_loader_iter)
            class_codes = arch_model(
                batched_input, run_type="meta_learn_test_support")
            keys = class_codes.keys()
            logger.info(keys)
            self.assertTrue("cls_conv" in keys)
            self.assertTrue("cls_bias" in keys)
            logger.info(f"cls bias size: {class_codes['cls_bias'].size()}")
            self.assertTrue(
                class_codes["cls_conv"].size() == class_kernel_size)
            self.assertTrue(
                class_codes["cls_bias"].size() == torch.Size((1, 1, 1, 1)))
