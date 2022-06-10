#!/usr/bin/env python3
import logging
import os
import unittest

import pkg_resources
import torch
from d2go.runner import create_runner
from detectron2.utils.logger import setup_logger
# from libfb.py import parutil
from sylph.runner.meta_fcos_runner import MetaFCOSRunner  # noqa
from sylph.utils import create_cfg
from detectron2.config import set_global_cfg

logger = logging.getLogger(__name__)


def once_setup():
    # set environment
    os.environ.get("SYLPH_TEST_MODE", default=True)
    config_file = "LVISv1-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml"
    config_file = pkg_resources.resource_filename(
        "sylph", os.path.join("configs", config_file)
    )
    logger.info(f"config_file {config_file}")

    # config_file = parutil.get_file_path(config_file)

    logger.info(f"config_file {config_file}")

    runner = create_runner("sylph.runner.MetaFCOSRunner")
    default_cfg = runner.get_default_cfg()
    cfg = create_cfg(default_cfg, config_file, None)
    # Reset train iterations
    cfg.SOLVER.MAX_ITER = 1
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.TEST.EVAL_PERIOD = 0  # do not test
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.META_LEARN.SHOT = 2
    cfg.MODEL.META_LEARN.EVAL_SHOT = 2
    cfg.DATALOADER.NUM_WORKERS = 1
    # register dataset
    runner.register(cfg)

    logger.info(f"cfg {cfg}")
    return runner, cfg


class TestFewShotDataLoader(unittest.TestCase):
    def setUp(self):
        setup_logger()
        self.runner, self.cfg = once_setup()
        set_global_cfg(self.cfg)

    # def test_few_shot_train_data_loader(self):
    #     # TODO: replace this to default configs
    #     data_loader = MetaFCOSRunner.build_episodic_learning_detection_train_loader(
    #         self.cfg
    #     )
    #     # Remove the output directory
    #     if os.path.exists("./output"):
    #         import shutil

    #         shutil.rmtree("./output")
    #     data_loader_iter = iter(data_loader)
    #     num_iters = 1
    #     for i in range(num_iters):
    #         if (i + 1) % 50 == 0:
    #             logger.info(f"i {i}")
    #         data_batch = next(data_loader_iter)
    #         logger.info(len(data_batch))
    #         self.assertEqual(len(data_batch), self.cfg.SOLVER.IMS_PER_BATCH)
    #         keys = data_batch[0].keys()
    #         logger.info(keys)
    #         self.assertTrue("support_set_target" in keys)
    #         self.assertTrue("support_set" in keys)
    #         self.assertTrue("query_set" in keys)
    #         logger.info(f"type: {type(data_batch[0]['support_set_target'])}")
    #         self.assertTrue(
    #             torch.is_tensor(data_batch[0]["support_set_target"]),
    #             "support set target is not torch.tensor type",
    #         )
    #         self.assertTrue(
    #             len(data_batch[0]["support_set"]) == self.cfg.MODEL.META_LEARN.SHOT,
    #             "support set length does not match",
    #         )
    #         self.assertTrue(
    #             len(data_batch[0]["query_set"]) == self.cfg.MODEL.META_LEARN.QUERY_SHOT,
    #             "query set length does not match",
    #         )

    def test_few_shot_test_support_set_base(self):
        dataset_name = "lvis_meta_val_all"
        # inference use all exampels to generate the code, 10 images at a time
        # set it to -1 if you want to test generating cls code using all shots
        self.cfg.MODEL.META_LEARN.BASE_EVAL_SHOT = 10
        self.cfg.MODEL.META_LEARN.USE_ALL_GTS_IN_BASE_CLASSES = True
        set_global_cfg(self.cfg)

        data_loader = (
            MetaFCOSRunner.build_episodic_learning_detection_test_support_set_base_loader(
                self.cfg, dataset_name
            )
        )

        # Remove the output directory
        if os.path.exists("./test_output"):
            import shutil

            shutil.rmtree("./test_output")
        data_loader_iter = iter(data_loader)
        num_iters = 1
        for i in range(num_iters):
            if (i + 1) % 50 == 0:
                logger.info(f"i {i}")
            data_batch = next(data_loader_iter)
            self.assertEqual(
                len(data_batch), 1, "test data loader only load one at a time"
            )
            keys = data_batch[0].keys()
            logger.info(
                f"keys: {keys}, len: {data_batch[0]['len']}, total_len: {data_batch[0]['total_len']}, total dataset dict: {len(data_loader)}")
            self.assertTrue("support_set_target" in keys)
            self.assertTrue("support_set" in keys)
            self.assertTrue("class_name" in keys)
            self.assertTrue(
                torch.is_tensor(data_batch[0]["support_set_target"]),
                "support set target is not torch.tensor type",
            )

    # def test_few_shot_test_support_set(self):
    #     dataset_name = "lvis_meta_val_novelv1"
    #     data_loader = (
    #         MetaFCOSRunner.build_episodic_learning_detection_test_support_set_loader(
    #             self.cfg, dataset_name, meta_test_seed=0
    #         )
    #     )
    #     # Remove the output directory
    #     if os.path.exists("./output"):
    #         import shutil

    #         shutil.rmtree("./output")
    #     data_loader_iter = iter(data_loader)
    #     num_iters = 1
    #     for i in range(num_iters):
    #         if (i + 1) % 50 == 0:
    #             logger.info(f"i {i}")
    #         data_batch = next(data_loader_iter)
    #         self.assertEqual(
    #             len(data_batch), 1, "test data loader only load one at a time"
    #         )
    #         keys = data_batch[0].keys()
    #         logger.info(f"keys: {keys}")
    #         self.assertTrue("support_set_target" in keys)
    #         self.assertTrue("support_set" in keys)
    #         self.assertTrue("class_name" in keys)
    #         self.assertTrue(
    #             torch.is_tensor(data_batch[0]["support_set_target"]),
    #             "support set target is not torch.tensor type",
    #         )
    #         self.assertTrue(
    #             len(data_batch[0]["support_set"])
    #             == self.cfg.MODEL.META_LEARN.EVAL_SHOT,
    #             "support set length does not match",
    #         )

    # def test_few_shot_test_query_set(self):
    #     dataset_name = "lvis_meta_val_novelv1"
    #     data_loader = (
    #         MetaFCOSRunner.build_episodic_learning_detection_test_query_loader(
    #             self.cfg, dataset_name, mapper=None
    #         )
    #     )
    #     # Remove the output directory
    #     if os.path.exists("./output"):
    #         import shutil

    #         shutil.rmtree("./output")
    #     data_loader_iter = iter(data_loader)
    #     num_iters = 1
    #     for i in range(num_iters):
    #         if (i + 1) % 50 == 0:
    #             logger.info(f"i {i}")
    #         data_batch = next(data_loader_iter)
    #         print(len(data_batch), data_batch)
    #         self.assertEqual(len(data_batch), 1)
    #         self.assertTrue(isinstance(data_batch[0], dict), "data has wrong data type")


if __name__ == "__main__":
    unittest.main()
