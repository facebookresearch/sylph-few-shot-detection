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
from typing import Dict, List

import pkg_resources
import torch
from d2go.runner import create_runner
from detectron2.config import set_global_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from sylph.evaluation.meta_learn_evaluation import inference_on_support_set_dataset_base
from sylph.runner.meta_fcos_runner import MetaFCOSRunner  # noqa
from sylph.utils import create_cfg

logger = logging.getLogger(__name__)


def once_setup(config_file: str):
    # set environment
    os.environ.get("SYLPH_TEST_MODE", default=True)
    config_file = pkg_resources.resource_filename(
        "sylph", os.path.join("configs", config_file)
    )

    logger.info(f"config_file {config_file}")

    runner = create_runner("sylph.runner.MetaFCOSRunner")
    default_cfg = runner.get_default_cfg()
    lvis_cfg = create_cfg(default_cfg, config_file, None)

    logger.info(f"cfg {lvis_cfg}")
    return runner, default_cfg

# TODO: change this to support test on 
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
        self.default_cfg.DATALOADER.NUM_WORKERS = 0  # avoid broken pipe
        set_global_cfg(self.default_cfg)

        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        # # # setup data loader
        if not self.default_cfg.MODEL.META_LEARN.EPISODIC_LEARNING:  # for pretraining
            data_loader = MetaFCOSRunner.build_detection_train_loader(
                self.default_cfg)
        else:
            data_loader = MetaFCOSRunner.build_episodic_learning_detection_train_loader(
                self.default_cfg
            )
        # Remove the output directory
        if os.path.exists("./test_output"):
            import shutil

            shutil.rmtree("./test_output")
        self._data_loader_iter = iter(data_loader)

        # test run_step
        with torch.enable_grad():
            data = next(self._data_loader_iter)
            loss_dict = model(data)
            logger.info(loss_dict)
            self.assertTrue(isinstance(loss_dict, dict))

    def _do_meta_learn_evaluation(self):
        # Reset train iterations
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0
        set_global_cfg(self.default_cfg)

        model = self.runner.build_model(self.default_cfg)
        model.train(False)
        self.runner.do_test(self.default_cfg, model, train_iter=None)

    def _do_meta_learn_train(self):
        # Reset train iterations
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0
        set_global_cfg(self.default_cfg)

        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        # # # setup data loader
        if not self.default_cfg.MODEL.META_LEARN.EPISODIC_LEARNING:  # for pretraining
            data_loader = MetaFCOSRunner.build_detection_train_loader(
                self.default_cfg)
        else:
            data_loader = MetaFCOSRunner.build_episodic_learning_detection_train_loader(
                self.default_cfg
            )
        # Remove the output directory
        if os.path.exists("./test_output"):
            import shutil

            shutil.rmtree("./test_output")
        self._data_loader_iter = iter(data_loader)

        # test run_step
        with torch.enable_grad():
            data = next(self._data_loader_iter)
            loss_dict = model(data)
            logger.info(loss_dict)
            self.assertTrue(isinstance(loss_dict, dict))

    def _setup_few_shot_configs(self):
        config_file = "LVISv1-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml"
        self.runner, self.default_cfg = once_setup(config_file)
        # Reset train iterations
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0

    def _setup_few_shot_configs_2(self, config_file: str):
        self.runner, self.default_cfg = once_setup(config_file)
        # Reset train iterations
        self.default_cfg.SOLVER.MAX_ITER = 1
        if not torch.cuda.is_available():
            self.default_cfg.MODEL.DEVICE = "cpu"
        self.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        self.default_cfg.SOLVER.IMS_PER_BATCH = 2
        self.default_cfg.MODEL.META_LEARN.SHOT = 2
        self.default_cfg.DATALOADER.NUM_WORKERS = 0

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
        if os.path.exists("./test_output"):
            import shutil

            shutil.rmtree("./test_output")
        self._data_loader_iter = iter(data_loader)
        # test run_step
        with torch.enable_grad():
            data = next(self._data_loader_iter)
            if run_type == "meta_learn_test_instance":
                instances = model(
                    data, class_code=class_code, run_type=run_type)
            elif run_type == "meta_learn_test_support":
                instances = model(data, run_type=run_type)
            else:
                instances = model(data, run_type=run_type)
            logger.info(f"instances: {instances} ")
            return instances

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_lvis_meta_learn_test_instances(self):
        self._setup_few_shot_configs()
        toy_class_codes = {
            "cls_conv": torch.rand(10, 256, 1, 1),
            "cls_bias": torch.rand(10),
        }
        kwargs = {
            "cfg": self.default_cfg,
            "dataset_name": self.default_cfg.DATASETS.TEST[0],
        }
        set_global_cfg(self.default_cfg)

        dataloader_fun = (
            MetaFCOSRunner.build_episodic_learning_detection_test_query_loader
        )
        instances = self._few_shot_test(
            data_loader_fun=dataloader_fun,
            class_code=toy_class_codes,
            run_type="meta_learn_test_instance",
            **kwargs,
        )
        self.assertTrue(isinstance(instances, List))

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_runner_forward_normalize_codes(self):
        from sylph.evaluation.meta_learn_evaluation import inference_normalization
        self._setup_few_shot_configs()
        set_global_cfg(self.default_cfg)
        # generate fake class codes
        class_codes = []
        for _ in range(3):
            class_code = {'class_code': {'cls_conv': torch.rand(
                1, 256, 1, 1), 'cls_bias': torch.rand(1)}}
            class_codes.append(class_code)
        model = self.runner.build_model(self.default_cfg)
        model.train(False)
        class_codes = inference_normalization(model, class_codes)
        self.assertTrue(isinstance(class_codes, List))
        self.assertTrue(len(class_codes) == 3)
        logger.info(f" normalized class codes: {class_codes}")

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    # def test_runner_lvis_meta_learn_test_class_codes(self):
    #     self._setup_few_shot_configs()
    #     kwargs = {
    #         "cfg": self.default_cfg,
    #         "dataset_name": "lvis_meta_val_all",
    #     }
    #     set_global_cfg(self.default_cfg)

    #     dataloader_fun = (
    #         MetaFCOSRunner.build_episodic_learning_detection_test_support_set_loader
    #     )
    #     class_codes = self._few_shot_test(
    #         data_loader_fun=dataloader_fun,
    #         run_type="meta_learn_test_support",
    #         **kwargs,
    #     )

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    # def test_runner_coco_meta_learn_test_class_codes(self):
    #     config_file = "COCO-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml"
    #     self._setup_few_shot_configs_2(config_file)
    #     kwargs = {
    #         "cfg": self.default_cfg,
    #         "dataset_name": "coco_meta_val_novel",
    #     }
    #     set_global_cfg(self.default_cfg)

    #     dataloader_fun = (
    #         MetaFCOSRunner.build_episodic_learning_detection_test_support_set_loader
    #     )
    #     class_codes = self._few_shot_test(
    #         data_loader_fun=dataloader_fun,
    #         run_type="meta_learn_test_support",
    #         **kwargs,
    #     )

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    # def test_runner_lvis_meta_learn_test_class_codes_base(self):
    #     self._setup_few_shot_configs()
    #     kwargs = {
    #         "cfg": self.default_cfg,
    #         "dataset_name": "lvis_meta_val_all",
    #     }
    #     set_global_cfg(self.default_cfg)
    #     dataloader_fun = (
    #         MetaFCOSRunner.build_episodic_learning_detection_test_support_set_base_loader
    #     )
    #     model = self.runner.build_model(self.default_cfg)
    #     model.train(False)
    #     # # # setup data loader
    #     data_loader = dataloader_fun(**kwargs)
    #     # Remove the output directory
    #     if os.path.exists("./output"):
    #         import shutil

    #         shutil.rmtree("./output")
    #     train_dataset_name = self.default_cfg.DATASETS.TRAIN[0]
    #     logger.info(f"train dataset: {train_dataset_name}")
    #     train_metadata = MetadataCatalog.get(train_dataset_name)
    #     base_id_map = train_metadata.thing_dataset_id_to_contiguous_id
    #     all_id_map = MetadataCatalog.get(kwargs["dataset_name"]).thing_dataset_id_to_contiguous_id
    #     logger.info(f"base_id_map: {base_id_map}")
    #     base_sub_class_codes = inference_on_support_set_dataset_base(model, data_loader, all_id_map, base_id_map, output_dir=None)
    #     class_codes = MetaFCOSRunner._gather_class_code(base_sub_class_codes)
    #     logger.info(f"gathered class codes: {class_codes}")

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    # def test_runner_lvis_pretrain(self):
    #     # TODO: replace this to default configs
    #     config_file = "LVISv1-Detection/Meta-FCOS/Meta-FCOS-pretrain.yaml"
    #     self.runner, self.default_cfg = once_setup(config_file)
    #     self._do_pretrain()

    # @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    # def test_runner_lvis_meta_learn(self):
    #     config_file = "LVISv1-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml"
    #     self.runner, self.default_cfg = once_setup(config_file)
    #     self._do_meta_learn_train()

    # #    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    # def test_runner_lvis_meta_learn_evaluation(self):
    #     config_file = "LVISv1-Detection/Meta-FCOS/Meta-FCOS-finetune.yaml"
    #     self.runner, self.default_cfg = once_setup(config_file)
    #     self._do_meta_learn_evaluation()
if __name__ == "__main__":
    unittest.main()
