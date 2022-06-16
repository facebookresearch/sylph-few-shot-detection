"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

#!/usr/bin/env python3
import logging
import unittest

from detectron2.utils.logger import setup_logger
from sylph.predictor import SylphPredictor
from sylph.runner.meta_fcos_runner import MetaFCOSRunner  # noqa

logger = logging.getLogger(__name__)

#TODO: reformate and test


class TestPredictor(unittest.TestCase):

    def setUp(self):
        setup_logger()
        # config has to be compatible with current version

        # TODO: set path to config and weight, and class code
        model_config = ""
        model_weight = ""
        class_code_path = ""
        test_dataset_names = {"all": "lvis_meta_val_all",
                              "base": "lvis_meta_train_basefc", "novel": "lvis_meta_train_novelr"}
        self.predictor = SylphPredictor(
            config_file=model_config,
            weight_path=model_weight,
            class_code_path=class_code_path,
            test_dataset_names=test_dataset_names
        )

    # def test_predictor(self):
    #     success_examples = self.predictor.test_novel_predictor(10)
    #     self.assertEqual(success_examples, 10, f"Only {success_examples} success runs instead of 10")

    # def test_many_shot_predictor_on_dataset(self):
    #     """
    #     Inference only on base classes
    #     """
    #     return None

    # def test_few_shot_predictor_on_dataset(self):
    #     """
    #     Include inference on novel and base classes
    #     """
    #     return None

    # def test_register_class_code(self):
    #     # Generate a randome support set
    #     return None

    # def test_few_shot_inference(self):
    #     """
    #     Inference with class code registered in test_register_class_code
    #     """
    #     return None
