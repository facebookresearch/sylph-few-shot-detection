#!/usr/bin/env python3
import logging
import unittest

from detectron2.utils.logger import setup_logger
from sylph.predictor import SylphPredictor
from sylph.runner.meta_fcos_runner import MetaFCOSRunner  # noqa

logger = logging.getLogger(__name__)


class TestPredictor(unittest.TestCase):
    def setUp(self):
        setup_logger()
        output_id = "20211105173218"  # f307393206,  "APr": 16.5,"APc": 23.7,"APf": 29.0,
        # config has to be compatible with current version
        model_config = f"manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test/{output_id}/e2e_train/config.yaml"
        model_weight = f"manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test/{output_id}/e2e_train/model_final.pth"
        class_code_path = f"manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test/{output_id}/e2e_train/inference/default/final"
        test_dataset_names = {"all": "lvis_meta_val_all", "base": "lvis_meta_train_basefc", "novel": "lvis_meta_train_novelr"}
        self.predictor = SylphPredictor(
            config_file=model_config,
            weight_path=model_weight,
            class_code_path=class_code_path,
            test_dataset_names = test_dataset_names
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
