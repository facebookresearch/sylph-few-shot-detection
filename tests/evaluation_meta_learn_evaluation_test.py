#!/usr/bin/env python3
import logging
import os
import unittest
from typing import Dict, List

import pkg_resources
import torch
from d2go.runner import create_runner
from detectron2.utils.logger import setup_logger
from sylph.runner.meta_fcos_runner import MetaFCOSRunner  # noqa
from sylph.utils import create_cfg
from sylph.evaluation.meta_learn_evaluation import COCO_OWD_Evaluator, COCO_OWD, COCOMetaEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO
import copy
#from detectron2.evaluation.coco_evaluation import COCOEvaluator

logger = logging.getLogger(__name__)


def once_setup(config_file: str):
    # config_file = "LVIS-Meta-FCOS-Detection/Meta_FCOS_MS_R_50_1x.yaml"
    config_file = pkg_resources.resource_filename(
        "sylph", os.path.join("configs", config_file)
    )

    logger.info(f"config_file {config_file}")

    runner = create_runner("sylph.runner.MetaFCOSRunner")
    default_cfg = runner.get_default_cfg()
    coco_cfg = create_cfg(default_cfg, config_file, None)
    # register dataset
    # runner.register(lvis_cfg)

    logger.info(f"cfg {coco_cfg}")
    return runner, coco_cfg


class TestMetaFCOSOWDEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        setup_logger()

        cls.runner, cls.default_cfg = once_setup(
            "COCO-Meta-FCOS-Detection/Base-Meta-FCOS-pretrain_owd.yaml")
        cls.default_cfg.SOLVER.MAX_ITER = 1
        cls.default_cfg.MODEL.DEVICE = "cpu"
        cls.default_cfg.TEST.EVAL_PERIOD = 0  # do not test
        cls.default_cfg.SOLVER.IMS_PER_BATCH = 2
        cls.default_cfg.DATALOADER.NUM_WORKERS = 0  # avoid broken pipe

        cls.test_dataset_name = "coco_pretrain_val_novel"
        data_loader = MetaFCOSRunner.build_detection_test_loader(
            cls.default_cfg, cls.test_dataset_name)
        _data_loader_iter = iter(data_loader)

        cls.data = next(_data_loader_iter)

        cls.novel_owd_agnostic_evaluator = MetaFCOSRunner.get_evaluator(
            cls.default_cfg, cls.test_dataset_name, './output')
        cls.novel_owd_evaluator = COCO_OWD_Evaluator(
            cls.test_dataset_name, use_fast_impl=False, agnostic_eval=False)
        cls.novel_evaluator = COCOMetaEvaluator(
            cls.test_dataset_name, use_fast_impl=False)

    def test_check_evaluator_type(self):
        self.assertIsInstance(self.novel_owd_agnostic_evaluator, COCO_OWD_Evaluator,
                              'COCO OWD Evaluator must be used for class agnostic proposals')
        self.assertTrue(self.novel_owd_agnostic_evaluator.agnostic_eval)

    def test_check_base_vs_novel(self):
        dataset_names = ["coco_pretrain_val_base", "coco_pretrain_val_novel"]
        for dataset_name in dataset_names:
            with self.subTest(dataset_name):
                metadata = MetadataCatalog.get(dataset_name)
                json_file = PathManager.get_local_path(metadata.json_file)
                base_cls_categories = set(
                    metadata.thing_dataset_id_to_contiguous_id.keys())
                coco_all = COCO(
                    json_file
                )

                filtered_coco_base = COCO_OWD(
                    json_file, filteredClassIds=set(metadata.thing_dataset_id_to_contiguous_id.keys()), agnostic_cls=True
                )

                assert set(filtered_coco_base.cats.keys()) == {
                    1
                }, "There should only be one class in agnostic eval"
                assert filtered_coco_base.cats[1] == {
                    "supercategory": "nothing",
                    "id": 1,
                    "name": "agnostic",
                }, f"only category should be class agnostic {filtered_coco_base.cats}"

                for ann_id, ann in coco_all.anns.items():
                    if ann["category_id"] in base_cls_categories:
                        assert ann_id in filtered_coco_base.anns.keys(
                        ), f'Annotation ID should be here {ann_id}'
                    else:
                        assert ann_id not in filtered_coco_base.anns.keys(
                        ), f'Annotation ID shouldnt be here {ann_id}'

    def test_evaluation(self):
        model = self.runner.build_model(self.default_cfg)
        model.eval()
        inputs = self.data
        with torch.no_grad():
            outputs = model(inputs)
        self.novel_owd_agnostic_evaluator.reset()
        self.novel_owd_agnostic_evaluator.process(inputs, outputs)
        results = self.novel_owd_agnostic_evaluator.evaluate()
        output_keys = {"AP", "AP50", "AP75", "APs", "APm", "APl",
                       "ARdet1", "ARdet10", "ARdet100", "ARs", "ARm", "ARl"}
        self.assertTrue(output_keys.issubset(
            set(results['bbox'].keys())), 'Not returning the AP and AR!')

    def test_evaluation_class_normal_identical(self):
        dataset_name = "coco_pretrain_val_novel"
        metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(metadata.json_file)
        coco_gt = COCO_OWD(
            json_file, filteredClassIds=set(
                metadata.thing_dataset_id_to_contiguous_id.keys())
        )

        predictions = []
        for image_id, annotations in coco_gt.imgToAnns.items():
            prediction = {"image_id": image_id}
            prediction["instances"] = []
            for id, actual_prediction in enumerate(annotations):
                prediction["instances"].append(
                    {
                        "image_id": image_id,
                        "category_id": metadata.thing_dataset_id_to_contiguous_id[
                            actual_prediction["category_id"]
                        ],
                        "bbox": actual_prediction["bbox"],
                        "id": id + 1,
                        "is_crowd": 0,
                        "score": 1,
                        "ignore": 0,
                    }
                )
            predictions.append(prediction)

        owd_evaluator = self.novel_owd_evaluator
        owd_evaluator._tasks = {"bbox"}
        owd_evaluator._results = {}
        owd_evaluator._eval_predictions(
            copy.deepcopy(predictions),
        )

        evaluator = self.novel_evaluator
        #evaluator._coco_api = coco_gt
        evaluator._tasks = {"bbox"}
        evaluator._results = {}
        evaluator._eval_predictions(
            copy.deepcopy(predictions)
        )

        bbox_owd_results = owd_evaluator._results['bbox']
        bbox_results = evaluator._results['bbox']

        self.assertDictEqual(bbox_owd_results, bbox_results)

    def test_evaluation_class_agnostic_identical(self):
        dataset_name = "coco_pretrain_val_novel"
        metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(metadata.json_file)
        coco_gt = COCO_OWD(
            json_file, filteredClassIds=set(
                metadata.thing_dataset_id_to_contiguous_id.keys())
        )

        predictions = []
        for image_id, annotations in coco_gt.imgToAnns.items():
            prediction = {"image_id": image_id}
            prediction["instances"] = []
            for id, actual_prediction in enumerate(annotations):
                prediction["instances"].append(
                    {
                        "image_id": image_id,
                        "category_id": metadata.thing_dataset_id_to_contiguous_id[
                            actual_prediction["category_id"]
                        ],
                        "bbox": actual_prediction["bbox"],
                        "id": id + 1,
                        "is_crowd": 0,
                        "score": 1,
                        "ignore": 0,
                    }
                )
            predictions.append(prediction)

        owd_evaluator = self.novel_owd_agnostic_evaluator
        owd_evaluator._tasks = {"bbox"}
        owd_evaluator._results = {}
        owd_evaluator._eval_predictions(
            copy.deepcopy(predictions),
        )

        evaluator = self.novel_evaluator
        #evaluator._coco_api = coco_gt
        evaluator._tasks = {"bbox"}
        evaluator._results = {}
        evaluator._eval_predictions(
            copy.deepcopy(predictions)
        )

        self.assertTrue(
            evaluator._results['bbox']['AP'] == owd_evaluator._results['bbox']['AP'])
        self.assertTrue(
            evaluator._results['bbox']['ARdet100'] == owd_evaluator._results['bbox']['ARdet100'])

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("./output"):
            import shutil
            shutil.rmtree("./output")
