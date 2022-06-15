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

logger = logging.getLogger(__name__)


def once_setup(config_file: str):
    config_file = pkg_resources.resource_filename(
        "sylph", os.path.join("configs", config_file)
    )

    logger.info(f"config_file {config_file}")

    runner = create_runner("sylph.runner.MetaFCOSRunner")
    default_cfg = runner.get_default_cfg()
    coco_cfg = create_cfg(default_cfg, config_file, None)
    # register dataset

    logger.info(f"cfg {coco_cfg}")
    return runner, coco_cfg


class TestMetaFCOSOWD(unittest.TestCase):

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

        data_loader = MetaFCOSRunner.build_detection_train_loader(
            cls.default_cfg)
        _data_loader_iter = iter(data_loader)
        cls.data = next(_data_loader_iter)

    def test_owd_config(self):
        self.assertTrue(set(self.default_cfg.MODEL.FCOS.BOX_QUALITY).issubset(
            ['iou', 'ctrness']), 'Box Quality must either be iou or ctrness')
        self.assertTrue(self.default_cfg.MODEL.PROPOSAL_GENERATOR.OWD,
                        'Must Turn on OWD Detection!')

    def test_owd_frozen_components(self):
        model = self.runner.build_model(self.default_cfg)

        self.assertTrue(all([not p.requires_grad for p in model.proposal_generator.fcos_head.cls_tower.parameters(
        )]), 'CLS Tower isnt frozen!')
        self.assertTrue(all([not p.requires_grad for p in model.proposal_generator.fcos_head.cls_logits.parameters(
        )]), 'CLS Logits Head isnt frozen!')

        localization_modules = {
            'bbox_tower': model.proposal_generator.fcos_head.bbox_tower.parameters(),
            'centerness_head': model.proposal_generator.fcos_head.ctrness.parameters(),
            'iou_pred_head': model.proposal_generator.fcos_head.iou_overlap.parameters(),
            'bbox_reg_head': model.proposal_generator.fcos_head.bbox_pred.parameters(),
        }

        for localization_head_name, params in localization_modules.items():
            self.assertTrue(all([p.requires_grad for p in params]),
                            f'{localization_head_name} should not be frozen!')

    def test_owd_train_iou(self):
        self.default_cfg.MODEL.FCOS.BOX_QUALITY = ['iou']
        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        losses = model(self.data)

        self.assertTrue('loss_fcos_iou' in losses.keys(),
                        'IOU Overlap Loss has to be returned during OWD!')
        self.assertTrue('loss_fcos_loc' in losses.keys(),
                        'Localization Loss has to be returned during OWD!')

        self.assertFalse('loss_fcos_cls' in losses.keys(
        ), f'OWD Should not return FCOS Class Loss! {losses.keys()}')
        self.assertFalse('loss_fcos_ctr' in losses.keys(
        ), 'Shouldnt have centerness loss when box quality = [iou]')
        self.assertTrue(all([item.grad_fn is not None and item.requires_grad for item in losses.values(
        )]), 'Gradients must have backprop enabled')

        with self.subTest('backprop test'):
            try:
                sum(losses.values()).backward()
            except Exception:
                raise AssertionError('BackProp Failed')

    def test_owd_train_ctrness(self):
        self.default_cfg.MODEL.FCOS.BOX_QUALITY = ['ctrness']
        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        losses = model(self.data)

        self.assertTrue('loss_fcos_ctr' in losses.keys(),
                        'Centerness Loss has to be returned during OWD!')
        self.assertTrue('loss_fcos_loc' in losses.keys(),
                        'Localization Loss has to be returned during OWD!')

        self.assertFalse('loss_fcos_cls' in losses.keys(
        ), f'OWD Should not return FCOS Class Loss! {losses.keys()}')
        self.assertFalse('loss_fcos_iou' in losses.keys(),
                         'Shouldnt have iou loss when box quality = [ctrness]')
        self.assertTrue(all([item.grad_fn is not None and item.requires_grad for item in losses.values(
        )]), 'Gradients must have backprop enabled')

        with self.subTest('backprop test'):
            try:
                sum(losses.values()).backward()
            except Exception:
                raise AssertionError('BackProp Failed')

    def test_owd_train_both(self):
        self.default_cfg.MODEL.FCOS.BOX_QUALITY = ['ctrness', 'iou']
        model = self.runner.build_model(self.default_cfg)
        model.train(True)
        losses = model(self.data)

        self.assertTrue('loss_fcos_ctr' in losses.keys(),
                        'Centerness Loss has to be returned during OWD!')
        self.assertTrue('loss_fcos_loc' in losses.keys(),
                        'Localization Loss has to be returned during OWD!')
        self.assertTrue('loss_fcos_iou' in losses.keys(),
                        'IOU Overlap Loss has to be returned during OWD!')

        self.assertFalse('loss_fcos_cls' in losses.keys(
        ), f'OWD Should not return FCOS Class Loss! {losses.keys()}')
        self.assertTrue(all([item.grad_fn is not None and item.requires_grad for item in losses.values(
        )]), 'Gradients must have backprop enabled')

        with self.subTest('backprop test'):
            try:
                sum(losses.values()).backward()
            except Exception:
                raise AssertionError('BackProp Failed')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("./output"):
            import shutil
            shutil.rmtree("./output")
