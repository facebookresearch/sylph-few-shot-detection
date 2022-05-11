#!/usr/bin/env python3

import logging
import os
import unittest

import pkg_resources
import torch
from d2go.runner import create_runner
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

# from egodet.runner.detr_runner import EgoDETRRunner  # noqa
from libfb.py import parutil
from sylph.modeling.code_generator.roi_encoder import ROIEncoder
from sylph.runner import MetaFCOSROIEncoderRunner  # noqa

logger = logging.getLogger(__name__)


def once_setup():
    config_file = "LVISv1-Detection/Meta-FCOS/Meta-FCOS-ROI-Encoder-finetune.yaml"
    config_file = pkg_resources.resource_filename(
        "sylph.model_zoo", os.path.join("configs", config_file)
    )
    config_file = parutil.get_file_path(config_file)

    logger.info(f"config_file {config_file}")

    runner = create_runner("sylph.runner.MetaFCOSROIEncoderRunner")
    cfg = runner.get_default_cfg()
    cfg.merge_from_file(config_file)

    # Reset train iterations
    cfg.SOLVER.MAX_ITER = 1
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.TEST.EVAL_PERIOD = 0  # do not test
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.META_LEARN.SHOT = 2
    cfg.MODEL.META_LEARN.EVAL_SHOT = 2
    cfg.DATALOADER.NUM_WORKERS = 4
    # test kernel_size
    cfg.MODEL.META_LEARN.CODE_GENERATOR.CLS_LAYER = ["GN", "", 3]
    cfg.MODEL.META_LEARN.CODE_GENERATOR.OUT_CHANNEL = 512

    logger.info(f"cfg {cfg}")
    return runner, cfg


class TestROIEncoder(unittest.TestCase):
    def setUp(self):
        setup_logger()
        self.runner, self.cfg = once_setup()

    # def test_roi_encoder_with_data_loader(self):
    #     return None

    def test_roi_transformer_hypernetwork(self):
        """
        Test from fake inputs
        """
        channels = 256
        strides = (8, 16, 32, 64)
        batch_size = 2
        num_shots = self.cfg.MODEL.META_LEARN.SHOT
        height, width = 1024, 1024
        # feature_shape = [] # feature size from FPN

        spatial_shapes = [[height // stride, width // stride] for stride in strides]
        spatial_shapes = torch.as_tensor(spatial_shapes)

        # (batch_size*num_shots, channels, height, width)
        # view(0, height * width, channels)

        support_set_image_features = [
            torch.rand(batch_size * num_shots, channels, h, w).to(self.cfg.MODEL.DEVICE)
            for h, w in spatial_shapes
        ]
        # support_memory = torch.cat(support_memory, dim=1)

        # support_memory.to(self.cfg.MODEL.DEVICE)
        # spatial_shapes.to(self.cfg.MODEL.DEVICE)

        # logger.info(f"support_memory shape {support_set_image_features.shape}")
        # logger.info(f"spatial_shapes shape {spatial_shapes.shape}")
        hypernetwork = ROIEncoder(
            self.cfg,
            feature_channels=channels,
            feature_levels=len(strides),
            strides=strides,
        )

        # support_imgs_ls = []
        gt_instances = []  # list of boxes
        for _i in range(batch_size):
            # support_imgs = []
            for _j in range(num_shots):
                instances = Instances((height, width))
                x1 = torch.randint(0, width - 1, (1,)).item()
                y1 = torch.randint(0, height - 1, (1,)).item()
                x2 = torch.randint(x1 + 1, width, (1,)).item()
                y2 = torch.randint(y1 + 1, height, (1,)).item()

                instances.gt_boxes = Boxes(torch.as_tensor([[x1, y1, x2, y2]]))

                logger.info(f"gt_boxes {instances}")

                gt_instances.append(instances)

        class_codes = hypernetwork(
            support_set_image_features,
            gt_instances,
        )
        class_weights, class_bias = class_codes["cls_conv"], class_codes["cls_bias"]

        logger.info(f"class_weights shape {class_weights.shape}")
        logger.info(f"class_bias shape {class_bias.shape}")

        self.assertEqual(
            class_weights.size(),
            torch.Size(
                (
                    batch_size,
                    self.cfg.MODEL.META_LEARN.CODE_GENERATOR.HEAD.OUTPUT_DIM,
                    1,
                    1,
                )
            ),
        )
        self.assertEqual(class_bias.size(), torch.Size((batch_size,)))
