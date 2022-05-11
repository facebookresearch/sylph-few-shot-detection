#!/usr/bin/env python3
from sylph.utils import create_cfg
import logging
import torch
from d2go.runner import create_runner
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
)
from detectron2.utils.visualizer import Visualizer
from sylph.utils import PathManagerImgLoader, PathManagerImgSave
import matplotlib.pyplot as plt
import os
from PIL import Image
from detectron2.utils.file_io import PathManager
from sylph.runner.meta_fcos_runner import MetaFCOSRunner
from sylph.evaluation.meta_learn_evaluation import format_class_codes_shared, inference_on_support_set_dataset

from typing import Dict, List
import numpy as np
from detectron2.data import detection_utils as utils
from detectron2.config import set_global_cfg




logger = logging.getLogger(__name__)

class SylphPredictor:
    """
    Modified from DefalutPredictor
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = SylphPredictor(cfg)
        # inputs = cv2.imread("input.jpg") # BGR
        # outputs = pred(inputs)
    """

    def __init__(self, config_file: str,  weight_path: str, class_code_path: str, runner_name: str="sylph.runner.MetaFCOSRunner", test_dataset_names: Dict = None):
        """
        test_dataset_names:Three choices: "base", "novel", "all"
        """
        logger.info("SylphPredictor initializing...")
        runner = create_runner(runner_name)
        self.cfg = create_cfg(runner.get_default_cfg(), config_file, None)
        self.cfg = self.cfg.clone()  # cfg can be modified by model
        assert self.cfg.MODEL.META_LEARN.EPISODIC_LEARNING, "This is not few-shot model"

        # Update the weight
        self.cfg.MODEL.WEIGHTS = weight_path
        if not torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.DATALOADER.NUM_WORKERS = 4 # fail if num_workers > 0 on CPU
        set_global_cfg(self.cfg)
        self.model = runner.build_model(self.cfg)
        # Prepare the model
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        self.class_code_path = class_code_path

        # prepare metadata, validatation data, and class codes
        self.user_metadata, self.user_class_codes = None, None
        self.metadatas, self.class_codes = {}, {}
        # Preload class codes, normally in 10 shots
        assert test_dataset_names is not None, "No test data"
        assert "all" in test_dataset_names, "split 'all' is not in the test dataset names"
        logger.info("start to load datasets")
        for split, dataset_name in test_dataset_names.items(): # if there are multiple novel datasets, only perserve the last one
            if split == "all":
                self.dicts = DatasetCatalog.get(dataset_name)
            self.metadatas[split] = MetadataCatalog.get(dataset_name)
            if split == "all":
                self.class_codes[split] = self._get_datasets_class_codes(self.metadatas[split], dataset_name)

        # filter the testing images by having overlapping with novel categories
        self.all_dataset_id_map = self.metadatas["all"].thing_dataset_id_to_contiguous_id
        self.novel_dataset_id_map = self.metadatas["novel"].thing_dataset_id_to_contiguous_id
        novel_dataset_ids = set(self.novel_dataset_id_map.keys())
        self.test_images = []
        for record in self.dicts[-1]:
            if len(record["annotations_cat_set"].intersection(novel_dataset_ids)) > 0:
                self.test_images.append(record)

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = self.cfg.INPUT.FORMAT
        logger.info(f"Input format: {self.input_format}")
        assert self.input_format in ["RGB", "BGR"], self.input_format
        self.registered_class_codes = None
        self.current_registered_class_id = 0
        self.total_registered_class_codes = 0

        # for easy access
        self.metadatas.update({ "user": self.user_metadata})
        self.class_codes.update({"user": self.user_class_codes})
        logger.info("SylphPredictor done initialization")

    def _generate_class_code_from_dataset(self, dataset_name: str, shot: int = 10):
        """
        Generate class codes on the given dataset for a given shot
        """
        default_shot = self.cfg.MODEL.META_LEARN.EVAL_SHOT
        self.cfg.MODEL.META_LEARN.EVAL_SHOT = shot
        data_loader = MetaFCOSRunner.build_episodic_learning_detection_test_support_set_loader(self.cfg, dataset_name, meta_test_seed=0)
        # data_loader = iter(data_loader)
        def _get_inference_dir_name(base_dir, inference_type, dataset_name):            return os.path.join(
                base_dir,
                inference_type,
                "default",
                "final",
                dataset_name,
            )
        output_dir=_get_inference_dir_name(self.cfg.OUTPUT_DIR, "inference", dataset_name)
        logger.info(f"Save class codes using {self.cfg.MODEL.META_LEARN.EVAL_SHOT} shots examples in {output_dir}")
        class_codes = inference_on_support_set_dataset(self.model, data_loader=data_loader, output_dir=output_dir)
        class_codes = format_class_codes_shared(class_codes, device=self.model.device)
        self.cfg.MODEL.META_LEARN.EVAL_SHOT = default_shot
        # dict "cls_conv" "cls_bias", tensor
        return class_codes

    def _generate_class_codes_from_a_support_set(self, support_set: List[Dict]):
        raise NotImplementedError("_generate_class_codes_from_a_support_set is not implemented")

    def _get_datasets_class_codes(self, metadata: Dict, dataset_name: str):
        """
        Read class codes from class_code_path (10 shots)
        """
        classes = metadata.thing_classes
        class_codes = []
        code_path = os.path.join(self.class_code_path, dataset_name, '0')
        for class_name in classes:
            file = os.path.join(code_path, f"{class_name}.pth")
            while not PathManager.exists(file):
                raise ValueError(f"{file} is missing")
            class_codes.append(
                torch.load(PathManager.open(file, mode="rb"), map_location="cpu")
            )
        logger.info(f"Got {len(class_codes)} class codes for prediction.")
        class_codes = format_class_codes_shared(class_codes, device=self.model.device)
        logger.info(f"class codes has keys: {class_codes.keys()}")
        assert "cls_conv" in class_codes, "conv is not in class_codes"
        return class_codes


    def _few_shot_inference(self, inputs: List[Dict], class_codes: Dict[str, torch.Tensor], visualize=False, output_dir=None):
        """
        Inference on all categories, but split the visualization into two: base and novel
        """
        if visualize:
            temp_path = "output" if output_dir is None else output_dir
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
        total_examples = 0
        num_examples = len(inputs)
        metadata = self.metadatas["all"]
        with torch.no_grad():
            for ids, dic in enumerate(inputs):
                logger.info("prediction progress: {}/{}".format(ids, num_examples))
                img_rgb = PathManagerImgLoader(dic["file_name"], img_color="RGB")
                img = utils.convert_PIL_to_numpy(img_rgb, format="BGR")
                # image = Image.open(join(scan_directory, f)).convert('RGB')
                # open_cv_image = np.array(image )
                # # Convert RGB to BGR
                # open_cv_image = open_cv_image[:, :, ::-1].copy()
                outputs = self._call_few_shot(img, class_codes)
                # visualize the output
                if visualize:
                    v = Visualizer(img[:, :, ::-1].copy(), metadata, scale=1)
                    out = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
                    plt.figure()
                    plt.imshow(out)
                    output_filename = os.path.join(temp_path, f"{ids}.png")
                    PathManagerImgSave(Image.fromarray(out, "RGB"), output_filename)
                total_examples += 1
        logger.info(f"total examples: {total_examples}")
        return total_examples

    def test_novel_predictor(self, num_examples: int = 10, rand: bool =True):
        examples = np.random.choice(self.test_images, 10, False)
        class_codes = self.class_codes["all"]
        return self._few_shot_inference(examples, class_codes)

    def register_class(self, support_set):
        """
        1. Write the initial images in manifold
        2. Run code generator and save class_code
        3. Update self.user_metadata
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            class_id:  a random id in range [1, 10000]
            contingous_id:
        """
        # self.assertTrue("support_set_target" in keys)
        #     self.assertTrue("support_set" in keys)
        #     self.assertTrue("query_set" in keys)
        return None

    def inference_on_registered_class(self, original_image):
        return self._call_few_shot(original_image, split="user")

    def inference_on_base_classes(self, original_image):
        return self._call_few_shot(original_image, split="base")

    def inference_on_novel_classes(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        return self._call_few_shot(original_image, split="novel")

    def _call_few_shot(self, original_image: np.ndarray, class_codes: Dict[str, torch.Tensor]):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            class_codes (Dict[str, torch.Tensor]): class_codes to be used as classifier head

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model(
                    [inputs], class_code=class_codes, run_type="meta_learn_test_instance"
                )
            predictions = predictions[0]
            return predictions

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
