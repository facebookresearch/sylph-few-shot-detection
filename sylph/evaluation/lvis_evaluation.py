# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import json
import logging
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import torch
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import LVISEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from lvis import LVIS
from sylph.data.data_injection.classes import unknown_category
from tabulate import tabulate
from detectron2.evaluation.lvis_evaluation import _evaluate_predictions_on_lvis



class FewshotLVIS(LVIS):
    def __init__(self, annotation_path, id_map=None):
        """Class for reading and visualizing annotations.
        Args:
            annotation_path (str): location of annotation file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading annotations.")

        self.dataset = self._load_json(annotation_path)

        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))

        self._create_index(id_map)

    def _create_index(self, id_map=None):
        self.logger.info("Creating index.")

        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)

        self.anns = {}
        self.cats = {}
        self.imgs = {}

        use_unknown = unknown_category["id"] in id_map
        self.logger.info(f"Use unknown category: {use_unknown}")

        if id_map is not None:  # save only categories in id_map
            img_ids = set()
            # process annotation
            for ann in self.dataset["annotations"]:
                if ann["category_id"] in id_map:  # or use unknown
                    # if use_unknown:
                    #     # change the category_id
                    #     ann["category_id"] = unknown_category["id"]
                    # replace neg_category_ids
                    self.img_ann_map[ann["image_id"]].append(ann)
                    self.anns[ann["id"]] = ann
                    # onky keep imgs with the right ids
                    img_ids.add(ann["image_id"])
                    self.cat_img_map[ann["category_id"]].append(ann["image_id"])

            for img in self.dataset["images"]:
                # neg_cat_ids=img["neg_category_ids"]
                # if len(set(neg_cat_ids).intersection(excluded_cat_ids)) == 0:
                #     img["neg_category_ids"].append(unknown_category["id"])
                if img["id"] in img_ids:
                    self.imgs[img["id"]] = img

            for cat in self.dataset["categories"]:
                if cat["id"] in id_map:
                    self.cats[cat["id"]] = cat
                # else:
                #     excluded_cat_ids.add(cat["id"])
            # if use_unknown: # add a faking unknown category
            #     self.cats[unknown_category["id"]] = unknown_category

            self.logger.info("Index created with id_map.")
            return

        for ann in self.dataset["annotations"]:
            self.img_ann_map[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann

        for img in self.dataset["images"]:
            self.imgs[img["id"]] = img

        for cat in self.dataset["categories"]:
            self.cats[cat["id"]] = cat

        for ann in self.dataset["annotations"]:
            self.cat_img_map[ann["category_id"]].append(ann["image_id"])

        self.logger.info("Index created.")


class FewshotLVISEvaluator(LVISEvaluator):
    def __init__(self, dataset_name, tasks=None, distributed=True, output_dir=None, per_category_ap: bool = True):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.info("Initializing FewshotLVISEvaluator")

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        # CHANGE
        id_map = (
            self._metadata.thing_dataset_id_to_contiguous_id
            if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id")
            else None
        )
        self._lvis_api = FewshotLVIS(json_file, id_map)
        # Test set json files do not contain annotations (evaluation must be
        # performed using the LVIS evaluation server).
        self._do_evaluation = len(self._lvis_api.get_ann_ids()) > 0
        self.per_category_ap=per_category_ap

    def _eval_predictions(self, predictions: List[Dict[str, Any]]):
        """
        Change: Replace _evaluate_predictions_on_lvis with _evaluate_predictions_on_lvis_with_per_category
        Evaluate predictions. Fill self._results with the metrics of the tasks.

        Args:
            predictions (list[dict]): list of outputs from the model
        """
        self._logger.info("Preparing results in the LVIS format ...")
        lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(lvis_results)

        # LVIS evaluator can be used to evaluate results for COCO dataset categories.
        # In this case `_metadata` variable will have a field with COCO-specific category mapping.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        else:
            # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
            for result in lvis_results:
                result["category_id"] += 1

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "lvis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        class_names = copy.deepcopy(self._metadata.get("thing_classes"))
        if "unknown" in class_names:
            class_names.remove("unknown")
        # TODO: skip evaluating on unknown for now
        for task in sorted(tasks):
            if not self.per_category_ap:
                evaluate_fun = _evaluate_predictions_on_lvis
            else:
                evaluate_fun = self._evaluate_predictions_on_lvis_with_per_category
            res = evaluate_fun(
                    self._lvis_api,
                    lvis_results,
                    task,
                    class_names=class_names,
                )
            self._results[task] = res

    def _evaluate_predictions_on_lvis_with_per_category(
        self, lvis_gt, lvis_results, iou_type, class_names: List[str] = None
    ):
        """
        Similar to `detectron2.evaluation._evaluate_predictions_on_lvis`, except
        calculating per-category AP at the end.

        Args:
            iou_type (str):
            kpt_oks_sigmas (list[float]):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        }[iou_type]

        logger = logging.getLogger(__name__)

        if len(lvis_results) == 0:  # TODO: check if needed
            logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        if iou_type == "segm":
            lvis_results = copy.deepcopy(lvis_results)
            # When evaluating mask AP, if the results contain bbox, LVIS API will
            # use the box area as the area of the instance, instead of the mask area.
            # This leads to a different definition of small/medium/large.
            # We remove the bbox field to let mask AP use mask area.
            for c in lvis_results:
                c.pop("bbox", None)

        from lvis import LVISEval, LVISResults

        lvis_results = LVISResults(lvis_gt, lvis_results)
        lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
        lvis_eval.run()
        lvis_eval.print_results()

        # Pull the standard metrics from the LVIS results
        results = lvis_eval.get_results()
        results = {metric: float(results[metric] * 100) for metric in metrics}
        logger.info(
            "Evaluation results for {}: \n".format(iou_type)
            + create_small_table(results)
        )

        # START OF CHANGE, basically follow the implementation in
        # `detectron2.evaluation.coco_evaluation.COCOEvaluator._derive_coco_results`

        # add a small check
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results

        self._logger.info("Start computing per-category AP.")
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = lvis_eval.eval["precision"]
        # CHANGE: precision has 4 dims (iou, recall, cls, area range),
        # w/o 5th dim (max dets) in `coco_eval.eval["precision"]`
        assert (
            len(class_names) == precisions.shape[2]
        ), f"class_names' length: {len(class_names)} does not equals to prediction size: {precisions.shape[2]}"

        # CHANGE: `short_name` can give a better layout
        def short_name(x):
            """
            Same as `detectron2.data.build.print_instances_class_histogram.short_name`
            """
            # make long class names shorter. useful for lvis
            if len(x) > 13:
                return x[:11] + ".."
            return x

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            # CHANGE:
            precision = precisions[:, :, idx, 0]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(
                # CHANGE: use `short_name`
                ("{}".format(short_name(name)), float(ap * 100))
            )

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)]
        )
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results
