"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

#!/usr/bin/env python3
import itertools
import json
import logging
import os

from detectron2.evaluation.coco_evaluation import (
    COCOEvaluator,
)
from detectron2.utils.file_io import PathManager
from pycocotools.cocoeval import COCOeval
from sylph.data.data_injection.classes import COCO_BASE_CLASSES, COCO_NOVEL_CLASSES


logger = logging.getLogger(__name__)



class COCOFewShotEvaluator(COCOEvaluator):
    """
    Evaluate instance detection outputs using COCO's metrics and APIs.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
    ):
        super().__init__(
            dataset_name,
            tasks,
            distributed,
            output_dir,
            use_fast_impl=use_fast_impl,
            kpt_oks_sigmas=kpt_oks_sigmas,
        )
        # add args for few-shot learning
        self._logger = logging.getLogger(__name__)
        self._logger.info("Customized COCOEvaluator for few-shot detection.")
        self._dataset_name = dataset_name

        self._is_splits = (
            "all" in dataset_name or "base" in dataset_name or "novel" in dataset_name
        )
        self._base_classes = COCO_BASE_CLASSES
        self._novel_classes = COCO_NOVEL_CLASSES

    def _eval_prediction_few_shot(self, predictions, img_ids=None):
        """
        Evaluate predictions for few shot training. Fill self._results with the metrics of the tasks.
        """
        assert self._is_splits is True, self._logger.info(
            "This is not a few shot split."
        )
        self._logger.info("Preparing results for few shot COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        self._logger.info(f"len of coco_results: {len(coco_results)}")
        tasks = self._tasks or self._tasks_from_predictions(coco_results)
        assert "bbox" in tasks, self._logger.info(
            "Only bbox is supported for few shot."
        )

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = (
                self._metadata.thing_dataset_id_to_contiguous_id
            )
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert (
                min(all_contiguous_ids) == 0
                and max(all_contiguous_ids) == num_classes - 1
            )

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        self._results["bbox"] = {}

        split = "all"
        if "base" in self._dataset_name:
            classes = self._base_classes
            names = self._metadata.get("base_classes")
            split = "base"
        elif "novel" in self._dataset_name:
            classes = self._novel_classes
            names = self._metadata.get("novel_classes")
            split = "novel"
        else:  # "all"
            classes = self._base_classes + self._novel_classes
            names = self._metadata.get("thing_classes")

        # call customized _evaluate_predictions_on_coco
        coco_eval = (
            _evaluate_predictions_on_coco(
                coco_gt=self._coco_api,
                coco_results=coco_results,
                iou_type="bbox",
                catIds=classes,
                imgIds=img_ids,
            )
            if len(coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )
        res_ = self._derive_coco_results(
            coco_eval,
            "bbox",
            class_names=names,
        )
        logger.info(
            f"Default evaluation res has keys: {res_.keys()}, and the res: {res_}"
        )
        self._results["bbox"] = res_

        if split == "all":  # show average validation accuracy among novel classes
            novel_names = self._metadata.get("novel_classes")
            nAP, bAP = 0.0, 0.0
            count = 0
            for metric in res_.keys():
                class_name = metric.split("-")
                if len(class_name) == 2:
                    class_name = class_name[-1]
                else:
                    continue
                if class_name in novel_names:
                    logger.info(f"Novel {class_name}: {res_[metric]}")
                    nAP += res_[metric]
                else:
                    logger.info(f"Base {class_name}: {res_[metric]}")
                    bAP += res_[metric]
                    count += 1
            assert count == 60, "The len of base class is not 60"
            logger.info(f"nAP: {nAP/len(novel_names)}")
            logger.info(f"bAP: {bAP/count}")

    def _eval_predictions(self, predictions, img_ids=None):
        if self._is_splits:
            self._eval_prediction_few_shot(predictions, img_ids)
        else:
            self._logger.info("Preparing results for few shot COCO format ...")
            super()._eval_predictions(predictions, img_ids)


def _evaluate_predictions_on_coco(
    coco_gt, coco_results, iou_type, catIds=None, imgIds=None
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    # For few shot, the number of cat is changed
    if catIds is not None:
        coco_eval.params.catIds = catIds
    if imgIds is not None:
        coco_eval.params.imgIds = imgIds

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
