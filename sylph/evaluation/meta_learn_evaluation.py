#!/usr/bin/env python3
import datetime
import itertools
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import ExitStack
from typing import Dict, Any, List

import torch
from detectron2.evaluation.coco_evaluation import (
    COCOEvaluator, COCOevalMaxDets, _evaluate_predictions_on_coco as _evaluate_predictions_on_coco_owd
)
from detectron2.evaluation.evaluator import inference_context, DatasetEvaluators
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table, log_every_n_seconds
from pycocotools.cocoeval import COCOeval
from sylph.data.data_injection.classes import (
    COCO_BASE_CLASSES,
    COCO_NOVEL_CLASSES,
)
from torch import nn
import copy
from detectron2.evaluation.fast_eval_api import COCOeval_opt
import numpy as np
import contextlib
import io
from pycocotools.coco import COCO
from d2go.utils.misc import tabulate


# def format_class_codes(class_codes, device):
#     """
#     class codes is a list of n novel classes
#     class_codes[0] has three keys: 'support_set_target': tensor([1]), 'class_name': 'bicycle', 'class_code':
#     class code is a list of total feature levels in FPN
#     """
#     feature_levels = len(class_codes[0]["class_code"])
#     num_classes = len(class_codes)
#     assert feature_levels == len(class_codes[0]["class_code"])
#     all_class_kernel = []
#     for level in range(feature_levels):
#         all_class_kernel_per_level = [None for i in range(num_classes)]
#         for class_code in class_codes:
#             class_contingous_id = class_code["support_set_target"]
#             all_class_kernel_per_level[class_contingous_id] = class_code["class_code"][
#                 level
#             ].to(device)
#         all_class_kernel_per_level = torch.cat(
#             all_class_kernel_per_level, dim=0
#         )  # 20, 256, 1, 1
#         all_class_kernel.append(all_class_kernel_per_level)

#     assert len(all_class_kernel) == feature_levels
#     assert all_class_kernel[0].size(0) == num_classes
#     assert len(all_class_kernel[0].size()) == 4
#     return all_class_kernel


def format_class_codes_shared(class_codes: List[Dict[str, Any]], device)->Dict[str, torch.tensor]:
    """
    Formating all class codes into a Dict with tensors as values.
    Args:
        class codes: a list of n novel class codes, each class code is shared for all levels in FPN, and has three keys:
        'support_set_target': tensor([1]), 'class_name': 'bicycle', 'class_code':
    Returns:
        Dict with potentially two keys:
        "cls_conv" code in dimension 1, 256, 1, 1
        "cls_bias" code in dimension 1, 1, 1, 1
        "cls_weight_norm" in dimension 1, 1, 1
        "
    """
    # feature_levels = len(class_codes[0]["class_code"])
    num_classes = len(class_codes)
    if num_classes == 0:
        return class_codes
    outs = defaultdict(list)
    # initiate
    for k in class_codes[0]["class_code"].keys():
        outs[k] = [None for _ in range(num_classes)]
    for code in class_codes:
        for dtype, value in code["class_code"].items():
            if dtype == "snnl":
                continue
            outs[dtype][code["support_set_target"]]=value.cpu()
    final_outs = {}
    for key, value in outs.items():
        # print(f"key: {key}, value: {value}")
        final_outs[key] = torch.cat(value, dim=0).to(device)
        if key == "cls_bias": # reduce the dimension to only 1
            final_outs[key] = final_outs[key].view(final_outs[key].numel())
    return final_outs

def inference_normalization(model, codes: Dict[str, Any]):
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info(
        f"Start normalizing class codes on {num_devices} devices"
    )

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        return model(batched_inputs=None, class_code=codes, run_type="meta_learn_normalize_code")

def inference_on_support_set_dataset_base(model, data_loader, all_id_map: Dict[int, Any], base_id_map: Dict[int, Any], output_dir: str=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        A list of class code
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info(
        f"Start generating class codes on {len(data_loader)} segments of support set on {num_devices} devices"
    )

    total = len(data_loader)  # inference data loader must have a fixed length

    if output_dir is not None:
        if not PathManager.exists(output_dir):
            PathManager.mkdirs(output_dir)
        logger.info(f"Save class codes in {output_dir}")

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    results = []
    cid_to_class_code = defaultdict(Dict)
    cid_to_class_name = defaultdict(str)
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        for idx, inputs in enumerate(data_loader):
            assert len(inputs) == 1, "inputs' batch size is not 1"
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            class_code = model(inputs, run_type="meta_learn_test_support")
            # logger.info(f"class code, bias dimension: {class_code['cls_bias'].size()}")
            # logger.info(f"class_code: {class_code}")
            # use module because it is wrapped in DataParallel, cant reach to attributes
            for k, v in class_code.items():
                if torch.is_tensor(v) and v.is_cuda:
                    class_code[k] = v.cpu()
            cid = inputs[0]["support_set_target"].item()
            # print(f"inputs: {inputs[0]}")
            cid_to_class_name[cid] = inputs[0]["class_name"]
            weight = (float)(inputs[0]["len"]) / inputs[0]["total_len"]
            if cid in cid_to_class_code:
                cid_to_class_code[cid]["cls_conv"]+= class_code["cls_conv"]*weight
                cid_to_class_code[cid]["cls_bias"]+= class_code["cls_bias"]*weight
                cid_to_class_code[cid]["acc_weight"] += weight
            else:
                cid_to_class_code[cid] = {}
                cid_to_class_code[cid]["cls_conv"] = class_code["cls_conv"]*weight
                cid_to_class_code[cid]["cls_bias"] = class_code["cls_bias"]*weight
                cid_to_class_code[cid]["acc_weight"] = weight
            # temp_result = {"support_set_target": cid, "class_name": cid_to_class_name[cid], "class_code": cid_to_class_code[cid]}
            # logger.info(f"temp_result, bias dimension: {cid_to_class_code[cid]['cls_bias'].size()}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / class code. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
    # logger.info(f"cid_to_class_code: {cid_to_class_code}")
    assert len(cid_to_class_code.keys()) == len(cid_to_class_name.keys())
    for con_id in cid_to_class_code.keys():
        # con_id = torch.tensor(con_id)
        assert con_id in cid_to_class_code
        assert con_id in cid_to_class_name
        result = {"support_set_target": con_id, "class_name": cid_to_class_name[con_id], "class_code": cid_to_class_code[con_id]}
        # write result into output
        # TODO: save to a particular path after reduce
        # if output_dir is not None:
        #     save_file = os.path.join(output_dir, f"{result['class_name']}.pth")
        #     logger.info(f"Save file: {save_file}")
        #     if PathManager.exists(save_file) and PathManager.isfile(save_file):
        #         # delete the file to be able to evaluate multiple times
        #         try:
        #             PathManager.rm(save_file)
        #         except OSError:
        #             logger.info(f"Failed to rm file: {save_file}")
        #             pass

        #     with PathManager.open(save_file, "wb") as save_path: # would this replace the existing file?
        #         torch.save(
        #                 result,
        #                 save_path,
        #             )
        results.append(result)
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / class code per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )
    return results

def inference_on_support_set_dataset(model, data_loader, output_dir: str=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        A list of class code
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info(
        f"Start generating class codes on {len(data_loader)} support sets on {num_devices} devices"
    )

    total = len(data_loader)  # inference data loader must have a fixed length

    if output_dir is not None:
        if not PathManager.exists(output_dir):
            PathManager.mkdirs(output_dir)
        logger.info(f"Save class codes in {output_dir}")

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    results = []
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        for idx, inputs in enumerate(data_loader):
            assert len(inputs) == 1, "inputs' batch size is not 1"
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            # save class id and class name
            result = {
                key: value for key, value in inputs[0].items() if key != "support_set"
            }
            start_compute_time = time.perf_counter()
            class_code = model(inputs, run_type="meta_learn_test_support")
            # use module because it is wrapped in DataParallel, cant reach to attributes
            for k, v in class_code.items():
                if torch.is_tensor(v) and v.is_cuda:
                    class_code[k] = v.cpu()
            result["class_code"] = class_code
            # write result into output
            if output_dir is not None:
                save_file = os.path.join(output_dir, f"{result['class_name']}.pth")
                if PathManager.exists(save_file) and PathManager.isfile(save_file):
                    # delete the file to be able to evaluate multiple times
                    PathManager.rm(save_file)
                with PathManager.open(save_file, "wb") as save_path:
                    torch.save(
                        result,
                        save_path,
                    )
            results.append(result)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / class code. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / class code per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )
    return results

def inference_on_dataset_with_class_codes(
    model, data_loader, evaluator, class_codes, cls_reweight=False, eval_with_pretrained_code = False
):
    """
    Modified from inference_on_dataset, class_codes is dict saving all conditional parameters
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)

    if eval_with_pretrained_code:
        assert class_codes is None
        class_code_type = "pretrained"
    else:
        assert class_codes is not None
        class_code_type = "predicted"
    logger.info(
        f"Start inference with {class_code_type} class codes on {len(data_loader)} images"
    )

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        # reweight class code
        if cls_reweight and not eval_with_pretrained_code:
            logger.info("cls code reweighting")
            class_codes["cls_conv"] = model(
                batched_inputs=None,
                class_code=class_codes["cls_conv"],
                run_type="meta_learn_test_support",
                cls_reweight=True,
            )
            logger.info(
                f"reweighted class code: {class_codes['cls_conv'].mean(), class_codes['cls_conv'].std()}"
            )

        for idx, inputs in enumerate(data_loader):

            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            # The only change
            outputs = model(
                inputs, class_code=class_codes, run_type="meta_learn_test_instance"
            )
            # logger.info(f"Test losses: {losses}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

class AREvaluator(COCOEvaluator):
    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
    ):
        super().__init__(
            dataset_name,
            tasks,
            distributed,
            output_dir,
            max_dets_per_image=None,
            use_fast_impl=use_fast_impl,
            kpt_oks_sigmas=kpt_oks_sigmas,
        )

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "ARdet1", "ARdet10", "ARdet100", "ARs", "ARm", "ARl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
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

class COCOMetaEvaluator(AREvaluator):
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
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
    ):
        super().__init__(
            dataset_name,
            tasks,
            distributed,
            output_dir,
            max_dets_per_image=None,
            use_fast_impl=use_fast_impl,
            kpt_oks_sigmas=kpt_oks_sigmas,
        )
        # add args for few-shot learning
        self._logger = logging.getLogger(__name__)
        self._logger.info("Customized COCOEvaluator for meta_learn detection.")
        self._dataset_name = dataset_name

        self.pretrain = True if "pretrain" in dataset_name else False

        # class ids
        self._base_classes = COCO_BASE_CLASSES
        self._novel_classes = COCO_NOVEL_CLASSES

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions for few-shot detection. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for meta-learning COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        self._logger.info(f"len of coco_results: {len(coco_results)}")
        tasks = self._tasks or self._tasks_from_predictions(coco_results)
        assert "bbox" in tasks, self._logger.info(
            "Only bbox is supported for meta-learn detection."
        )

        # unmap the category ids for COCO
        assert hasattr(
            self._metadata, "thing_dataset_id_to_contiguous_id"
        ), "Add thing_dataset_id_to_contiguous_id metadata"
        dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        classes = list(dataset_id_to_contiguous_id.keys())
        all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
        num_classes = len(all_contiguous_ids)
        assert (
            min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1
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

        # Replace the classes for evaluation.
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

        res = self._derive_coco_results(
            coco_eval, "bbox", class_names=self._metadata.get("thing_classes")
        )
        self._results["bbox"] = res


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


class COCO_OWD_Evaluator(AREvaluator):
    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        agnostic_eval=False,
    ):
        super().__init__(
            dataset_name,
            tasks,
            distributed,
            output_dir,
            max_dets_per_image=None,
            use_fast_impl=use_fast_impl,
            kpt_oks_sigmas=kpt_oks_sigmas,
        )
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO_OWD(json_file, set(self._metadata.thing_dataset_id_to_contiguous_id.keys()), agnostic_eval)
        self.agnostic_eval = agnostic_eval

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)
        assert 'bbox' in tasks, f'only bbox supported tasks: {tasks}'

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                if not self.agnostic_eval:
                    result["category_id"] = reverse_id_mapping[category_id]
                else:
                    result["category_id"] = 1

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

        if not self.agnostic_eval:
            logging.info('Below is class-dependent evaluation')
        else:
            logging.info('Below is class-agnostic evaluation')

        task = 'bbox'
        coco_eval = (
            _evaluate_predictions_on_coco_owd(
                self._coco_api,
                coco_results,
                task,
                kpt_oks_sigmas=self._kpt_oks_sigmas,
                use_fast_impl=self._use_fast_impl,
                img_ids=img_ids,
                max_dets_per_image=self._max_dets_per_image,
            )
            if len(coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )

        if not self.agnostic_eval:
            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
        else:
            res = self._derive_coco_results(
                coco_eval, task
            )
        self._results[task] = res

class COCO_OWD(COCO):
    def __init__(self, annotation_file, filteredClassIds=None, agnostic_cls=False):
        super().__init__(annotation_file)
        if filteredClassIds is not None and "annotations" in self.dataset:
            if "annotations" in self.dataset:
                self.dataset["annotations"] = list(
                    filter(
                        lambda annotation: annotation["category_id"] in filteredClassIds,
                        self.dataset["annotations"],
                    )
                )
            if "categories" in self.dataset:
                self.dataset['categories'] = list(
                    filter(
                        lambda category: category["id"] in filteredClassIds,
                        self.dataset["categories"],
                    )
                )
            if agnostic_cls:
                for ann in self.dataset["annotations"]:
                    ann["category_id"] = 1
                self.dataset["categories"] = [{'supercategory': 'nothing', 'id': 1, 'name': 'agnostic'}]

        self.createIndex(True)

    def createIndex(self, filtered=False):
        if filtered:
            super().createIndex()
        else:
            pass
