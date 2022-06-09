#!/usr/bin/env python3
import logging
import os
from collections import defaultdict, OrderedDict
from copy import deepcopy
from typing import Dict, Any

# @manual=//mobile-vision/d2go/d2go:d2go
import d2go.utils.abnormal_checker as abnormal_checker
import detectron2.utils.comm as comm
# import egodet.data.transform  # noqa
import numpy as np
import torch
from d2go.config import temp_defrost
from d2go.data.dataset_mappers import build_dataset_mapper
from d2go.data.transforms.build import build_transform_gen
from d2go.data.utils import maybe_subsample_n_images
from d2go.modeling import kmeans_anchors, model_ema
# from d2go.projects.adet.adet_runner import AdelaiDetRunner
from sylph.runner.adet_runner import AdelaiDetRunner
# @manual=//mobile-vision/d2go/d2go:d2go
from d2go.runner.default_runner import _get_tbx_writer, GeneralizedRCNNRunner
from d2go.utils.flop_calculator import add_print_flops_callback
from d2go.utils.misc import get_tensorboard_log_dir

# @manual=//vision/fair/detectron2/detectron2:detectron2
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.config import set_global_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import SimpleTrainer, hooks
from detectron2.evaluation import (
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from detectron2.utils.file_io import PathManager
from sylph.config.config import CfgNode as CN
from sylph.data.build import (
    build_meta_detection_train_loader,
    build_meta_detection_test_support_set_loader,
    build_meta_detection_test_support_set_base_loader,
    build_detection_test_loader,
)
from sylph.data.dataset_mapper.meta_learn_dataset_mapper import (  # noqa
    MetalearnDatasetMapper,  # noqa
)  # noqa
from sylph.evaluation.evaluation import FewShotVisualizationEvaluator
from sylph.evaluation.lvis_evaluation import FewshotLVISEvaluator
from sylph.evaluation.meta_learn_evaluation import (
    COCO_OWD_Evaluator,
    inference_on_support_set_dataset,
    inference_on_dataset_with_class_codes,
    format_class_codes_shared,
    inference_on_support_set_dataset_base,
    inference_normalization,
)
from sylph.evaluation.visualization import EpisodicLearningDataLoaderVisWrapper
from sylph.modeling.code_generator.code_generator import CodeGenerator  # noqa
from sylph.modeling.code_generator.roi_encoder import ROIEncoder  # noqa
from sylph.modeling.code_generator.utils import reduce_class_code, replace_class_code
from sylph.modeling.meta_arch.meta_one_stage_detector import (  # noqa
    MetaOneStageDetector,
)
from sylph.modeling.meta_fcos.fcos import MetaFCOS  # noqa
from sylph.runner.default_configs import (
    add_base_config,
    add_fcos_config,
    add_default_meta_learn_config,
    add_code_genertor_config,
    add_roi_encoder_config,
    add_tfa_config,
)


logger = logging.getLogger(__name__)

# TODO: rename this to DefaultFewShotRunner


class MetaFCOSRunner(AdelaiDetRunner):
    """
    * inject coco datasets to support meta-learning
    * add three data loaders for episodic learning training and testing stages.
    * customize do_train and do_test to do both pretraining and episodic learning
    * customize evaluator
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.info("MetaFCOSRunner initialized. ")

    def get_default_cfg(self):
        _C = super().get_default_cfg()
        # Convert the config node to sylph library type
        _C = CN(_C)
        _C = add_base_config(_C)
        _C = add_fcos_config(_C)
        _C = add_tfa_config(_C)
        _C = add_default_meta_learn_config(_C)
        _C = add_code_genertor_config(_C)
        _C = add_roi_encoder_config(_C)
        return _C

    @staticmethod
    def get_evaluator(cfg, dataset_name, output_folder, **kwargs):
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        logger.info(f"Evaluator type: {evaluator_type}")
        if evaluator_type == "coco_meta_learn":
            return COCO_OWD_Evaluator(
                dataset_name=dataset_name,
                output_dir=output_folder,
                kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
                use_fast_impl=False,
                agnostic_eval=cfg.MODEL.PROPOSAL_GENERATOR.OWD,
            )
            """if cfg.MODEL.PROPOSAL_GENERATOR.OWD:
                return COCO_OWD_Evaluator(
                    dataset_name=dataset_name,
                    output_dir=output_folder,
                    kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
                    use_fast_impl=False,
                    agnostic_eval=True,
                )
            return COCOMetaEvaluator(
                dataset_name=dataset_name,
                output_dir=output_folder,
                kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
                use_fast_impl=False,
            )"""
        elif evaluator_type == "lvis_meta_learn":
            return FewshotLVISEvaluator(dataset_name, output_dir=output_folder)
        elif evaluator_type == "tao_meta_learn":
            return FewshotLVISEvaluator(dataset_name, output_dir=output_folder)
        else:
            return GeneralizedRCNNRunner.get_evaluator(
                cfg, dataset_name, output_folder, **kwargs
            )

    @staticmethod
    def get_few_shot_test_mapper(cfg, is_train, need_annotation=True):
        """
        modified from get_mapper with additional arg need_annotation
        """
        tfm_gens = build_transform_gen(cfg, is_train)
        mapper = build_dataset_mapper(
            cfg, is_train, tfm_gens=tfm_gens, need_annotation=need_annotation
        )
        return mapper

    @classmethod
    def build_episodic_learning_detection_train_loader(cls, cfg):
        """
        Support set and query set will be sampled together
        """
        mapper = cls.get_mapper(cfg, is_train=True)
        dataloader = build_meta_detection_train_loader(cfg, mapper)
        logger.info("Using dataset mapper:\n{}".format(mapper))
        if comm.is_main_process():
            data_loader_type = EpisodicLearningDataLoaderVisWrapper
            if data_loader_type is not None:
                tbx_writer = _get_tbx_writer(
                    get_tensorboard_log_dir(cfg.OUTPUT_DIR))
                data_loader = data_loader_type(cfg, tbx_writer, dataloader)
            return data_loader
        return dataloader

    @classmethod
    def build_episodic_learning_detection_test_support_set_loader(
        cls, cfg, dataset_name, meta_test_seed: int = 0
    ):
        logger.info(
            "Building episodic learning detection test support set loader for dataset: {} ...".format(
                dataset_name
            )
        )
        mapper = cls.get_few_shot_test_mapper(
            cfg, is_train=False, need_annotation=True)
        logger.info(
            "Using dataset mapper:\n{} in testing support set".format(mapper))
        return build_meta_detection_test_support_set_loader(
            cfg, dataset_name, mapper=mapper, meta_test_seed=meta_test_seed
        )

    @classmethod
    def build_episodic_learning_detection_test_support_set_base_loader(
        cls, cfg, dataset_name
    ):
        logger.info(
            "Building episodic learning detection test support set loader for dataset: {} ...".format(
                dataset_name
            )
        )
        mapper = cls.get_few_shot_test_mapper(
            cfg, is_train=False, need_annotation=True)
        logger.info(
            "Using dataset mapper:\n{} in testing support set".format(mapper))
        return build_meta_detection_test_support_set_base_loader(
            cfg, dataset_name, mapper=mapper
        )

    @classmethod
    def build_episodic_learning_detection_test_query_loader(
        cls, cfg, dataset_name, mapper=None
    ):
        logger.info(
            "Building episodic learning detection test query image loader for dataset: {} ...".format(
                dataset_name
            )
        )
        mapper = cls.get_few_shot_test_mapper(
            cfg,
            is_train=False,
            need_annotation=True,  # Set true to see the testing loss
        )

        logger.info(
            "Using dataset mapper:\n{} in testing query images".format(mapper))
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    def _weight_preprocess(self, cfg):
        """
        Deciding weights to load by modules instead of loading the whole pretrained model.
        """
        filter_modules = cfg.MODEL.WEIGHTS_FILTER_BY_MODULE
        assert isinstance(filter_modules, list)

        if len(filter_modules) == 0:
            logger.info("No weight filtering.")
            return
        weight_path = cfg.MODEL.WEIGHTS
        logger.info(f"initial weight path: {weight_path}")

        if "manifold" != weight_path.split("://")[0]:
            logger.info("Path is not manifold")
            return
        if not PathManager.exists(weight_path):
            logger.info(f"Weight path: {weight_path} does not exist")
            return

        local_weight_path = PathManager.get_local_path(weight_path)
        pretrained_model = torch.load(
            local_weight_path, map_location=torch.device("cpu")
        )

        # filter by module
        from collections import OrderedDict

        filtered_model = {}
        for key, value in pretrained_model["model"].items():
            if any([True for module in filter_modules if module in key]):
                continue
            else:
                filtered_model[key] = value
        filtered_model = OrderedDict(filtered_model)
        filtered_model = {"model": filtered_model}
        logger.info(
            f"After filtering, the model: {filtered_model['model'].keys()}")
        # save path
        save_path = weight_path.split("://")
        if len(save_path) == 2:
            save_path[1] = os.path.join(
                "/".join(save_path[1].split("/")[0:-1]), "filtered_model.pth"
            )
            save_path = "://".join(save_path)
        else:
            save_path = os.path.join(
                "/".join(weight_path.split("/")[0:-1]), "filtered_model.pth"
            )
        logger.info(f"save_path: {save_path}")
        # TODO: sometimes it fails to remove existing file, have to do it manually
        if PathManager.exists(save_path):
            logger.info(f"Remove existing file: {save_path}")
            PathManager.rm(save_path)
        torch.save(
            filtered_model,
            PathManager.open(save_path, "wb"),
        )
        logger.info(f"save_path: {save_path}")
        return save_path

    def do_train_per_stage(
        self, cfg, model, resume, iter_start: int = 0, iter_end: int = 0
    ):
        torch.cuda.empty_cache()
        # STAGE 1: set stage 1 parameters state, freeze detector network

        # Episodic learning inferrence for image and features is not supported in forward
        add_print_flops_callback(cfg, model, disable_after_callback=True)

        optimizer = self.build_optimizer(cfg, model)
        scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Add weight filtering
        filtered_weight_path = self._weight_preprocess(cfg)
        weight_path = cfg.MODEL.WEIGHTS
        if filtered_weight_path is not None:
            weight_path = filtered_weight_path

        checkpointer = self.build_checkpointer(
            cfg,
            model,
            save_dir=cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        checkpoint = checkpointer.resume_or_load(weight_path, resume=resume)
        # END OF CHANGE
        start_iter = (
            checkpoint.get("iteration", -1)
            if resume and checkpointer.has_checkpoint()
            else -1
        )
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        start_iter += 1
        max_iter = cfg.SOLVER.MAX_ITER
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )
        data_loader = self.build_episodic_learning_detection_train_loader(cfg)
        # batch size * {'support set', 'query set'}

        def _get_model_with_abnormal_checker(model):
            if not cfg.ABNORMAL_CHECKER.ENABLED:
                return model

            tbx_writer = _get_tbx_writer(
                get_tensorboard_log_dir(cfg.OUTPUT_DIR))
            writers = abnormal_checker.get_writers(cfg, tbx_writer)
            checker = abnormal_checker.AbnormalLossChecker(start_iter, writers)
            ret = abnormal_checker.AbnormalLossCheckerWrapper(model, checker)
            return ret

        # few shot trainer
        trainer = SimpleTrainer(
            _get_model_with_abnormal_checker(model), data_loader, optimizer
        )

        trainer_hooks = [
            hooks.IterationTimer(),
            model_ema.EMAHook(cfg, model) if cfg.MODEL_EMA.ENABLED else None,
            self._create_data_loader_hook(cfg),
            self._create_after_step_hook(
                cfg, model, optimizer, scheduler, periodic_checkpointer
            ),
            hooks.EvalHook(
                cfg.TEST.EVAL_PERIOD,
                lambda: self.do_test(cfg, model, train_iter=trainer.iter),
            ),
            kmeans_anchors.compute_kmeans_anchors_hook(self, cfg),
            self._create_qat_hook(
                cfg) if cfg.QUANTIZATION.QAT.ENABLED else None,
        ]

        if comm.is_main_process():
            tbx_writer = _get_tbx_writer(
                get_tensorboard_log_dir(cfg.OUTPUT_DIR))
            writers = [
                CommonMetricPrinter(max_iter),
                JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
                tbx_writer,
            ]
            trainer_hooks.append(hooks.PeriodicWriter(writers))
        trainer.register_hooks(trainer_hooks)
        trainer.train(start_iter, max_iter)

        trained_cfg = cfg.clone()
        with temp_defrost(trained_cfg):
            trained_cfg.MODEL.WEIGHTS = checkpointer.get_checkpoint_file()
        return {"model_final": trained_cfg}

    @classmethod
    def _gather_class_code(cls, sub_class_codes: Dict[str, Any], reduce: bool = False):
        """
        Gather class code from different processes.
        """
        if get_world_size() > 1:
            all_gather_class_codes_list = [
                None for _ in range(get_world_size())]
            torch.distributed.all_gather_object(
                all_gather_class_codes_list, sub_class_codes
            )
            out_codes = [
                cls_code
                for sub_cls_codes in all_gather_class_codes_list
                for cls_code in sub_cls_codes
            ]
            # if len(out_codes) == 0:
            #     return out_codes

            # assert "class_code" in out_codes[0]
            # all_keys = out_codes[0]["class_code"].keys()
            # cid_to_class_code_lst = defaultdict(list) # each item is a list of dict
            # cid_to_other_field = defaultdict(dict)
            # for cls_code in out_codes:
            #     cid = cls_code["support_set_target"]
            #     assert "class_code" in cls_code
            #     cid_to_class_code_lst[cid].append(cls_code["class_code"])
            #     if cid in cid_to_other_field.items():
            #         # ensure all the field equals
            #         for key, value in cid_to_other_field[cid]:
            #             assert cls_code[key] == value, f"key, value pair: {key, value} does not match in {cid_to_other_field}"
            #     else:
            #         cid_to_other_field[cid] = deepcopy(cls_code)
            #         # delete cls code
            #         del cid_to_other_field[cid]["class_code"]

            # # reduce a list to a single sum
            # results = []
            # for cid, code_lst in cid_to_class_code_lst.items():
            #     result = cid_to_other_field[cid]
            #     result["class_code"] = {key: functools.reduce(lambda x, y: x + y[key], code_lst, 0) for key in all_keys}
            #     if result["class_code"]["acc_weight"] != 1.0:
            #         acc_weight = result["class_code"]["acc_weight"]
            #         logger.info(f"category id: {cid}, is using only {acc_weight}, rebalance it")
            #         result["class_code"]["cls_conv"] = result["class_code"]["cls_conv"]/acc_weight
            #         result["class_code"]["cls_bias"] = result["class_code"]["cls_bias"]/acc_weight
            #         result["class_code"]["acc_weight"] = 1.0
            #     results.append(result)
            # return results
        else:
            out_codes = sub_class_codes

        if not reduce:
            return out_codes
        # print(f"code before reduce: {out_codes}")
        # reduce
        logger.info(
            f"gathered code dim: {out_codes[0]['class_code']['cls_bias'].size()}")
        return reduce_class_code(out_codes)

    def do_train(self, cfg, model, resume):
        """
        Similar to `Detectron2GoRunner.do_train` except removing `hooks.EvalHook`
        from `trainer_hooks` to avoid OOM during evaluation on full testset.
        """
        set_global_cfg(cfg)
        if not cfg.MODEL.META_LEARN.EPISODIC_LEARNING:  # for pretraining
            return super().do_train(cfg, model, resume)
        return self.do_train_per_stage(cfg, model, resume)

    def _do_test_meta_learning(self, cfg, model, train_iter=None, model_tag="default"):
        """
        Modified from _do_test, used to generate class code and inference on query images
        """
        assert len(cfg.DATASETS.TEST)
        assert cfg.OUTPUT_DIR

        is_final = (train_iter is None) or (
            train_iter == cfg.SOLVER.MAX_ITER - 1)

        logger.info(
            f"Running evaluation for model tag {model_tag} at iter {train_iter}..."
        )

        def _get_inference_dir_name(base_dir, inference_type, dataset_name, seed=""):
            return os.path.join(
                base_dir,
                inference_type,
                model_tag,
                str(train_iter) if train_iter is not None else "final",
                dataset_name,
                str(seed),
            )

        add_print_flops_callback(cfg, model, disable_after_callback=True)

        results = OrderedDict()
        results[model_tag] = OrderedDict()
        print(f"model tag: {model_tag}, train_iter: {train_iter}")
        num_repeat_test = 1
        if is_final:  # start the multiple tests at the final round
            num_repeat_test = cfg.TEST.REPEAT_TEST
        test_AP_history = defaultdict(list)  # used to cal the vairance
        for seed in range(num_repeat_test):
            logger.info(f"{seed} out of {num_repeat_test} tests.")
            results[f"seed{seed}"] = OrderedDict()
            for dataset_name in cfg.DATASETS.TEST:
                eval_with_pretrained_code = (
                    "base" in dataset_name
                    and cfg.MODEL.META_LEARN.EVAL_WITH_PRETRAINED_CODE
                )
                if eval_with_pretrained_code:
                    logger.info(
                        f"inference with pretrained class code on {dataset_name}"
                    )
                # Evaluator will create output folder, no need to create here
                output_folder = _get_inference_dir_name(
                    cfg.OUTPUT_DIR, "inference", dataset_name, seed
                )
                # -----change    start------------
                # register class codes
                classes = MetadataCatalog.get(dataset_name).thing_classes
                id_map = MetadataCatalog.get(
                    dataset_name).thing_dataset_id_to_contiguous_id
                class_codes = None
                if not eval_with_pretrained_code:
                    val_support_set_loader = (
                        self.build_episodic_learning_detection_test_support_set_loader(
                            cfg, dataset_name, seed
                        )
                    )
                    # Pass output_folder to save class codes for predictor
                    sub_class_codes = inference_on_support_set_dataset(
                        model, val_support_set_loader, output_dir=output_folder
                    )
                    few_shot_class_codes = self._gather_class_code(
                        sub_class_codes)
                    # logger.info(f"few_shot_class_codes: {few_shot_class_codes}")
                    # print(f"few_shot_class_codes: {few_shot_class_codes}")
                    if cfg.MODEL.META_LEARN.USE_ALL_GTS_IN_BASE_CLASSES:
                        # TODO: build the base support set loader
                        train_dataset_name = cfg.DATASETS.TRAIN[0]
                        base_id_map = MetadataCatalog.get(
                            train_dataset_name).thing_dataset_id_to_contiguous_id
                        base_support_set_loader = self.build_episodic_learning_detection_test_support_set_base_loader(
                            cfg, dataset_name)
                        base_sub_class_codes = inference_on_support_set_dataset_base(
                            model=model, data_loader=base_support_set_loader, all_id_map=id_map, base_id_map=base_id_map, output_dir=output_folder)
                        base_class_codes = self._gather_class_code(
                            base_sub_class_codes, reduce=True)
                        class_codes = replace_class_code(
                            few_shot_class_codes, base_class_codes, device=model.device)
                    else:
                        class_codes = few_shot_class_codes
                    # normalize class codes
                    # Put class codes on model
                    # model.train(False)
                    class_codes = inference_normalization(model, class_codes)
                    # class_codes = model(batched_inputs=None, class_code=class_codes, run_type="meta_learn_normalize_code")
                    assert len(class_codes) == len(
                        classes
                    ), f"Got {len(class_codes)} class codes for prediction, but expect to be {len(classes)}."
                    # print(f"normalized codes: {class_codes[0:2]}")
                    class_codes = format_class_codes_shared(
                        class_codes, device=model.device
                    )

                # NOTE: creating evaluator after dataset is loaded as there might be dependency.  # noqa
                data_loader = self.build_episodic_learning_detection_test_query_loader(
                    cfg, dataset_name
                )
                evaluator = self.get_evaluator(
                    cfg, dataset_name, output_folder=output_folder
                )

                if not isinstance(evaluator, DatasetEvaluators):
                    evaluator = DatasetEvaluators([evaluator])

                if comm.is_main_process():
                    tbx_writer = _get_tbx_writer(
                        get_tensorboard_log_dir(cfg.OUTPUT_DIR)
                    )
                    logger.info("Adding visualization evaluator ...")
                    vis_eval_type = FewShotVisualizationEvaluator
                    if vis_eval_type is not None:
                        mapper = self.get_few_shot_test_mapper(
                            cfg, is_train=False, need_annotation=True
                        )
                        # embeddings = (
                        #     class_codes["cls_conv"]
                        #     .view(class_codes["cls_conv"].size()[0:2])
                        #     .detach()
                        #     .cpu()
                        # )
                        evaluator._evaluators.append(
                            vis_eval_type(
                                cfg,
                                tbx_writer,
                                mapper,
                                dataset_name,
                                train_iter=train_iter,
                                tag_postfix=f"seed{seed}",
                                # class_codes=embeddings,
                                # metadata=classes,
                            )
                        )
                results_per_dataset = inference_on_dataset_with_class_codes(
                    model,
                    data_loader,
                    evaluator,
                    class_codes,
                    eval_with_pretrained_code=eval_with_pretrained_code,
                )
                # -----change    ends------------
                if comm.is_main_process():
                    # results write to tensorboard
                    results[f"seed{seed}"][dataset_name] = results_per_dataset
                    test_AP_history[dataset_name].append(
                        results_per_dataset["bbox"]["AP"])
                    if seed == 0:
                        # init
                        results[model_tag][dataset_name] = deepcopy(
                            results_per_dataset)
                    else:
                        for k in results[model_tag][dataset_name]["bbox"]:
                            results[model_tag][dataset_name]["bbox"][
                                k
                            ] += results_per_dataset["bbox"][k]
                            if seed == num_repeat_test - 1:
                                # average
                                results[model_tag][dataset_name]["bbox"][
                                    k
                                ] /= num_repeat_test
                    if is_final:
                        # compute avg and std
                        if seed == num_repeat_test - 1:
                            averaged_ap = defaultdict(list)
                            for s in range(num_repeat_test):
                                for k, v in results[f"seed{s}"][dataset_name]["bbox"].items():
                                    if k in ["AP", "APr", "APc", "APf"]:
                                        averaged_ap[k].append(v)
                            for k, lst in averaged_ap.items():
                                ap_history = np.array(lst)
                                results[model_tag][dataset_name]["bbox"][f"{k}_avg"] = ap_history.mean(
                                )
                                results[model_tag][dataset_name]["bbox"][f"{k}_std"] = ap_history.std(
                                )
                        print(
                            f"[{model_tag}]-[{dataset_name}]-{results[model_tag][dataset_name]}")
                        # print the averaged result
                        print_csv_format(results[model_tag][dataset_name])

                if is_final and cfg.TEST.AUG.ENABLED:
                    # In the end of training, run an evaluation with TTA
                    # Only support some R-CNN models.
                    output_folder = _get_inference_dir_name(
                        cfg.OUTPUT_DIR, "inference_TTA", dataset_name, seed
                    )

                    logger.info(
                        "Running inference with test-time augmentation ...")
                    data_loader = self.build_detection_test_loader(
                        cfg, dataset_name, mapper=lambda x: x
                    )
                    evaluator = self.get_evaluator(
                        cfg, dataset_name, output_folder=output_folder
                    )
                    inference_on_dataset(
                        GeneralizedRCNNWithTTA(
                            cfg, model), data_loader, evaluator
                    )

        if is_final and cfg.TEST.EXPECTED_RESULTS and comm.is_main_process():
            # assert len(results) == 1, "Results verification only supports one dataset!"
            verify_results(cfg, results[model_tag][cfg.DATASETS.TEST[0]])

        # write results to tensorboard
        if comm.is_main_process() and results:
            from detectron2.evaluation.testing import flatten_results_dict

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                tbx_writer = _get_tbx_writer(
                    get_tensorboard_log_dir(cfg.OUTPUT_DIR))
                tbx_writer._writer.add_scalar(
                    "eval_{}".format(k), v, train_iter)

        if comm.is_main_process():
            tbx_writer = _get_tbx_writer(
                get_tensorboard_log_dir(cfg.OUTPUT_DIR))
            tbx_writer._writer.flush()
        return results

    def do_test(self, cfg, model, train_iter=None):
        set_global_cfg(cfg)
        episodic_learning = cfg.MODEL.META_LEARN.EPISODIC_LEARNING
        # Pretraining test
        if not episodic_learning:
            results = super().do_test(cfg, model, train_iter)
        else:
            # Meta-learning test
            # 1. obtain class code
            # 2. apply class code and predict
            results = OrderedDict()
            with maybe_subsample_n_images(cfg) as new_cfg:
                # default model
                cur_results = self._do_test_meta_learning(
                    new_cfg, model, train_iter=train_iter, model_tag="default"
                )
                results.update(cur_results)

                # model with ema weights
                if cfg.MODEL_EMA.ENABLED:
                    logger.info("Run evaluation with EMA.")
                    with model_ema.apply_model_ema_and_restore(model):
                        cur_results = self._do_test_meta_learning(
                            new_cfg, model, train_iter=train_iter, model_tag="ema"
                        )
                        results.update(cur_results)

        return results
        # process the results
        # Add function to average AP cross different  set of classes
        for model_tag in results.keys():
            for dataset_name in results[model_tag].keys():
                # for  results[model_tag][dataset_name]["bbox"]
                dataset_split = dataset_name.split('_')
                assert len(dataset_split) > 3, logger.info(
                    "data split is less than 4")
                class_split = dataset_split[3]
                if class_split != "all":
                    continue
                # process
                base_classes = cfg.DATASETS.BASE_CLASSES_SPLIT
                novel_classes = cfg.DATASETS.NOVEL_CLASSES_SPLIT
                base_classes = MetadataCatalog.get(base_classes).thing_classes
                novel_classes = MetadataCatalog.get(
                    novel_classes).thing_classes

                #
