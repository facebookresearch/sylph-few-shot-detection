#!/usr/bin/env python3

import logging
import os
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)

import fblearner.flow.api as flow

# @dep=//mobile-vision/d2go/d2go:d2go-all-projects
from d2go.runner import create_runner
from d2go.utils.misc import get_tensorboard_log_dir
from detectron2.config import set_global_cfg
from fblearner.flow.api import types
from fblearner.flow.core.future import fut

# @dep=//fblearner/flow/projects/common:workflow-lib
from fblearner.flow.projects.mobile_vision.detectron2go.core import operators
from fblearner.flow.projects.mobile_vision.detectron2go.core.common import (
    DistributedConfig,
    OperatorArgument,
    Resources,
    SingleCPUResources,
    SingleGPUResources,
)

# @dep=//fblearner/flow/projects/mobile_vision/utils:utils
from fblearner.flow.projects.mobile_vision.utils import checkpoint_utils
from fblearner.flow.projects.mobile_vision.utils.config_utils import (
    cfg_cloner,
    clone_overwrite_cfg,
    create_cfg,
    merge_from_file_updater,
    merge_from_list_updater,
    pick_first_config_file,
    update_cfg,
)
from fblearner.flow.projects.mobile_vision.utils.metrics_table import make_2d_html_table


def get_tensorboard_vis_metrics(working_dir):
    log_dir = get_tensorboard_log_dir(working_dir)
    return types.VISUALIZATION_METRICS.new(log_dir)


def update_metrics(old_metrics, new_metrics, namespace=None):
    # NOTE: new_metrics should not conflict with old_metrics, since the top-level keys
    # are model name and model names are unique. Maybe need to add some check for this.
    if namespace is not None:
        new_metrics = {
            "{}/{}".format(new_metrics, k): v for k, v in new_metrics.itmes()
        }
    old_metrics.update(new_metrics)
    return old_metrics


@flow.flow_async()
@flow.registered(owners=["oncall+fai4ar"])
@flow.typed(
    returns=types.Schema(
        output_dir=types.STRING,
        vis_metrics=types.NULLABLE(types.VISUALIZATION_METRICS),
        model_configs=types.DICT(types.STRING, types.STRING),
        accuracy_table_raw=types.DICT(types.TEXT, types.ANY),
        accuracy_table_html=types.HTML,
        best_model_data=types.NULLABLE(types.DICT(types.TEXT, types.ANY)),
    )
)
@checkpoint_utils.auto_output_dir()
def e2e_workflow(
    config_file: str,
    output_dir: str = "auto://MANIFOLD_MV_WORKFLOW",
    runner_name: Optional[str] = None,
    pre_training_opts: Optional[List[str]] = None,
    post_training_opts: Optional[List[str]] = None,
    e2e_train: Optional[OperatorArgument] = None,
    eval_pytorch: Optional[OperatorArgument] = None,
    async_validation: Optional[OperatorArgument] = None,
):
    workflow_wd = output_dir
    logger.info("Using workflow_wd: {}\n".format(workflow_wd))
    runner = create_runner(runner_name)
    logger.info("Using runner: {}\n".format(runner))

    logger.info("Merge from pre_training_opts: \n{}\n".format(pre_training_opts))
    base_cfg = create_cfg(runner.get_default_cfg(), config_file, pre_training_opts)

    accuracy_table = OrderedDict()
    model_configs = {}

    train_cfg = None
    if e2e_train:
        logger.info("Running e2e_train with:\n{}\n".format(e2e_train))
        train_dc = e2e_train.dist_config or e2e_train.resources.infer_gpu_dist_config()
        train_cfg = update_cfg(
            base_cfg,
            cfg_cloner,
            merge_from_list_updater(e2e_train.overwrite_opts),
        )
        logger.info(f"train_cfg:\n {train_cfg}")
        set_global_cfg(train_cfg)
        e2e_train_result = operators.train_net_as_gang(
            train_cfg,
            output_dir=os.path.join(workflow_wd, "e2e_train"),
            runner=runner,
            resources=e2e_train.resources,
            dist_config=train_dc,
            custom_name="e2e_train",
            # binary specific optional arguments
            **e2e_train.extra_args,
        )

        # when e2e_train is given, update base_cfg to use newly trained config file
        base_cfg = update_cfg(
            base_cfg,
            fut(merge_from_file_updater)(
                fut(pick_first_config_file)(e2e_train_result.model_configs)
            ),
        )
        set_global_cfg(train_cfg)
        accuracy_table["e2e_train"] = e2e_train_result.accuracy
        model_configs = e2e_train_result.model_configs

    if async_validation:
        logger.info("Running async_validation with \n{}\n".format(async_validation))
        async_cfg = train_cfg or base_cfg
        async_validation_fut = operators.async_validation(
            cfg=async_cfg,
            output_dir=os.path.join(workflow_wd, "async_validation"),
            runner=runner,
            resources=async_validation.resources,
            dist_config=async_validation.dist_config,
            checkpoint_dir=(
                async_validation.extra_args.pop("checkpoint_dir", None)
                or os.path.join(workflow_wd, "e2e_train")
            ),
            custom_name="async_validation",
        )

    logger.info("Merge from post_training_opts: \n{}\n".format(post_training_opts))
    base_cfg = clone_overwrite_cfg(base_cfg, post_training_opts)

    if eval_pytorch:
        logger.info("Running eval_pytorch with:\n{}\n".format(eval_pytorch))
        eval_cfg = clone_overwrite_cfg(base_cfg, eval_pytorch.overwrite_opts)
        set_global_cfg(eval_cfg)
        eval_pytorch_result = operators.test_net_as_gang(
            eval_cfg,
            output_dir=os.path.join(workflow_wd, "eval_pytorch"),
            runner=runner,
            resources=eval_pytorch.resources,
            dist_config=eval_pytorch.dist_config,
            custom_name="eval_pytorch",
        )
        accuracy_table["eval_pytorch"] = eval_pytorch_result.accuracy

    if async_validation:
        accuracy_table = fut(lambda a, b: dict(a, **b))(
            accuracy_table, async_validation_fut["results_all_models"]
        )

    return flow.Output(
        output_dir=workflow_wd,
        vis_metrics=get_tensorboard_vis_metrics(workflow_wd),
        model_configs=model_configs,
        accuracy_table_raw=accuracy_table,
        accuracy_table_html=fut(make_2d_html_table)(accuracy_table),
        best_model_data=async_validation_fut["results_best_model"]
        if async_validation
        else None,
    )
