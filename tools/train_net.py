"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

#!/usr/bin/env python3

"""
Detection Training Script.
"""

import logging

import detectron2.utils.comm as comm
import torch
# from classy_vision.fb.generic.util import initialize_logger
from d2go.distributed import launch

# change this to use customized setup.py
from setup import (
    basic_argument_parser,
    post_mortem_if_fail_for_main,
    prepare_for_launch,
    setup_after_launch,
)
from d2go.utils.misc import print_metrics_table, dump_trained_model_configs
from detectron2.engine.defaults import create_ddp_model
from sylph.config.config import CfgNode  # noqa

logger = logging.getLogger(__name__)


def main(
    cfg: CfgNode,
    output_dir,
    runner=None,
    eval_only=False,
    # NOTE: always enable resume when running on cluster
    resume=True,
):
    logger.info(f"torch.cuda.is_available() {torch.cuda.is_available()}")

    setup_after_launch(cfg, output_dir, runner)
    logger.info("setup_after_launch done")
    model = runner.build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if eval_only:
        checkpointer = runner.build_checkpointer(
            cfg, model, save_dir=output_dir)
        # checkpointer.resume_or_load() will skip all additional checkpointable
        # which may not be desired like ema states
        if resume and checkpointer.has_checkpoint():
            checkpoint = checkpointer.resume_or_load(
                cfg.MODEL.WEIGHTS, resume=resume)
        else:
            checkpoint = checkpointer.load(cfg.MODEL.WEIGHTS)
        train_iter = checkpoint.get("iteration", None)
        model.eval()
        metrics = runner.do_test(cfg, model, train_iter=train_iter)
        print_metrics_table(metrics)
        return {
            "accuracy": metrics,
            "model_configs": {},
            "metrics": metrics,
        }

    model = create_ddp_model(
        model,
        fp16_compression=cfg.MODEL.DDP_FP16_GRAD_COMPRESS,
        device_ids=None if cfg.MODEL.DEVICE == "cpu" else [
            comm.get_local_rank()],
        broadcast_buffers=False,
        find_unused_parameters=cfg.MODEL.DDP_FIND_UNUSED_PARAMETERS,
    )

    trained_cfgs = runner.do_train(cfg, model, resume=resume)
    metrics = runner.do_test(cfg, model)
    print_metrics_table(metrics)

    # dump config files for trained models
    trained_model_configs = dump_trained_model_configs(
        cfg.OUTPUT_DIR, trained_cfgs)
    return {
        # for e2e_workflow
        "accuracy": metrics,
        # for unit_workflow
        "model_configs": trained_model_configs,
        "metrics": metrics,
    }


def run_with_cmdline_args(args):
    cfg, output_dir, runner = prepare_for_launch(args)
    launch(
        post_mortem_if_fail_for_main(main),
        num_processes_per_machine=args.num_processes,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        backend=args.dist_backend,
        args=(cfg, output_dir, runner, args.eval_only, args.resume),
    )


def cli():
    parser = basic_argument_parser(requires_output_dir=False)
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    run_with_cmdline_args(parser.parse_args())


if __name__ == "__main__":
    # initialize_logger()
    cli()
