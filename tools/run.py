#!/usr/bin/env python3

"""Workflow launcher script.

Run locally:
```
CONFIG_FILE=path_to_yaml
./run.py \
   --name 'owd_run' --num-gpus 8 \
   --config-file "sylph://COCO-Meta-FCOS-Detection/Base-Meta-FCOS-pretrain_owd.yaml" \
   --output-dir /tmp/testing
   --gpu-type P100
```

Run remotely:
```
CONFIG_FILE=path_to_yaml
./run.py \
   --config-file $CONFIG_FILE \
   --entitlement gpu_prod --name 'my_remote_training_run' \
   --nodes 1 --num-gpus 8 --gpu-type P100 \
   --output-dir /mnt/vol/gfsai-east/users/$USER/d2go/runs/ \
   --run-as-secure-group extreme_vision \
   --canary
```

Optional argument:
--workflow {e2e_workflow}   Specify which workflow to launch in workflow.py
                            default is mobile_vision.detectron2go.workflow.e2e_workflow
--eval                      Flag for evaluation only
--async-val                 Flag for async validation
--skip-build
OVERRIDE.PARAM VAL          overwrite config

"""


import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime


try:
    from shlex import quote as shlex_quote
except ImportError:
    from pipes import quote as shlex_quote


def parse_args():
    parser = argparse.ArgumentParser(description="beta runner")
    parser.add_argument(
        "--canary", action="store_true", help="Run a canary instead of a local test"
    )
    parser.add_argument("--force-build", action="store_true", help="Force a full build")
    parser.add_argument(
        "--skip-build", action="store_true", help="Skip a full build"
    )
    parser.add_argument("--entitlement", help="GPU entitlement to use", type=str)
    parser.add_argument(
        "--run-as-secure-group", help="Secure group to run as", type=str
    )

    parser.add_argument(
        "--nodes", help="number of nodes to run on", type=int, default=1
    )

    parser.add_argument(
        "--num-gpus", help="number of gpus per node", type=int, default=8
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="V100",
        choices=["M40", "P100", "V100", "V100_32G", "A100"],
    )
    parser.add_argument("--name", help="name for this run", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--build-mode", type=str)

    parser.add_argument(
        "--workflow",
        type=str,
        default="e2e_workflow",
        choices=["e2e_workflow", "d2go_workflow", "meta_fcos_e2e_workflow"],
    )
    parser.add_argument(
        "--config-file", help="YAML configuration file", type=str, required=True
    )
    parser.add_argument(
        "--runner",
        help="D2GO Runner name",
        type=str,
        default="sylph.runner.MetaFCOSRunner",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run eval instead of training"
    )

    parser.add_argument("--async-val", action="store_true", help="Run async val")

    parser.add_argument(
        "--resume", action="store_true", help="Resume an interrupted run"
    )

    parser.add_argument(
        "config_overrides",
        help="See lib/core/config.py for all options you can override",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


def v(maybe_none, default):
    """Shorter one-liner for a || b"""
    return maybe_none if maybe_none else default


def main():
    args = parse_args()

    if args.canary:
        assert args.entitlement, "Must provide entitlement for canary mode"
        assert args.run_as_secure_group, (
            "Must provide secure group for canary mode, see "
            "https://fb.workplace.com/groups/flow.fyi/permalink/2424065044308763/"
        )

    output_dir = (
        args.output_dir
        if args.resume
        else os.path.join(args.output_dir, datetime.today().strftime("%Y%m%d%H%M%S"))
    )

    if args.eval:
        run_type = "eval_pytorch"
    else:
        run_type = "e2e_train"
    # Example:fblearner/flow/projects/mobile_vision/detectron2go/core/examples/all_steps.json
    if args.entitlement == "vll":
        args.entitlement = "ar_rp_vll"
        args.gpu_type = "V100"
    elif args.entitlement == "ncg":
        args.entitlement = "ar_rp_ncg"
        args.gpu_type = "A100"
    params = {
        "config_file": args.config_file,
        "output_dir": output_dir,
        "runner_name": args.runner,
        run_type: {
            "dist_config": {
                "num_machines": args.nodes,
                "num_processes_per_machine": args.num_gpus,
                "gang_schedule": False,
                "gang_affinity": False,
            },
            "resources": {
                "memory": "225g",
                "capabilities": _get_capabilities(args.gpu_type),
            },
            "overwrite_opts": args.config_overrides,
        },
    }

    if args.async_val:
        assert not args.eval, "Run async_val only when training"
        params["async_validation"] = {
            "dist_config": {"num_processes_per_machine": args.num_gpus},
            "extra_args": {"metrics": ["AP"]},
        }

    # --custom-build-options="-c misc.strip_binaries=debug-non-line" --buck-target {bucket_target}
    cmd = """\
/usr/local/bin/flow-cli {mode} {workflow}@few_shot_detection {force_build} {skip_build} \
--name "{name}" --run-as-secure-group "{run_as_secure_group}" \
--parameters-json {param_json} {entitlement} \
--mode {build_mode}
""".format(
        mode="canary" if args.canary else "test-locally",
        workflow={
            "e2e_workflow": "vision.few_shot_detection.workflow.e2e_workflow",
            "d2go_workflow": "mobile_vision.detectron2go.workflow.e2e_workflow",
            "meta_fcos_e2e_workflow": "vision.few_shot_detection.meta_fcos_workflow.e2e_workflow",
        }[args.workflow],
        # project_name = "meta_fcos",
        force_build="--force-build-yes-really" if args.force_build else "",
        skip_build="--skip-build" if args.skip_build else "",
        name=args.name,
        run_as_secure_group=args.run_as_secure_group,
        param_json=shlex_quote(json.dumps(params)),
        entitlement="--entitlement {}".format(args.entitlement) if args.canary else "",
        build_mode=v(args.build_mode, "opt" if args.canary else "dev-nosan"), #opt-split-dwarf
    )

    print("Executing: {}".format(cmd))
    subprocess.call(shlex.split(cmd))


def _get_capabilities(gpu_type):
    if gpu_type == "M40" or gpu_type == "GPU_M40_HOST":
        return ["GPU_M40_HOST"]
    elif gpu_type == "P100" or gpu_type == "GPU_P100_HOST":
        return ["GPU_P100_HOST"]
    elif gpu_type == "A100" or gpu_type == "GPU_A100_HOST":
        return ["GPU_A100_HOST"]
    elif gpu_type == "V100" or gpu_type == "GPU_V100_HOST":
        return ["GPU_V100_HOST"]
    elif gpu_type == "V100_32G" or gpu_type == "GPU_V100_32G_HOST":
        return ["GPU_V100_32G_HOST"]
    else:
        return None


if __name__ == "__main__":
    main()
