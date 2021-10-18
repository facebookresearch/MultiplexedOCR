import os, sys
multiplexer_dir = os.path.abspath('./')
print(multiplexer_dir)
# multiplexer_dir = "/private/home/jinghuang/code/ocr/multiplexer/"
if multiplexer_dir not in sys.path:
    sys.path.append(multiplexer_dir)

import argparse
import datetime

import config_utils
import submitit_utils
from virtual_fs import virtual_os as os
from tools.train_net import detectron2_launch
import getpass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch text spotting training workflow"
    )

    parser.add_argument(
        "--base_work_dir",
        default=f"/checkpoint/{getpass.getuser()}/flow",
        help="Base working directory",
    )
    parser.add_argument(
        "--batch_job_name",
        default=None,
        help="The name of the batch of the training jobs",
    )

    parser.add_argument("--dataset", default=None, help="Dataset name")
    parser.add_argument(
        "--dataset_ratios", default=None, help="Dataset ratios, e.g., 1:2:5"
    )
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument(
        "--partition",
        default="learnaccel",
        help="partition for the flow job. Examples: learnaccel, devaccel",
    )
    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument(
        "--gpu-type", default="volta32gb", choices=["volta32gb"]
    )
    parser.add_argument("--language_heads", default=None, help="Language heads")
    parser.add_argument(
        "--language_heads_enabled", default=None, help="Enabled language heads"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    parser.add_argument(
        "--min_size_train",
        default=None,
        help="string format for INPUT.MIN_SIZE_TRAIN, e.g., 512:1024:2048",
    )
    parser.add_argument("--name", default="spn_train", help="Flow name")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--num_machines", default=1, type=int)
    parser.add_argument(
        "--num_cpus", default=40, type=int, help="Number of CPUs **per machine**"
    )
    parser.add_argument(
        "--num_gpus", default=8, type=int, help="Number of GPUs **per machine**"
    )
    parser.add_argument("--ram-gb", default=200, type=int)
    parser.add_argument("--retry", default=1, type=int, help="Number of retries")
    parser.add_argument(
        "--run_type",
        default="flow",
        choices=["flow", "local"],
        help="Whether launch job in flow or run local",
    )
    parser.add_argument(
        "--solver_steps", default=None, help="Modify SOLVER.STEPS and SOLVER.MAX_ITER"
    )
    parser.add_argument(
        "--train_from_scratch",
        action="store_true",
        help="If enabled, MODEL.WEIGHT will be set to be empty",
    )
    parser.add_argument(
        "--unfreezed_seq_heads",
        default=None,
        help=(
            "Unfreezed sequential recognition heads."
            "If not set, default to be the same as language_heads_enabled"
        ),
    )
    parser.add_argument(
        "--work_dir",  # TODO: remove
        default=None,
        help="Work directory (if none, a random remote work_dir will be created)",
    )
    parser.add_argument(
        "--yaml",
        default="config.yaml",
        help="Default YAML configuration file",
    )

    parser.add_argument(
        "--yaml_dir",
        default="configs",
        help="The directory containing YAML configuration file",
    )
    parser.add_argument("opts", help="See config/defaults.py for all options", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert args.num_machines >= 1

    if args.language_heads_enabled is None:
        # If language_heads_enabled is not specified, default to language_heads
        args.language_heads_enabled = args.language_heads

    if args.unfreezed_seq_heads is None:
        # If unfreezed_seq_heads is not specified, default to language_heads_enabled
        args.unfreezed_seq_heads = args.language_heads_enabled

    # args.job_list = args.dataset.split("+")
    args.job_list = args.unfreezed_seq_heads.split("+")
    args.job_num = len(args.job_list)
    # assert len(args.unfreezed_seq_heads_list) == len(args.job_list)

    args.name_list = []
    for job in args.job_list:
        name = args.name + "_" + job
        args.name_list.append(name)

    if args.batch_job_name is None:
        # use date as the default batch job name
        args.batch_job_name = datetime.date.today().strftime("%Y%m%d")

    args.work_dir_list = []

    max_id = 100
    batch_job_dir = os.path.join(args.base_work_dir, args.batch_job_name)
    for job in args.job_list:
        id = 1
        while id < max_id:
            work_dir = os.path.join(batch_job_dir, f"{job}-{id}")
            if os.path.exists(work_dir):
                id += 1
            else:
                os.makedirs(work_dir, exist_ok=False)
                args.work_dir_list.append(work_dir)
                break
        assert (
            id < max_id
        ), f"Could not find an available id for {batch_job_dir}/{job}"


    args.forwarded_opts = ""

    # Temporary alert for migration to newer yaml configs
    assert args.yaml != "multi_seq_lang.yaml", (
        "multi_seq_lang.yaml has been renamed as multi_seq_lang_v1.yaml."
        "Please use multi_seq_lang_v2.yaml for more concise char maps that are ordered by frequency."
        "You can set DATASETS.AUG=True to enable default data augmentation"
        "Also, you can use multi_seq_lang_v2_aug.yaml to play with experimental CropperV2 augmentation"
    )

    return args


if __name__ == "__main__":
    args = parse_args()
    print("Args: {}".format(args))

    # binary_file = config_utils.create_train_binary_file(args)

    for job_id in range(args.job_num):
        args.job_id = job_id
        args.work_dir = args.work_dir_list[job_id]
        args.name = args.name_list[job_id]

        args.config_file = config_utils.create_config_file(args)

        if args.run_type == "flow":
            submitit_utils.launch_single_node(detectron2_launch, args)
        else:
            detectron2_launch(args)

