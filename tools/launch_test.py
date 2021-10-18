import argparse
import subprocess

import config_utils
import submitit_utils
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Launch text spotting test workflow")

    parser.add_argument(
        "--base_work_dir",
        default=f"/checkpoint/{os.getusername()}/flow",
        help="Base working directory",
    )

    parser.add_argument("--dataset", default=None, help="Dataset name")
    parser.add_argument(
        "--dataset_ratios", default=None, help="Dataset ratios, e.g., 1:2:5"
    )
    parser.add_argument(
        "--partition",
        default="learnaccel",
        help="partition for the flow job. Examples: learnaccel, devaccel",
    )
    parser.add_argument(
        "--gpu-type", default="volta32gb", choices=["volta32gb"]
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--num_machines", default=1, type=int)
    parser.add_argument(
        "--num_cpus", default=40, type=int, help="Number of CPUs **per machine**"
    )
    parser.add_argument(
        "--num_gpus", default=8, type=int, help="Number of GPUs **per machine**"
    )
    parser.add_argument("--ram-gb", default=200, type=int)
    parser.add_argument("--retry", default=1, type=int, help="Number of retries")
    parser.add_argument("--language_heads", default=None, help="Language heads")
    parser.add_argument(
        "--language_heads_enabled", default=None, help="Enabled language heads"
    )
    parser.add_argument("--name", default="spn_test", help="Flow name")
    parser.add_argument(
        "--run_type",
        default="flow",
        choices=["flow", "local"],
        help="Whether launch job in flow or run local",
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
        "--work_dir",
        default=None,
        help="Work directory (if none, a random remote work_dir will be created)",
    )

    parser.add_argument(
        "--yaml", default="config.yaml", help="Default YAML configuration file"
    )

    parser.add_argument(
        "--yaml_dir",
        default=None,
        help="The directory containing YAML configuration file",
    )

    parser.add_argument(
        "opts",
        help="See config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    args.job_list = args.dataset.split("+")
    args.job_num = len(args.job_list)

    args.name_list = []
    for job in args.job_list:
        name = args.name + "_" + (job if len(job) < 80 else (job[:79] + "_etc"))
        args.name_list.append(name)

    args.work_dir_list = []
    if args.work_dir is None:
        for _ in args.job_list:
            work_dir = config_utils.create_random_remote_work_dir(args.base_work_dir)
            args.work_dir_list.append(work_dir)
    else:
        if args.job_num == 1:
            args.work_dir_list.append(args.work_dir)
        else:
            for _ in args.job_list:
                work_dir = config_utils.create_random_remote_work_dir(args.work_dir)
                args.work_dir_list.append(work_dir)

    if args.language_heads_enabled is None:
        # If language_heads_enabled is not specified, default to language_heads
        args.language_heads_enabled = args.language_heads

    if args.unfreezed_seq_heads is None:
        # If unfreezed_seq_heads is not specified, default to language_heads_enabled
        args.unfreezed_seq_heads = args.language_heads_enabled

    args.forwarded_opts = ""

    return args


if __name__ == "__main__":
    args = parse_args()

    print("Args: {}".format(args))
    binary_file = config_utils.create_test_binary_file(args)
    for job_id in range(args.job_num):
        args.dataset = args.job_list[job_id]
        args.work_dir = args.work_dir_list[job_id]
        args.name = args.name_list[job_id]
        config_file = config_utils.create_config_file(args)

        if args.run_type == "flow":
            submitit_utils.launch_single_node(args, binary_file, config_file)
        else:
            assert args.run_type == "local"
            run_cmd = " ".join(
                [
                    binary_file,
                    f"--num-gpus {args.num_gpus}",
                    f"--config-file {config_file}",
                    f"{args.forwarded_opts}",
                ]
            )

            print("Run command: {}".format(run_cmd))
            subprocess.check_call(run_cmd, stderr=subprocess.STDOUT, shell=True)
