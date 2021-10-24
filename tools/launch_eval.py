import argparse
import getpass
import subprocess

import config_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Launch text spotting workflow")

    parser.add_argument(
        "--base_work_dir",
        default=f"/checkpoint/{getpass.getuser()}/flow/multiplexer/eval",
        help="Base working directory",
    )

    parser.add_argument("--benchmark", default="mlt19", choices=["mlt19"], help="Benchmark name")

    parser.add_argument("--gpu-type", default=None, choices=[None, "volta32gb"])

    parser.add_argument("--name", default="multiplexer_eval", help="Flow name")
    parser.add_argument(
        "--partition",
        default="learnaccel",
        help="partition for the flow job. Examples: learnaccel, devaccel",
    )
    parser.add_argument(
        "--run_type",
        default="flow",
        choices=["flow", "local"],
        help="Whether launch job in flow or run local",
    )

    parser.add_argument(
        "--work_dir",
        default=None,
        help="Work directory (if none, a random remote work_dir will be created)",
    )

    parser.add_argument(
        "opts",
        help="See config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    args.job_num = 1

    if args.work_dir is None and args.run_type != "local":
        args.work_dir = config_utils.create_random_remote_work_dir(args.base_work_dir)

    if args.benchmark == "mlt19":
        args.benchmark_target = "mlt19_eval"
    else:
        raise Exception("Benchmark target unspecified for: {}".format(args.benchmark))

    args.forwarded_opts = " ".join(args.opts).replace("==", "--")

    return args


if __name__ == "__main__":
    args = parse_args()
    print("Args: {}".format(args))
    # binary_file = config_utils.create_eval_binary_file(args)

    if args.run_type == "flow":
        raise NotImplementedError("flow mode not supported for eval")
        # submitit_utils.launch_job(mlt19_eval, args)
    else:
        assert args.run_type == "local"
        run_cmd = " ".join(
            [
                "python3",
                "multiplexer/evaluation/mlt19/mlt19_eval.py",
                "{}".format(args.forwarded_opts),
            ]
        )

    print("Run command: {}".format(run_cmd))
    subprocess.check_call(run_cmd, stderr=subprocess.STDOUT, shell=True)
