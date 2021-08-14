# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import getpass
from zipfile import ZipFile

from multiplexer.evaluation.mlt19.prepare_results import prepare_results_for_evaluation
from virtual_fs import virtual_os as os
from virtual_fs import virtual_shutil as shutil
from virtual_fs.virtual_io import open


def parse_args():
    parser = argparse.ArgumentParser(description="arg parser")

    parser.add_argument(
        "--cache_dir",
        default="/mnt/vol/gfsai-east/aml/mobile-vision/gpu/jinghuang/ocr/models/outputs/SPN/multiplexer/official/inference/mlt19_val_test/cache_files/",
        help="cache dir",
    )

    parser.add_argument(
        "--charmask",
        default="off",
        choices=["on", "off"],
        help="Whether charmask recognition head is enabled",
    )

    parser.add_argument(
        "--char_map_version",
        default="none",
        choices=["none", "v1", "v2", "v3"],
        help="Char map version",
    )

    parser.add_argument(
        "--confidence_type",
        default="det",
        choices=["det", "seq", "lang"],
        help="Type of confidence to be output in final output",
    )

    parser.add_argument(
        "--filter_heuristic",
        default="equal",
        choices=["equal", "more_latin"],
        help="Confidence filter heuristic (could be language-dependent)",
    )

    parser.add_argument("-gt", default=None, help="ground truth zip file")

    parser.add_argument(
        "--intermediate_results",
        default=f"/checkpoint/{getpass.getuser()}/outputs/SPN/multiplexer/official/inference/icdar15_test/trained_model_icdar15_intermediate_results/",
        help="intermediate results, can be dir or zip",
    )

    parser.add_argument(
        "--tmp_dir",
        default="/tmp/multiplexer/intermediate/",
        help="temp intermediate results dir",
    )

    parser.add_argument(
        "--language_eval_tasks",
        default="All",
        help=(
            "Language evaluation tasks."
            " When there are multiple languages in a task, connect them with ':', e.g., en:zh"
            " When there are multiple tasks of different combo of languages, connect them with '+',"
            " For example, All+ar+en:fr:de:it+zh:ja:ko+hi:bn means we have five language-wise tasks:"
            " (1) All denotes all languages;"
            " (2) ar denotes Arabic-only eval;"
            " (3) en:fr:de:it denotes the eval for the four Latin-based languages;"
            " (4) zh:ja:ko denotes the eval for the three East Asian languages;"
            " (5) hi:bn denotes the eval for the two South Asian languages;"
            " Other than 'All', it will also expand 'Individual' to be ar+en+fr+zh+de+ko+ja+it+bn+hi"
        ),
    )

    parser.add_argument(
        "--lexicon",
        default="none",
        choices=["none", "mlt19_train", "all"],
        help="lexicon name",
    )

    parser.add_argument("-o", default=None, help="output directory")

    parser.add_argument("--overlap", default=0.2, help="IoU threshold for NMS")

    parser.add_argument(
        "--protocol",
        default="intermediate",
        choices=["direct", "intermediate"],
        help="Evaluation protocol (direct/intermediate)",
    )

    parser.add_argument("-s", default=None, help="submission zip file")

    parser.add_argument("--score_det", type=float, default=0, help="detection confidence threshold")

    parser.add_argument(
        "--score_rec_charmask",
        type=float,
        default=0,
        help="charmask recognition head confidence threshold",
    )

    parser.add_argument(
        "--score_rec_seq",
        type=float,
        default=0,
        help="sequential recognition head confidence threshold",
    )

    parser.add_argument(
        "--seq",
        default="on",
        choices=["on", "off"],
        help="Whether sequential recognition head is enabled",
    )

    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Validation set or test set",
    )

    parser.add_argument(
        "--task",
        default="task4",
        choices=["task1", "task3", "task4"],
        help="Evaluation task",
    )

    parser.add_argument(
        "--zip_per_gpu",
        action="store_true",
        default=False,
        help="results were saved in separate files from each gpu",
    )

    args = parser.parse_args()

    if args.charmask == "on":
        args.use_rec_charmask = True
    else:
        args.use_rec_charmask = False

    if args.seq == "on":
        args.use_rec_seq = True
    else:
        args.use_rec_seq = False

    if args.task == "task1":
        from multiplexer.evaluation.mlt19.task1.script import mlt19_eval_task1, rrc_evaluation_funcs

        args.eval_task = mlt19_eval_task1
    elif args.task == "task3":
        from multiplexer.evaluation.mlt19.task3.script import mlt19_eval_task3, rrc_evaluation_funcs

        args.eval_task = mlt19_eval_task3

    elif args.task == "task4":
        from multiplexer.evaluation.mlt19.task4.script import mlt19_eval_task4, rrc_evaluation_funcs

        args.eval_task = mlt19_eval_task4
    else:
        raise Exception("Unknown task: {}".format(args.task))

    args.print_eval_result = rrc_evaluation_funcs.print_eval_result

    if args.gt is None:
        # Default ground truth zip file if not specified
        args.gt = f"/checkpoint/{getpass.getuser()}/datasets/MLT19/eval/val/{args.task}/gt.zip"
        print(f"[Info] Using default gt zip file: {args.gt}")

    language_task_str_list = args.language_eval_tasks.split("+")
    args.language_tasks = []
    supported_languages = ["ar", "en", "fr", "zh", "de", "ko", "ja", "it", "bn", "hi"]
    for language_task_str in language_task_str_list:

        if language_task_str == "All":
            # None means all languages are to be processed
            args.language_tasks.append(None)
        elif language_task_str == "Individual":
            assert (
                args.split == "val"
            ), "language-wise evaluation is only supported for validation set"
            # 'Individual' is equivalent to ar+en+fr+zh+de+ko+ja+it+bn+hi
            # i.e., we will evaluate the individual performance for each language
            for lang in supported_languages:
                args.language_tasks.append([lang])
        else:
            assert (
                args.split == "val"
            ), "language-wise evaluation is only supported for validation set"
            language_task = language_task_str.split(":")
            for lang in language_task:
                assert lang in supported_languages, f"Unsupported language: {lang}"

            args.language_tasks.append(language_task)

    if args.protocol == "intermediate":
        assert args.s is None, "Intermediate mode will auto-generate the submission file."
    else:
        assert args.protocol == "direct"
        assert args.s is not None, "Submission zip file is required for direct evaluation mode."

    if args.lexicon == "none":
        args.lexicon = None
    else:
        assert args.char_map_version != "none", (
            "Please specify the char map version (matching the version in CHAR_MAP.DIR"
            "in your config) in order for the lexicon to function properly"
        )

    # if input is a zip file, then assume it's zipped results, unzip to tmp folder
    if os.path.splitext(args.intermediate_results)[1] == ".zip":
        # if temp dir exists, remove everything
        if os.path.isdir(args.tmp_dir):
            shutil.rmtree(args.tmp_dir)

        if not os.path.isdir(args.tmp_dir):
            os.makedirs(args.tmp_dir)

        if args.zip_per_gpu:
            assert not os.path.isfile(args.intermediate_results)
            for i in range(8):
                base_path = os.path.splitext(args.intermediate_results)[0]
                zip_file = base_path + "_part{}.zip".format(i)
                assert os.path.isfile(zip_file), f"{zip_file} doesn't exist or is not a file!"
                # unzip all files to temp dir
                print("unzip intermediate results from: {} to {}...".format(zip_file, args.tmp_dir))
                with open(zip_file, "rb") as buffer:
                    with ZipFile(buffer, "r") as zip_ref:
                        zip_ref.extractall(args.tmp_dir)
        else:
            assert os.path.isfile(args.intermediate_results)
            # unzip all files to temp dir
            print(
                "unzip intermediate results from: {} to {}...".format(
                    args.intermediate_results, args.tmp_dir
                )
            )
            with open(args.intermediate_results, "rb") as buffer:
                with ZipFile(buffer, "r") as zip_ref:
                    zip_ref.extractall(args.tmp_dir)

        # point intermediate results dir to temp dir
        args.intermediate_results = args.tmp_dir

    return args


if __name__ == "__main__":
    args = parse_args()

    eval_results = []

    for language_task in args.language_tasks:
        if args.protocol == "intermediate":
            args.s = prepare_results_for_evaluation(
                results_dir=args.intermediate_results,
                task=args.task,
                cache_dir=args.cache_dir,
                score_det=args.score_det,
                score_rec_seq=args.score_rec_seq,
                score_rec_charmask=args.score_rec_charmask,
                overlap=args.overlap,
                use_rec_seq=args.use_rec_seq,
                use_rec_charmask=args.use_rec_charmask,
                confidence_type=args.confidence_type,
                languages=language_task,
                split=args.split,
                filter_heuristic=args.filter_heuristic,
                lexicon=args.lexicon,
                char_map_version=args.char_map_version,
            )

        if args.split == "val":
            eval_result = args.eval_task(
                pred_zip_file=args.s,
                gt_zip_file=args.gt,
                output_dir=args.o,
                languages=language_task,
            )

            eval_results.append(eval_result)

    if args.split == "val":
        print("~~~Final Evaluation Summary~~~")

        for (language_task, eval_result) in zip(args.language_tasks, eval_results):
            args.print_eval_result(eval_result=eval_result, languages=language_task)
    else:
        assert args.split == "test"
        print("Next: submit the zip files to ICDAR MLT19 website for official evaluation")

    # clean up temp dir if exists
    # if os.path.isdir(args.tmp_dir):
    #     rmtree(args.tmp_dir)
