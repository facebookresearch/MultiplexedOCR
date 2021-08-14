# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
# from multiplexer.utils.env import setup_environment  # noqa F401 isort:skip

import argparse

from multiplexer.checkpoint import DetectionCheckpointer
from multiplexer.config import cfg
from multiplexer.data import make_data_loader
from multiplexer.engine.launch import launch
from multiplexer.engine.text_inference import inference
from multiplexer.modeling import build_model
from multiplexer.utils.collect_env import collect_env_info
from multiplexer.utils.comm import get_rank
from multiplexer.utils.logging import setup_logger
from virtual_fs import virtual_os as os

# from multiplexer.data.datasets import extract_datasets
# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError("Use APEX for mixed precision via apex.amp")


def test(cfg, distributed):
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == "float16"
    amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    checkpointer = DetectionCheckpointer(cfg, model)
    checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            os.makedirs(output_folder, exist_ok=True)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    model_name = cfg.MODEL.WEIGHT.split("/")[-1]
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        inference(
            model,
            data_loader_val,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            model_name=model_name,
            cfg=cfg,
        )
        # synchronize()


def main(cfg, args):
    args.distributed = args.num_gpus > 1
    if args.distributed:
        args.local_rank = get_rank() % args.num_gpus

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("multiplexer", output_dir, get_rank())
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    test(cfg, args.distributed)


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--no-color", action="store_true", help="disable colorful logging")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus per machine")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    # parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


def detectron2_launch(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if cfg.MODEL.DEVICE == "cuda":
        # set cfg.TEST.IMS_PER_BATCH to at least number of gpus available
        # (to ensure each gpu has at least 1 image)
        world_size = args.num_gpus * args.num_machines
        if cfg.TEST.IMS_PER_BATCH < world_size:
            print(
                "[WARNING] cfg.TEST.IMS_PER_BATCH is too small, "
                "setting to total number of gpus: {}".format(world_size)
            )
            cfg.TEST.IMS_PER_BATCH = world_size

        cfg.freeze()

        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(cfg, args),
        )
    else:
        cfg.freeze()
        test(cfg, distributed=False)


if __name__ == "__main__":
    detectron2_launch(parse_args())
