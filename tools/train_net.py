# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
# from multiplexer.utils.env import setup_environment  # isort:skip

import argparse

import torch

from multiplexer.checkpoint import DetectionCheckpointer
from multiplexer.config import cfg
from multiplexer.data import make_data_loader
from multiplexer.engine.launch import launch
from multiplexer.engine.train_loop import do_train
from multiplexer.modeling import build_model
from multiplexer.solver import make_lr_scheduler, make_optimizer
from multiplexer.utils.collect_env import collect_env_info
from multiplexer.utils.comm import get_rank
from multiplexer.utils.logging import Logger, setup_logger
from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

# from multiplexer.data.datasets import extract_datasets
# from multiplexer.engine.inference import inference
# from multiplexer.utils.comm import get_world_size, synchronize


try:
    from apex import amp
except ImportError:
    raise ImportError("Use APEX for multi-precision via apex.amp")


def train(cfg, local_rank, distributed, tb_logger):
    # torch.autograd.set_detect_anomaly(True)  # For debugging
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = "O1" if use_mixed_precision else "O0"
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectionCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.SOLVER.RESUME)
    # Note: even if cfg.SOLVER.RESUME is False, resume would be enabled if last_checkpoint
    # is detected under save_dir since it's likely the job got preempted/restarted
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        tb_logger,
        cfg,
        local_rank,
    )

    return model


# def test(cfg, model, distributed):
#     if distributed:
#         model = model.module
#     torch.cuda.empty_cache()  # TODO check if it helps
#     iou_types = ("bbox",)
#     if cfg.MODEL.MASK_ON:
#         iou_types = iou_types + ("segm",)
#     output_folders = [None] * len(cfg.DATASETS.TEST)
#     if cfg.OUTPUT_DIR:
#         dataset_names = cfg.DATASETS.TEST
#         for idx, dataset_name in enumerate(dataset_names):
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
#             mkdir(output_folder)
#             output_folders[idx] = output_folder
#     data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
#     for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
#         inference(
#             model,
#             data_loader_val,
#             iou_types=iou_types,
#             box_only=cfg.MODEL.RPN_ONLY,
#             device=cfg.MODEL.DEVICE,
#             expected_results=cfg.TEST.EXPECTED_RESULTS,
#             expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
#             output_folder=output_folder,
#         )
#         synchronize()


def main(cfg, args):

    args.distributed = args.num_gpus > 1
    if args.distributed:
        args.local_rank = get_rank() % args.num_gpus
    else:
        args.local_rank = 0

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("multiplexer", output_dir, get_rank())
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    tb_logger = Logger(cfg.OUTPUT_DIR, get_rank())
    train(cfg, args.local_rank, args.distributed, tb_logger)

    # if not args.skip_test:
    #     test(cfg, model, args.distributed)


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no-color", action="store_true", help="disable colorful logging")
    parser.add_argument("--num-gpus", type=int, default=8, help="number of gpus per machine")
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
    cfg.freeze()
    # extract_datasets(cfg)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            cfg,
            args,
        ),
    )


def pytorch_launch(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    args.distributed = args.num_gpus > 1

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("multiplexer", output_dir, get_rank())
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    tb_logger = Logger(cfg.OUTPUT_DIR, get_rank())
    train(cfg, args.local_rank, args.distributed, tb_logger)


if __name__ == "__main__":
    detectron2_launch(parse_args())
    # pytorch_launch(parse_args())
