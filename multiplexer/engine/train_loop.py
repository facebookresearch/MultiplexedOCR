#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import sys
import time
import traceback
from collections import OrderedDict
import torch
import torch.distributed as dist
from apex import amp

from multiplexer.utils.comm import get_world_size
from multiplexer.utils.metric_logger import MetricLogger

logger = logging.getLogger("multiplexer.train_loop")


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def continuable_generator(gen):
    # This is used to catch the exceptions in the DataLoader generator
    count_failures = 0
    stop_iter = False
    while count_failures < 20:
        try:
            yield next(gen)
        except StopIteration:
            stop_iter = True
            break
        except Exception:
            logger.info("Exception for failure {}:".format(count_failures))
            exc_info = sys.exc_info()
            count_failures += 1
            # Display the *original* exception
            traceback.print_exception(*exc_info)
            del exc_info

    # Don't catch non-StopIteration exceptions any more if they occur more than 20 times
    if not stop_iter:
        yield next(gen)


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def freeze_bn(model, cfg):
    # Special handling for freezing BN

    if cfg.MODEL.SEG.BN_FROZEN:
        print("[Info] Freezing Mean/Var of BatchNorm in SEG head.")
        try:
            if hasattr(model, "module"):
                # For DistributedDataParallel,
                # the original model is now at self.model.module instead of self.model
                model.module.proposal.head.seg_out.apply(set_bn_eval)
            else:
                model.proposal.head.seg_out.apply(set_bn_eval)
        except AttributeError:
            print(
                (
                    "[Warning] proposal.head.seg_out doesn't exist in the current model,"
                    " skipped freezing"
                )
            )
    if cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR_FROZEN:
        print(
            "[Info] Freezing Mean/Var of BatchNorm in the feature extractor of the ROI mask head."
        )
        try:
            if hasattr(model, "module"):
                # For DistributedDataParallel,
                # the original model is now at self.model.module instead of self.model
                model.module.roi_heads.mask.feature_extractor.apply(set_bn_eval)
            else:
                model.roi_heads.mask.feature_extractor.apply(set_bn_eval)
        except AttributeError:
            print(
                (
                    "[Warning] roi_heads.mask.feature_extractor doesn't exist in the current model,"
                    " skipped freezing"
                )
            )
            
def infer_grouping(model, cfg):
    lang_head_weights = model.module.roi_heads.mask.predictor.language_grouper.lang_head_weights
    # print(lang_head_weights)
    group_dict = {}
    for i in range(len(cfg.SEQUENCE.LANGUAGES)):
        group_id = torch.argmax(lang_head_weights[i]).item()
        if group_id not in group_dict:
            group_dict[group_id] = []
        group_dict[group_id].append(cfg.SEQUENCE.LANGUAGES[i])
    # print(f"# {OrderedDict(sorted(group_dict.items()))}")
    return OrderedDict(sorted(group_dict.items()))


def do_train(
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
):
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()

    freeze_bn(model, cfg)

    start_training_time = time.time()
    end = time.time()
    # for iteration in range(start_iter, max_iter):
    #     try:
    #         (images, targets, _) = next(iter(data_loader))
    #     except StopIteration:
    #         print("[Warning] Hit StopIteration at iter {}".format(iteration))
    #         break
    
    current_grouping = None
    count_grouping_change = 0

    for iteration, (images, targets, _) in continuable_generator(
        enumerate(data_loader, start_iter)
    ):
        if images is None:
            logger.info("[WARNING] empty batch at iter {}, skipping".format(iteration))
            continue

        data_time = time.time() - end
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()

        # losses.backward()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        if cfg.SOLVER.USE_ADAM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        try:
            if cfg.MODEL.ROI_MASK_HEAD.PREDICTOR == "GroupedMaskRCNNC4Predictor":
                new_grouping = infer_grouping(model, cfg)
                if current_grouping is None:
                    logger.info(f"[{count_grouping_change}/{iteration}] grouping changed from {current_grouping} to {new_grouping}")
                    current_grouping = new_grouping
                else:
                    if new_grouping != current_grouping:
                        count_grouping_change += 1
                        logger.info(f"[{count_grouping_change}/{iteration}] grouping changed from {current_grouping} to {new_grouping}")
                        current_grouping = new_grouping
        except Exception as e:
            logger.info(f"[Debug] Infer grouping failed: {str(e)}")
            pass

        if local_rank == 0 and (
            iteration % cfg.SOLVER.DISPLAY_FREQ == 0 or iteration == (max_iter - 1)
        ):
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            for tag, value in loss_dict_reduced.items():
                tb_logger.scalar_summary(tag, value.item(), iteration)
        if local_rank == 0 and iteration % checkpoint_period == 0 and iteration > 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

    if local_rank == 0:
        checkpointer.save("model_{:07d}".format(iteration), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
