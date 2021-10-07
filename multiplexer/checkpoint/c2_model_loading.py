# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import pickle
from collections import OrderedDict

import torch


def _rename_basic_resnet_weights(layer_keys):
    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [k.replace(".w", ".weight") for k in layer_keys]
    layer_keys = [k.replace(".bn", "_bn") for k in layer_keys]
    layer_keys = [k.replace(".b", ".bias") for k in layer_keys]
    layer_keys = [k.replace("_bn.s", "_bn.scale") for k in layer_keys]
    layer_keys = [k.replace(".biasranch", ".branch") for k in layer_keys]
    layer_keys = [k.replace("bbox.pred", "bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("cls.score", "cls_score") for k in layer_keys]
    layer_keys = [k.replace("res.conv1_", "conv1_") for k in layer_keys]

    # RPN / Faster RCNN
    layer_keys = [k.replace(".biasbox", ".bbox") for k in layer_keys]
    layer_keys = [k.replace("conv.rpn", "rpn.conv") for k in layer_keys]
    layer_keys = [k.replace("rpn.bbox.pred", "rpn.bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("rpn.cls.logits", "rpn.cls_logits") for k in layer_keys]

    # Affine-Channel -> BatchNorm enaming
    layer_keys = [k.replace("_bn.scale", "_bn.weight") for k in layer_keys]

    # Make torchvision-compatible
    layer_keys = [k.replace("conv1_bn.", "bn1.") for k in layer_keys]

    layer_keys = [k.replace("res2.", "layer1.") for k in layer_keys]
    layer_keys = [k.replace("res3.", "layer2.") for k in layer_keys]
    layer_keys = [k.replace("res4.", "layer3.") for k in layer_keys]
    layer_keys = [k.replace("res5.", "layer4.") for k in layer_keys]

    layer_keys = [k.replace(".branch2a.", ".conv1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2a_bn.", ".bn1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b.", ".conv2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b_bn.", ".bn2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c.", ".conv3.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c_bn.", ".bn3.") for k in layer_keys]

    layer_keys = [k.replace(".branch1.", ".downsample.0.") for k in layer_keys]
    layer_keys = [k.replace(".branch1_bn.", ".downsample.1.") for k in layer_keys]

    return layer_keys


def _rename_fpn_weights(layer_keys, stage_names):
    for mapped_idx, stage_name in enumerate(stage_names, 1):
        suffix = ""
        if mapped_idx < 4:
            suffix = ".lateral"
        layer_keys = [
            k.replace(
                "fpn.inner.layer{}.sum{}".format(stage_name, suffix),
                "fpn_inner{}".format(mapped_idx),
            )
            for k in layer_keys
        ]
        layer_keys = [
            k.replace("fpn.layer{}.sum".format(stage_name), "fpn_layer{}".format(mapped_idx))
            for k in layer_keys
        ]

    layer_keys = [k.replace("rpn.conv.fpn2", "rpn.conv") for k in layer_keys]
    layer_keys = [k.replace("rpn.bbox_pred.fpn2", "rpn.bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("rpn.cls_logits.fpn2", "rpn.cls_logits") for k in layer_keys]

    return layer_keys


def _rename_weights_for_resnet(weights, stage_names):
    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())

    # for X-101, rename output to fc1000 to avoid conflicts afterwards
    layer_keys = [k if k != "pred_b" else "fc1000_b" for k in layer_keys]
    layer_keys = [k if k != "pred_w" else "fc1000_w" for k in layer_keys]

    # performs basic renaming: _ -> . , etc
    layer_keys = _rename_basic_resnet_weights(layer_keys)

    # FPN
    layer_keys = _rename_fpn_weights(layer_keys, stage_names)

    # Mask R-CNN
    layer_keys = [k.replace("mask.fcn.logits", "mask_fcn_logits") for k in layer_keys]
    layer_keys = [k.replace(".[mask].fcn", "mask_fcn") for k in layer_keys]
    layer_keys = [k.replace("conv5.mask", "conv5_mask") for k in layer_keys]

    # Keypoint R-CNN
    layer_keys = [k.replace("kps.score.lowres", "kps_score_lowres") for k in layer_keys]
    layer_keys = [k.replace("kps.score", "kps_score") for k in layer_keys]
    layer_keys = [k.replace("conv.fcn", "conv_fcn") for k in layer_keys]

    # Rename for our RPN structure
    layer_keys = [k.replace("rpn.", "rpn.head.") for k in layer_keys]

    key_map = {k: v for k, v in zip(original_keys, layer_keys)}

    logger = logging.getLogger(__name__)
    logger.info("Remapping C2 weights")
    max_c2_key_size = max([len(k) for k in original_keys if "_momentum" not in k])

    new_weights = OrderedDict()
    for k in original_keys:
        v = weights[k]
        if "_momentum" in k:
            continue
        # if 'fc1000' in k:
        #     continue
        w = torch.from_numpy(v)
        # if "bn" in k:
        #     w = w.view(1, -1, 1, 1)
        logger.info("C2 name: {: <{}} mapped name: {}".format(k, max_c2_key_size, key_map[k]))
        new_weights[key_map[k]] = w

    return new_weights


def _load_c2_pickled_weights(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    if "blobs" in data:
        weights = data["blobs"]
    else:
        weights = data
    return weights


def _rename_conv_weights_for_deformable_conv_layers(state_dict, cfg):
    import re

    logger = logging.getLogger(__name__)
    logger.info("Remapping conv weights for deformable conv weights")
    layer_keys = sorted(state_dict.keys())
    for ix, stage_with_dcn in enumerate(cfg.MODEL.RESNETS.STAGE_WITH_DCN, 1):
        if not stage_with_dcn:
            continue
        for old_key in layer_keys:
            pattern = ".*layer{}.*conv2.*".format(ix)
            r = re.match(pattern, old_key)
            if r is None:
                continue
            for param in ["weight", "bias"]:
                if old_key.find(param) == -1:
                    continue
                new_key = old_key.replace("conv2.{}".format(param), "conv2.conv.{}".format(param))
                logger.info(
                    "pattern: {}, old_key: {}, new_key: {}".format(pattern, old_key, new_key)
                )
                state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]
    return state_dict


_C2_STAGE_NAMES = {
    "R-50": ["1.2", "2.3", "3.5", "4.2"],
    "R-101": ["1.2", "2.3", "3.22", "4.2"],
}


def load_c2_format(cfg, f):
    # TODO make it support other architectures
    state_dict = _load_c2_pickled_weights(f)
    conv_body = cfg.MODEL.BACKBONE.CONV_BODY
    arch = conv_body.replace("-C4", "").replace("-FPN", "")
    stages = _C2_STAGE_NAMES[arch]
    state_dict = _rename_weights_for_resnet(state_dict, stages)
    # ***********************************
    # for deformable convolutional layer
    state_dict = _rename_conv_weights_for_deformable_conv_layers(state_dict, cfg)
    # ***********************************
    return dict(model=state_dict)


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(current_keys), len(loaded_keys))
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )
