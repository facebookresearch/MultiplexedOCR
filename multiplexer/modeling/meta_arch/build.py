# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from multiplexer.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")


def build_model(cfg):
    meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
    return meta_arch(cfg)
