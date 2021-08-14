# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .language_grouper_binary import BinaryLanguageGrouper
from .language_grouper_multi_v0 import MultiV0LanguageGrouper

_LANGUAGE_GROUPER = {
    "BinaryLanguageGrouper": BinaryLanguageGrouper,
    "MultiV0LanguageGrouper": MultiV0LanguageGrouper,
}


def make_language_grouper(cfg):
    language_grouper = _LANGUAGE_GROUPER[cfg.MODEL.LANGUAGE_GROUPER.NAME](cfg)
    if cfg.MODEL.LANGUAGE_GROUPER.FROZEN:
        for p in language_grouper.parameters():
            p.requires_grad = False
    return language_grouper
