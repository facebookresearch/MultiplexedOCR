# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class MultiV0LanguageGrouper(nn.Module):
    def __init__(self, cfg, do_init_weights=True):
        super(MultiV0LanguageGrouper, self).__init__()
        self.cfg = cfg

        self.num_languages = cfg.MODEL.LANGUAGE_HEAD.NUM_CLASSES
        self.num_heads = len(cfg.SEQUENCE.LANGUAGES_ENABLED)

        self.tau = cfg.MODEL.LANGUAGE_GROUPER.GUMBLE_SOFTMAX_TAU
        self.loss_weight = cfg.MODEL.LANGUAGE_GROUPER.LOSS_WEIGHT
        self.min_tasks = cfg.MODEL.LANGUAGE_GROUPER.MIN_TASKS

        self.lang_head_weights = nn.Parameter(torch.ones(self.num_languages, self.num_heads))

        if do_init_weights:
            self.init_weights()

    def forward(self, word_lang_probs):
        if self.training:
            lang_head_probs = F.gumbel_softmax(self.lang_head_weights, tau=self.tau, hard=False)
        else:
            lang_head_probs = F.softmax(self.lang_head_weights, dim=1)

        word_head_probs = torch.matmul(word_lang_probs, lang_head_probs)

        # encourage each head to process at least one language
        losses = {}

        for i, lang in enumerate(self.cfg.SEQUENCE.LANGUAGES_ENABLED):
            losses[f"loss_grp_{lang}"] = self.loss_weight * F.relu(
                self.min_tasks - torch.sum(lang_head_probs[:, i])
            )

        # print(f"[Debug] word_lang_probs = {word_lang_probs}")
        # print(f"[Debug] lang_head_probs = {lang_head_probs}")
        # print(f"[Debug] word_head_probs = {word_head_probs}")
        # print(f"[Debug] losses = {losses}")

        return word_head_probs, losses

    def init_weights(self):
        pass
