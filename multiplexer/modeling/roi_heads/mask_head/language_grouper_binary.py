# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class BinaryLanguageGrouper(nn.Module):
    def __init__(self, cfg, do_init_weights=True):
        super(BinaryLanguageGrouper, self).__init__()
        # self.cfg = cfg

        self.num_languages = cfg.MODEL.LANGUAGE_HEAD.NUM_CLASSES

        self.tau = cfg.MODEL.LANGUAGE_GROUPER.GUMBLE_SOFTMAX_TAU
        self.loss_weight = cfg.MODEL.LANGUAGE_GROUPER.LOSS_WEIGHT
        self.min_tasks = cfg.MODEL.LANGUAGE_GROUPER.MIN_TASKS

        self.lang_head_weights = nn.Parameter(torch.ones(self.num_languages, 2))

        if do_init_weights:
            self.init_weights()

    def forward(self, word_lang_probs):
        lang_head_probs = F.gumbel_softmax(self.lang_head_weights, tau=self.tau, hard=False)

        word_head_probs = torch.matmul(word_lang_probs, lang_head_probs)

        # encourage each head to process at least one language
        losses = {}
        losses["loss_group_1"] = self.loss_weight * F.relu(self.min_tasks - torch.sum(lang_head_probs[:, 0]))
        losses["loss_group_2"] = self.loss_weight * F.relu(self.min_tasks - torch.sum(lang_head_probs[:, 1]))

        return word_head_probs, losses

    def init_weights(self):
        pass
