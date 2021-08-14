# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn.functional as F
from torch import nn

from .language_predictor_base import BaseLanguagePredictor


class V1LanguagePredictor(BaseLanguagePredictor):
    def __init__(self, cfg, do_init_weights=True):
        super(V1LanguagePredictor, self).__init__(cfg=cfg, do_init_weights=False)
        input_c = cfg.MODEL.LANGUAGE_HEAD.INPUT_C  # default: 256
        input_h = cfg.MODEL.LANGUAGE_HEAD.INPUT_H  # default: 40
        input_w = cfg.MODEL.LANGUAGE_HEAD.INPUT_W  # default: 40
        conv1_c = cfg.MODEL.LANGUAGE_HEAD.CONV1_C  # default: 64
        conv2_c = cfg.MODEL.LANGUAGE_HEAD.CONV2_C  # default: 32

        assert input_h % 8 == 0
        assert input_w % 8 == 0

        fc1_in = (input_h // 8) * (input_w // 8) * conv2_c

        self.conv1 = nn.Conv2d(input_c, conv1_c, 2, 2, 0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_c, conv2_c, 2, 2, 0)
        self.fc1 = nn.Linear(fc1_in, 64)
        self.fc2 = nn.Linear(64, self.num_classes)

        if do_init_weights:
            self.init_weights()

    def forward(self, x):
        # [n, 256, 32, 32] => [n, 64, 16, 16] (input=32x32)
        # [n, 256, 48, 48] => [n, 64, 24, 24] (input=48x48)
        x = F.relu(self.conv1(x))
        # [n, 64, 16, 16] => [n, 64, 8, 8] (input=32x32)
        # [n, 64, 24, 24] => [n, 64, 12, 12] (input=48x48)
        x = self.maxpool(x)
        # [n, 64, 8, 8] => [n, 32, 4, 4] (input=32x32)
        # [n, 64, 12, 12] => [n, 32, 6, 6] (input=48x48)
        x = F.relu(self.conv2(x))
        # [n, 32, 4, 4] => [n, 512] (input=32x32)
        # [n, 32, 6, 6] => [n, 1152] (input=48x48)
        x = x.view(x.size(0), -1)
        # [n, 512] => [n, 64] (input=32x32)
        # [n, 1152] => [n, 64] (input=32x32)
        x = F.relu(self.fc1(x))
        # [n, 64] => [n, num_class]
        x = self.fc2(x)

        return x
        # to convert to probabilities:
        # return F.softmax(x, dim=1)
