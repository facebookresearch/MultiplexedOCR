import torch
import torch.nn.functional as F
from torch import nn

from multiplexer.modeling.roi_heads.mask_head.language_predictor_base import BaseLanguagePredictor


class V5LanguagePredictor(BaseLanguagePredictor):
    def __init__(self, cfg, do_init_weights=True):
        # Compared to V3, the main change is the support for dynamic input size.
        super(V5LanguagePredictor, self).__init__(cfg=cfg, do_init_weights=False)
        input_c = cfg.MODEL.LANGUAGE_HEAD.INPUT_C  # default: 512
        input_h = cfg.MODEL.LANGUAGE_HEAD.INPUT_H  # default: 3
        self.input_w = cfg.MODEL.LANGUAGE_HEAD.INPUT_W  # default: 40
        conv1_c = cfg.MODEL.LANGUAGE_HEAD.CONV1_C  # default: 128
        conv2_c = cfg.MODEL.LANGUAGE_HEAD.CONV2_C  # default: 64

        assert input_h % 3 == 0
        assert self.input_w % 4 == 0

        fc1_in = (input_h // 3) * (self.input_w // 4) * conv2_c

        self.conv1 = nn.Conv2d(input_c, conv1_c, kernel_size=(3, 2), stride=(3, 2), padding=0)
        self.conv2 = nn.Conv2d(conv1_c, conv2_c, kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.fc1 = nn.Linear(fc1_in, 64)
        self.fc2 = nn.Linear(64, self.num_classes)

        if do_init_weights:
            self.init_weights()

    def forward(self, x):
        assert (
            x.shape[3] <= self.input_w
        ), f"Input patch length {x.shape[3]} > max supported length {self.input_w}"

        # [n, 512, 3, <=40] => [n, 512, 3, 40] with 0-padding
        if x.shape[3] < self.input_w:
            x = torch.cat(
                (
                    x,
                    torch.zeros(
                        x.shape[0],
                        x.shape[1],
                        x.shape[2],
                        self.input_w - x.shape[3],
                        device=x.device,
                    ),
                ),
                dim=3,
            )
        # [n, 512, 3, 40] => [n, 128, 1, 20]
        x = F.relu(self.conv1(x))
        # [n, 128, 1, 20] => [n, 64, 1, 10]
        x = F.relu(self.conv2(x))

        # [n, 64, 1, 10] => [n, 640]
        x = x.view(x.size(0), -1)
        # [n, 640] => [n, 64]
        x = F.relu(self.fc1(x))
        # [n, 64] => [n, num_class]
        x = self.fc2(x)

        return x
        # to convert to probabilities:
        # return F.softmax(x, dim=1)
