import torch
import torch.nn.functional as F
from torch import nn

from multiplexer.modeling.roi_heads.mask_head.language_predictor_v5 import V5LanguagePredictor


class V6LanguagePredictor(V5LanguagePredictor):
    def __init__(self, cfg, do_init_weights=True):
        # Compared to V5, the main change is using elu instead of relu.
        super(V6LanguagePredictor, self).__init__(cfg=cfg, do_init_weights=True)

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
        x = F.elu(self.fc1(x))
        # [n, 64] => [n, num_class]
        x = self.fc2(x)

        return x
        # to convert to probabilities:
        # return F.softmax(x, dim=1)
