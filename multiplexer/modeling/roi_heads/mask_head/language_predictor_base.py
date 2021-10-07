# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torch import nn


class BaseLanguagePredictor(nn.Module):
    def __init__(self, cfg, do_init_weights=True):
        super(BaseLanguagePredictor, self).__init__()
        self.cfg = cfg

        self.num_classes = cfg.MODEL.LANGUAGE_HEAD.NUM_CLASSES

        if do_init_weights:
            self.init_weights()

    def forward(self, x):
        # [n, 256, 32, 32] => [n, 256, 8, 8]
        x = self.avgpool(x)
        # [n, 256, 8, 8] => [n, 64, 8, 8]
        x = self.conv(x)
        # [n, 64, 8, 8] => [n, 4096]
        x = x.view(x.size(0), -1)
        # [n, 4096] => [n, num_class]
        x = self.lang_logits(x)

        return x
        # to convert to probabilities:
        # return F.softmax(x, dim=1)

    def init_weights(self):
        # need special handling of init_weights for BatchNorm
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                try:
                    nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
                except ValueError as e:
                    if "bn" in name or "downsample.1" in name:
                        continue  # skip BatchNorms in res_layer
                    else:
                        raise Exception(f"Exception for weight {name}: {e}")
