#!/usr/bin/env python3

# from multiplexer.layers import Conv2d, ConvTranspose2d
from torch import nn
from torch.nn import functional as F


class BaseMaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, do_init_weights=True):
        super(BaseMaskRCNNC4Predictor, self).__init__()
        # num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.cfg = cfg
        num_classes = 1
        self.dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        self.dim_in = self.dim_reduced

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            self.num_inputs = self.dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            self.num_inputs = res2_out_channels * stage2_relative_factor

        conv5_arch = self.cfg.MODEL.ROI_MASK_HEAD.CONV5_ARCH
        mask_fcn_input_dim = self.cfg.MODEL.ROI_MASK_HEAD.MASK_FCN_INPUT_DIM
        # note: according to D2, after PyTorch 1.7 the custom Conv2d is no longer needed
        if conv5_arch != "none_a":
            if conv5_arch == "conv_a":
                self.conv5_mask = nn.Conv2d(self.num_inputs, mask_fcn_input_dim, 1, 1, 0)
            else:
                assert conv5_arch.startswith("transpose")
                self.conv5_mask = nn.ConvTranspose2d(self.num_inputs, mask_fcn_input_dim, 2, 2, 0)

            if cfg.MODEL.ROI_MASK_HEAD.PREDICTOR_TRUNK_FROZEN:
                for p in self.conv5_mask.parameters():
                    p.requires_grad = False

            self.mask_fcn_logits = nn.Conv2d(mask_fcn_input_dim, num_classes, 1, 1, 0)

            if cfg.MODEL.ROI_MASK_HEAD.PREDICTOR_TRUNK_FROZEN:
                for p in self.mask_fcn_logits.parameters():
                    p.requires_grad = False

        if do_init_weights:
            self.init_weights()

    def forward(self, x, decoder_targets=None, word_targets=None):
        if self.cfg.MODEL.ROI_MASK_HEAD.CONV5_ARCH != "none_a":
            x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)

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
                    weight_names = ["bn", "downsample.1", "transformer_encoder"]
                    if any(s in name for s in weight_names):
                        continue  # skip known BatchNorms in res_layer and transformer layers
                    else:
                        raise Exception(f"Exception for weight {name}: {e}")
