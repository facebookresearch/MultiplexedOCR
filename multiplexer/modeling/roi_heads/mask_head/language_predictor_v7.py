# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# experimental
import torch
import torch.nn.functional as F
from torch import nn

from multiplexer.modeling.roi_heads.mask_head.language_predictor_v5 import V5LanguagePredictor


# TODO: extract this one with and the one in roi_seq_predictor_ctc into the same class
class BidirectionalLSTM(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(in_ch, hidden_ch, bidirectional=True)
        if out_ch is not None:
            self.fc = nn.Linear(hidden_ch * 2, out_ch)
        else:
            self.fc = None

    def forward(self, input):
        # input size: [W; BS; in_ch]
        output, _ = self.rnn(input)
        # output size: [W; BS; hidden_ch * 2] (bi-bidirectional)

        if self.fc is not None:
            w, bs, hc = output.size()
            # view in size: [W * BS; hidden_ch * 2]
            output_view = output.view(w * bs, hc)
            # output size: [W * BS; out_ch]
            output = self.fc(output_view)
            # separate width and batch size: [W; BS; out_ch]
            output = output.view(w, bs, -1)

        return output


class V7LanguagePredictor(V5LanguagePredictor):
    def __init__(self, cfg, do_init_weights=True):
        super(V7LanguagePredictor, self).__init__(cfg=cfg, do_init_weights=False)

        self.bilstm_hidden_size = 192
        self.bilstm_output_size = 192
        self.lstm0_c = 256
        self.pre_lstm_kernel_height = 3

        self.pre_lstm_conv = nn.Conv2d(
            512,
            self.lstm0_c,
            kernel_size=(self.pre_lstm_kernel_height, 1),
            stride=1,
        )

        # note: original ctc-based rec-head uses self.num_classes + 1 to support dummy label
        self.lstm = nn.Sequential(
            BidirectionalLSTM(self.lstm0_c, self.bilstm_hidden_size, self.bilstm_output_size),
            BidirectionalLSTM(self.bilstm_output_size, self.bilstm_hidden_size, self.num_classes),
        )

        # self.ctc_reduction = "sum_manual"  # "sum"
        # reduction = self.ctc_reduction
        # if "manual" in self.ctc_reduction:
        #     reduction = "none"
        # self.criterion_seq_decoder = nn.CTCLoss(reduction=reduction, zero_infinity=True)

        if do_init_weights:
            self.init_weights()

    def forward(self, x):
        # n x 512 x 3 x <=40
        x = self.pre_lstm_conv(x)

        # shape before squeeze: n x ch x 1 x w(<=40)]
        x = torch.squeeze(x, 2)
        # shape after squeeze: n x ch x w

        x = x.permute(2, 0, 1).contiguous()
        # shape after permute: w x n x ch
        preds = self.lstm(x)
        # output size is w x n x cl

        # w x n x cl => n x cl
        aggregated_preds = torch.mean(F.relu(preds), dim=0)

        return aggregated_preds
        # to convert to probabilities:
        # return F.softmax(sum_preds, dim=1)
