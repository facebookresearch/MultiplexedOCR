# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import math

# import numpy as np
import torch
from torch import nn

from multiplexer.layers.res_blocks import res_layer

# from multiplexer.layers import Conv2d
from multiplexer.utils.chars import num2char as num2char_deprecated
from multiplexer.utils.languages import lang_code_to_char_map_class

# from warpctc_pytorch import CTCLoss


# logger = logging.getLogger(__name__)


def batch_remove_duplicates_and_blanks(batch_predicted_labels, batch_predicted_probs, prob_agg):
    # For CTC sequence transcriptions
    # Returns prob_agg (e.g. max) probability predicted for that decoded character
    assert len(batch_predicted_labels) == len(batch_predicted_probs)
    batch_predicted_labels_clean, batch_predicted_probs_clean = [], []
    for ii, l in enumerate(batch_predicted_labels):
        probs_curr_decoded_char = []
        predicted_labels_clean, predicted_probs_clean = [], []
        for i in range(len(l)):
            # Append the previous probability
            if i > 0 and l[i - 1] != 0:
                probs_curr_decoded_char.append(batch_predicted_probs[ii][i - 1])
            # AggProb1. Switching from non-zero to any character means we have
            # decoded a character.
            if i > 0 and l[i - 1] != 0 and l[i] != l[i - 1] and probs_curr_decoded_char:
                predicted_probs_clean.append(prob_agg(probs_curr_decoded_char))
                probs_curr_decoded_char = []
            # AggProb2. If we reached the end, we have also decoded a character.
            # There is a case when both AggProb1 and AggProb2 hit.
            if i == len(l) - 1 and l[i] != 0:
                probs_curr_decoded_char.append(batch_predicted_probs[ii][i])
                predicted_probs_clean.append(prob_agg(probs_curr_decoded_char))
            # Append predicted labels, as in remove_duplicates_and_blanks()
            if (i == 0 and l[i] != 0) or (l[i] != 0 and l[i] != l[i - 1]):
                predicted_labels_clean.append(l[i])
        batch_predicted_labels_clean.append(predicted_labels_clean)
        batch_predicted_probs_clean.append(predicted_probs_clean)
    return batch_predicted_labels_clean, batch_predicted_probs_clean


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


class CTCSequencePredictor(nn.Module):
    def __init__(
        self,
        cfg,
        dim_in,
        language="en_num",
        num_char=36,
        embed_size=38,  # not used
        hidden_size=256,
        arch=None,
        frozen=False,
    ):
        super(CTCSequencePredictor, self).__init__()
        self.cfg = cfg
        self.language = language
        self.num_char = num_char
        self.hidden_size = hidden_size
        self.arch = arch
        self.frozen = frozen
        self.bilstm_hidden_size = self.hidden_size
        self.bilstm_output_size = self.hidden_size
        self.lstm0_c = 256

        # if not self.cfg.SEQUENCE.SHARED_CONV5_MASK:
        #     self.conv5_mask = ConvTranspose2d(256, 256, 2, 2, 0)

        if arch == "ctc_b":
            assert cfg.MODEL.ROI_MASK_HEAD.CONV5_ARCH != "transpose_a"
            self.seq_encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, 3, padding=1),  # 40 x 40
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),  # 20 x 40
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),  # 10 x 40
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),  # 5 x 40
            )
        elif arch == "ctc_a":
            assert cfg.MODEL.ROI_MASK_HEAD.CONV5_ARCH == "transpose_a"
            self.seq_encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, 3, padding=1),  # 80 x 80
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 40 x 40
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),  # 20 x 40
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),  # 10 x 40
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),  # 5 x 40
            )
        if arch.startswith("ctc_lstm"):
            if arch == "ctc_lstm_res4":
                self.seq_encoder = res_layer()
                # self.relu = nn.ReLU(inplace=True)
            elif arch == "ctc_lstm_relu":
                self.relu = nn.ReLU(inplace=True)
            elif arch == "ctc_lstm_a":
                self.lstm0_c = 512

            self.pre_lstm_kernel_height = 3

            self.pre_lstm_conv = nn.Conv2d(
                512,
                self.lstm0_c,
                kernel_size=(self.pre_lstm_kernel_height, 1),
                stride=1,
            )
        elif "transformer" not in arch:
            self.pre_lstm_kernel_height = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_H // 8

            # average pooling to reduce height of feature map to 1
            # self.avr_pool = nn.AvgPool2d((self.pre_lstm_kernel_height, 1))

            self.pre_lstm_conv = nn.Conv2d(
                dim_in,
                self.lstm0_c,
                kernel_size=(self.pre_lstm_kernel_height, 1),
                stride=1,
                padding=0,
            )  # 1 x 40
            self.relu = nn.ReLU(inplace=True)

        if arch == "ctc_transformer":
            self.pre_lstm_kernel_height = 3

            self.pre_lstm_conv = nn.Conv2d(
                512,  # same as prod rec model
                self.lstm0_c,
                kernel_size=(self.pre_lstm_kernel_height, 1),
                stride=1,
            )

            print("Initializing Transfomer Encoder Layers")
            self.transformer_encoder = nn.Sequential(
                nn.TransformerEncoderLayer(d_model=self.lstm0_c, nhead=8, batch_first=True),
                nn.TransformerEncoderLayer(d_model=self.lstm0_c, nhead=8, batch_first=True),
            )
            self.post_transformer_conv1d = nn.Conv1d(
                self.lstm0_c, self.num_char + 1, kernel_size=1, stride=1
            )
        else:
            self.lstm = nn.Sequential(
                BidirectionalLSTM(self.lstm0_c, self.bilstm_hidden_size, self.bilstm_output_size),
                BidirectionalLSTM(
                    self.bilstm_output_size, self.bilstm_hidden_size, self.num_char + 1
                ),
            )

        self.ctc_reduction = "sum_manual"  # "sum"
        reduction = self.ctc_reduction
        if "manual" in self.ctc_reduction:
            reduction = "none"
        self.criterion_seq_decoder = nn.CTCLoss(reduction=reduction, zero_infinity=True)
        # self.criterion_seq_decoder = CTCLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

        if self.language in lang_code_to_char_map_class:
            self.char_map_class = lang_code_to_char_map_class[self.language]
            self.char_map_class.init(char_map_path=self.cfg.CHAR_MAP.DIR)

    def num2char(self, num):
        if hasattr(self, "char_map_class"):
            return self.char_map_class.num2char(num)
        return num2char_deprecated(num, self.language)

    def forward(
        self,
        x,
        decoder_targets=None,
        word_targets=None,  # not used
        use_beam_search=False,  # not used
        language_weights=None,
        coverage=None,  # not used
    ):
        # input shape: [BS; ch; H(=80); W(=80)]
        # the goal of this part is to reduce tensor height to 1 for LSTM
        # there could be multiple design choices here, like average pooling,
        # a few layers of convs with downsampling, or their combinations
        # here we use 4 convs with maxpooling to reduce feature shape to 5 x 40
        # then a conv with 5x1 kernel to convert it to 1 x 40
        # so input sequence len to LSTM is 40, meaning we can decode word up to 40 chars
        # note targets max length is 32, but that's fine since CTC can handle dup preds
        if hasattr(self, "seq_encoder"):
            x = self.seq_encoder(x)

        x = self.pre_lstm_conv(x)

        if hasattr(self, "relu"):
            x = self.relu(x)

        # shape before squeeze: [BS; ch; 1; W(=40)]
        x = torch.squeeze(x, 2)
        # shape after squeeze: [BS; ch; W]

        if "transformer" in self.arch:
            x = x.permute(0, 2, 1).contiguous()  # <= Transformer [BS; W; ch]
            x = self.transformer_encoder(x)  # [BS; W; ch]
            x = x.permute(0, 2, 1).contiguous()  # [BS; ch; W]
            x = self.post_transformer_conv1d(x)  # [BS; num_chars; W]
            preds = x.permute(2, 0, 1).contiguous()  # [W; BS; num_chars]
        else:
            x = x.permute(2, 0, 1).contiguous()
            # shape after permute: [W; BS; ch]
            preds = self.lstm(x)
            # output size is [W; BS; num_chars]

        if self.training:
            batch_size = preds.shape[1]
            target_length = preds.shape[0]
            preds_size = torch.IntTensor([target_length] * batch_size)

            # remove all blank label in targets, that seem to cause NaN
            decoder_targets[decoder_targets == 0] = self.num_char + 1

            # length for label is inferred from decoder_targets that looks like this:
            # [ 43,  13,  39,  39,  14,  13,  19,  19, 121, 121, 121, 121, 121, 121, ... ]
            # for consistency with seq2seq labels, word_targets now looks like this:
            # [ 43,  13,  39,  39,  14,  13,  19,  19, 121,   0,   0,   0,   0,   0, ... ]
            # so length will be 8 for this sample
            # if we want to clean up codebase, word_targets for CTC loss should be changed in
            # multiplexer/structures/segmentation_mask.py
            # and length can be passed as an input, then we don't need decoder_targets here
            length_for_loss = torch.count_nonzero(decoder_targets != self.num_char + 1, dim=1).int()

            # for Warp CTC, flatten and remove unneeded labels (nn.CTCLoss can work with 2D label)
            flatten_targets = decoder_targets.reshape(-1)
            flatten_targets = flatten_targets[flatten_targets != self.num_char + 1].int()

            # NOTE: PyTorch CTCLoss needs log_softmax on input
            # raw_loss_seq_decoder = self.criterion_seq_decoder(
            #     preds.log_softmax(dim=2), word_targets, preds_size, length_for_loss
            # )
            torch.backends.cudnn.enabled = False  # avoid PyTorch CTCLoss bug
            raw_loss_seq_decoder = self.criterion_seq_decoder(
                preds.log_softmax(dim=2), flatten_targets.cpu(), preds_size, length_for_loss.cpu()
            )
            torch.backends.cudnn.enabled = True  # avoid PyTorch CTCLoss bug
            # NOTE: use warp_ctc seems to be more stable
            # raw_loss_seq_decoder = self.criterion_seq_decoder(
            #     preds, flatten_targets.cpu(), preds_size, length_for_loss.cpu()
            # )

            if self.ctc_reduction == "sum_manual":
                # manual reduction to incorporate language_weights

                if language_weights is not None:
                    # print(f"[Debug] raw_loss_seq_decoder (original) {self.language} = {raw_loss_seq_decoder}")
                    # print(f"[Debug] language_weights {self.language} = {language_weights}")
                    raw_loss_seq_decoder = raw_loss_seq_decoder * language_weights
                    # print(f"[Debug] raw_loss_seq_decoder (after) {self.language} = {raw_loss_seq_decoder}")

                loss_seq_decoder = raw_loss_seq_decoder.sum()
                # print(f"[Debug] loss_seq_decoder (after) {self.language} = {loss_seq_decoder}")
            if self.ctc_reduction == "mean_manual":
                # manual reduction to incorporate language_weights

                # when there are zeros in length_for_loss, the corresponding loss should be 0
                # the original CTCLoss implementation take the guard at the last step with zero_infinity=True
                # here we make sure 0/0 == 0 by turning zero lengths into ones, i.e., 0/max(0, 1) = 0
                positive_length_for_loss = torch.clamp(length_for_loss, min=1)
                raw_loss_seq_decoder = raw_loss_seq_decoder / positive_length_for_loss

                if language_weights is not None:
                    raw_loss_seq_decoder = raw_loss_seq_decoder * language_weights

                loss_seq_decoder = raw_loss_seq_decoder.sum()
            elif self.ctc_reduction == "sum":
                assert (
                    language_weights is None
                ), "Please use ctc_reduction == 'sum_manual' to incorporate language_weights"
                if torch.isnan(raw_loss_seq_decoder) or torch.isinf(raw_loss_seq_decoder):
                    # still getting nan in very rare case but training doesn't seem to be affected
                    from virtual_fs.virtual_io import open

                    dbg_info = {
                        "preds": preds,
                        "flatten_targets": flatten_targets.cpu(),
                        "preds_size": preds_size,
                        "length_for_loss": length_for_loss.cpu(),
                    }
                    dbg_file = f"/checkpoint/jinghuang/tmp/ctc_dbg/ctc_dbg_{self.language}.pt"
                    with open(dbg_file, "wb") as buffer:
                        torch.save(dbg_info, buffer)
                    print("WARNING: raw loss is nan or inf: ", raw_loss_seq_decoder)
                    print(f"[Debug] preds = {preds}")
                    print(f"[Debug] flatten_targets = {flatten_targets}")
                    print(f"[Debug] preds_size = {preds_size}")
                    print(f"[Debug] length_for_loss = {length_for_loss}")
                    print(f"[Debug] Saved the above debug info to {dbg_file}.")

                # squeeze to change loss from shape (1) to single number to match with other losses
                # loss_seq_decoder = raw_loss_seq_decoder.cuda().squeeze()
                loss_seq_decoder = raw_loss_seq_decoder.squeeze()

            loss_seq_decoder = loss_seq_decoder / batch_size
            loss_seq_decoder = self.cfg.SEQUENCE.LOSS_WEIGHT * loss_seq_decoder
            # print(raw_loss_seq_decoder, loss_seq_decoder)
            if torch.isnan(loss_seq_decoder) or torch.isinf(loss_seq_decoder):
                print("WARNING: loss is nan or inf: ", loss_seq_decoder)

            return loss_seq_decoder

        else:
            # greedy decoder, similar to what's done in prod recognition
            # TODO: may need to split in a separate module and rewritten for torchscript
            #       see example in prod deployment script
            pred_softmax = nn.Softmax(dim=2)(preds)
            pred_probs, pred_labels = pred_softmax.topk(k=1, dim=2, largest=True, sorted=True)
            pred_probs = pred_probs.squeeze(2).t().data.cpu().numpy()
            pred_labels = pred_labels.squeeze(2).t().data.cpu().numpy()

            pred_labels, pred_probs = batch_remove_duplicates_and_blanks(
                pred_labels, pred_probs, prob_agg=max
            )

            words = []
            decoded_scores = []
            for labels, probs in zip(pred_labels, pred_probs):
                word = []
                char_scores = []
                for label, prob in zip(labels, probs):
                    if label == 0:
                        word.append("~")
                        char_scores.append(0.0)
                    else:
                        word.append(self.num2char(label))
                        char_scores.append(prob)
                # NOTE: sometimes pred_labels are all zero? thus this is needed
                if len(word) == 0:
                    word.append("~")
                    char_scores.append(0.0)
                words.append("".join(word))
                decoded_scores.append(char_scores)

            # CTC won't have the detailed_decoded_scores returned by seq2seq decoder
            # return words, decoded_scores, pred_softmax
            return words, decoded_scores, None
