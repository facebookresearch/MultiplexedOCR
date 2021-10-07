# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import math
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from multiplexer.utils.chars import num2char as num2char_deprecated
from multiplexer.utils.languages import lang_code_to_char_map_class

logger = logging.getLogger(__name__)


gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")


def reduce_mul(score_list: List[float]):
    out: float = 1.0
    for x in score_list:
        out *= x
    return out


def check_all_done(
    seqs: List[
        Tuple[
            List[Tuple[int, float, torch.Tensor, List[torch.Tensor]]],
            float,
            torch.Tensor,
            bool,
        ]
    ]
):
    for seq in seqs:
        if not seq[-1]:
            return False
    return True


class NLLLossWithPenalizingIndex(nn.NLLLoss):
    def __init__(self, ignore_index=-100, reduction="mean", penalize_index=-200):
        super(NLLLossWithPenalizingIndex, self).__init__(
            weight=None, size_average=None, reduction=reduction
        )
        assert (
            self.reduction == "mean" or self.reduction == "none"
        ), "reduction option '{}' not supported".format(self.reduction)
        self.ignore_index = ignore_index
        self.penalize_index = penalize_index

    def forward(self, input, target):
        out = torch.zeros_like(target, dtype=torch.float)
        for i in range(len(target)):
            if target[i] == self.ignore_index:
                out[i] = 0
            elif target[i] == self.penalize_index:
                # penalize non-ignored unsupported index.
                # Since math.log(0) is undefined and math.log(1e-5) == -11.5,
                # here we essentially assign a prob that is smaller than 1e-5 to this index.
                out[i] = -12
            else:
                assert (
                    target[i] >= 0 and target[i] < input.shape[1]
                ), "Index {} is out of bound".format(target[i])
                out[i] = input[i][target[i]]
        return -(out.sum() / len(out) if self.reduction == "mean" else out)


class BaseSequencePredictor(nn.Module):
    def __init__(
        self,
        cfg,
        dim_in,
        language="en_num",
        num_char=36,
        embed_size=38,
        hidden_size=256,
        arch="seq2seq_a",
        frozen=False,
    ):
        super(BaseSequencePredictor, self).__init__()
        self.cfg = cfg
        self.language = language
        self.num_char = num_char
        self.hidden_size = hidden_size
        self.arch = arch
        self.frozen = frozen

        if not self.cfg.SEQUENCE.SHARED_CONV5_MASK:
            self.conv5_mask = nn.ConvTranspose2d(256, 256, 2, 2, 0)

        if arch == "seq2seq_b":
            self.pre_seq_encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        elif arch == "seq2seq_c":
            self.pre_seq_encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                # nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                # nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),
            )

        if cfg.SEQUENCE.TWO_CONV:
            self.seq_encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                nn.Conv2d(dim_in, hidden_size, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.seq_encoder = nn.Sequential(
                nn.Conv2d(dim_in, hidden_size, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
            )
        x_onehot_size = int(cfg.SEQUENCE.RESIZE_WIDTH / 2)
        y_onehot_size = int(cfg.SEQUENCE.RESIZE_HEIGHT / 2)
        self.seq_decoder = BahdanauAttnDecoderRNN(
            hidden_size=hidden_size,
            embed_size=embed_size,
            output_size=self.num_char + 2,
            n_layers=1,
            dropout_p=0.1,
            onehot_size=(y_onehot_size, x_onehot_size),
        )

        if cfg.SEQUENCE.DECODER_LOSS == "NLLLossWithPenalizingIndex":
            self.criterion_seq_decoder = NLLLossWithPenalizingIndex(
                ignore_index=-1, reduction="none", penalize_index=0
            )
        else:
            assert cfg.SEQUENCE.DECODER_LOSS == "NLLLoss"
            self.criterion_seq_decoder = nn.NLLLoss(ignore_index=-1, reduction="none")

        self.rescale = nn.Upsample(
            size=(cfg.SEQUENCE.RESIZE_HEIGHT, cfg.SEQUENCE.RESIZE_WIDTH),
            mode="bilinear",
            align_corners=False,
        )

        self.x_onehot = nn.Embedding(x_onehot_size, x_onehot_size)
        self.x_onehot.weight.data = torch.eye(x_onehot_size)
        self.y_onehot = nn.Embedding(y_onehot_size, y_onehot_size)
        self.y_onehot.weight.data = torch.eye(y_onehot_size)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

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
        word_targets=None,
        use_beam_search=False,
        language_weights=None,
        coverage=None,
    ):
        device = x.device
        if not self.cfg.SEQUENCE.SHARED_CONV5_MASK:
            x = F.relu(self.conv5_mask(x))
        rescale_out = self.rescale(x)

        if self.arch in ["seq2seq_b", "seq2seq_c", "seq2seq_d"]:
            rescale_out = self.pre_seq_encoder(rescale_out)

        seq_decoder_input = self.seq_encoder(rescale_out)
        x_onehot_size = int(self.cfg.SEQUENCE.RESIZE_WIDTH / 2)
        y_onehot_size = int(self.cfg.SEQUENCE.RESIZE_HEIGHT / 2)
        y_t, x_t = torch.meshgrid(
            torch.arange(y_onehot_size, device=device),
            torch.arange(x_onehot_size, device=device),
        )
        x_onehot_embedding = (
            self.x_onehot(x_t)
            .transpose(0, 2)
            .transpose(1, 2)
            .repeat(seq_decoder_input.size(0), 1, 1, 1)
        )
        y_onehot_embedding = (
            self.y_onehot(y_t)
            .transpose(0, 2)
            .transpose(1, 2)
            .repeat(seq_decoder_input.size(0), 1, 1, 1)
        )
        seq_decoder_input_loc = torch.cat(
            [seq_decoder_input, x_onehot_embedding, y_onehot_embedding], 1
        )
        seq_decoder_input_reshape = (
            seq_decoder_input_loc.view(
                seq_decoder_input_loc.size(0), seq_decoder_input_loc.size(1), -1
            )
            .transpose(0, 2)
            .transpose(1, 2)
        )
        if self.training:
            bos_onehot = np.zeros((seq_decoder_input_reshape.size(1), 1), dtype=np.int32)
            bos_onehot[:, 0] = self.cfg.SEQUENCE.BOS_TOKEN
            decoder_input = torch.tensor(bos_onehot.tolist(), device=device)
            decoder_hidden = torch.zeros(
                (seq_decoder_input_reshape.size(1), self.hidden_size), device=device
            )
            use_teacher_forcing = (
                True if random.random() < self.cfg.SEQUENCE.TEACHER_FORCE_RATIO else False
            )
            target_length = decoder_targets.size(1)
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    if not torch.all(decoder_input >= -2):
                        print(
                            "\n".join(
                                [
                                    "[Error] Weird decoder_input detected:"
                                    "{}".format(decoder_input),
                                    "decoder_targets = {}".format(decoder_targets),
                                    "di = {}".format(di),
                                    "decoder_targets[:, {}] = {}".format(
                                        di - 1, decoder_targets[:, di - 1]
                                    ),
                                ]
                            )
                        )
                        assert 0 == 1

                    (decoder_output, decoder_hidden, decoder_attention,) = self.seq_decoder(
                        word_input=decoder_input,
                        last_hidden=decoder_hidden,
                        encoder_outputs=seq_decoder_input_reshape,
                    )
                    if di == 0:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    decoder_input = decoder_targets[:, di]  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    (decoder_output, decoder_hidden, decoder_attention,) = self.seq_decoder(
                        word_input=decoder_input,
                        last_hidden=decoder_hidden,
                        encoder_outputs=seq_decoder_input_reshape,
                    )
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(1).detach()  # detach from history as input
                    if di == 0:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
            if language_weights is not None:
                loss_seq_decoder = loss_seq_decoder * language_weights

            loss_seq_decoder = loss_seq_decoder.sum() / loss_seq_decoder.size(0)
            loss_seq_decoder = self.cfg.SEQUENCE.LOSS_WEIGHT * loss_seq_decoder

            return loss_seq_decoder
        else:
            words = []
            decoded_scores = []
            detailed_decoded_scores = []
            # real_length = 0
            if use_beam_search:
                for batch_index in range(seq_decoder_input_reshape.size(1)):
                    decoder_hidden = torch.zeros((1, self.hidden_size), device=device)
                    word = []
                    char_scores = []
                    detailed_char_scores = []
                    top_seqs = self.beam_search(
                        seq_decoder_input_reshape[:, batch_index : batch_index + 1, :],
                        decoder_hidden,
                        beam_size=6,
                        max_len=self.cfg.SEQUENCE.MAX_LENGTH,
                        device=device,
                    )
                    top_seq = top_seqs[0]
                    for character in top_seq[1:]:
                        character_index = character[0]
                        if character_index == self.num_char + 1:
                            char_scores.append(character[1])
                            detailed_char_scores.append(character[2])
                            break
                        else:
                            if character_index == 0:
                                word.append("~")
                                char_scores.append(0.0)
                            else:
                                word.append(self.num2char(character_index))
                                char_scores.append(character[1])
                                detailed_char_scores.append(character[2])
                    words.append("".join(word))
                    decoded_scores.append(char_scores)
                    detailed_decoded_scores.append(detailed_char_scores)
            else:
                for batch_index in range(seq_decoder_input_reshape.size(1)):
                    bos_onehot = np.zeros((1, 1), dtype=np.int32)
                    bos_onehot[:, 0] = self.cfg.SEQUENCE.BOS_TOKEN
                    decoder_input = torch.tensor(bos_onehot.tolist(), device=device)
                    decoder_hidden = torch.zeros((1, self.hidden_size), device=device)
                    word = []
                    char_scores = []
                    for _di in range(self.cfg.SEQUENCE.MAX_LENGTH):
                        (decoder_output, decoder_hidden, decoder_attention,) = self.seq_decoder(
                            word_input=decoder_input,
                            last_hidden=decoder_hidden,
                            encoder_outputs=seq_decoder_input_reshape[
                                :, batch_index : batch_index + 1, :
                            ],
                        )
                        # decoder_attentions[di] = decoder_attention.data
                        topv, topi = decoder_output.data.topk(1)
                        char_scores.append(topv.item())
                        if topi.item() == self.num_char + 1:
                            break
                        else:
                            if topi.item() == 0:
                                word.append("~")
                            else:
                                word.append(self.num2char(topi.item()))

                        # real_length = di
                        decoder_input = topi.squeeze(1).detach()
                    words.append("".join(word))
                    decoded_scores.append(char_scores)
            return (
                words,
                decoded_scores,
                detailed_decoded_scores if use_beam_search else None,
            )

    def beam_search_step(self, encoder_context, top_seqs, k, device=gpu_device):
        all_seqs = []
        for seq in top_seqs:
            seq_score = reduce_mul([_score for _, _score, _, _ in seq])
            if seq[-1][0] == self.num_char + 1:
                all_seqs.append((seq, seq_score, seq[-1][2], True))
                continue
            decoder_hidden = seq[-1][-1][0]
            onehot = np.zeros((1, 1), dtype=np.int32)
            onehot[:, 0] = seq[-1][0]
            decoder_input = torch.tensor(onehot.tolist(), device=device)
            decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                word_input=decoder_input,
                last_hidden=decoder_hidden,
                encoder_outputs=encoder_context,
            )
            detailed_char_scores = decoder_output.detach().cpu().numpy()
            # print(decoder_output.shape)
            scores, candidates = decoder_output.data[:, 1:].topk(k)
            for i in range(k):
                character_score = scores[:, i]
                character_index = candidates[:, i]
                score = seq_score * character_score.item()
                char_score = seq_score * detailed_char_scores
                rs_seq = seq + [
                    (
                        character_index.item() + 1,
                        character_score.item(),
                        char_score,
                        [decoder_hidden],
                    )
                ]
                done = character_index.item() == self.num_char
                all_seqs.append((rs_seq, score, char_score, done))
        all_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)
        topk_seqs = [seq for seq, _, _, _ in all_seqs[:k]]
        all_done = check_all_done(all_seqs[:k])
        return topk_seqs, all_done

    def beam_search(
        self,
        encoder_context,
        decoder_hidden,
        beam_size=6,
        max_len=32,
        device=gpu_device,
    ):
        char_score = np.zeros(self.num_char + 2)
        top_seqs = [[(self.cfg.SEQUENCE.BOS_TOKEN, 1.0, char_score, [decoder_hidden])]]
        # loop
        for _ in range(max_len):
            top_seqs, all_done = self.beam_search_step(encoder_context, top_seqs, beam_size, device)
            if all_done:
                break
        return top_seqs


class Attn(nn.Module):
    def __init__(self, method, hidden_size, embed_size, onehot_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.attn = nn.Linear(2 * hidden_size + onehot_size, hidden_size)
        # self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden:
            previous hidden state of the decoder, in shape (B, hidden_size)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (H*W, B, hidden_size)
        :return
            attention energies in shape (B, H*W)
        """
        max_len = encoder_outputs.size(0)
        # this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)  # (B, H*W, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, H*W, hidden_size)
        attn_energies = self.score(H, encoder_outputs)  # compute attention score (B, H*W)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax (B, 1, H*W)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(
            self.attn(torch.cat(tensors=[hidden, encoder_outputs], dim=2))
        )  # (B, H*W, 2*hidden_size+H+W)->(B, H*W, hidden_size)
        energy = energy.transpose(2, 1)  # (B, hidden_size, H*W)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # (B, 1, hidden_size)
        energy = torch.bmm(v, energy)  # (B, 1, H*W)
        return energy.squeeze(1)  # (B, H*W)


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size,
        embed_size,
        output_size,
        n_layers=1,
        dropout_p=0,
        bidirectional=False,
        onehot_size=(8, 32),
    ):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        if output_size == embed_size:
            # embedding.weight.data.shape is (output_size, embed_size) !!
            # It's not possible to assign identity matrix to the weights
            # if output_size != embed_size, thus we cannot use onehot
            # embedding if embed_size < output_size.
            self.embedding.weight.data = torch.eye(embed_size)

        # self.dropout = nn.Dropout(dropout_p)
        self.word_linear = nn.Linear(embed_size, hidden_size)
        self.attn = Attn("concat", hidden_size, embed_size, onehot_size[0] + onehot_size[1])
        self.rnn = nn.GRUCell(2 * hidden_size + onehot_size[0] + onehot_size[1], hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        """
        :param word_input:
            word input for current time step, in shape (B)
            Note: B can be different in different iterations.
            Example: tensor([52, 52, 52, 38, 52, 52, 46, 52, 45]) (B = 9)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B, hidden_size)
        :param encoder_outputs:
            encoder outputs in shape (H*W, B, C)
        :return
            decoder output
        """
        # Get the embedding of the current input word (last output word)
        assert torch.all(word_input >= -2), "Weird word_input detected:\n{}".format(word_input)
        # Note: when embed_size != output size,
        # word_embedded_onehot is not onehot embedding any more,
        # but just random embedding (see more explanation above)
        word_embedded_onehot = self.embedding(word_input).view(
            1, word_input.size(0), -1
        )  # (1, B, embed_size).

        word_embedded = self.word_linear(word_embedded_onehot)  # (1, B, hidden_size)

        attn_weights = self.attn(last_hidden, encoder_outputs)  # (B, 1, H * W)

        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1)
        )  # (B, 1, H*W) * (B, H*W, C) = (B, 1, C)

        context = context.transpose(0, 1)  # (1, B, C)

        # Combine embedded input word and attended context, run through RNN
        # 2 * hidden_size + W + H: 256 + 256 + 32 + 8 = 552
        # [1, B, hidden_size], [1, B, C] => [1, B, hidden_size + C]
        rnn_input = torch.cat(tensors=(word_embedded, context), dim=2)

        last_hidden = last_hidden.view(last_hidden.size(0), -1)  # (B, hidden_size)

        rnn_input = rnn_input.view(word_input.size(0), -1)  # (B, hidden_size + C)

        hidden = self.rnn(rnn_input, last_hidden)  # (B, hidden_size)

        if not self.training:
            output = F.softmax(self.out(hidden), dim=1)
        else:
            output = F.log_softmax(self.out(hidden), dim=1)  # [B, char_size]

        # Return final output, hidden state
        # print(output.shape)
        return output, hidden, attn_weights
