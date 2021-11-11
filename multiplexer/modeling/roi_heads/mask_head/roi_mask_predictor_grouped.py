#!/usr/bin/env python3

import random

import numpy as np
import torch
from torch.nn import functional as F

from multiplexer.modeling.roi_heads.mask_head.language_grouper_builder import make_language_grouper
from multiplexer.modeling.roi_heads.mask_head.language_predictor_builder import (
    make_language_predictor,
)
from multiplexer.modeling.roi_heads.mask_head.roi_mask_predictor_multi_seq import (
    MultiSeqMaskRCNNC4Predictor,
)
from multiplexer.structures.word_result import WordResult


class GroupedMaskRCNNC4Predictor(MultiSeqMaskRCNNC4Predictor):
    def __init__(self, cfg, do_init_weights=True):
        super(GroupedMaskRCNNC4Predictor, self).__init__(cfg, do_init_weights=False)

        self.language_predictor = make_language_predictor(cfg=cfg)

        self.language_grouper = make_language_grouper(cfg=cfg)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        if do_init_weights:
            self.init_weights()

    def gt_prob_matrix(self, gt_lang_id):
        probs = torch.zeros(len(gt_lang_id), self.cfg.MODEL.LANGUAGE_HEAD.NUM_CLASSES)
        for i, lang_id in enumerate(gt_lang_id):
            probs[i][lang_id] = 1.0
        return probs.to(gt_lang_id.device)

    def forward(self, x, decoder_targets=None, word_targets=None, gt_language_targets=None):
        language_logits = self.language_predictor(x)

        if self.cfg.MODEL.ROI_MASK_HEAD.CONV5_ARCH != "none_a":
            mask_pred = F.relu(self.conv5_mask(x))
            mask_logits = self.mask_fcn_logits(mask_pred)
            # for CTC Loss, use the tensor before the last transposed conv
            if not self.cfg.SEQUENCE.DECODER_LOSS == "CTCLoss":
                x = mask_pred
        else:
            mask_logits = None

        if self.training:
            # During training:
            loss_seq_dict = {}

            device = language_logits.device

            num_words = language_logits.shape[0]

            assert num_words > 0

            keep_indices = []

            if gt_language_targets is not None:
                assert (
                    len(gt_language_targets) == num_words
                ), f"Dimension mismatch: {len(gt_language_targets)} != {num_words}"
                for word_id in range(0, num_words):
                    try:
                        gt_language = gt_language_targets[word_id].languages[0]
                    except IndexError:
                        print(f"gt_language_targets = {gt_language_targets}")
                        print(f"word_id = {word_id}")
                        print(f"gt_language_targets[{word_id}] = {gt_language_targets[word_id]}")
                        gt_language = gt_language_targets[word_id].languages[0]  # raise
                    if gt_language != "none":
                        keep_indices.append(word_id)

            if len(keep_indices) == 0:
                # TODO: think of a better fix
                if num_words > 0:
                    print(
                        (
                            "[Warning] No valid word/language in this batch,"
                            " keep one to avoid require_gradient error for now"
                        )
                    )
                    keep_indices.append(0)

            num_filtered = num_words - len(keep_indices)
            if num_filtered > 0:
                language_logits = language_logits[keep_indices]
                gt_language_targets = gt_language_targets[keep_indices]

                if random.random() < 0.001:
                    # log every 1000
                    print(
                        "[Info] Filtered {} out of {} targets using none-language criteria".format(
                            num_filtered, num_words
                        )
                    )

                kept_x = x[keep_indices]
                num_words = language_logits.shape[0]
                # need to handle the case when num_words == 0 below
            else:
                kept_x = x

            gt_lang_id = torch.zeros(num_words).long()
            best_coverage = torch.zeros(num_words)

            for word_id in range(0, num_words):
                if gt_language_targets is not None:
                    gt_language = gt_language_targets[word_id].languages[0]
                    if gt_language in self.lang_to_id:
                        # the gt language is supported by current model
                        gt_lang_id[word_id] = self.lang_to_id[gt_language]
                        continue

                # either there's no gt language, or the gt language is not supported,
                # we use the following heuristics to figure out
                # the best potential gt language
                lang_id = 0
                max_count = 0
                best_language = self.default_language
                for language in self.cfg.SEQUENCE.LANGUAGES:
                    coverage = torch.sum((word_targets[language] > 0).int(), dim=1) - 1
                    if coverage[word_id] > best_coverage[word_id]:
                        best_coverage[word_id] = coverage[word_id]
                        gt_lang_id[word_id] = lang_id
                        best_language = language
                        max_count = 1
                    elif coverage[word_id] == best_coverage[word_id]:
                        if best_language == self.default_language:
                            # Prefer default language when equal
                            pass
                        elif language == self.default_language:
                            # Prefer default language when equal
                            gt_lang_id[word_id] = lang_id
                            best_language = language
                        else:
                            # Classical "Uniformly Pick Max Integer"
                            max_count += 1
                            if random.random() < 1.0 / max_count:
                                gt_lang_id[word_id] = lang_id
                                best_language = language

                    lang_id += 1

            gt_lang_id = gt_lang_id.to(device)

            # Compute loss for language prediction head
            loss_seq_dict[
                "loss_language_pred"
            ] = self.cfg.MODEL.LANGUAGE_HEAD.LOSS_WEIGHT * self.cross_entropy_loss(
                language_logits, gt_lang_id
            )

            # Language Grouper and loss
            word_lang_probs = self.gt_prob_matrix(gt_lang_id)
            # word_lang_probs = F.gumbel_softmax(language_logits, tau=1.0, hard=False)
            word_head_probs, loss_grouper = self.language_grouper(word_lang_probs)

            loss_seq_dict.update(loss_grouper)

            # Compute loss for each rec head
            for head_id, language in enumerate(self.cfg.SEQUENCE.LANGUAGES_ENABLED):
                # NOTE: for ctc loss, 0 is blank char, but using -1 will cause NaN
                # the current fix is to remove all "ignore" labels
                # (done in the head on decoder_targets, word_targets is not used in ctc head)

                kept_word_target = (
                    word_targets[language][keep_indices]
                    if num_filtered > 0
                    else word_targets[language]
                )
                kept_decoder_target = (
                    decoder_targets[language][keep_indices]
                    if num_filtered > 0
                    else decoder_targets[language]
                )

                loss_seq_dict[f"loss_seq_{language}"] = getattr(self, f"seq_{language}")(
                    x=kept_x,
                    decoder_targets=kept_decoder_target,
                    word_targets=kept_word_target,
                    language_weights=(
                        word_head_probs[:, head_id] + self.cfg.SEQUENCE.LOSS_WEIGHT_BASE
                    ),
                )

            return {
                "char_mask_logits": None,
                "mask_logits": mask_logits,
                "loss_seq_dict": loss_seq_dict,
            }

        # During inferencing:
        # convert to probabilities
        word_lang_probs = F.softmax(language_logits, dim=1)
        word_head_probs, loss_grouper = self.language_grouper(word_lang_probs)

        # language_id:
        #   predicted by language prediction head regardless of the enabled heads
        # language_id_enabled:
        #   the one with the highest language prediction score among the enabled heads

        num_words = language_logits.shape[0]

        rec_head_to_word_map = {}

        word_result_list = []

        for k in range(num_words):
            word_result = WordResult()
            language_id = torch.argmax(word_lang_probs[k]).item()
            language_prob = word_lang_probs[k][language_id].item()

            head_id = torch.argmax(word_head_probs[k]).item()
            head_prob = word_head_probs[k][head_id].item()

            if head_id not in rec_head_to_word_map:
                rec_head_to_word_map[head_id] = []

            rec_head_to_word_map[head_id].append(k)

            word_result.language_id = language_id
            word_result.language = self.cfg.SEQUENCE.LANGUAGES[language_id]
            word_result.language_prob = language_prob
            word_result.language_id_enabled = head_id
            word_result.language_enabled = self.cfg.SEQUENCE.LANGUAGES_ENABLED[head_id]
            word_result.language_prob_enabled = head_prob

            word_result_list.append(word_result)

        # print(f"rec_head_to_word_map = {rec_head_to_word_map}")

        for language_id_enabled in rec_head_to_word_map:
            language = self.cfg.SEQUENCE.LANGUAGES_ENABLED[language_id_enabled]
            indices = rec_head_to_word_map[language_id_enabled]
            kept_x = x[indices]

            seq_words, seq_scores, detailed_seq_scores = getattr(self, f"seq_{language}")(
                x=kept_x, use_beam_search=self.cfg.SEQUENCE.BEAM_SEARCH
            )

            for i in range(len(indices)):
                word_result_list[indices[i]].seq_word = seq_words[i]
                word_result_list[indices[i]].seq_score = sum(seq_scores[i]) / float(
                    len(seq_scores[i])
                )
                if detailed_seq_scores is not None:
                    word_result_list[indices[i]].detailed_seq_scores = np.squeeze(
                        np.array(detailed_seq_scores[i]), axis=1
                    )
                else:
                    word_result_list[indices[i]].detailed_seq_scores = None

        # When run_all_heads == False,
        # word_result_list will store one recognition result per word.
        return {
            "char_mask_logits": None,
            "mask_logits": mask_logits,
            "word_result_list": word_result_list,
        }
