#!/usr/bin/env python3

import random

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from multiplexer.modeling.roi_heads.mask_head.language_predictor_builder import (
    make_language_predictor,
)
from multiplexer.modeling.roi_heads.mask_head.roi_mask_predictor_multi_seq import (
    MultiSeqMaskRCNNC4Predictor,
)
from multiplexer.structures.word_result import WordResult
from multiplexer.utils.languages import LANGUAGE_COMBO


class MultiSeqLangMaskRCNNC4Predictor(MultiSeqMaskRCNNC4Predictor):
    def __init__(self, cfg, do_init_weights=True):
        super(MultiSeqLangMaskRCNNC4Predictor, self).__init__(cfg, do_init_weights=False)

        self.run_all_heads = cfg.MULTIPLEXER.TEST.RUN_ALL_HEADS
        if not self.run_all_heads:
            self.build_rec_head_map()

        self.language_predictor = make_language_predictor(cfg=cfg)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        if do_init_weights:
            self.init_weights()

    def build_rec_head_map(self):
        if self.cfg.SEQUENCE.LANGUAGES_ENABLED == self.cfg.SEQUENCE.LANGUAGES:
            self.enabled_all_rec_heads = True
            self.rec_head_map = None
        else:
            self.enabled_all_rec_heads = False
            # rec_head_map is a 1-to-N mapping from recognition head to a subset of languages;
            # it's possible that some languages do not have a dedicated recognition head.
            self.rec_head_map = {}
            for rec_id, language_rec in enumerate(self.cfg.SEQUENCE.LANGUAGES_ENABLED):
                if language_rec in LANGUAGE_COMBO:
                    covered_language_set = set(LANGUAGE_COMBO[language_rec]) - set(
                        self.cfg.SEQUENCE.LANGUAGES_ENABLED
                    )
                    if language_rec in self.cfg.SEQUENCE.LANGUAGES:
                        # In this case, this unified rec head has
                        # its own corresponding output from LID
                        # and acts like a non-unified head
                        covered_language_set.add(language_rec)

                    assert len(covered_language_set) > 0, (
                        f"[Error] Rec-head seq_{language_rec} is unnecessary"
                        " since all sub-languages have dedicated heads"
                    )
                else:
                    covered_language_set = {language_rec}

                self.rec_head_map[rec_id] = [
                    id
                    for id, language in enumerate(self.cfg.SEQUENCE.LANGUAGES)
                    if language in covered_language_set
                ]

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

            keep_indices = []

            if gt_language_targets is not None:
                assert len(gt_language_targets) == num_words, f"Dimension mismatch: {len(gt_language_targets)} != {num_words}"
                for word_id in range(0, num_words):
                    try:
                        gt_language = gt_language_targets[word_id].languages[0]
                    except IndexError:
                        print(f"gt_language_targets = {gt_language_targets}")
                        print(f"word_id = {word_id}")
                        print(f"gt_language_targets[{word_id}] = {gt_language_targets[word_id]}")
                        print(f"gt_language_targets[{word_id}].languages = {gt_language_targets[word_id].languages}")
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

            # Compute loss for language prediction head
            if num_words > 0:
                loss_seq_dict[
                    "loss_language_pred"
                ] = self.cfg.MODEL.LANGUAGE_HEAD.LOSS_WEIGHT * self.cross_entropy_loss(
                    language_logits, gt_lang_id.to(device=device)
                )
            else:
                # Not working
                zero_loss = torch.tensor(0.0).to(device=device)
                # https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074/6
                loss_seq_dict["loss_language_pred"] = Variable(zero_loss, requires_grad=True)

            # Compute loss for each rec head
            for language in self.cfg.SEQUENCE.LANGUAGES_ENABLED:
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

                if num_words > 0:
                    loss_seq_dict["loss_seq_{}".format(language)] = getattr(
                        self, "seq_{}".format(language)
                    )(
                        x=kept_x,
                        decoder_targets=kept_decoder_target,
                        word_targets=kept_word_target,
                    )
                else:
                    # Not working
                    zero_loss = torch.tensor(0.0).to(device=device)
                    # https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074/6
                    loss_seq_dict["loss_seq_{}".format(language)] = Variable(
                        zero_loss, requires_grad=True
                    )

            return {
                "char_mask_logits": None,
                "mask_logits": mask_logits,
                "loss_seq_dict": loss_seq_dict,
            }

        # During inferencing:
        # convert to probabilities
        language_probs = F.softmax(language_logits, dim=1)

        if self.run_all_heads:
            decoded_chars_list = []
            decoded_scores_list = []
            detailed_decoded_scores_list = []

            for language in self.cfg.SEQUENCE.LANGUAGES_ENABLED:
                decoded_chars, decoded_scores, detailed_decoded_scores = getattr(
                    self, "seq_{}".format(language)
                )(x, use_beam_search=self.cfg.SEQUENCE.BEAM_SEARCH)
                decoded_chars_list.append(decoded_chars)
                decoded_scores_list.append(decoded_scores)
                detailed_decoded_scores_list.append(detailed_decoded_scores)

            # When run_all_heads == True,
            # seq_outputs_list, seq_scores_list and detailed_seq_scores_list
            # will store len(SEQUENCE.LANGUAGES_ENABLED) recognition results per word.
            return {
                "char_mask_logits": None,
                "seq_outputs_list": decoded_chars_list,
                "seq_scores_list": decoded_scores_list,
                "detailed_seq_scores_list": detailed_decoded_scores_list,
                "language_probs": language_probs,
                "mask_logits": mask_logits,
            }
        else:
            # language_id:
            #   predicted by language prediction head regardless of the enabled heads
            # language_id_enabled:
            #   the one with the highest language prediction score among the enabled heads

            num_words = language_logits.shape[0]

            rec_head_to_word_map = {}

            word_result_list = []

            for k in range(num_words):
                word_result = WordResult()
                language_id = torch.argmax(language_probs[k]).item()
                language_prob = language_probs[k][language_id].item()
                if self.enabled_all_rec_heads:
                    language_id_enabled = language_id
                    language_prob_enabled = language_prob
                else:
                    language_id_enabled = 0
                    language_prob_enabled = 0
                    for rec_id, _ in enumerate(self.cfg.SEQUENCE.LANGUAGES_ENABLED):
                        for pred_id in self.rec_head_map[rec_id]:
                            if language_probs[k][pred_id] > language_prob_enabled:
                                language_prob_enabled = language_probs[k][pred_id].item()
                                language_id_enabled = rec_id
                if language_id_enabled not in rec_head_to_word_map:
                    rec_head_to_word_map[language_id_enabled] = []

                rec_head_to_word_map[language_id_enabled].append(k)

                word_result.language_id = language_id
                word_result.language = self.cfg.SEQUENCE.LANGUAGES[language_id]
                word_result.language_prob = language_prob
                word_result.language_id_enabled = language_id_enabled
                word_result.language_enabled = self.cfg.SEQUENCE.LANGUAGES_ENABLED[
                    language_id_enabled
                ]
                word_result.language_prob_enabled = language_prob_enabled

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
