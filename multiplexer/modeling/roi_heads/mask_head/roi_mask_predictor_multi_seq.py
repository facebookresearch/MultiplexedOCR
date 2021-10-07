# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import random

import torch
from torch.nn import functional as F

from multiplexer.modeling.roi_heads.mask_head.roi_mask_predictor_base import BaseMaskRCNNC4Predictor
from multiplexer.modeling.roi_heads.mask_head.roi_seq_predictor_base import BaseSequencePredictor
from multiplexer.modeling.roi_heads.mask_head.roi_seq_predictor_ctc import CTCSequencePredictor
from multiplexer.utils.languages import LANGUAGE_COMBO, get_language_config


class MultiSeqMaskRCNNC4Predictor(BaseMaskRCNNC4Predictor):
    def __init__(self, cfg, do_init_weights=True):
        super(MultiSeqMaskRCNNC4Predictor, self).__init__(cfg, do_init_weights=False)
        assert cfg.SEQUENCE.SEQ_ON, "Sequence head must be on for this predictor!"

        # Build two-way mapping between language names and ids
        # Multiple language names can be matched to the same id,
        # if the language name is in LANGUAGE_COMBO
        self.lang_to_id = {}
        self.id_to_lang = {}
        id = 0
        for language in cfg.SEQUENCE.LANGUAGES:
            self.lang_to_id[language] = id

            # Example: if we only have u_la1 in SEQUENCE.LANGUAGES but not 'fr',
            # we can map 'fr' in the gt to the id of 'u_la1' directly
            if language in LANGUAGE_COMBO:
                for sub_language in LANGUAGE_COMBO[language]:
                    if sub_language not in cfg.SEQUENCE.LANGUAGES:
                        self.lang_to_id[sub_language] = id

            # reverse map
            self.id_to_lang[id] = language
            id += 1

        # Assign default language
        default_languages = ["en", "la", "u_la1", cfg.SEQUENCE.LANGUAGES[0]]
        for lang in default_languages:
            if lang in cfg.SEQUENCE.LANGUAGES:
                self.default_language = lang
                break

        # Add heads
        for language in cfg.SEQUENCE.LANGUAGES_ENABLED:
            language_cfg = get_language_config(cfg, language)

            frozen = (cfg.SEQUENCE.LANGUAGES_UNFREEZED is not None) and (
                language not in cfg.SEQUENCE.LANGUAGES_UNFREEZED
            )

            if cfg.SEQUENCE.DECODER_LOSS == "CTCLoss" or language_cfg.ARCH.startswith("ctc_"):
                self.seq_predictor = CTCSequencePredictor
            else:
                self.seq_predictor = BaseSequencePredictor

            setattr(
                self,
                "seq_{}".format(language),
                self.seq_predictor(
                    cfg=cfg,
                    dim_in=self.dim_in,
                    language=language,
                    num_char=language_cfg.NUM_CHAR,
                    embed_size=language_cfg.EMBED_SIZE,
                    hidden_size=language_cfg.HIDDEN_SIZE,
                    arch=language_cfg.ARCH,
                    frozen=frozen,
                ),
            )

        if do_init_weights:
            self.init_weights()

    def forward(self, x, decoder_targets=None, word_targets=None, gt_language_targets=None):
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
            num_words = x.shape[0]

            keep_indices = []

            for word_id in range(0, num_words):
                if gt_language_targets is not None:
                    gt_language = gt_language_targets[word_id].languages[0]
                    if gt_language != "none":
                        keep_indices.append(word_id)

            num_filtered = num_words - len(keep_indices)
            if num_filtered > 0:
                # gt_language_targets = gt_language_targets[keep_indices]

                if random.random() < 0.001:
                    # log every 1000
                    print(
                        "[Info] Filtered {} out of {} targets using none-language criteria".format(
                            num_filtered, num_words
                        )
                    )

                kept_x = x[keep_indices]
                num_words = kept_x.shape[0]
                # Need to handle the case when num_words == 0 below
            else:
                kept_x = x

            loss_seq_dict = {}
            for language in self.cfg.SEQUENCE.LANGUAGES_ENABLED:
                # fix word targets -1 to be 0 for ctc loss
                # NOTE: 0 is blank char for ctc, but use -1 will cause NaN
                if self.cfg.SEQUENCE.DECODER_LOSS == "CTCLoss":
                    word_targets[language][word_targets[language] == -1] = 0

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

                loss_seq_dict["loss_seq_{}".format(language)] = (
                    getattr(self, "seq_{}".format(language))(
                        x=kept_x,
                        decoder_targets=kept_decoder_target,
                        word_targets=kept_word_target,
                    )
                    if num_words > 0
                    else torch.tensor(0.0).to(device=x.device)
                )

            return {
                "char_mask_logits": None,
                "mask_logits": mask_logits,
                "loss_seq_dict": loss_seq_dict,
            }

        # During inferencing:
        decoded_chars_list = []
        decoded_scores_list = []
        detailed_decoded_scores_list = []

        for language in self.cfg.SEQUENCE.LANGUAGES_ENABLED:
            if not self.cfg.MODEL.TORCHSCRIPT_ONLY and self.cfg.SEQUENCE.BEAM_SEARCH:
                decoded_chars, decoded_scores, detailed_decoded_scores = getattr(
                    self, "seq_{}".format(language)
                )(x, use_beam_search=self.cfg.SEQUENCE.BEAM_SEARCH)
            else:
                decoded_chars, decoded_scores, detailed_decoded_scores = getattr(
                    self, "seq_{}".format(language)
                )(x)
            decoded_chars_list.append(decoded_chars)
            decoded_scores_list.append(decoded_scores)
            detailed_decoded_scores_list.append(detailed_decoded_scores)

        return {
            "char_mask_logits": None,
            "seq_outputs_list": decoded_chars_list,
            "seq_scores_list": decoded_scores_list,
            "detailed_seq_scores_list": detailed_decoded_scores_list,
            "language_probs": None,
            "mask_logits": mask_logits,
        }
