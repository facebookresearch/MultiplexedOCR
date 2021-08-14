# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List

import torch


class WordResult(object):
    def __init__(self):
        self.box = []
        self.rotated_box = []
        self.rotated_box_5d = None
        self.polygon = []
        self.det_score = 0.0

        self.language_id = -1
        self.language = "none"
        self.language_prob = 0.0
        self.language_id_enabled = -1
        self.language_enabled = "none"
        self.language_prob_enabled = 0.0

        # seq_word is for normal inference and testing in python
        # seq_word_labels is for inference in torchscript, decoding will happen outside
        self.seq_word = None
        self.seq_word_labels: List[int] = []
        self.seq_score: float = 0.0
        self.detailed_seq_scores = None

    @torch.jit.unused
    def should_be_filtered(
        self,
        name,
        det_conf_thresh=0,
        seq_conf_thresh=0,
        det_conf_thresh2=None,
        seq_conf_thresh2=None,
    ):
        score_det = self.det_score
        score_rec_seq = self.seq_score
        word = self.seq_word

        filtered = False

        if score_det < det_conf_thresh or score_rec_seq < seq_conf_thresh:
            filtered = True

        if filtered:
            if det_conf_thresh2 is not None:
                assert seq_conf_thresh2 is not None
                if score_det > det_conf_thresh2 and score_rec_seq > seq_conf_thresh2:
                    filtered = False

        status = "filtered" if filtered else "kept"
        lang = self.language  # predicted language
        rec_lang = self.language_enabled  # actual rec head
        score_str = "[det={:.3f}, seq={:.3f}]".format(score_det, score_rec_seq)

        print(f"[{name}][{status}][{lang}][{rec_lang}][{score_str}] {word}")

        return filtered

    def __repr__(self):
        seq_word = self.seq_word if self.seq_word is not None else self.seq_word_labels
        return ", ".join(
            [
                f"WordResult(word='{seq_word}'",
                f"lang={self.language} ({self.language_id}, {self.language_prob:.3f})",
                f"head={self.language_enabled} ({self.language_id_enabled}",
                f"{self.language_prob_enabled:.3f})",
                f"det={self.det_score:.3f}",
                f"seq={self.seq_score:.3f})",
            ]
        )
