#!/usr/bin/env python3

# from .roi_mask_predictor import (
#     CharMaskRCNNC4Predictor,
#     MaskRCNNC4Predictor,
#     SeqMaskRCNNC4Predictor,
#     SeqRCNNC4Predictor,
# )
from .roi_mask_predictor_grouped import GroupedMaskRCNNC4Predictor
from .roi_mask_predictor_multi_seq import MultiSeqMaskRCNNC4Predictor

# from .roi_mask_predictor_multi_seq_1_char_mask import MultiSeq1CharMaskRCNNC4Predictor
from .roi_mask_predictor_multi_seq_lang import MultiSeqLangMaskRCNNC4Predictor

# from .roi_mask_predictor_multiplexed import MultiplexedMaskRCNNC4Predictor
# from .roi_mask_predictor_seq_char_mask import SeqCharMaskRCNNC4Predictor


_ROI_MASK_PREDICTOR = {
    "GroupedMaskRCNNC4Predictor": GroupedMaskRCNNC4Predictor,
    # "MaskRCNNC4Predictor": MaskRCNNC4Predictor,
    # "CharMaskRCNNC4Predictor": CharMaskRCNNC4Predictor,
    # "SeqCharMaskRCNNC4Predictor": SeqCharMaskRCNNC4Predictor,
    # "SeqMaskRCNNC4Predictor": SeqMaskRCNNC4Predictor,
    # "SeqRCNNC4Predictor": SeqRCNNC4Predictor,
    "MultiSeqMaskRCNNC4Predictor": MultiSeqMaskRCNNC4Predictor,
    # "MultiplexedMaskRCNNC4Predictor": MultiplexedMaskRCNNC4Predictor,
    "MultiSeqLangMaskRCNNC4Predictor": MultiSeqLangMaskRCNNC4Predictor,
    # "MultiSeq1CharMaskRCNNC4Predictor": MultiSeq1CharMaskRCNNC4Predictor,
}


def make_roi_mask_predictor(cfg):
    predictor = _ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]

    return predictor(cfg)
