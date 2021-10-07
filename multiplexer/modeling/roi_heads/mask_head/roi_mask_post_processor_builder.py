#!/usr/bin/env python3

# from .roi_mask_post_processor_multi_seq import MultiSeqPostProcessor
from .roi_mask_post_processor_multi_seq_1_char_mask import MultiSeq1CharMaskPostProcessor

_ROI_MASK_POST_PROCESSORS = {
    # "MultiSeqPostProcessor": MultiSeqPostProcessor,
    "MultiSeq1CharMaskPostProcessor": MultiSeq1CharMaskPostProcessor,
}


def make_roi_mask_post_processor(cfg):
    # TODO: add MODEL.ROI_MASK_HEAD.POST_PROCESSOR to config
    post_processor = _ROI_MASK_POST_PROCESSORS[cfg.MODEL.ROI_MASK_HEAD.POST_PROCESSOR]
    return post_processor(cfg=cfg, masker=None)
