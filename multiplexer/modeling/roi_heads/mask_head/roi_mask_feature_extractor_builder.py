# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from multiplexer.modeling.roi_heads.box_head.roi_box_feature_extractors import (
    ResNet50Conv5ROIFeatureExtractor,
)
from multiplexer.modeling.roi_heads.mask_head.roi_mask_feature_extractor_base import (
    MaskRCNNFPNFeatureExtractor,
)

# from multiplexer.modeling.roi_heads.mask_head.roi_mask_feature_extractor_resnet18_rect_v0 import (
#     Resnet18RectV0FeatureExtractor,
# )
# from multiplexer.modeling.roi_heads.mask_head.roi_mask_feature_extractor_resnet18_rect_v1 import (
#     Resnet18RectV1FeatureExtractor,
# )
# from multiplexer.modeling.roi_heads.mask_head.roi_mask_feature_extractor_resnet18_rect_v2 import (
#     Resnet18RectV2FeatureExtractor,
# )
# from .roi_mask_feature_extractor_resnet18_rotated_v0 import (
#     Resnet18RotatedV0FeatureExtractor,
# )
# from .roi_mask_feature_extractor_resnet18_square_v0 import (
#     Resnet18SquareV0FeatureExtractor,
# )
# from multiplexer.modeling.roi_heads.mask_head.roi_mask_feature_extractor_rotated import (
#     RotatedMaskRCNNFPNFeatureExtractor,
# )

_ROI_MASK_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "MaskRCNNFPNFeatureExtractor": MaskRCNNFPNFeatureExtractor,
    # "Resnet18SquareV0FeatureExtractor": Resnet18SquareV0FeatureExtractor,
    # "Resnet18RectV0FeatureExtractor": Resnet18RectV0FeatureExtractor,
    # "Resnet18RectV1FeatureExtractor": Resnet18RectV1FeatureExtractor,
    # "Resnet18RectV2FeatureExtractor": Resnet18RectV2FeatureExtractor,
    # "Resnet18RotatedV0FeatureExtractor": Resnet18RotatedV0FeatureExtractor,
    # "RotatedMaskRCNNFPNFeatureExtractor": RotatedMaskRCNNFPNFeatureExtractor,
}


def make_roi_mask_feature_extractor(cfg):
    func = _ROI_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR]

    feature_extractor = func(cfg)
    if cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR_FROZEN:
        for p in feature_extractor.parameters():
            p.requires_grad = False

    return feature_extractor
