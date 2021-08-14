# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .build import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .cropped_rcnn import CroppedRCNN
from .rcnn import GeneralizedRCNN
