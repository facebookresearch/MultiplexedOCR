# Copyright (c) Facebook, Inc. and its affiliates.
from .build import build_backbone, BACKBONE_REGISTRY  # noqa F401 isort:skip

from .fpn import FPN
from .resnet import ResNet