# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn

from .build import PROPOSAL_GENERATOR_REGISTRY


@PROPOSAL_GENERATOR_REGISTRY.register()
class SPN(nn.Module):
    pass
