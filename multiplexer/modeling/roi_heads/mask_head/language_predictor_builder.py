# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# from .language_predictor_v0 import V0LanguagePredictor
from .language_predictor_v1 import V1LanguagePredictor

# from .language_predictor_v2 import V2LanguagePredictor
# from .language_predictor_v3 import V3LanguagePredictor
# from .language_predictor_v4 import V4LanguagePredictor
from .language_predictor_v5 import V5LanguagePredictor
from .language_predictor_v6 import V6LanguagePredictor
from .language_predictor_v7 import V7LanguagePredictor
from .language_predictor_v8 import V8LanguagePredictor

_LANGUAGE_PREDICTOR = {
    # "V0LanguagePredictor": V0LanguagePredictor,
    "V1LanguagePredictor": V1LanguagePredictor,
    # "V2LanguagePredictor": V2LanguagePredictor,
    # "V3LanguagePredictor": V3LanguagePredictor,
    # "V4LanguagePredictor": V4LanguagePredictor,
    "V5LanguagePredictor": V5LanguagePredictor,
    "V6LanguagePredictor": V6LanguagePredictor,
    "V7LanguagePredictor": V7LanguagePredictor,
    "V8LanguagePredictor": V8LanguagePredictor,
}


def make_language_predictor(cfg):
    predictor = _LANGUAGE_PREDICTOR[cfg.MODEL.LANGUAGE_HEAD.PREDICTOR]
    language_head = predictor(cfg)
    if cfg.MODEL.LANGUAGE_HEAD.FROZEN:
        for p in language_head.parameters():
            p.requires_grad = False
    return language_head
