#!/usr/bin/env python3

# from .language_predictors_v0 import V0LanguagePredictor
# from .language_predictors_v1 import V1LanguagePredictor
# from .language_predictors_v2 import V2LanguagePredictor
# from .language_predictors_v3 import V3LanguagePredictor
# from .language_predictors_v4 import V4LanguagePredictor
# from .language_predictors_v5 import V5LanguagePredictor


_LANGUAGE_PREDICTOR = {
    # "V0LanguagePredictor": V0LanguagePredictor,
    # "V1LanguagePredictor": V1LanguagePredictor,
    # "V2LanguagePredictor": V2LanguagePredictor,
    # "V3LanguagePredictor": V3LanguagePredictor,
    # "V4LanguagePredictor": V4LanguagePredictor,
    # "V5LanguagePredictor": V5LanguagePredictor,
}


def make_language_predictor(cfg):
    predictor = _LANGUAGE_PREDICTOR[cfg.MODEL.LANGUAGE_HEAD.PREDICTOR]
    language_head = predictor(cfg)
    if cfg.MODEL.LANGUAGE_HEAD.FROZEN:
        for p in language_head.parameters():
            p.requires_grad = False
    return language_head
