# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class HungarianCharMap(CharMap):
    MAX_CHAR_NUM = 118

    @classmethod
    def contain_char_exclusive(cls, char):
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        if char in "ÁáÉéÍíÓóÖöŐőÚúÜüŰű":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#%&*()+-_={{}}\\:\";'?,./":
            return True

        # Quotation marks for Hungarian, https://en.wikipedia.org/wiki/Quotation_mark
        if char in "„”":
            return True

        return False
