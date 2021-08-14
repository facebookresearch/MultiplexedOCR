# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class SymbolCharMap(CharMap):
    MAX_CHAR_NUM = 58

    @classmethod
    def contain_char_exclusive(cls, char):
        # Punctuations (keyboard order)
        if char in "~!@#$%^&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (not on keyboard, within 256)
        if char in "¥§®·×":
            return True

        # Currency
        if ord(char) in [8361, 65509]:
            return True

        # Full-width punctuations
        if ord(char) in [65283, 65288]:
            return True

        return ord(char) in [
            1548,
            1644,
            2404,
            8204,
            8212,
            8226,
            8251,
            8364,
            8739,
            9601,
            9642,
            9825,
            9836,
            12290,
            12539,
        ]

    @classmethod
    def contain_char_shared(cls, char):
        return False
