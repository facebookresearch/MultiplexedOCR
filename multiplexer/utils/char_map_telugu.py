# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class TeluguCharMap(CharMap):
    MAX_CHAR_NUM = 128

    # Telugu script: https://en.wikipedia.org/wiki/Telugu_(Unicode_block)
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0C00" <= char <= "\u0C7F":
            return True

        # Indian rupee sign
        if char == "₹":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "!@#*()_+-=|\\:\";'<>?,./":
            return True

        # Zero width non-joiner/joiner
        if ord(char) in [8204]:
            return True

        return False
