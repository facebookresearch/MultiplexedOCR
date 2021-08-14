# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class TamilCharMap(CharMap):
    MAX_CHAR_NUM = 108

    # tamil script: https://en.wikipedia.org/wiki/Tamil_(Unicode_block)
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0B80" <= char <= "\u0BFF":
            return True

        # Indian rupee sign
        if char == "₹":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Zero width non-joiner/joiner
        if ord(char) in [8203]:
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$*()_-=[]:\";'<>?,./":
            return True

        return False
