# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class MalayalamCharMap(CharMap):
    MAX_CHAR_NUM = 128

    # malayalam script: https://en.wikipedia.org/wiki/Malayalam_script
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0D00" <= char <= "\u0D7F":
            return True

        # Indian rupee sign
        if char == "₹":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Zero width non-joiner/joiner
        if ord(char) in [8204, 8205]:
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "!@#&*()_+`-[]\\:\";'?,./":
            return True

        return False
