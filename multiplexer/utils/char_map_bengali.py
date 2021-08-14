# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class BengaliCharMap(CharMap):
    MAX_CHAR_NUM = 118

    # bengali script: https://en.wikipedia.org/wiki/Bengali_alphabet#Unicode
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0980" <= char <= "\u09FF":
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

        # Zero width non-joiner/joiner
        if ord(char) in [8204]:
            return True

        # Punctuations (keyboard order)
        if char in "~!@#%*()_+`-{{}}[]\\:\";'?,./":
            return True

        # Other punctuations
        if char in "।‘’":
            return True

        return False
