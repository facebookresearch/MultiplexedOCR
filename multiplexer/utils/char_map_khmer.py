# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class KhmerCharMap(CharMap):
    MAX_CHAR_NUM = 138

    # khmer script: https://en.wikipedia.org/wiki/Khmer_alphabet
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u1780" <= char <= "\u17FF" or "\u19E0" <= char <= "\u19FF":
            return True
        else:
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
        if char in '!@#$%&*()+-={{}}|[]:"<>?,./':
            return True

        # Punctuations (<256)
        if char in "®":
            return True

        return False
