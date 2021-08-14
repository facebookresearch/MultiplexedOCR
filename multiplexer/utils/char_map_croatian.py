# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class CroatianCharMap(CharMap):
    MAX_CHAR_NUM = 108

    # https://en.wikipedia.org/wiki/Gaj%27s_Latin_alphabet
    @classmethod
    def contain_char_exclusive(cls, char):
        if char in ["Q", "q", "W", "w", "X", "x"]:
            return False
        elif "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        if ord(char) in [268, 269, 262, 263, 382, 272, 273, 352, 353, 381, 382]:
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        if char in "QqWwXx":
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "!@#*()_+-|[]:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "©":
            return True

        return False
