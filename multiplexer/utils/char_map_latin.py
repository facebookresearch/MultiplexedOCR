# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class LatinCharMap(CharMap):
    MAX_CHAR_NUM = 242

    @classmethod
    def contain_char_exclusive(cls, char):
        if cls.contain_char_shared(char):
            return False

        if ord(char) < 256:
            return True

        # Greek small
        if 945 <= ord(char) <= 969:
            return True

        # Greek capital
        if ord(char) in [923, 928, 931, 934, 937]:
            return True

        # Cyrillic capital
        if ord(char) in [1041, 1048, 1049, 1051, 1055, 1059, 1062, 1071]:
            return True

        # Cyrillic small
        if ord(char) in [1079, 1092, 1102]:
            return True

        # Letter like symbols
        if ord(char) in [8470, 8496]:
            return True

        # Roman Numeral
        if ord(char) in [8545, 8546]:
            return True

        # Mathematical operators
        if ord(char) in [8721, 8747]:
            return True

        # Currency
        if ord(char) in [8353, 8361, 8364, 65509]:
            return True

        # Trademark
        if ord(char) in [8482]:
            return True

        # Other Latin
        return ord(char) in [
            256,
            269,
            275,
            283,
            333,
            338,
            339,
            352,
            363,
            376,
            441,
            466,
            593,
            643,
            658,
            7776,
            7838,
        ]

    @classmethod
    def contain_char_shared(cls, char):
        # Punctuations (keyboard order)
        if char in "~!@#%&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "°±·×÷":
            return True

        # Punctuations (>256)
        if ord(char) in [8216, 8592, 8593, 8594, 8595]:
            return True

        # Full-width characters corresponding to basic half-width characters
        if ord("！") <= ord(char) <= ord("～"):
            # [65281, 65374] are full-width characters corresponding to [33, 126]
            # https://en.wikipedia.org/wiki/Halfwidth_and_Fullwidth_Forms_(Unicode_block)
            return True

        return False
