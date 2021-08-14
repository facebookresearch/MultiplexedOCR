# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class HebrewCharMap(CharMap):
    MAX_CHAR_NUM = 108

    # https://utf8-chartable.de/unicode-utf8-table.pl?start=1408&number=128&names=-&utf8=0x
    substitution_map = {
        "{}{}".format(chr(215), chr(144)): chr(1488),  # א
        "{}{}".format(chr(215), chr(157)): chr(1501),  # ם
    }

    # hebrew script: https://en.wikipedia.org/wiki/Hebrew_alphabet#Unicode_and_HTML
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0590" <= char <= "\u05FF" or "\uFB1D" <= char <= "\uFB4F":
            return True

        # Shekel sign (currency)
        if char in "₪":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#&*()_+-\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "©":
            return True

        return False
