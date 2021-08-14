# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class SpanishCharMap(CharMap):
    MAX_CHAR_NUM = 118

    # https://www.alt-codes.net/spanish_alt_codes/
    @classmethod
    def contain_char_exclusive(cls, char):
        return (
            "A" <= char <= "Z"
            or "a" <= char <= "z"
            or ord(char)
            in [
                161,
                191,
                193,
                201,
                205,
                209,
                211,
                218,
                220,
                225,
                233,
                237,
                241,
                243,
                250,
                252,
            ]
        )

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#*()_+-\\:\";'?,./":
            return True

        # Punctuations (not on keyboard, within 256)
        if char in "®":
            return True

        return False
