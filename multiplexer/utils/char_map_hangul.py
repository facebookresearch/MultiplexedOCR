# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class HangulCharMap(CharMap):
    MAX_CHAR_NUM = 2498

    # hangul script (Korean): https://en.wikipedia.org/wiki/Hangul
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    @classmethod
    def contain_char_exclusive(cls, char):
        # Won sign
        if ord(char) in [8361]:
            return True

        # Hangul
        return (
            "\uAC00" <= char <= "\uD7AF"
            or "\u1100" <= char <= "\u11FF"
            or "\u3130" <= char <= "\u318F"
            or "\uA960" <= char <= "\uA97F"
            or "\uD7B0" <= char <= "\uD7FF"
        )

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$%^&*()_+`-={{}}|[]\\<>:\";'?,./":
            return True

        # Punctuations (<256)
        if char in "©°·×":
            return True

        # Punctuations (others)
        if ord(char) in [
            8217,
            8221,
            8226,
            8251,
            9642,
            9675,
            9679,
            12298,
            12299,
            12300,
            12301,
            12302,
            12303,
            12539,
        ]:
            return True

        # Full-width characters
        if ord(char) in [65287]:
            return True

        return False
