# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class FrenchCharMap(CharMap):
    MAX_CHAR_NUM = 158

    # https://en.wikipedia.org/wiki/French_orthography
    @classmethod
    def contain_char_exclusive(cls, char):
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        # Diacritics and ligatures
        if char in "ÀàÂâÆæÇçÉéÈèÊêËëÎîÏïÔôŒœÙùÛûÜüŸÿ":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$%&*()_+-[]:\";'?,./":
            return True

        # Punctuations (<256)
        if char in "®":
            return True

        return False
