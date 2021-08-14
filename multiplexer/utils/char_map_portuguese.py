# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class PortugueseCharMap(CharMap):
    MAX_CHAR_NUM = 118

    @classmethod
    def contain_char_exclusive(cls, char):
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        # https://en.wikipedia.org/wiki/Portuguese_orthography
        # Diacritics
        if char in "çáéíóúâêôãõàèìòù":
            return True
        if char in "ÇÁÉÓÂÃÊÕ":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "!@#$*()_+-=[]:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "¡®":
            return True

        # Other characters (<10000)
        if char in "‡‰€":
            return True

        return False
