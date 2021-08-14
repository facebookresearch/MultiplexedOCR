# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class ThaiCharMap(CharMap):
    MAX_CHAR_NUM = 208

    # thai encoding;
    # Thai is a Unicode block containing characters for the Thai, Lanna Tai,
    # and Pali languages.
    # It is based on the Thai Industrial Standard 620-2533.
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0E01" <= char <= "\u0E3A" or "\u0E3F" <= char <= "\u0E5B":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$%^&*()_+-=[]\\:\";'<>?,./":
            return True

        # Punctuations (>256)
        if ord(char) in [8226, 10003]:
            return True

        return False
