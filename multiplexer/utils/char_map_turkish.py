# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class TurkishCharMap(CharMap):
    MAX_CHAR_NUM = 108

    @classmethod
    def contain_char_exclusive(cls, char):
        # characters not directly in Turkish
        if char in "QqWwXx":
            return False

        # Turkish alphabet
        # https://en.wikipedia.org/wiki/Turkish_alphabet
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        if char in "ÇçĞğÖöŞşÜü":
            return True

        # https://en.wikipedia.org/wiki/Dotted_and_dotless_I
        if char in "İı":
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
        if char in "~!@#%&*()_+-=|[]:\";'?,./":
            return True

        return False
