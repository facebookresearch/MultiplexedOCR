# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class GujaratiCharMap(CharMap):
    MAX_CHAR_NUM = 118

    # Gujarati script: https://en.wikipedia.org/wiki/Gujarati_alphabet
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0A80" <= char <= "\u0AFF":
            return True

        # https://en.wikipedia.org/wiki/Danda
        if char == "।":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations
        if char in "~!@#%*()_+-=|\\:\";'?,./":
            return True

        return False
