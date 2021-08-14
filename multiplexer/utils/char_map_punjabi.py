# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class PunjabiCharMap(CharMap):
    MAX_CHAR_NUM = 118

    # Punjabi script: https://en.wikipedia.org/wiki/Gurmukhi_(Unicode_block)
    # the official name for the script should be Gurmukhi, but seems Punjabi
    # is the only major language there, so naming it Punjabi for simplicity
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0A00" <= char <= "\u0A7F":
            return True

        # https://en.wikipedia.org/wiki/Danda
        if char in "।॥":
            return True

        # https://en.wikipedia.org/wiki/Khanda_(Sikh_symbol)
        if char == "☬":
            return True

        # Indian rupee sign
        if char == "₹":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#*()_+`-{{}}|[]\\:\"'?,./":
            return True

        return False
