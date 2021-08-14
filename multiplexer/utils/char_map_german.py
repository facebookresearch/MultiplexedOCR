# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class GermanCharMap(CharMap):
    MAX_CHAR_NUM = 118

    # German uses three letter-diacritic combinations (Ä/ä, Ö/ö, Ü/ü)
    # using the umlaut and one ligature (ß (called Eszett (sz) or scharfes S, sharp s))
    # which are officially considered distinct letters of the alphabet.
    # The capital ẞ was declared an official letter of the German alphabet
    # on 29 June 2017.
    # https://www.alt-codes.net/german_alt_codes/
    @classmethod
    def contain_char_exclusive(cls, char):
        return (
            "A" <= char <= "Z"
            or "a" <= char <= "z"
            or ord(char) in [196, 214, 220, 223, 228, 246, 252, 7838]
        )

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$&%*()_+-={{}}[]\\:\";'<>?,./":
            return True

        # Quotation marks for German, https://en.wikipedia.org/wiki/Quotation_mark
        if char in "„”":
            return True

        # Copyright
        if char in "©":
            return True

        return False
