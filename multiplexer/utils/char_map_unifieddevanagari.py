# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map import CharMap


class UnifiedDevanagariCharMap(CharMap):
    MAX_CHAR_NUM = 178

    # devanagari encoding = Hindi + Sanskrit + ... (apprarently 120+ languages
    # use this according to wikipedia: https://en.wikipedia.org/wiki/Devanagari)
    # Languages with explicit OCR support: Hindi, Marathi
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0900" <= char <= "\u097F" or "\uA8E0" <= char <= "\uA8FF":
            return True

        # https://en.wikipedia.org/wiki/Danda
        if char in "॥":
            return True

        # Indian rupee sign
        if char == "₹":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Zero width non-joiner/joiner
        if ord(char) in [8203, 8204, 8205]:
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$%^&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "©":
            return True

        return False
