# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map import CharMap


class UnifiedBGHMPCharMap(CharMap):
    MAX_CHAR_NUM = 398

    @classmethod
    def contain_char_exclusive(cls, char):
        # Devanagari (Hindi/Marathi)
        if "\u0900" <= char <= "\u097F" or "\uA8E0" <= char <= "\uA8FF":
            return True

        # Bengali
        if "\u0980" <= char <= "\u09FF":
            return True

        # Gujarati
        if "\u0A80" <= char <= "\u0AFF":
            return True

        # Punjabi
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

        # Other punctuations
        if char in "।‘’":
            return True

        return False
