# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map import CharMap
from .char_map_greek import GreekCharMap
from .char_map_unifiedcyrillic import UnifiedCyrillicCharMap


class UnifiedCGCharMap(CharMap):
    MAX_CHAR_NUM = 258

    @classmethod
    def contain_char_exclusive(cls, char):
        return UnifiedCyrillicCharMap.contain_char_exclusive(
            char
        ) or GreekCharMap.contain_char_exclusive(char)

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$%^&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "©«·»":
            return True

        # https://en.wikipedia.org/wiki/Quotation_mark
        if char in "„“”":
            return True

        # Other symbols
        if ord(char) in [8212, 8226, 8470]:
            return True

        return False
