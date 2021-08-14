# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map import CharMap
from .char_map_hangul import HangulCharMap


class UnifiedKECharMap(CharMap):
    MAX_CHAR_NUM = 2498

    @classmethod
    def contain_char_exclusive(cls, char):
        return HangulCharMap.contain_char_exclusive(char)

    @classmethod
    def contain_char_shared(cls, char):
        if HangulCharMap.contain_char_shared(char):
            return True

        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        return False
