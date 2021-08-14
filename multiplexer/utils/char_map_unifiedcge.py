# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map import CharMap
from .char_map_unifiedcg import UnifiedCGCharMap


class UnifiedCGECharMap(CharMap):
    MAX_CHAR_NUM = 278

    @classmethod
    def contain_char_exclusive(cls, char):
        return UnifiedCGCharMap.contain_char_exclusive(char)

    @classmethod
    def contain_char_shared(cls, char):
        # the only difference between UnifiedCGECharMap and UnifiedCGCharMap is
        # UnifiedCGECharMap supports basic English characters
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        return UnifiedCGCharMap.contain_char_shared(char)
