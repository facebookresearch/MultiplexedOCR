# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map_unifiedcj import UnifiedCJCharMap


class UnifiedCJECharMap(UnifiedCJCharMap):
    MAX_CHAR_NUM = 8998

    @classmethod
    def contain_char_exclusive(cls, char):
        return super(UnifiedCJECharMap, cls).contain_char_exclusive(char)

    @classmethod
    def contain_char_shared(cls, char):
        # the only difference between UnifiedCJECharMap and UnifiedCJCharMap is
        # UnifiedCJECharMap supports basic English characters
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        return super(UnifiedCJECharMap, cls).contain_char_shared(char)
