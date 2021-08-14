# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map_unifiedapu import UnifiedAPUCharMap


class UnifiedAPUECharMap(UnifiedAPUCharMap):
    MAX_CHAR_NUM = 298

    @classmethod
    def contain_char_exclusive(cls, char):
        return super(UnifiedAPUECharMap, cls).contain_char_exclusive(char)

    @classmethod
    def contain_char_shared(cls, char):
        # the only difference between UnifiedAPUECharMap and UnifiedAPUCharMap is
        # UnifiedAPUECharMap supports basic English characters
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        return super(UnifiedAPUECharMap, cls).contain_char_shared(char)
