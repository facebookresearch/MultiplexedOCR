# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map import CharMap
from .languages import BurmeseCharMap, KhmerCharMap, ThaiCharMap


class UnifiedBKTCharMap(CharMap):
    MAX_CHAR_NUM = 398

    @classmethod
    def contain_char_exclusive(cls, char):
        return (
            ThaiCharMap.contain_char_exclusive(char)
            or BurmeseCharMap.contain_char_exclusive(char)
            or KhmerCharMap.contain_char_exclusive(char)
        )

    @classmethod
    def contain_char_shared(cls, char):
        # Zero width non-joiner/joiner
        if ord(char) in [8203, 8204]:
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$%^&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "©®":
            return True

        # Punctuations (>256)
        if ord(char) in [8226, 10003]:
            return True

        return False
