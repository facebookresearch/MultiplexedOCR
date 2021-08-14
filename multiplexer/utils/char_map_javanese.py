# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class JavaneseCharMap(CharMap):
    MAX_CHAR_NUM = 108

    @classmethod
    def contain_char_exclusive(cls, char):
        return "A" <= char <= "Z" or "a" <= char <= "z"

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#&*()_+-={{}}[]\\:\"'<>?,./":
            return True

        return False
