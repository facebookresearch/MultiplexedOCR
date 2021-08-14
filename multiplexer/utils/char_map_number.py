# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class NumberCharMap(CharMap):
    MAX_CHAR_NUM = 18

    @classmethod
    def contain_char_exclusive(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

    @classmethod
    def contain_char_shared(cls, char):
        # Punctuations (keyboard order)
        if char in "-,./":
            return True

        return False
