# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)

# Note: Different from other CharMaps, this CharMap does not
# save a json file, but use hard-coded mapping within the class


class EnglishNumber36CharMap(CharMap):
    MAX_CHAR_NUM = 36

    @classmethod
    def char2num(cls, char, read_only=True, confirmed_by_gt=False, max_char_num=-1):
        if "0" <= char <= "9":
            return ord(char) - ord("0") + 1

        if "a" <= char <= "z":
            return ord(char) - ord("a") + 11

        # This CharMap doesn't distinguish between upper case and lower case
        if "A" <= char <= "Z":
            return ord(char.lower()) - ord("a") + 11

        return 0

    @classmethod
    def contain_char_exclusive(cls, char):
        return "a" <= char <= "z" or "A" <= char <= "Z" or "0" <= char <= "9"

    @classmethod
    def contain_char_shared(cls, char):
        return False

    @classmethod
    def load_default(cls, retry=0):
        if len(cls.chars) == 0:
            cls.chars = []
            for ch in "0123456789abcdefghijklmnopqrstuvwxyz":
                cls.chars.append(ch)
            cls.count = len(cls.chars)
