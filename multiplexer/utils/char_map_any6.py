# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map import CharMap


class Any6CharMap(CharMap):
    MAX_CHAR_NUM = 10998

    @classmethod
    def contain_char(cls, char):
        return True
