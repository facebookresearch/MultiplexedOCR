# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class GreekCharMap(CharMap):
    MAX_CHAR_NUM = 118

    # Greek script: https://en.wikipedia.org/wiki/Greek_and_Coptic
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0370" <= char <= "\u03FF" or "\u1F00" <= char <= "\u1FFF":
            return True
        else:
            return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#&*()_+`-={{}}[]\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "«»":
            return True

        return False
