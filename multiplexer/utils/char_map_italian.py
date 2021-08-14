# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class ItalianCharMap(CharMap):
    MAX_CHAR_NUM = 108

    # https://www.alt-codes.net/italian_alt_codes/
    @classmethod
    def contain_char_exclusive(cls, char):
        return (
            "A" <= char <= "Z"
            or "a" <= char <= "z"
            or ord(char) in [192, 200, 201, 204, 210, 217, 224, 232, 233, 236, 242, 249]
        )

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#&*()_+-=|\\:\";'<>?,./":
            return True

        # Punctuations (not on keyboard, within 256)
        if char in "®":
            return True

        return False
