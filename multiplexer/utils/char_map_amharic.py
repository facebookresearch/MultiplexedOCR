# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class AmharicCharMap(CharMap):
    MAX_CHAR_NUM = 368

    # https://en.wikipedia.org/wiki/Ethiopic_(Unicode_block)
    @classmethod
    def contain_char_exclusive(cls, char):
        return "\u1200" <= char <= "\u137f"

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!#$%*()_+-={{}}[]\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "«·»÷":
            return True

        # Punctuations (>256)
        if ord(char) in [8209, 10003]:
            return True

        # Quotation marks for Amharic, https://en.wikipedia.org/wiki/Quotation_mark
        if char in "“”":
            return True

        # Full-width characters
        if char in ["！"]:
            return True

        return False
