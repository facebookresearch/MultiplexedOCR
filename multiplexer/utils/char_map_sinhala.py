from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class SinhalaCharMap(CharMap):
    MAX_CHAR_NUM = 118

    # sinhala: https://en.wikipedia.org/wiki/Sinhala_(Unicode_block)
    @classmethod
    def contain_char_exclusive(cls, char):
        if "\u0D80" <= char <= "\u0DFF":
            return True
        else:
            return False

    @classmethod
    def contain_char_shared(cls, char):
        # Zero width non-joiner/joiner
        if ord(char) in [8203, 8205]:
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#*()_+-=|\\:\";'?,./":
            return True

        return False
