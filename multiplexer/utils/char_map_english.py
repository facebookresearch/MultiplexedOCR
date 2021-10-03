from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class EnglishCharMap(CharMap):
    MAX_CHAR_NUM = 118

    @classmethod
    def contain_char_exclusive(cls, char):
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        # Dollar/Pound sign
        if char in "$£":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#%&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "®°±·×÷":
            return True

        return False
