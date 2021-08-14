# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class BurmeseCharMap(CharMap):
    MAX_CHAR_NUM = 158

    # burmese encoding = Burmese + Mon + Karen + Kayah + Shan + Palaung language
    # from wikipedia: "it is also used to write Pali and Sanskrit in Myanmar"
    # don't know how it works exactly but should be careful if need to support those
    # Languages with explicit OCR support: Burmese
    @classmethod
    def contain_char_exclusive(cls, char):
        if (
            "\u1000" <= char <= "\u109F"
            or "\uAA60" <= char <= "\uAA7F"
            or "\uA9E0" <= char <= "\uA9FF"
        ):
            return True
        elif "\ue000" <= char <= "\uf8ff":
            # Extended range for "fake" NFC
            return True
        else:
            return False

    @classmethod
    def contain_char_shared(cls, char):
        # Zero width non-joiner/joiner
        if ord(char) in [8203, 8204]:
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$%*()+-={{}}|[]\\:\"'<>?,./":
            return True

        return False
