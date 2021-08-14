# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class UrduCharMap(CharMap):
    MAX_CHAR_NUM = 158

    # e.g., https://www.codetable.net/decimal/1729
    # https://utf8-chartable.de/unicode-utf8-table.pl?start=1536&number=128&names=-&utf8=0x
    substitution_map = {
        "{}{}".format(chr(217), chr(129)): chr(1601),  # ف
        "{}{}".format(chr(219), chr(129)): chr(1729),  # ہ
    }

    # arabic encoding = Arabic + Urdu + Persian language
    # Languages with explicit OCR support: Arabic, Urdu
    @classmethod
    def contain_char_exclusive(cls, char):
        return (
            "\u0600" <= char <= "\u06FF"
            or "\u0750" <= char <= "\u077F"
            or "\u08A0" <= char <= "\u08FF"
            or "\uFB50" <= char <= "\uFDFF"
            or "\uFE70" <= char <= "\uFEFF"
        )

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "!@$()_-|[]:\"',./":
            return True

        return False
