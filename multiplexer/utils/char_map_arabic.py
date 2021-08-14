# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class ArabicCharMap(CharMap):
    MAX_CHAR_NUM = 198

    # e.g., https://www.codetable.net/decimal/1729
    # https://utf8-chartable.de/unicode-utf8-table.pl?start=1536&number=128&names=-&utf8=0x
    substitution_map = {
        chr(217) + chr(129): chr(1601),  # ف
        chr(217) + chr(143): chr(1615),  # "\u064F"
        chr(219) + chr(129): chr(1729),  # ہ
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

        # Punctuations
        if char in "~!@#$%*()_+-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (>256)
        if ord(char) in [8226]:
            return True

        return False
