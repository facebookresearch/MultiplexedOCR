# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class CyrillicCharMap(CharMap):
    MAX_CHAR_NUM = 128

    # e.g., https://www.codetable.net/decimal/1089
    # https://www.utf8-chartable.de/unicode-utf8-table.pl?start=1024&number=128&names=-&utf8=string-literal
    substitution_map = {
        "{}{}".format(chr(208), chr(129)): chr(1025),  # Ё
        "{}{}".format(chr(208), chr(144)): chr(1040),  # А
        "{}{}".format(chr(208), chr(157)): chr(1053),  # Н
        "{}{}".format(chr(209), chr(129)): chr(1089),  # с
        "{}{}".format(chr(209), chr(141)): chr(1101),  # э
        "{}{}".format(chr(209), chr(143)): chr(1103),  # я
    }

    # cyrillic encoding =
    #     Russian + Belorussian + Bulgarian + Abkhasian + Serbian language
    # Languages with explicit OCR support: Russian, Bulgarian
    @classmethod
    def contain_char_exclusive(cls, char):
        if (
            "\u0400" <= char <= "\u04FF"
            or "\u0500" <= char <= "\u052F"
            or "\u2DE0" <= char <= "\u2DFF"
            or "\uA640" <= char <= "\uA69F"
            or "\u1C80" <= char <= "\u1C8F"
        ):
            return True
        else:
            return False

    @classmethod
    def contain_char_shared(cls, char):
        return False
