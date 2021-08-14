# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class DutchCharMap(CharMap):
    MAX_CHAR_NUM = 98

    @classmethod
    def contain_char_exclusive(cls, char):
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        # A small number of Dutch words use Ë/É
        # https://en.wiktionary.org/wiki/Category:Dutch_terms_spelled_with_%C3%8B (780)
        # https://en.wiktionary.org/wiki/Category:Dutch_terms_spelled_with_%C3%89 (88)
        if char in "ËëÉé":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # A small number of Dutch words use Ü
        # https://en.wiktionary.org/wiki/Category:Dutch_terms_spelled_with_%C3%9C (19)
        if char in "Üü":
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations
        if char in "!@#&*()_+`-|:\";'?,./":
            return True

        return False
