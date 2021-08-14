# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class KanaCharMap(CharMap):
    MAX_CHAR_NUM = 208

    # Japanese script:
    # http://www.rikai.com/library/kanjitables/kanji_codes.unicode.shtml
    # Hiragana and Katakana is fine, but Kanji is tricky because many are
    # shared with Chinese and it's hard to decide language on character-level
    # due to CJK Unification (current implementation will always go to Chinese).
    # Similarly Full-width characters and punctuations are also shared.
    # but half-width Katakana we can know it's Japanese:
    # https://en.wikipedia.org/wiki/Half-width_kana
    @classmethod
    def contain_char(cls, char):
        # Hiragana and Katakana
        return "\u3040" <= char <= "\u30FF" or "\uFF66" <= char <= "\uFF9D"
