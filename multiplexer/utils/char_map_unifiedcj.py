# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map import CharMap


class UnifiedCJCharMap(CharMap):
    MAX_CHAR_NUM = 8998

    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    @classmethod
    def contain_char_exclusive(cls, char):
        # Chinese/Kanji characters
        # [19968, 25343], [25344, 30719], [30720, 36095], [36096, 40959]
        # Equivalently:
        # if (
        #     "\u4E00" <= char <= "\u62FF"
        #     or "\u6300" <= char <= "\u77FF"
        #     or "\u7800" <= char <= "\u8CFF"
        #     or "\u8D00" <= char <= "\u9FFF"
        # ):
        #     return True
        if 19968 <= ord(char) <= 40959:
            return True

        # Hiragana and Katakana
        if "\u3040" <= char <= "\u30FF" or "\uFF66" <= char <= "\uFF9D":
            return True

        # CJK Symbols and Punctuation [\u3000, \u303F] = [12288, 12351]
        # https://www.compart.com/en/unicode/block/U+3000
        if 12288 <= ord(char) <= 12351:
            return True

        # Yuan/Yen sign
        if char == "¥" or char == "￥":
            return True

        # Bopomofo characters
        # https://en.wikipedia.org/wiki/Bopomofo_(Unicode_block)
        if 12549 <= ord(char) <= 12588:
            return True

        # CJK Unified Ideographs Extension A
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_A
        if 13312 <= ord(char) <= 19893:
            return True

        # Full-width characters corresponding to basic half-width characters
        if ord("！") <= ord(char) <= ord("～"):
            # [65281, 65374] are full-width characters corresponding to [33, 126]
            # https://en.wikipedia.org/wiki/Halfwidth_and_Fullwidth_Forms_(Unicode_block)
            return True

        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_B
        if ord(char) in [134071]:
            return True

        # https://en.wikipedia.org/wiki/CJK_Compatibility
        if 13056 <= ord(char) <= 13310:
            return True

        # Other full-width characters
        if ord(char) in [65073, 65105]:
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Number forms
        if 8544 <= ord(char) <= 8547:
            return True

        # Circled numbers
        if 9312 <= ord(char) <= 9331:
            return True

        # Space
        if char == " ":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#$%&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "®°·×":
            return True

        # Punctuations
        if ord(char) in [
            8208,
            8212,
            8213,
            8216,
            8220,
            8221,
            8226,
            8230,
            8251,
            8592,
            8594,
            9675,
            9678,
            9679,
            9702,
            9834,
        ]:
            return True

        # Letterlike Symbols
        if ord(char) in [8451]:
            return True

        return False
