# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from multiplexer.utils.char_map import CharMap


class UnifiedLatin1CharMap(CharMap):
    MAX_CHAR_NUM = 498

    @classmethod
    def contain_char_exclusive(cls, char):
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        if 191 <= ord(char) <= 382:
            return True

        if 416 <= ord(char) <= 417:
            return True

        if 431 <= ord(char) <= 438:
            return True

        if 452 <= ord(char) <= 591:
            return True

        if 7680 <= ord(char) <= 7929:
            return True

        # Dollar/Pound sign
        if char in "$£":
            return True

    @classmethod
    def contain_char_shared(cls, char):
        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#%^&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (<256)
        if char in "¡©«®°±²´·»":
            return True

        # Punctuations (>256)
        if ord(char) in [
            698,
            8216,
            8217,
            8221,
            8222,
            8225,
            8226,
            8240,
            8364,
            8369,
            8470,
            8482,
            8592,
            8593,
            8594,
            8595,
            8901,
        ]:
            return True

        # Full-width characters corresponding to basic half-width characters
        if ord("！") <= ord(char) <= ord("～"):
            # [65281, 65374] are full-width characters corresponding to [33, 126]
            # https://en.wikipedia.org/wiki/Halfwidth_and_Fullwidth_Forms_(Unicode_block)
            return True

        return False
