from multiplexer.utils.char_map import CharMap
from multiplexer.utils.char_map_kannada import KannadaCharMap
from multiplexer.utils.char_map_telugu import TeluguCharMap


class UnifiedKTCharMap(CharMap):
    MAX_CHAR_NUM = 218

    @classmethod
    def contain_char_exclusive(cls, char):
        return KannadaCharMap.contain_char_exclusive(char) or TeluguCharMap.contain_char_exclusive(
            char
        )

    @classmethod
    def contain_char_shared(cls, char):
        # Zero width non-joiner/joiner
        if ord(char) in [8203, 8204, 8205]:
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#&*()_+-=|\\:\";'<>?,./":
            return True

        return False
