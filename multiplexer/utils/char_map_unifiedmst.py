from d2ocr.utils.char_map import CharMap
from d2ocr.utils.languages import MalayalamCharMap, SinhalaCharMap, TamilCharMap


class UnifiedMSTCharMap(CharMap):
    MAX_CHAR_NUM = 298

    @classmethod
    def contain_char_exclusive(cls, char):
        return (
            MalayalamCharMap.contain_char_exclusive(char)
            or SinhalaCharMap.contain_char_exclusive(char)
            or TamilCharMap.contain_char_exclusive(char)
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
        if char in "~!@#$%^&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        return False
