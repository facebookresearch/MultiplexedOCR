from .char_map import CharMap


class Any4CharMap(CharMap):
    MAX_CHAR_NUM = 10998

    @classmethod
    def contain_char(cls, char):
        return True
