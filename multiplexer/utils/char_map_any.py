import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class AnyCharMap(CharMap):
    MAX_CHAR_NUM = 10998

    @classmethod
    def contain_char(cls, char):
        return True
