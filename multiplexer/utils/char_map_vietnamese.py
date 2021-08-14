# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

from .char_map import CharMap

logger = logging.getLogger(__name__)


class VietnameseCharMap(CharMap):
    MAX_CHAR_NUM = 258

    @classmethod
    def contain_char_exclusive(cls, char):
        if char in ["F", "f", "J", "j", "W", "w", "Z", "z"]:
            return False
        if "A" <= char <= "Z" or "a" <= char <= "z":
            return True

        if char in "Đđ":
            return True

        # Không / Ngang
        if char in "ĂăÂâÊêÔôƠơƯư":
            return True
        # Huyền
        if char in "ÀàẰằẦầÈèỀềÌìÒòỒồỜờÙùỪừỲỳ":
            return True
        # Sắc
        if char in "ÁáẮắẤấÉéẾếÍíÓóỐốỚớÚúỨứÝý":
            return True
        # Hỏi
        if char in "ẢảẲẳẨẩẺẻỂểỈỉỎỏỔổỞởỦủỬửỶỷ":
            return True
        # Ngã
        if char in "ÃãẴẵẪẫẼẽỄễĨĩÕõỖỗỠỡŨũỮữỸỹ":
            return True
        # Nặng
        if char in "ẠạẶặẬậẸẹỆệỊịỌọỘộỢợỤụỰựỴỵ":
            return True

        return False

    @classmethod
    def contain_char_shared(cls, char):
        if char in ["F", "f", "J", "j", "W", "w", "Z", "z"]:
            return True

        # Digits
        if "0" <= char <= "9":
            return True

        # Punctuations (keyboard order)
        if char in "~!@#^&*()_+`-={{}}|[]\\:\";'<>?,./":
            return True

        # Punctuations (not on keyboard, within 256)
        if char in "£°":
            return True

        return False
