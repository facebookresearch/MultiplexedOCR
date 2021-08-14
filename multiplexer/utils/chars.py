# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from multiplexer.utils.languages import LANGUAGE_COMBO, lang_code_to_char_map_class


def char2num_en_num(char):
    if char in "0123456789":
        num = ord(char) - ord("0") + 1
    elif char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
        num = ord(char.lower()) - ord("a") + 11
    else:
        num = 0
    return num


def char2num(
    char,
    encode_language="en_num",
    gt_language=None,
    max_char_num=-1,
    char_map_class=None,
):
    """
    Arguments:
        char_map_class: the char_map_class corresponding to the encode_language
    """
    if encode_language in lang_code_to_char_map_class:
        confirmed_by_gt = False
        if encode_language == gt_language:
            # we know from ground truth that this char belongs to this language
            # a typical example is kanji in Japanese vs Chinese characters
            # where we need to rely on gt to judge which one the character belongs to
            confirmed_by_gt = True
        elif encode_language in LANGUAGE_COMBO:
            # unified character sets
            if gt_language in LANGUAGE_COMBO[encode_language]:
                confirmed_by_gt = True
        else:
            # When encode_language is 'any', any characters should be supported
            # thus disable the read_only mode
            confirmed_by_gt = encode_language == "any"

        read_only = not confirmed_by_gt

        assert (
            char_map_class is not None
        ), "Please specify the char_map_class corresponding to the encode_language!"

        return char_map_class.char2num(
            char,
            read_only=read_only,
            confirmed_by_gt=confirmed_by_gt,
            max_char_num=max_char_num,
        )

    assert encode_language == "en_num", "Unknown language: {}".format(encode_language)

    return char2num_en_num(char)


def num2char(num, language="en_num"):
    if language in lang_code_to_char_map_class:
        return lang_code_to_char_map_class[language].num2char(num)

    assert language == "en_num", "Unknown language: {}".format(language)
    chars = "_0123456789abcdefghijklmnopqrstuvwxyz"
    char = chars[num]

    return char
