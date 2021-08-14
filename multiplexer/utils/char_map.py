# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json
import logging
import random
import threading
import time

from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

logger = logging.getLogger(__name__)


class CharMap(object):
    char_map = {}
    chars = []
    count = 0

    MAX_CHAR_NUM = 0
    substitution_map = {}
    update_lock = threading.Lock()

    @classmethod
    def char2num(cls, char, read_only=True, confirmed_by_gt=False, max_char_num=-1):
        # When read_only is True, we don't alter the alphabet dict if the character
        # is not found
        # When confirmed_by_gt is True, we know that the character is labeled as
        # this language by ground truth
        # max_char_num specifies the max character number the model can support
        cls.load_default()
        char = cls.normalize_char(char)
        if cls.contain_char(char):
            if max_char_num == -1:
                max_char_num = cls.MAX_CHAR_NUM

            if char not in cls.char_map:
                if read_only:
                    return 0

                if cls.count >= max_char_num:
                    logger.info(
                        "[{}] Max char num {} reached, not adding {} to charmap.".format(
                            cls.__name__, max_char_num, char
                        )
                    )
                    return 0

                with cls.update_lock:
                    cls.char_map[char] = cls.count + 1
                    cls.chars.append(char)
                    cls.count += 1
                    logger.info(
                        "[{}] Found new character {} and mapped it to {}".format(
                            cls.__name__, char, cls.count
                        )
                    )

                    cls.normalize()  # avoid multi-threading issues
                    is_valid = cls.valid()

                if not is_valid:
                    logger.warning("[{}] Char map contaminated, reloading".format(cls.__name__))
                    with cls.update_lock:
                        cls.count = 0
                        cls.char_map = {}
                        cls.chars = []

                    return cls.char2num(char, read_only, confirmed_by_gt)

                with cls.update_lock:
                    assert cls.count <= cls.MAX_CHAR_NUM, "[{}] Character count exceeds {}!".format(
                        cls.__name__, cls.MAX_CHAR_NUM
                    )

                    new_json = cls.get_new_json()

                    try:
                        with open(new_json, "w", encoding="utf8") as f:
                            json.dump(cls.char_map, f, ensure_ascii=False, indent=2)
                        logger.info(f"Saved to {new_json}")
                    except PermissionError:
                        print("[Warning] Permission error for {}, skipped saving".format(new_json))

            result = cls.char_map[char]
            if result < 0:
                print(
                    "[Warning][{}] Char map contaminated with negative values, reloading".format(
                        cls.__name__
                    )
                )
                with cls.update_lock:
                    cls.char_map = {}
                    cls.chars = []
                    cls.count = 0

                return cls.char2num(char, read_only, confirmed_by_gt)
            elif result > max_char_num:
                logger.info(
                    (
                        "[{}] Index for {} is {},"
                        " which exceeds the max char num {} supported by the model,"
                        " return 0 instead."
                    ).format(cls.__name__, char, result, max_char_num)
                )
                # 0 means the character is ignored
                return 0
            else:
                return result

        else:
            if confirmed_by_gt:
                if random.random() < 0.01:
                    # log every 100
                    logger.info(
                        "[Warning]"
                        + "'{}' (ord = {}) is labeled as {} in gt".format(
                            char, ord(char), cls.__name__
                        )
                        + " but not confirmed by contain_char()."
                    )
            return 0

    @classmethod
    def contain_char(cls, char):
        # contain_char is a rough check

        # normalize char
        char = cls.normalize_char(char)

        # the language exclusively contain these characters
        if cls.contain_char_exclusive(char):
            return True

        # shared characters
        return cls.contain_char_shared(char)

    @classmethod
    def contain_char_exclusive(cls, char):
        return False

    @classmethod
    def contain_char_shared(cls, char):
        return False

    @classmethod
    def get_new_json(cls):
        # write-only candidate new json that contains more characters
        basename = os.path.basename(cls.json_file)
        new_dir = os.path.join(os.path.dirname(cls.json_file), "new")
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        new_json = os.path.join(new_dir, basename)
        return new_json

    @classmethod
    def get_utf8_word(cls, word):
        try:
            new_word = word.encode("cp1252").decode("utf8")
            return new_word
        except (UnicodeEncodeError, UnicodeDecodeError):
            for sub in cls.substitution_map:
                if sub in word:
                    # print("Replacing sub {} with {} in {}".format(
                    #     sub, substitution_map[sub], word))
                    parts = word.split(sub)
                    new_parts = [cls.get_utf8_word(part) for part in parts]
                    return cls.substitution_map[sub].join(new_parts)
            return word

    @classmethod
    def init(cls, char_map_path=None):
        if not hasattr(cls, "json_file") and char_map_path is not None:
            # only init once
            # e.g. SymbolCharMap.__name__[:-7].lower() == "symbol"
            cls.json_file = os.path.join(char_map_path, cls.__name__[:-7].lower() + ".json")

    @classmethod
    def load_default(cls, retry=0):
        if len(cls.chars) == 0:
            cls.update_lock.acquire()
            assert hasattr(
                cls, "json_file"
            ), f"Please call init() with char_map_path for {cls.__name__} first"
            if os.path.isfile(cls.json_file):
                with open(cls.json_file) as f:
                    cls.char_map = json.load(f)

                cls.normalize()
                logger.info(
                    "[Info] Loaded char_map with {} characters from {}.".format(
                        cls.count, cls.json_file
                    )
                )

                if not cls.valid():
                    logger.warning("[{}] Char map contaminated, reloading".format(cls.__name__))

                    cls.char_map = {}
                    cls.chars = []
                    cls.count = 0

                    if retry < 6:
                        print(
                            "[{}][Warning] Sleeping for {} seconds ...".format(
                                cls.__name__, 2 ** retry
                            )
                        )
                        # release the lock before retry
                        cls.update_lock.release()
                        time.sleep(2 ** retry)
                        cls.load_default(retry=retry + 1)
                        return
                    else:
                        print(
                            (
                                "[{}][Warning] Max retries reached and char map is still invalid, "
                                "give up loading and start new char map"
                            ).format(cls.__name__)
                        )
                else:
                    assert cls.count <= cls.MAX_CHAR_NUM, "[{}] Character count exceeds {}!".format(
                        cls.__name__, cls.MAX_CHAR_NUM
                    )
            cls.update_lock.release()  # release the lock

    @classmethod
    def normalize(cls):
        cls.chars = [k for k, v in sorted(cls.char_map.items(), key=lambda x: x[1])]

        # verify 1-to-1 mapping
        for i in range(len(cls.chars)):
            if cls.char_map[cls.chars[i]] != i + 1:
                print(
                    ("[{}] Auto-correct map for {} from {} to {}").format(
                        cls.__name__, cls.chars[i], cls.char_map[cls.chars[i]], i + 1
                    )
                )
                cls.char_map[cls.chars[i]] = i + 1

        cls.count = len(cls.chars)

    @classmethod
    def normalize_char(cls, char):
        return char

        # # Quick filter
        # if ord(char) < ord("！"):
        #     return char

        # new_char = char
        # # Normalize full-width chars into half-width chars
        # # https://en.wikipedia.org/wiki/Halfwidth_and_Fullwidth_Forms_(Unicode_block)

        # if ord("！") <= ord(char) <= ord("～"):
        #     # Map [65281, 65374] to [33, 126]
        #     new_char = chr(ord(char) - ord("！") + ord("!"))
        # elif char == "￥":
        #     new_char = "¥"

        # if char != new_char:
        #     if random.random() < 0.001:
        #         # log every 1000
        #         logger.info(
        #             (
        #                 "[Info] Full-width char {} (ord: {})"
        #                 " is converted to half-width char {} (ord: {})"
        #             ).format(char, ord(char), new_char, ord(new_char))
        #         )

        # return new_char

    @classmethod
    def num2char(cls, num):
        cls.load_default()
        if num > 0 and num <= cls.count:
            return cls.chars[num - 1]
        else:
            return "_"  # CAUTION!

    @classmethod
    def random_string(cls, min_len=1, max_len=16, min_idx=-1, max_idx=-1):
        """Generate a random string of characters"""
        cls.load_default()
        length = random.randint(min_len, max_len)
        if min_idx < 0 or min_idx >= cls.count:
            min_idx = 0
        if max_idx < 0 or max_idx >= cls.count:
            max_idx = cls.count
        assert min_idx < max_idx
        return "".join(random.choices(cls.chars[min_idx:max_idx], k=length))

    @classmethod
    def valid(cls):
        for ch in cls.char_map:
            if not cls.contain_char(ch):
                raise Exception(
                    f"[{cls.__name__}] Invalid character {ch} found in the loaded charmap!"
                )
                # return False

        return True
