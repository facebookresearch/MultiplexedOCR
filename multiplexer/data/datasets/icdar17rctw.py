import random

from ...utils.char_map_arabic import ArabicCharMap
from ...utils.char_map_chinese import ChineseCharMap
from .icdar15 import Icdar15Dataset


class Icdar17RCTWDataset(Icdar15Dataset):
    def __init__(
        self,
        name,
        use_charann,
        imgs_dir,
        gts_dir,
        transforms=None,
        ignore_difficult=False,
        expected_imgs_in_dir=0,
        num_samples=0,
    ):
        """
        num_samples: number of (the subset of) samples.
            When it's <= 0 we use all the samples.
        """
        super(Icdar17RCTWDataset, self).__init__(
            name,
            use_charann,
            imgs_dir,
            gts_dir,
            transforms,
            ignore_difficult,
            expected_imgs_in_dir,
        )

        self.image_lists = self.get_image_list(num_samples)

    def line2boxes(self, line):
        parts = line.strip().split(",")
        if "\xef\xbb\xbf" in parts[0]:
            # UTF-8 Byte order mark
            parts[0] = parts[0][3:]
        if "\ufeff" in parts[0]:
            # UTF-8 Byte order mark
            parts[0] = parts[0].replace("\ufeff", "")

        assert len(parts) >= 10, "[Error][RCTW17] line = {}, parts = {}".format(line, parts)

        if len(parts) > 10:
            word = ",".join(parts[9:])
            if random.random() < 0.01:
                # log every 100
                print(
                    "[Warning][RCTW17] line = {}, parts = {}, word = {}".format(line, parts, word)
                )
        else:
            word = parts[9]
        # unlike ICDAR15, transcription is wrapped in double quotes
        word = word.strip('"')

        # NOTE: parts[8] is 'difficulty', which is ignored for now

        quadrilateral = [int(float(x)) for x in parts[:8]]

        if word == "###":
            language = "none"
        else:
            language = "zh"
            # Verify if the ground truth language is mis-labeled
            verified = False
            for i in range(len(word)):
                if ChineseCharMap.contain_char_exclusive(word[i]):
                    verified = True
                    break

            if not verified:
                language = "any"

                if random.random() < 0.01:
                    # log every 100
                    ords = ""
                    for i in range(len(word)):
                        ords += "{},".format(ord(word[i]))
                    print(
                        (
                            "[Warning][RCTW17] Auto-corrected word "
                            "{} (ords: {}) from Chinese to Any."
                        ).format(word, ords)
                    )
            # Check right-to-left words
            is_right_to_left = False
            for i in range(len(word)):
                if ArabicCharMap.contain_char_exclusive(word[i]):
                    is_right_to_left = True
                    break
            if is_right_to_left:
                word = word[::-1]

        return quadrilateral, language, word
