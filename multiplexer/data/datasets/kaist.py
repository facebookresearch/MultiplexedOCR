import random
import xml.etree.ElementTree as ET

import numpy as np

from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

from ...utils.char_map_arabic import ArabicCharMap
from ...utils.char_map_hangul import HangulCharMap
from ...utils.char_map_latin import LatinCharMap
from .icdar import IcdarDataset


class KAISTDataset(IcdarDataset):
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
        super(KAISTDataset, self).__init__(
            name,
            use_charann,
            imgs_dir,
            gts_dir,
            transforms,
            ignore_difficult,
            expected_imgs_in_dir,
        )

        self.image_lists = self.get_image_list(num_samples)

    def get_image_list(self, num_samples=0):
        if num_samples > 0:
            assert num_samples <= len(self.image_lists), "Number of samples exceeds max samples!"
            return random.sample(self.image_lists, num_samples)
        else:
            return self.image_lists

    def load_gt_from_txt(self, gt_path):
        words, boxes, char_boxes, segmentations, labels, languages = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # Parse from xml file
        dirname, fname = os.path.split(gt_path)
        xml_fname = os.path.join(dirname, fname[3:-4] + ".xml")
        try:
            with open(xml_fname, "rb") as buffer:
                root = ET.parse(buffer).getroot()
        except ET.ParseError as e:
            return {"exception": str(e)}

        # Extract gt data from xml
        lines = []
        for w in root.findall("./image/words/word"):
            word = ""
            for c in w.findall("character"):
                word += c.attrib["char"]

            line = w.attrib
            line["word"] = word
            lines.append(line)

        for line in lines:
            try:
                rect, language, word = self.line2boxes(line)
            except Exception:
                raise Exception("Error parsing gt file {}".format(gt_path))

            # if word == "###" and not self.ignore_difficult:
            #     continue
            if word == "###":
                word = ""
                assert language == "none"

            min_x = min(rect[::2])
            min_y = min(rect[1::2])
            max_x = max(rect[::2])
            max_y = max(rect[1::2])
            box = [min_x, min_y, max_x, max_y]
            segmentations.append([rect])
            boxes.append(box)
            words.append(word)
            labels.append(1)
            languages.append(language)

        num_boxes = len(boxes)
        if len(boxes) > 0:
            keep_boxes = np.zeros((num_boxes, 5))
            keep_boxes[:, :4] = np.array(boxes)
            keep_boxes[:, 4] = range(num_boxes)
            # the 5th column is the box label,
            # same as the 10th column of all char_boxes which belong to the box
            boxes = np.array(keep_boxes)

            if not self.use_charann:
                char_box = np.zeros((10,), dtype=np.float32)
                if len(char_boxes) == 0:
                    for _ in range(len(words)):
                        char_boxes.append([char_box])
        else:
            words.append("")
            boxes = np.zeros((1, 5), dtype=np.float32)
            char_boxes = [[np.zeros((10,), dtype=np.float32)]]
            segmentations = [[np.zeros((8,), dtype=np.float32)]]
            labels = [1]
            languages.append("none")

        return {
            "words": words,
            "boxes": boxes,
            "char_boxes": char_boxes,
            "segmentations": segmentations,
            "labels": labels,
            "languages": languages,
        }

    def line2boxes(self, line):
        word = line["word"]
        x = int(line["x"])
        y = int(line["y"])
        w = int(line["width"])
        h = int(line["height"])

        quadrilateral = [x, y, x + w, y, x + w, y + h, x, y + h]

        if word == "###":
            language = "none"
        else:
            # Auto-infer the gt language
            verified = False
            for i in range(len(word)):
                if HangulCharMap.contain_char_exclusive(word[i]):
                    language = "ko"
                    verified = True
                    break

            if not verified:
                for i in range(len(word)):
                    if LatinCharMap.contain_char_exclusive(word[i]):
                        language = "la"
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
                            "[Warning][KAIST] Auto-corrected word "
                            "{} (ords: {}) from Korean/Latin to Any."
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
