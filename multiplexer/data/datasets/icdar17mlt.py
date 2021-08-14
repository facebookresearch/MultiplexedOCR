# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import random

import numpy as np

from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

from ...utils.char_map_arabic import ArabicCharMap
from ...utils.languages import lang_code_to_char_map_class, name_to_code
from .icdar import IcdarDataset

logger = logging.getLogger(__name__)

# import torch
# from d2ocr.structures.bounding_box import BoxList
# from d2ocr.structures.segmentation_mask import (
#     SegmentationCharMask,
#     SegmentationMask,
# )
# from PIL import Image, ImageDraw


class Icdar17MLTDataset(IcdarDataset):
    def __init__(
        self,
        name,
        use_charann,
        imgs_dir,
        gts_dir,
        transforms=None,
        ignore_difficult=False,
        expected_imgs_in_dir=0,
        split="val",
        language="",
    ):
        super(IcdarDataset, self).__init__(name=name, transforms=transforms)
        self.use_charann = use_charann
        self.imgs_dir = imgs_dir
        self.language = language
        self.language_list = ["ar", "en", "fr", "zh", "de", "ko", "ja", "it", "bn"]
        self.image_lists = self.get_image_list(split, expected_imgs_in_dir)

        self.gts_dir = gts_dir
        self.min_proposal_size = 2
        self.char_classes = "_0123456789abcdefghijklmnopqrstuvwxyz"
        self.vis = False
        self.ignore_difficult = ignore_difficult
        if self.ignore_difficult and self.gts_dir is not None and "train" in self.gts_dir:
            self.image_lists = self.filter_image_lists()

    def filter_image_lists(self):
        new_image_lists = []
        for img_path in self.image_lists:
            has_positive = False
            im_name = os.path.basename(img_path)
            gt_path = os.path.join(self.gts_dir, im_name + ".txt")
            if not os.path.isfile(gt_path):
                gt_path = os.path.join(self.gts_dir, "gt_" + im_name.split(".")[0] + ".txt")
            lines = open(gt_path, "r").readlines()
            for line in lines:
                strs, loc, _language = self.line2boxes(line)
                word = strs[0]
                if word == "###":
                    continue
                else:
                    has_positive = True
            if has_positive:
                new_image_lists.append(img_path)
        return new_image_lists

    def get_image_list(self, split, expected_imgs_in_dir):
        image_names = os.listdir(self.imgs_dir)
        num_imgs = len(image_names)
        if expected_imgs_in_dir > 0 and expected_imgs_in_dir != num_imgs:
            msg = "[Warning] Expected {} images in dir {}, but found {}.".format(
                expected_imgs_in_dir, self.imgs_dir, num_imgs
            )
            logger.info(msg)
        else:
            logger.info("Total #images in the dir {}: {}".format(self.imgs_dir, num_imgs))

        output_image_list = []

        if self.language == "":
            # If language is not specified
            return [os.path.join(self.imgs_dir, img) for img in image_names]
        elif self.language == "random50":
            languages = []
        else:
            # only one language
            languages = [self.language]

        if self.language == "random50":
            assert split == "val", "Not implemented"
            output_image_list = random.sample(image_names, 50)
        else:
            # If language is specified
            if split == "train":
                num_images_per_language = 800
                assert len(image_names) == num_images_per_language * len(self.language_list)
                for image_name in image_names:
                    # e.g., img_3200.jpg -> img_3200
                    name = image_name.split(".")[-2]
                    # e.g., img_3200 -> 3200
                    id = int(name.split("_")[-1])
                    # e.g., 3200 -> 3, 3201 -> 4
                    lang_id = (id - 1) // num_images_per_language
                    # e.g., language_list[4] == "de"
                    if self.language_list[lang_id] in languages:
                        output_image_list.append(image_name)
            elif split == "val":
                raise NotImplementedError
            elif split == "test":
                raise NotImplementedError
            else:
                raise ValueError("Unknown split name: {}".format(split))

        logger.info(
            "Total number of images for language {}: {}".format(
                self.language, len(output_image_list)
            )
        )

        # add the path info
        output_image_list = [os.path.join(self.imgs_dir, img) for img in output_image_list]

        return output_image_list

    def load_gt_from_txt(self, gt_path):
        words, boxes, char_boxes, segmentations, labels, languages, rotated_boxes = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        lines = open(gt_path).readlines()
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
            rotated_box = self.polygon_to_rotated_box(rect)
            segmentations.append([rect])
            boxes.append(box)
            words.append(word)
            labels.append(1)
            languages.append(language)
            rotated_boxes.append(rotated_box)

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
            rotated_boxes = np.zeros((1, 5), dtype=np.float32)

        return {
            "words": words,
            "boxes": boxes,
            "char_boxes": char_boxes,
            "segmentations": segmentations,
            "labels": labels,
            "languages": languages,
            "rotated_boxes": rotated_boxes,
        }

    def line2boxes(self, line):
        parts = line.strip().split(",")
        if "\xef\xbb\xbf" in parts[0]:
            parts[0] = parts[0][3:]
            assert 0 == 1, "special character 1 found!"
        if "\ufeff" in parts[0]:
            parts[0] = parts[0].replace("\ufeff", "")
            assert 0 == 1, "special character 2 found!"

        assert len(parts) >= 10, "[Error][MLT17] line = {}, parts = {}".format(line, parts)

        if len(parts) > 10:
            word = ",".join(parts[9:])
            if random.random() < 0.02:
                # log every 50
                print("[Warning][MLT17] line = {}, parts = {}, word = {}".format(line, parts, word))
        else:
            word = parts[9]

        quadrilateral = [int(float(x)) for x in parts[:8]]
        language_name = parts[8]

        if word == "###":
            language = "none"
        else:
            language = name_to_code(language_name)  # e.g., Latin => la
            # Verify if the ground truth language is mis-labeled
            if language in lang_code_to_char_map_class:
                char_map_class = lang_code_to_char_map_class[language]
                verified = False
                for i in range(len(word)):
                    # Note: Chinese/Kanji is considered exclusive to
                    # both Chinese and Japanese
                    if char_map_class.contain_char_exclusive(word[i]):
                        verified = True
                        break

                # Trust language annotation for empty string
                if word == "":
                    verified = True

                if not verified:
                    language = "any"
                    ords = ""
                    for i in range(len(word)):
                        ords += "{},".format(ord(word[i]))
                    if random.random() < 0.05:
                        # log every 20
                        print(
                            (
                                "[Warning][MLT17] Auto-corrected word "
                                "{} (ords: {}) from {} to Any."
                            ).format(word, ords, language_name)
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
