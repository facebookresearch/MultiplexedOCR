# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging
import random

import numpy as np

from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

from ...utils.char_map_arabic import ArabicCharMap
from ...utils.languages import lang_code_to_char_map_class, name_to_code
from .icdar import IcdarDataset

logger = logging.getLogger(__name__)


class Icdar19ArTDataset(IcdarDataset):
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
        total_text_train_dir=None,
        total_text_test_dir=None,
        total_text_map_file=None,
        variation="default",
    ):
        super(IcdarDataset, self).__init__(name=name, transforms=transforms)
        self.use_charann = use_charann
        self.imgs_dir = imgs_dir
        self.language = language
        self.total_text_train_dir = total_text_train_dir
        self.total_text_test_dir = total_text_test_dir
        self.total_text_map_file = total_text_map_file
        self.variation = variation
        self.image_lists = self.get_image_list(split, expected_imgs_in_dir)

        self.gts_dir = gts_dir
        self.json_annotations = None
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
            gt_path = os.path.join(self.gts_dir, "gt_" + im_name.split(".")[0] + ".txt")
            if not os.path.isfile(gt_path):
                # Trust it if it hasn't been extracted from the json
                new_image_lists.append(img_path)
                continue

            lines = open(gt_path, "r").readlines()
            for word in lines[1::2]:
                if word == "###":
                    continue
                else:
                    has_positive = True
                    break
            if has_positive:
                new_image_lists.append(img_path)
        return new_image_lists

    def get_total_text_train_ids(self):
        assert self.total_text_train_dir is not None
        names = os.listdir(self.total_text_train_dir)
        num_gt = len(names)
        assert num_gt == 1255, "Expected {} files in dir {}, but found {}.".format(
            1255, self.total_text_train_dir, num_gt
        )
        # img5393.jpg => 5393
        return [int(name.split(".")[0][3:]) for name in names]

    def get_total_text_test_ids(self):
        assert self.total_text_test_dir is not None
        names = os.listdir(self.total_text_test_dir)
        num_gt = len(names)
        assert num_gt == 300, "Expected {} files in dir {}, but found {}.".format(
            300, self.total_text_test_dir, num_gt
        )
        # img5393.jpg => 5393
        return [int(name.split(".")[0][3:]) for name in names]

    def get_total_text_map(self):
        with open(self.total_text_map_file, "r") as f:
            lines = f.readlines()
        art_to_total_text_map = {}
        for line in lines:
            parts = line.split(" ")
            # poly_gt_img1293.txt => 1293
            total_text_id = int(parts[0].split(".")[0][11:])
            # gt_5425.txt => 5425
            art_id = int(parts[1].split(".")[0][3:])
            art_to_total_text_map[art_id] = total_text_id
        assert len(art_to_total_text_map) == 1555
        return art_to_total_text_map

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

        if self.variation == "with_total_text_test":
            print("[Warning] Not filtering total text test set in ArT19")
            return [os.path.join(self.imgs_dir, img) for img in image_names]
        elif self.variation == "default":
            output_image_list = []
            total_text_test_ids = self.get_total_text_test_ids()
            total_text_map = self.get_total_text_map()
            count = 0
            for img in image_names:
                # "img101.jpg" => 101
                img_id = int(img.split(".")[0][3:])
                if img_id in total_text_map.keys():
                    if total_text_map[img_id] in total_text_test_ids:
                        logger.warning(
                            "Filtered ArT-{} since it's TotalTextTest-{}".format(
                                img_id, total_text_map[img_id]
                            )
                        )
                        count += 1
                        continue

                output_image_list.append(img)

            assert count == 300, "Expected to filter 300 imgs from TotalTextTest, found {}".format(
                count
            )
            assert len(output_image_list) == expected_imgs_in_dir - 300
            logger.info("Filter 300 imgs in ArT19 that belong to TotalTextTest.")
        elif self.variation == "total_text_train_only":
            total_text_train_ids = self.get_total_text_train_ids()
            total_text_map = self.get_total_text_map()
            output_image_list = []

            for img in image_names:
                # "img101.jpg" => 101
                img_id = int(img.split(".")[0][3:])
                if img_id in total_text_map.keys():
                    if total_text_map[img_id] in total_text_train_ids:
                        output_image_list.append(img)
                        if random.random() < 0.01:
                            # log every 100
                            logger.info(
                                "[ArT19] Using ArT-{} (TotalTextTrain-{})".format(
                                    img_id, total_text_map[img_id]
                                )
                            )
                        continue

            assert (
                len(output_image_list) == 1255
            ), "Expected to retain 1255 imgs from TotalTextTrain, found {}".format(
                len(output_image_list)
            )
            logger.info("Using 1255 imgs in ArT19 that belong to TotalTextTrain.")
        elif self.variation == "random50":
            # Test only
            output_image_list = random.sample(image_names, 50)
        else:
            raise Exception("Unknown variation choice: {}".format(self.variation))

        logger.info(
            "Total number of images for ArT variation {}: {}".format(
                self.variation, len(output_image_list)
            )
        )

        # add the path info
        output_image_list = [os.path.join(self.imgs_dir, img) for img in output_image_list]

        return output_image_list

    def load_gt_from_txt(self, gt_path):
        words, boxes, char_boxes, segmentations, labels, languages = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        if os.path.isfile(gt_path):
            lines = open(gt_path).readlines()
        else:
            print("[Warning] {} doesn't exist, creating it from json".format(gt_path))
            lines = []
            if self.json_annotations is None:
                gt_json = os.path.join(self.gts_dir, "../train_labels.json")
                with open(gt_json, "r") as f:
                    self.json_annotations = json.load(f)
            # e.g., gt_gt_1985.txt --> gt_1985
            im_name = os.path.basename(gt_path).split(".")[0][3:]
            assert im_name in self.json_annotations, "Unable to locate {} in gt json".format(
                im_name
            )
            annotation_per_image = self.json_annotations[im_name]

            with open(gt_path, "w") as gt_file:
                for ann in annotation_per_image:
                    word = ann["transcription"]
                    points = ann["points"]
                    language = ann["language"]
                    # serialize annotation to one line representation

                    # Line 1: {bboxes},language
                    line = ""
                    for point in points:
                        line += "{},{},".format(point[0], point[1])
                    line += language
                    lines.append(line)
                    gt_file.write(line + "\n")

                    # Line 2: word (which might contain ',')
                    lines.append(word)
                    gt_file.write(word + "\n")

        for (line1, line2) in zip(lines[0::2], lines[1::2]):
            try:
                polygon, language, word = self.line2boxes(line1, line2)
            except ValueError:
                raise ValueError(
                    "ValueError decoding lines: \n{}\n{} in {}".format(line1, line2, gt_path)
                )

            if len(polygon) < 6:
                if "gt_2805" in gt_path:
                    logger.info(f"Skipped known non-polygon in {gt_path}: {polygon}")
                    continue
                else:
                    assert len(polygon) < 6, f"Non-polygon {polygon} found in {gt_path}"

            # if word == "###" and not self.ignore_difficult:
            #     continue
            if word == "###":
                word = ""
                assert language == "none"

            min_x = min(polygon[::2])
            min_y = min(polygon[1::2])
            max_x = max(polygon[::2])
            max_y = max(polygon[1::2])
            box = [min_x, min_y, max_x, max_y]
            segmentations.append([polygon])
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

    def line2boxes(self, line1, line2):
        parts = line1.strip().split(",")
        polygon = [int(x) for x in parts[:-1]]
        language_name = parts[-1]
        word = line2.strip()

        if (
            len(word) == 3
            and ord(word[0]) == 65283
            and ord(word[1]) == 65283
            and ord(word[2]) == 65283
        ):
            # correct full-width ignore label to "###"
            print(("[Warning] Auto-corrected word {} to ###").format(word))
            word = "###"

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
                    if random.random() < 0.1:
                        # log every 10
                        print(
                            (
                                "[Warning][ArT19] Auto-corrected word "
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

        return polygon, language, word
