# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging

import numpy as np

from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

from ...utils.char_map_chinese import ChineseCharMap
from .icdar import IcdarDataset
from .icdar19art import Icdar19ArTDataset

logger = logging.getLogger(__name__)


class Icdar19LSVTDataset(Icdar19ArTDataset):
    def __init__(
        self,
        name,
        use_charann,
        imgs_dir,
        gts_dir,
        transforms=None,
        ignore_difficult=False,
        expected_imgs_in_dir=0,
    ):
        super(IcdarDataset, self).__init__(name=name, transforms=transforms)
        self.use_charann = use_charann
        self.image_lists = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)]
        num_imgs = len(self.image_lists)
        if expected_imgs_in_dir > 0 and expected_imgs_in_dir != num_imgs:
            msg = "[Warning] Expected {} images in dir {}, but found {}.".format(
                expected_imgs_in_dir, imgs_dir, num_imgs
            )
            logger.info(msg)
        else:
            logger.info("Total #images in the dir {}: {}".format(imgs_dir, num_imgs))

        self.gts_dir = gts_dir
        self.json_annotations = None
        self.min_proposal_size = 2
        self.char_classes = "_0123456789abcdefghijklmnopqrstuvwxyz"
        self.vis = False
        self.ignore_difficult = ignore_difficult
        if self.ignore_difficult and self.gts_dir is not None and "train" in self.gts_dir:
            self.image_lists = self.filter_image_lists()

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

        if os.path.isfile(gt_path):
            lines = open(gt_path).readlines()
        else:
            print("[Warning] {} doesn't exist, creating it from json".format(gt_path))
            lines = []
            if self.json_annotations is None:
                gt_json = os.path.join(self.gts_dir, "../train_full_labels.json")
                gt_json = os.path.normpath(gt_json)
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

                    # infer language is Chinese or Latin
                    # assign language name here, will change to code in line2boxes()
                    language = "Latin"
                    if word == "###":
                        assert ann["illegibility"]
                        language = "Any"
                    else:
                        for i in range(len(word)):
                            if ChineseCharMap.contain_char_exclusive(word[i]):
                                language = "Chinese"
                                break

                    # serialize annotation to two line representation
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
            rotated_box = self.polygon_to_rotated_box(polygon)
            segmentations.append([polygon])
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
