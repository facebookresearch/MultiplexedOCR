# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import random
import time
import zipfile

import numpy as np
import torch
from PIL import Image

from multiplexer.structures.bounding_box import BoxList
from multiplexer.structures.language_list import LanguageList
from multiplexer.structures.segmentation_mask import SegmentationCharMask, SegmentationMask
from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

from ...utils.char_map_arabic import ArabicCharMap
from ...utils.languages import code_to_name, lang_code_to_char_map_class, name_to_code
from .polygon_ocr_dataset import PolygonOcrDataset

logger = logging.getLogger(__name__)


class SynthTextMLTZipDataset(PolygonOcrDataset):
    def __init__(
        self,
        name,
        split,
        imgs_dir,
        gts_dir,
        transforms=None,
        ignore_difficult=False,
        use_char_ann=False,
        language="",
    ):
        super(SynthTextMLTZipDataset, self).__init__(name=name, transforms=transforms)
        assert split == "train" or split == "test", "Unknown split: {}".format(split)
        self.split = split
        self.imgs_dir = imgs_dir
        self.gts_dir = gts_dir
        self.ignore_difficult = ignore_difficult  # dummy
        self.use_char_ann = use_char_ann

        self.language = language

        # img_zip = zipfile.ZipFile('/mnt/vol/gfsai-oregon/aml/ocr/datasets/MLT19_Synthetic/Latin.zip')
        # print (img_zip.namelist())

        self.language_list = ["ar", "bn", "hi", "ja", "ko", "la", "zh"]
        self.image_lists = self.get_image_list(split)

        self.filter_list = []

    def get_image_list(self, split):
        # Number of images for different languages:
        # Arabic 51099
        # Bangla 48490
        # Hindi 12164 (originally 32540, but 20376 are filtered due to bad annotations)
        # Japanese 20510
        # Korean 40432
        # Latin 49766
        # Chinese 30138

        if self.language == "" or self.language == "random50":
            languages = ["ar", "bn", "hi", "ja", "ko", "la", "zh"]
        else:
            # only one language
            languages = [self.language]

        output_image_list = []
        for language in languages:
            lang_name = code_to_name(language)

            keep_list_file = os.path.join(self.gts_dir, f"{lang_name}_keep.list")

            if os.path.exists(keep_list_file):
                # Online filtering is too slow for SynthMLT-Hindi, so we use the pre-generated keep-list.
                with open(keep_list_file, "r") as f:
                    output_image_list += f.readlines()
            else:
                assert language != "hi", "SynthMLT-Hindi needs to be filtered before being used."
                with open(os.path.join(self.imgs_dir, "{}.zip".format(lang_name)), "rb") as buffer:
                    img_zip = zipfile.ZipFile(buffer)

                # Only add the image files (filtering out folders and txt)
                output_image_list += [
                    name for name in img_zip.namelist() if str.split(name, ".")[-1] == "jpg"
                ]

        if self.language == "random50":
            assert split == "test", "Not implemented"
            # randomly sample 50 images, for testing purpose only
            output_image_list = random.sample(output_image_list, 50)

        logger.info(
            "Total number of images for language {}: {}".format(
                self.language, len(output_image_list)
            )
        )

        return output_image_list

    def __getitem__(self, item, retry=0):
        if item in self.filter_list:
            if random.random() < 0.01:
                # log every 100
                print("[SynthMLT] data {} already filtered (retry {})".format(item, retry))
            if retry < 1000:
                return self.__getitem__(
                    item=random.randint(0, len(self.image_lists) - 1), retry=retry + 1
                )
            else:
                raise

        img_name = os.path.join(self.imgs_dir, self.image_lists[item])
        lang_name, im_name = str.split(self.image_lists[item], "/")

        if os.path.isfile(img_name):
            with open(img_name, "rb") as img_buffer:
                img = Image.open(img_buffer).convert("RGB")
        else:
            try:
                img_zip_name = os.path.join(self.imgs_dir, "{}.zip".format(lang_name))
                with open(img_zip_name, "rb") as buffer:
                    img_zip = zipfile.ZipFile(buffer)
            except FileNotFoundError:
                print("[Warning] File {} not found for retry {}".format(img_zip_name, retry))
                if retry < 12:
                    print("[Warning] Sleeping for {} seconds ...".format(2 ** retry))
                    time.sleep(2 ** retry)
                    return self.__getitem__(
                        item=random.randint(0, len(self.image_lists) - 1),
                        retry=retry + 1,
                    )
                else:
                    raise
            with open(self.image_lists[item]) as buffer:
                with img_zip.open(buffer) as img_buffer:
                    img = Image.open(img_buffer).convert("RGB")
            print("[Info] Saving {}".format(self.image_lists[item]))
            with open(img_name, "wb") as buffer:
                img.save(buffer)

        # load gt
        boxes = []
        segmentations = []
        words = []
        languages = []
        char_boxes = []

        gt_name = lang_name + "/" + str.split(im_name, ".")[0] + ".txt"  # .jpg -> .txt
        gt_path = os.path.join(
            self.gts_dir, "{}_gt".format(lang_name), str.split(im_name, ".")[0] + ".txt"
        )
        if os.path.isfile(gt_path):
            lines = open(gt_path).readlines()
        else:
            lines = []
            with open(os.path.join(self.gts_dir, "{}_gt.zip".format(lang_name)), "rb") as buffer:
                gt_zip = zipfile.ZipFile(buffer)

            try:
                with gt_zip.open(gt_name) as f:
                    for line in f:
                        lines.append(line.decode())  # bytes -> str
            except KeyError:
                print("[Warning] Cannot open {}".format(gt_name))
                if retry < 5:
                    return self.__getitem__(
                        item=random.randint(0, len(self.image_lists) - 1),
                        retry=retry + 1,
                    )
                else:
                    raise
            with open(gt_path, "w") as gt_file:
                for line in lines:
                    gt_file.write(line)

        for line in lines:
            try:
                rect, language, word = self.line2boxes(line)
            except Exception:
                raise Exception("Error parsing gt file {}".format(gt_path))

            if (lang_name == "Hindi" and language == "any") or " " in word:
                self.filter_list.append(item)
                if random.random() < 0.05:
                    # log every 20
                    print(
                        "[SynthMLT{}][#filtered: {}] Filtered {} with bad annotation `{}`.".format(
                            lang_name, len(self.filter_list), gt_path, word
                        )
                    )
                return self.__getitem__(
                    item=random.randint(0, len(self.image_lists) - 1),
                    retry=retry + 1,
                )

            min_x = min(rect[::2])
            min_y = min(rect[1::2])
            max_x = max(rect[::2])
            max_y = max(rect[1::2])
            segmentations.append([rect])
            boxes.append([min_x, min_y, max_x, max_y])
            words.append(word)
            languages.append(language)

            # Visualize bounding box
            # draw.rectangle((min_x, min_y, max_x, max_y), None, "#0f0")

        if len(boxes) > 0:
            boxes = np.array(boxes)
            if not self.use_char_ann:
                char_box = np.zeros((10,), dtype=np.float32)
                if len(char_boxes) == 0:
                    for _ in range(len(words)):
                        char_boxes.append([char_box])
        else:
            print("[Warning] len(boxes) = {} for retry {}".format(len(boxes), retry))
            if retry < 3:
                return self.__getitem__(
                    item=random.randint(0, len(self.image_lists) - 1), retry=retry + 1
                )

            words.append("")
            boxes = np.zeros((1, 4), dtype=np.float32)
            char_boxes = [[np.zeros((10,), dtype=np.float32)]]
            segmentations = [[np.zeros((8,), dtype=np.float32)]]
            languages.append("None")

        target = BoxList(boxes, img.size, mode="xyxy", use_char_ann=self.use_char_ann)

        labels = torch.ones(len(boxes))
        target.add_field("labels", labels)

        masks = SegmentationMask(segmentations, img.size)
        target.add_field("masks", masks)

        char_masks = SegmentationCharMask(
            chars_boxes=char_boxes,
            words=words,
            use_char_ann=self.use_char_ann,
            size=img.size,
        )
        target.add_field("char_masks", char_masks)

        language_list = LanguageList(languages)
        target.add_field("languages", language_list)

        # img = Image.alpha_composite(img, txt)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, img_name

    def __len__(self):
        return len(self.image_lists)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        pass

    def line2boxes(self, line):
        parts = line.strip().split(",")
        if "\xef\xbb\xbf" in parts[0]:
            parts[0] = parts[0][3:]
            assert 0 == 1, "special character 1 found!"
        if "\ufeff" in parts[0]:
            parts[0] = parts[0].replace("\ufeff", "")
            assert 0 == 1, "special character 2 found!"

        assert len(parts) >= 10, "[Error][SynthMLT] line = {}, parts = {}".format(line, parts)

        if len(parts) > 10:
            word = ",".join(parts[9:])
            if random.random() < 0.02:
                # log every 50
                print(
                    "[Warning][SynthMLT] line = {}, parts = {}, word = {}".format(line, parts, word)
                )
        else:
            word = parts[9]

        quadrilateral = [int(float(x)) for x in parts[:8]]
        language_name = parts[8]

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

            if not verified:
                language = "any"
                ords = ""
                for i in range(len(word)):
                    ords += "{},".format(ord(word[i]))
                if random.random() < 0.05:
                    # log every 20
                    print(
                        (
                            "[Warning][SynthMLT] Auto-corrected word "
                            "{} (ords: {}) from {} to Any."
                        ).format(word, ords, language_name)
                    )

        is_right_to_left = False
        for i in range(len(word)):
            if ArabicCharMap.contain_char_exclusive(word[i]):
                is_right_to_left = True
                break
        if is_right_to_left:
            word = word[::-1]
        return quadrilateral, language, word
