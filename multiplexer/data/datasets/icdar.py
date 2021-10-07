# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import random
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from multiplexer.structures.bounding_box import BoxList
from multiplexer.structures.language_list import LanguageList
from multiplexer.structures.rotated_box_list import RotatedBoxList
from multiplexer.structures.segmentation_mask import SegmentationCharMask, SegmentationMask
from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

from .polygon_ocr_dataset import PolygonOcrDataset

logger = logging.getLogger(__name__)


class IcdarDataset(PolygonOcrDataset):
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
                # charbbs = []
                strs, loc = self.line2boxes(line)
                word = strs[0]
                if word == "###":
                    continue
                else:
                    has_positive = True
            if has_positive:
                new_image_lists.append(img_path)
        return new_image_lists

    def get_item_name(self, item):
        return self.image_lists[item]

    def __getitem__(self, item, retry=0):
        im_name = os.path.basename(self.image_lists[item])
        try:
            img = self.load_image(path=self.image_lists[item])
        except Exception as e:
            file_name = self.image_lists[item]
            logger.warning(
                "[{}] Failed to load {} in retry {}, with exception {}".format(
                    self.name, file_name, retry, str(e)
                )
            )
            if retry < 40:
                logger.warning("[Warning] Sleeping for {:.0f} seconds ...".format(1.2 ** retry))
                time.sleep(1.2 ** retry)
                return self.__getitem__(
                    item=random.randint(0, len(self.image_lists) - 1), retry=retry + 1
                )
            else:
                raise e

        if self.gts_dir is not None:
            gt_path = os.path.join(self.gts_dir, "gt_" + im_name.split(".")[0] + ".txt")
            try:
                gt = self.load_gt_from_txt(gt_path)
            except FileNotFoundError as e:
                # You can throw other non-retriable exceptions in load_gt_from_txt
                gt["exception"] = str(e)

            if "exception" in gt:
                if retry < 20:
                    logger.warning(
                        (
                            f"Exception loading {gt_path} for retry {retry}: {gt['exception']}\n"
                            f"Sleeping for {1.5 ** retry} seconds ..."
                        )
                    )
                    time.sleep(1.5 ** retry)
                    return self.__getitem__(
                        item=random.randint(0, len(self.image_lists) - 1),
                        retry=retry + 1,
                    )
                else:
                    raise

            words = gt["words"]
            boxes = gt["boxes"]
            char_boxes = gt["char_boxes"]
            segmentations = gt["segmentations"]
            labels = gt["labels"]
            languages = gt["languages"]

            target = BoxList(boxes[:, :4], img.size, mode="xyxy", use_char_ann=self.use_charann)
            if self.ignore_difficult:
                labels = torch.from_numpy(np.array(labels))
            else:
                labels = torch.ones(len(boxes))
            target.add_field("labels", labels)

            masks = SegmentationMask(segmentations, img.size)
            target.add_field("masks", masks)

            char_masks = SegmentationCharMask(
                char_boxes,
                words=words,
                use_char_ann=(self.use_charann and words[0] != ""),
                size=img.size,
            )
            target.add_field("char_masks", char_masks)

            assert len(languages) > 0, "[Debug] {}: languages size = 0!".format(gt_path)
            language_list = LanguageList(languages)
            target.add_field("languages", language_list)

            if "rotated_boxes" in gt:
                rotated_boxes_5d = torch.tensor(gt["rotated_boxes"])
                target.add_field("rotated_boxes_5d", RotatedBoxList(rotated_boxes_5d, img.size))

        else:
            target = None

        if self.transforms is not None:
            try:
                img, target = self.transforms(img, target)
            except ValueError as e:
                raise Exception(
                    "[{}] Failed to transform {} (id: {}), with exception {}".format(
                        self.name, im_name, item, str(e)
                    )
                )

        if self.vis:
            self.visualize_old(img=img, im_name=im_name, target=target)

        item_name = self.get_item_name(item)
        return img, target, item_name

    def creat_color_map(self, n_class, width):
        splits = int(np.ceil(np.power((n_class * 1.0), 1.0 / 3)))
        maps = []
        for i in range(splits):
            r = int(i * width * 1.0 / (splits - 1))
            for j in range(splits):
                g = int(j * width * 1.0 / (splits - 1))
                for k in range(splits - 1):
                    b = int(k * width * 1.0 / (splits - 1))
                    maps.append([r, g, b])
        return np.array(maps)

    def __len__(self):
        return len(self.image_lists)

    def load_gt_from_txt(self, gt_path):
        words, boxes, charsboxes, segmentations, labels = [], [], [], [], []
        lines = open(gt_path).readlines()
        for line in lines:
            charbbs = []
            strs, loc = self.line2boxes(line)
            word = strs[0]
            if word == "###":
                if self.ignore_difficult:
                    rect = list(loc[0])
                    min_x = min(rect[::2]) - 1
                    min_y = min(rect[1::2]) - 1
                    max_x = max(rect[::2]) - 1
                    max_y = max(rect[1::2]) - 1
                    box = [min_x, min_y, max_x, max_y]
                    segmentations.append([loc[0, :]])
                    tindex = len(boxes)
                    boxes.append(box)
                    words.append(word)
                    labels.append(-1)
                    charbbs = np.zeros((10,), dtype=np.float32)
                    if loc.shape[0] > 1:
                        # for _i in range(1, loc.shape[0]):
                        #     charbb[9] = tindex
                        #     charbbs.append(charbb.copy())
                        charsboxes.append(charbbs)
                else:
                    continue
            else:
                rect = list(loc[0])
                min_x = min(rect[::2]) - 1
                min_y = min(rect[1::2]) - 1
                max_x = max(rect[::2]) - 1
                max_y = max(rect[1::2]) - 1
                box = [min_x, min_y, max_x, max_y]
                segmentations.append([loc[0, :]])
                tindex = len(boxes)
                boxes.append(box)
                words.append(word)
                labels.append(1)
                c_class = self.char2num(strs[1:])
                charbb = np.zeros((10,), dtype=np.float32)
                if loc.shape[0] > 1:
                    for i in range(1, loc.shape[0]):
                        charbb[:8] = loc[i, :]
                        charbb[8] = c_class[i - 1]
                        charbb[9] = tindex
                        charbbs.append(charbb.copy())
                    charsboxes.append(charbbs)
        num_boxes = len(boxes)
        if len(boxes) > 0:
            keep_boxes = np.zeros((num_boxes, 5))
            keep_boxes[:, :4] = np.array(boxes)
            keep_boxes[:, 4] = range(num_boxes)
            # the 5th column is the box label,
            # same as the 10th column of all charsboxes which belong to the box
            if self.use_charann:
                return words, np.array(keep_boxes), charsboxes, segmentations, labels
            else:
                charbbs = np.zeros((10,), dtype=np.float32)
                if len(charsboxes) == 0:
                    for _ in range(len(words)):
                        charsboxes.append([charbbs])
                return words, np.array(keep_boxes), charsboxes, segmentations, labels
        else:
            words.append("")
            charbbs = np.zeros((10,), dtype=np.float32)
            return (
                words,
                np.zeros((1, 5), dtype=np.float32),
                [[charbbs]],
                [[np.zeros((8,), dtype=np.float32)]],
                [1],
            )

    def load_image(self, path):
        with open(path, "rb") as buffer:
            return Image.open(buffer).convert("RGB")

    def line2boxes(self, line):
        parts = line.strip().split(",")
        if "\xef\xbb\xbf" in parts[0]:
            parts[0] = parts[0][3:]
        if "\ufeff" in parts[0]:
            parts[0] = parts[0].replace("\ufeff", "")
        x1 = np.array([int(float(x)) for x in parts[::9]])
        y1 = np.array([int(float(x)) for x in parts[1::9]])
        x2 = np.array([int(float(x)) for x in parts[2::9]])
        y2 = np.array([int(float(x)) for x in parts[3::9]])
        x3 = np.array([int(float(x)) for x in parts[4::9]])
        y3 = np.array([int(float(x)) for x in parts[5::9]])
        x4 = np.array([int(float(x)) for x in parts[6::9]])
        y4 = np.array([int(float(x)) for x in parts[7::9]])
        strs = parts[8::9]
        loc = np.vstack((x1, y1, x2, y2, x3, y3, x4, y4)).transpose()
        return strs, loc

    def check_charbbs(self, charbbs):
        xmins = np.minimum.reduce([charbbs[:, 0], charbbs[:, 2], charbbs[:, 4], charbbs[:, 6]])
        xmaxs = np.maximum.reduce([charbbs[:, 0], charbbs[:, 2], charbbs[:, 4], charbbs[:, 6]])
        ymins = np.minimum.reduce([charbbs[:, 1], charbbs[:, 3], charbbs[:, 5], charbbs[:, 7]])
        ymaxs = np.maximum.reduce([charbbs[:, 1], charbbs[:, 3], charbbs[:, 5], charbbs[:, 7]])
        return np.logical_and(
            xmaxs - xmins > self.min_proposal_size,
            ymaxs - ymins > self.min_proposal_size,
        )

    def check_charbb(self, charbb):
        xmins = min(charbb[0], charbb[2], charbb[4], charbb[6])
        xmaxs = max(charbb[0], charbb[2], charbb[4], charbb[6])
        ymins = min(charbb[1], charbb[3], charbb[5], charbb[7])
        ymaxs = max(charbb[1], charbb[3], charbb[5], charbb[7])
        return xmaxs - xmins > self.min_proposal_size and ymaxs - ymins > self.min_proposal_size

    def char2num(self, chars):
        # chars ['h', 'e', 'l', 'l', 'o']
        nums = [self.char_classes.index(c.lower()) for c in chars]
        return nums

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """

        im_name = os.path.basename(self.image_lists[item])
        img = Image.open(self.image_lists[item])
        width, height = img.size
        img_info = {"im_name": im_name, "height": height, "width": width}
        return img_info

    def visualize_old(self, img, im_name, target):
        new_im = img.numpy().copy().transpose([1, 2, 0]) + [
            102.9801,
            115.9465,
            122.7717,
        ]
        new_im = Image.fromarray(new_im.astype(np.uint8)).convert("RGB")
        mask = target.extra_fields["masks"].polygons[0].convert("mask")
        mask = Image.fromarray((mask.numpy() * 255).astype(np.uint8)).convert("RGB")
        if self.use_charann:
            m, _ = target.extra_fields["char_masks"].chars_boxes[0].convert("char_mask")
            color = self.creat_color_map(37, 255)
            color_map = color[m.numpy().astype(np.uint8)]
            char = Image.fromarray(color_map.astype(np.uint8)).convert("RGB")
            char = Image.blend(char, new_im, 0.5)
        else:
            char = new_im
        new = Image.blend(char, mask, 0.5)
        img_draw = ImageDraw.Draw(new)
        for box in target.bbox.numpy():
            box = list(box)
            box = box[:2] + [box[2], box[1]] + box[2:] + [box[0], box[3]] + box[:2]
            img_draw.line(box, fill=(255, 0, 0), width=2)
        new.save("./vis/char_" + im_name)
