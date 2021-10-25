import math
import random

import numpy as np

from virtual_fs.virtual_io import open

from .icdar import IcdarDataset


class MSRATD500Dataset(IcdarDataset):
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
        super(MSRATD500Dataset, self).__init__(
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
        parts = line.strip().split()
        if "\xef\xbb\xbf" in parts[0]:
            # UTF-8 Byte order mark
            parts[0] = parts[0][3:]
        if "\ufeff" in parts[0]:
            # UTF-8 Byte order mark
            parts[0] = parts[0].replace("\ufeff", "")

        assert len(parts) >= 7, "[Error][MSRA-TD500] line = {}, parts = {}".format(line, parts)

        if len(parts) > 7:
            word = ",".join(parts[6:])
            if random.random() < 0.01:
                # log every 100
                print(
                    "[Warning][MSRA-TD500] line = {}, parts = {}, word = {}".format(
                        line, parts, word
                    )
                )
        else:
            word = "###"
            language = "none"

        x, y, w, h, angle = list(map(float, parts[2:7]))
        cx = x + w / 2
        cy = y + h / 2

        x_ul, y_ul = self.rotate_point((cx, cy), (x, y), angle)
        x_ur, y_ur = self.rotate_point((cx, cy), (x + w, y), angle)
        x_ll, y_ll = self.rotate_point((cx, cy), (x, y + h), angle)
        x_lr, y_lr = self.rotate_point((cx, cy), (x + w, y + h), angle)
        points = [x_ul, y_ul, x_ur, y_ur, x_lr, y_lr, x_ll, y_ll]
        quadrilateral = list(map(int, points))

        return quadrilateral, language, word

    @staticmethod
    def rotate_point(center, point, angle):
        cx, cy = center
        px, py = point

        x = cx + (px - cx) * math.cos(angle) - (py - cy) * math.sin(angle)
        y = cy + (px - cx) * math.sin(angle) + (py - cy) * math.cos(angle)
        return x, y
