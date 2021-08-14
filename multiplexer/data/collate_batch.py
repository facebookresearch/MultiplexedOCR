# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from multiplexer.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        # filter bad data in batch
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None, None, None

        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        # if transposed_batch[1] is None:
        #     images = to_image_list(transposed_batch[0], self.size_divisible)
        #     targets = transposed_batch[1]
        #     img_ids = transposed_batch[2]
        # else:
        #     images, targets = to_image_target_list(
        # transposed_batch[0], self.size_divisible, transposed_batch[1])
        #     img_ids = transposed_batch[2]
        return images, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images before inference
    """

    def __call__(self, batch):
        return list(zip(*batch))
