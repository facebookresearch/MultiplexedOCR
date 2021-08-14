# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transform as T


def build_transforms(cfg, is_train=True):
    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        # flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        # flip_prob = 0
        # rotate_prob = 0.5
        rotate_prob = cfg.DATASETS.RANDOM_ROTATE_PROB
        pixel_aug_prob = 0.2
        random_crop_prob = cfg.DATASETS.RANDOM_CROP_PROB
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        sqr_size = cfg.INPUT.SQR_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        # flip_prob = 0
        rotate_prob = 0
        pixel_aug_prob = 0
        random_crop_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    if is_train:
        augmenter = cfg.DATASETS.AUGMENTER.NAME
        if not cfg.DATASETS.AUG:
            # if augmentation is disabled during training, use the ResizerV0 by default
            augmenter = "ResizerV0"
        else:
            if cfg.DATASETS.FIX_CROP:
                assert (
                    cfg.DATASETS.AUGMENTER.NAME == "CropperV0"
                ), "Please set AUGMENTER.NAME = CropperV0 for fixed crop"
    else:
        # common options during testing: ResizerV0, SquareResizerV0
        augmenter = cfg.DATASETS.AUGMENTER.TEST

    if augmenter == "CropperV2":
        # Main features:
        # Use RandomCropV2
        # RandomCrop happens after RandomRotate
        transform = T.Compose(
            [
                T.RandomBrightness(pixel_aug_prob),
                T.RandomContrast(pixel_aug_prob),
                T.RandomHue(pixel_aug_prob),
                T.RandomSaturation(pixel_aug_prob),
                T.RandomGamma(pixel_aug_prob),
                T.RandomRotate(
                    prob=rotate_prob,
                    max_theta=cfg.DATASETS.MAX_ROTATE_THETA,
                    fix_rotate=cfg.DATASETS.FIX_ROTATE,
                ),
                T.RandomCropV2(
                    prob=random_crop_prob,
                    min_width_ratio=cfg.DATASETS.AUGMENTER.MIN_WIDTH_RATIO,
                    min_height_ratio=cfg.DATASETS.AUGMENTER.MIN_HEIGHT_RATIO,
                    min_box_num_ratio=cfg.DATASETS.AUGMENTER.MIN_BOX_NUM_RATIO,
                ),
                T.Resize(min_size, max_size, cfg.INPUT.STRICT_RESIZE),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    elif augmenter == "CropperV1":
        transform = T.Compose(
            [
                T.RandomCrop(random_crop_prob),
                T.RandomBrightness(pixel_aug_prob),
                T.RandomContrast(pixel_aug_prob),
                T.RandomHue(pixel_aug_prob),
                T.RandomSaturation(pixel_aug_prob),
                T.RandomGamma(pixel_aug_prob),
                T.RandomRotate(
                    rotate_prob,
                    max_theta=cfg.DATASETS.MAX_ROTATE_THETA,
                    fix_rotate=cfg.DATASETS.FIX_ROTATE,
                ),
                T.Resize(min_size, max_size, cfg.INPUT.STRICT_RESIZE),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    elif augmenter == "CropperV0":
        transform = T.Compose(
            [
                T.RandomCrop(1.0, crop_min_size=512, crop_max_size=640, max_trys=50),
                T.RandomBrightness(pixel_aug_prob),
                T.RandomContrast(pixel_aug_prob),
                T.RandomHue(pixel_aug_prob),
                T.RandomSaturation(pixel_aug_prob),
                T.RandomGamma(pixel_aug_prob),
                T.RandomRotate(rotate_prob),
                T.Resize(min_size, max_size, cfg.INPUT.STRICT_RESIZE),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    elif augmenter == "SquareResizerV0":
        assert not is_train, "sqr_size is not supported during training yet"
        # resize only, but aware of sqr_size
        transform = T.Compose(
            [
                T.SquareAwareResize(min_size, max_size, sqr_size),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        assert augmenter == "ResizerV0", f"Unrecognized augmenter name: {augmenter}"
        # resize only
        transform = T.Compose(
            [
                T.Resize(min_size, max_size, cfg.INPUT.STRICT_RESIZE),
                T.ToTensor(),
                normalize_transform,
            ]
        )

    return transform
