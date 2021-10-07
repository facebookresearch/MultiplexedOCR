# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    # public datasets
    PUBLIC_DATASET_ROOT = "/checkpoint/jinghuang/datasets"
    SYNTHTEXT_MLT_ZIP_ROOT = f"{PUBLIC_DATASET_ROOT}/MLT19_Synthetic/"
    MSRATD500_ROOT = f"{PUBLIC_DATASET_ROOT}/MSRA-TD500"
    ICDAR15_ROOT = f"{PUBLIC_DATASET_ROOT}/ICDAR15"
    MLT17_ROOT = f"{PUBLIC_DATASET_ROOT}/MLT17"
    RCTW17_ROOT = f"{PUBLIC_DATASET_ROOT}/RCTW17"
    ArT19_ROOT = f"{PUBLIC_DATASET_ROOT}/ArT19"
    LSVT19_ROOT = f"{PUBLIC_DATASET_ROOT}/LSVT19"
    MLT19_ROOT = f"{PUBLIC_DATASET_ROOT}/MLT19"
    ICDAR13_ROOT = f"{PUBLIC_DATASET_ROOT}/ICDAR13"
    KAIST_ROOT = f"{PUBLIC_DATASET_ROOT}/KAIST"
    TOTAL_TEXT_ROOT = f"{PUBLIC_DATASET_ROOT}/total_text"

    SYNTHTEXT_MLT_ZIP_LANGUAGES = ["ar", "bn", "hi", "ja", "ko", "la", "zh"]
    MLT17_LANGUAGES = ["ar", "en", "fr", "zh", "de", "ko", "ja", "it", "bn", "random50"]
    MLT19_LANGUAGES = [
        "ar",
        "en",
        "fr",
        "zh",
        "de",
        "ko",
        "ja",
        "it",
        "bn",
        "hi",
        "first5",
        "second5",
        "random50",
    ]

    DATASETS = {
        "mlt17_train": {
            "factory": "Icdar17MLTDataset",
            "args": {
                "name": "mlt17_train",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT17_ROOT, "train", "imgs"),
                "gts_dir": os.path.join(MLT17_ROOT, "train", "gts"),
                "expected_imgs_in_dir": 7200,
            },
        },
        # Note: mlt17_val_train and mlt17_val_test are exactly the same dataset
        # It's a common practice in academia to include mlt17 val into training
        # as there's no validation benchmark
        "mlt17_val_train": {
            "factory": "Icdar17MLTDataset",
            "args": {
                "name": "mlt17_val_train",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT17_ROOT, "val", "imgs"),
                "gts_dir": os.path.join(MLT17_ROOT, "val", "gts"),
                "expected_imgs_in_dir": 1800,
            },
        },
        "mlt17_val_test": {
            "factory": "Icdar17MLTDataset",
            "args": {
                "name": "mlt17_val_test",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT17_ROOT, "val", "imgs"),
                "gts_dir": os.path.join(MLT17_ROOT, "val", "gts"),
                "expected_imgs_in_dir": 1800,
            },
        },
        "mlt17_test": {
            "factory": "Icdar17MLTDataset",
            "args": {
                "name": "mlt17_test",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT17_ROOT, "test", "imgs"),
                "gts_dir": None,
                "expected_imgs_in_dir": 9000,
            },
        },
        "rctw17_train": {
            "factory": "Icdar17RCTWDataset",
            "args": {
                "name": "rctw17_train",
                "use_charann": False,
                "imgs_dir": os.path.join(RCTW17_ROOT, "train", "imgs"),
                "gts_dir": os.path.join(RCTW17_ROOT, "train", "gts"),
                "expected_imgs_in_dir": 8034,
            },
        },
        "art19_train": {
            "factory": "Icdar19ArTDataset",
            "args": {
                "name": "art19_train",
                "use_charann": False,
                "imgs_dir": os.path.join(ArT19_ROOT, "train", "train_images"),
                "gts_dir": os.path.join(ArT19_ROOT, "train", "gts"),
                "total_text_test_dir": os.path.join(TOTAL_TEXT_ROOT, "test", "images"),
                "total_text_map_file": os.path.join(
                    ArT19_ROOT, "install_files", "Total_Text_ID_vs_ArT_ID.list"
                ),
                "expected_imgs_in_dir": 5603,
            },
        },
        "art19_total_text_train": {
            "factory": "Icdar19ArTDataset",
            "args": {
                "name": "art19_total_text_train",
                "use_charann": False,
                "imgs_dir": os.path.join(ArT19_ROOT, "train", "train_images"),
                "gts_dir": os.path.join(ArT19_ROOT, "train", "gts"),
                "total_text_train_dir": os.path.join(TOTAL_TEXT_ROOT, "train", "images"),
                "total_text_test_dir": os.path.join(TOTAL_TEXT_ROOT, "test", "images"),
                "total_text_map_file": os.path.join(
                    ArT19_ROOT, "install_files", "Total_Text_ID_vs_ArT_ID.list"
                ),
                "expected_imgs_in_dir": 5603,
                "variation": "total_text_train_only",
            },
        },
        "art19_with_total_text_test_train": {
            "factory": "Icdar19ArTDataset",
            "args": {
                "name": "art19_with_total_text_test_train",
                "use_charann": False,
                "imgs_dir": os.path.join(ArT19_ROOT, "train", "train_images"),
                "gts_dir": os.path.join(ArT19_ROOT, "train", "gts"),
                "expected_imgs_in_dir": 5603,
                "variation": "with_total_text_test",
            },
        },
        "art19_test": {
            "factory": "Icdar19ArTDataset",
            "args": {
                "name": "art19_test",
                "use_charann": False,
                "imgs_dir": os.path.join(ArT19_ROOT, "test", "test_images"),
                "gts_dir": os.path.join(ArT19_ROOT, "test", "gts"),
                "expected_imgs_in_dir": 4563,
                "variation": "with_total_text_test",
            },
        },
        "lsvt19_train": {
            "factory": "Icdar19LSVTDataset",
            "args": {
                "name": "lsvt19_train",
                "use_charann": False,
                "imgs_dir": os.path.join(LSVT19_ROOT, "train", "imgs"),
                "gts_dir": os.path.join(LSVT19_ROOT, "train", "gts"),
                "expected_imgs_in_dir": 30000,
            },
        },
        "mlt19_train": {
            "factory": "Icdar19MLTDataset",
            "args": {
                "name": "mlt19_train",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT19_ROOT, "train", "imgs"),
                "gts_dir": os.path.join(MLT19_ROOT, "train", "gts"),
                "expected_imgs_in_dir": 10000,
            },
        },
        # Note: mlt19_val_train and mlt19_val_test are exactly the same dataset
        # Please only include mlt19_val_train in the final step of training (before
        # submitting to test benchmark),
        # since including this will lead to wrong MLT19 validation benchmark result
        "mlt19_val_train": {
            "factory": "Icdar19MLTDataset",
            "args": {
                "name": "mlt19_val_train",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT19_ROOT, "val", "val_imgs"),
                "gts_dir": os.path.join(MLT19_ROOT, "val", "val_gts"),
                "expected_imgs_in_dir": 2000,
                "split": "val",  # validation dataset
            },
        },
        "mlt19_val_test": {
            "factory": "Icdar19MLTDataset",
            "args": {
                "name": "mlt19_val_test",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT19_ROOT, "val", "val_imgs"),
                "gts_dir": os.path.join(MLT19_ROOT, "val", "val_gts"),
                "expected_imgs_in_dir": 2000,
                "split": "val",  # validation dataset
            },
        },
        "mlt19_test": {
            "factory": "Icdar19MLTDataset",
            "args": {
                "name": "mlt19_test",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT19_ROOT, "test", "imgs"),
                "gts_dir": None,
                "expected_imgs_in_dir": 10000,
                "split": "test",  # testing dataset
            },
        },
        # "icdar_2013_train": ("icdar2013/train_images", "icdar2013/train_gts"),
        # "icdar_2013_test": ("icdar2013/test_images", "icdar2013/test_gts"),
        "rotated_ic13_test": (
            "icdar2013/rotated_test_images",
            "icdar2013/rotated_test_gts",
        ),
        "icdar13_train": {
            "args": {
                "name": "icdar13_train",
                "use_charann": False,
                "imgs_dir": os.path.join(ICDAR13_ROOT, "train/imgs"),
                "gts_dir": os.path.join(ICDAR13_ROOT, "train/anno"),
                "expected_imgs_in_dir": 229,
            },
            "factory": "Icdar13Dataset",
        },
        "icdar13_test": {
            "args": {
                "name": "icdar13_test",
                "use_charann": False,
                "imgs_dir": os.path.join(ICDAR13_ROOT, "test/imgs"),
                "gts_dir": os.path.join(ICDAR13_ROOT, "test/anno"),
                "expected_imgs_in_dir": 233,
            },
            "factory": "Icdar13Dataset",
        },
        "icdar15_train": {
            "args": {
                "name": "icdar15_train",
                "use_charann": False,
                "imgs_dir": os.path.join(ICDAR15_ROOT, "train", "imgs"),
                "gts_dir": os.path.join(ICDAR15_ROOT, "train", "anno"),
                "expected_imgs_in_dir": 1000,
            },
            "factory": "Icdar15Dataset",
        },
        "icdar15_test": {
            "args": {
                "name": "icdar15_test",
                "use_charann": False,
                "imgs_dir": os.path.join(ICDAR15_ROOT, "test", "imgs"),
                "gts_dir": os.path.join(ICDAR15_ROOT, "test", "anno"),
                "expected_imgs_in_dir": 500,
            },
            "factory": "Icdar15Dataset",
        },
        "icdar15_random50_test": {
            "args": {
                "use_charann": False,
                "imgs_dir": os.path.join(ICDAR15_ROOT, "test", "imgs"),
                "gts_dir": os.path.join(ICDAR15_ROOT, "test", "anno"),
                "expected_imgs_in_dir": 500,
                "num_samples": 50,
            },
            "factory": "Icdar15Dataset",
        },
        "msra_td500_train": {
            "args": {
                "name": "msra_td500_train",
                "use_charann": False,
                "imgs_dir": os.path.join(MSRATD500_ROOT, "train", "imgs"),
                "gts_dir": os.path.join(MSRATD500_ROOT, "train", "gts"),
                "expected_imgs_in_dir": 300,
            },
            "factory": "IcdarMSRATD500Dataset",
        },
        "msra_td500_test": {
            "args": {
                "name": "msra_td500_train",
                "use_charann": False,
                "imgs_dir": os.path.join(MSRATD500_ROOT, "test", "imgs"),
                "gts_dir": os.path.join(MSRATD500_ROOT, "test", "gts"),
                "expected_imgs_in_dir": 200,
            },
            "factory": "IcdarMSRATD500Dataset",
        },
        "kaist_train": {
            "args": {
                "name": "kaist_train",
                "use_charann": False,
                "imgs_dir": os.path.join(KAIST_ROOT, "train", "imgs"),
                "gts_dir": os.path.join(KAIST_ROOT, "train", "gts"),
                "expected_imgs_in_dir": 2427,
            },
            "factory": "IcdarKAISTDataset",
        },
        "synthtext_train": ("SynthText", "SynthText_GT_E2E"),
        "synthtext_test": ("SynthText", "SynthText_GT_E2E"),
        "textvqa_full_train": ("train/imgs", None, 21953),
        "textvqa_full_val_test": ("val/imgs", None, 3166),
        "textvqa_full_test": ("test/imgs", None, 3353),
        "total_text_test": {
            "args": {
                "name": "total_text_test",
                "use_charann": False,
                "imgs_dir": os.path.join(TOTAL_TEXT_ROOT, "test", "images"),
                "gts_dir": None,
                "expected_imgs_in_dir": 300,
            },
            "factory": "TotaltextDataset",
        },
        "total_text_random50_test": {
            "args": {
                "name": "total_text_random50_test",
                "use_charann": False,
                "imgs_dir": os.path.join(TOTAL_TEXT_ROOT, "test", "images"),
                "gts_dir": None,
                "expected_imgs_in_dir": 300,
                "variation": "random50",
            },
            "factory": "TotaltextDataset",
        },
        "scut-eng-char_train": (
            "scut-eng-char/train_images",
            "scut-eng-char/train_gts",
        ),
    }

    # ICDAR19 MLT
    for language in MLT19_LANGUAGES:
        DATASETS["mlt19_" + language + "_train"] = {
            "factory": "Icdar19MLTDataset",
            "args": {
                "name": "mlt19_" + language + "_train",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT19_ROOT, "train", "imgs"),
                "gts_dir": os.path.join(MLT19_ROOT, "train", "gts"),
                "expected_imgs_in_dir": 10000,
                "split": "train",  # training datasets
                "language": language,
            },
        }

    for language in MLT19_LANGUAGES:
        DATASETS["mlt19_" + language + "_test"] = {
            "factory": "Icdar19MLTDataset",
            "args": {
                "name": "mlt19_" + language + "_test",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT19_ROOT, "test", "imgs_" + language),
                "gts_dir": None,  # we don't have ground truth for now
                "expected_imgs_in_dir": 1000,  # 1000 images in each sub-folder
                "split": "test",  # testing datasets
                "language": language,
            },
        }

    for language in MLT19_LANGUAGES:
        DATASETS["mlt19_" + language + "_val_train"] = {
            "factory": "Icdar19MLTDataset",
            "args": {
                "name": "mlt19_" + language + "_val_train",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT19_ROOT, "val", "val_imgs"),
                "gts_dir": os.path.join(MLT19_ROOT, "val", "val_gts"),
                "expected_imgs_in_dir": 2000,
                "split": "val",  # validation datasets, used as training
                "language": language,
            },
        }

    for language in MLT19_LANGUAGES:
        DATASETS["mlt19_" + language + "_val_test"] = {
            "factory": "Icdar19MLTDataset",
            "args": {
                "name": "mlt19_" + language + "_val_test",
                "use_charann": False,
                "imgs_dir": os.path.join(MLT19_ROOT, "val", "val_imgs"),
                "gts_dir": os.path.join(MLT19_ROOT, "val", "val_gts"),
                "expected_imgs_in_dir": 2000,
                "split": "val",  # validation datasets, used as testing
                "language": language,
            },
        }

    @staticmethod
    def get(name):
        return DatasetCatalog.DATASETS[name]


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = (
        "output/train/{}coco_2014_train%3A{}"
        + "coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    )
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/") :]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        if "resnet34" in name or "resnet18" in name:
            return name
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/") :]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
