#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

# from yacs.config import CfgNode as CN
from virtual_fs.virtual_config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.SEG_ON = False
_C.MODEL.CHAR_MASK_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.TRAIN_DETECTION_ONLY = False
_C.MODEL.RESNET34 = False
_C.MODEL.TORCHSCRIPT_ONLY = False

# ---------------------------------------------------------------------------- #
# Proposal generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.PROPOSAL_GENERATOR = CN()
# Current proposal generators include "RPN", "RRPN", "SPN" and "RSPN"
_C.MODEL.PROPOSAL_GENERATOR.NAME = "SPN"

# Language-specific heads
_C.MODEL.LANGUAGE = "en_num_36"

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""

_C.MULTIPLEXER = CN()
_C.MULTIPLEXER.LANGUAGE_WEIGHT_MODE = "soft"
_C.MULTIPLEXER.LOSS_FORMAT = "separate"
_C.MULTIPLEXER.TEST = CN()
_C.MULTIPLEXER.TEST.RUN_ALL_HEADS = False

# Character-map related configs
_C.CHAR_MAP = CN()
# char map directory
_C.CHAR_MAP.DIR = ""

_C.SEQUENCE = CN()
_C.SEQUENCE.SEQ_ON = False
# When SEQUENCE.SHARED_CONV5_MASK is True (default setting),
# all seq heads share the same conv5 mask layer
# Otherwise if SEQUENCE.SHARED_CONV5_MASK is False,
# each seq head will have its own conv5 mask layer, meaning they are almost fully decoupled,
# which might significantly increase the parameter number.
_C.SEQUENCE.SHARED_CONV5_MASK = True
_C.SEQUENCE.NUM_SEQ_HEADS = 1
# Enabled languages for language identification purpose
# TODO: Move this to _C.MODEL.LANGUAGE_HEAD.LANGUAGES
_C.SEQUENCE.LANGUAGES = ("ar", "bn", "hi", "ja", "ko", "la", "zh", "symbol")
# Enabled languages for recognition purpose
# This must be a subset of _C.SEQUENCE.LANGUAGES
_C.SEQUENCE.LANGUAGES_ENABLED = ("ar", "bn", "hi", "ja", "ko", "la", "zh", "symbol")
# List of unfreezed sequential heads
# For backward compatibility of the configs, when it's set to be None,
# it means the seq heads for all languages are NOT frozen;
# When it's an empty list, it means the seq heads for all languages are frozen;
_C.SEQUENCE.LANGUAGES_UNFREEZED = None  # requires yacs >= 0.1.8
_C.SEQUENCE.NUM_CHAR = 36
_C.SEQUENCE.EMBED_SIZE = 38
_C.SEQUENCE.HIDDEN_SIZE = 256
# Whether beam search is enabled at test time
_C.SEQUENCE.BEAM_SEARCH = True
_C.SEQUENCE.BOS_TOKEN = 0
_C.SEQUENCE.MAX_LENGTH = 32
_C.SEQUENCE.TEACHER_FORCE_RATIO = 1.0
_C.SEQUENCE.TWO_CONV = False
_C.SEQUENCE.MEAN_SCORE = False
_C.SEQUENCE.RESIZE_HEIGHT = 16
_C.SEQUENCE.RESIZE_WIDTH = 64
_C.SEQUENCE.DECODER_LOSS = "NLLLoss"
_C.SEQUENCE.LOSS_WEIGHT = 0.5

_C.SEQUENCE.EN_NUM = CN()
_C.SEQUENCE.EN_NUM.ARCH = "seq2seq_a"
_C.SEQUENCE.EN_NUM.NUM_CHAR = 36
_C.SEQUENCE.EN_NUM.EMBED_SIZE = 38
_C.SEQUENCE.EN_NUM.HIDDEN_SIZE = 256

_C.SEQUENCE.EN_NUM_36 = CN()
_C.SEQUENCE.EN_NUM_36.ARCH = "seq2seq_a"
_C.SEQUENCE.EN_NUM_36.NUM_CHAR = 36
_C.SEQUENCE.EN_NUM_36.EMBED_SIZE = 38
_C.SEQUENCE.EN_NUM_36.HIDDEN_SIZE = 256

_C.SEQUENCE.AMHARIC = CN()
_C.SEQUENCE.AMHARIC.ARCH = "seq2seq_a"
_C.SEQUENCE.AMHARIC.NUM_CHAR = 370
_C.SEQUENCE.AMHARIC.EMBED_SIZE = 50
_C.SEQUENCE.AMHARIC.HIDDEN_SIZE = 256

_C.SEQUENCE.ANY = CN()
_C.SEQUENCE.ANY.ARCH = "seq2seq_a"
_C.SEQUENCE.ANY.NUM_CHAR = 11000
_C.SEQUENCE.ANY.EMBED_SIZE = 100
_C.SEQUENCE.ANY.HIDDEN_SIZE = 512

_C.SEQUENCE.ARABIC = CN()
_C.SEQUENCE.ARABIC.ARCH = "seq2seq_a"
_C.SEQUENCE.ARABIC.NUM_CHAR = 150
_C.SEQUENCE.ARABIC.EMBED_SIZE = 30
_C.SEQUENCE.ARABIC.HIDDEN_SIZE = 256

_C.SEQUENCE.BENGALI = CN()
_C.SEQUENCE.BENGALI.ARCH = "seq2seq_a"
_C.SEQUENCE.BENGALI.NUM_CHAR = 120
_C.SEQUENCE.BENGALI.EMBED_SIZE = 30
_C.SEQUENCE.BENGALI.HIDDEN_SIZE = 256

_C.SEQUENCE.BULGARIAN = CN()
_C.SEQUENCE.BULGARIAN.ARCH = "seq2seq_a"
_C.SEQUENCE.BULGARIAN.NUM_CHAR = 110
_C.SEQUENCE.BULGARIAN.EMBED_SIZE = 30
_C.SEQUENCE.BULGARIAN.HIDDEN_SIZE = 192

_C.SEQUENCE.BURMESE = CN()
_C.SEQUENCE.BURMESE.ARCH = "seq2seq_a"
_C.SEQUENCE.BURMESE.NUM_CHAR = 160
_C.SEQUENCE.BURMESE.EMBED_SIZE = 30
_C.SEQUENCE.BURMESE.HIDDEN_SIZE = 192

_C.SEQUENCE.CHINESE = CN()
_C.SEQUENCE.CHINESE.ARCH = "seq2seq_a"
_C.SEQUENCE.CHINESE.NUM_CHAR = 6100
_C.SEQUENCE.CHINESE.EMBED_SIZE = 50
_C.SEQUENCE.CHINESE.HIDDEN_SIZE = 256

_C.SEQUENCE.CROATIAN = CN()
_C.SEQUENCE.CROATIAN.ARCH = "seq2seq_a"
_C.SEQUENCE.CROATIAN.NUM_CHAR = 110
_C.SEQUENCE.CROATIAN.EMBED_SIZE = 100
_C.SEQUENCE.CROATIAN.HIDDEN_SIZE = 192

_C.SEQUENCE.CYRILLIC = CN()
_C.SEQUENCE.CYRILLIC.ARCH = "seq2seq_a"
_C.SEQUENCE.CYRILLIC.NUM_CHAR = 130
_C.SEQUENCE.CYRILLIC.EMBED_SIZE = 30
_C.SEQUENCE.CYRILLIC.HIDDEN_SIZE = 256

_C.SEQUENCE.DEVANAGARI = CN()
_C.SEQUENCE.DEVANAGARI.ARCH = "seq2seq_a"
_C.SEQUENCE.DEVANAGARI.NUM_CHAR = 130
_C.SEQUENCE.DEVANAGARI.EMBED_SIZE = 30
_C.SEQUENCE.DEVANAGARI.HIDDEN_SIZE = 256

_C.SEQUENCE.DUTCH = CN()
_C.SEQUENCE.DUTCH.ARCH = "seq2seq_a"
_C.SEQUENCE.DUTCH.NUM_CHAR = 100
_C.SEQUENCE.DUTCH.EMBED_SIZE = 30
_C.SEQUENCE.DUTCH.HIDDEN_SIZE = 192

_C.SEQUENCE.ENGLISH = CN()
_C.SEQUENCE.ENGLISH.ARCH = "seq2seq_a"
_C.SEQUENCE.ENGLISH.NUM_CHAR = 120
_C.SEQUENCE.ENGLISH.EMBED_SIZE = 30
_C.SEQUENCE.ENGLISH.HIDDEN_SIZE = 256

_C.SEQUENCE.FRENCH = CN()
_C.SEQUENCE.FRENCH.ARCH = "seq2seq_a"
_C.SEQUENCE.FRENCH.NUM_CHAR = 160
_C.SEQUENCE.FRENCH.EMBED_SIZE = 30
_C.SEQUENCE.FRENCH.HIDDEN_SIZE = 192

_C.SEQUENCE.GERMAN = CN()
_C.SEQUENCE.GERMAN.ARCH = "seq2seq_a"
_C.SEQUENCE.GERMAN.NUM_CHAR = 120
_C.SEQUENCE.GERMAN.EMBED_SIZE = 30
_C.SEQUENCE.GERMAN.HIDDEN_SIZE = 192

_C.SEQUENCE.GREEK = CN()
_C.SEQUENCE.GREEK.ARCH = "seq2seq_a"
_C.SEQUENCE.GREEK.NUM_CHAR = 120
_C.SEQUENCE.GREEK.EMBED_SIZE = 30
_C.SEQUENCE.GREEK.HIDDEN_SIZE = 192

_C.SEQUENCE.GUJARATI = CN()
_C.SEQUENCE.GUJARATI.ARCH = "seq2seq_a"
_C.SEQUENCE.GUJARATI.NUM_CHAR = 120
_C.SEQUENCE.GUJARATI.EMBED_SIZE = 30
_C.SEQUENCE.GUJARATI.HIDDEN_SIZE = 192

_C.SEQUENCE.HANGUL = CN()
_C.SEQUENCE.HANGUL.ARCH = "seq2seq_a"
_C.SEQUENCE.HANGUL.NUM_CHAR = 2000
_C.SEQUENCE.HANGUL.EMBED_SIZE = 50
_C.SEQUENCE.HANGUL.HIDDEN_SIZE = 256

_C.SEQUENCE.HEBREW = CN()
_C.SEQUENCE.HEBREW.ARCH = "seq2seq_a"
_C.SEQUENCE.HEBREW.NUM_CHAR = 110
_C.SEQUENCE.HEBREW.EMBED_SIZE = 30
_C.SEQUENCE.HEBREW.HIDDEN_SIZE = 192

_C.SEQUENCE.HUNGARIAN = CN()
_C.SEQUENCE.HUNGARIAN.ARCH = "seq2seq_a"
_C.SEQUENCE.HUNGARIAN.NUM_CHAR = 120
_C.SEQUENCE.HUNGARIAN.EMBED_SIZE = 30
_C.SEQUENCE.HUNGARIAN.HIDDEN_SIZE = 192

_C.SEQUENCE.INDONESIAN = CN()
_C.SEQUENCE.INDONESIAN.ARCH = "seq2seq_a"
_C.SEQUENCE.INDONESIAN.NUM_CHAR = 100
_C.SEQUENCE.INDONESIAN.EMBED_SIZE = 30
_C.SEQUENCE.INDONESIAN.HIDDEN_SIZE = 192

_C.SEQUENCE.ITALIAN = CN()
_C.SEQUENCE.ITALIAN.ARCH = "seq2seq_a"
_C.SEQUENCE.ITALIAN.NUM_CHAR = 110
_C.SEQUENCE.ITALIAN.EMBED_SIZE = 30
_C.SEQUENCE.ITALIAN.HIDDEN_SIZE = 192

_C.SEQUENCE.JAPANESE = CN()
_C.SEQUENCE.JAPANESE.ARCH = "seq2seq_a"
_C.SEQUENCE.JAPANESE.NUM_CHAR = 3300
_C.SEQUENCE.JAPANESE.EMBED_SIZE = 50
_C.SEQUENCE.JAPANESE.HIDDEN_SIZE = 256

_C.SEQUENCE.JAVANESE = CN()
_C.SEQUENCE.JAVANESE.ARCH = "seq2seq_a"
_C.SEQUENCE.JAVANESE.NUM_CHAR = 110
_C.SEQUENCE.JAVANESE.EMBED_SIZE = 30
_C.SEQUENCE.JAVANESE.HIDDEN_SIZE = 168

_C.SEQUENCE.KANA = CN()
_C.SEQUENCE.KANA.ARCH = "seq2seq_a"
_C.SEQUENCE.KANA.NUM_CHAR = 210
_C.SEQUENCE.KANA.EMBED_SIZE = 30
_C.SEQUENCE.KANA.HIDDEN_SIZE = 256

_C.SEQUENCE.KANNADA = CN()
_C.SEQUENCE.KANNADA.ARCH = "seq2seq_a"
_C.SEQUENCE.KANNADA.NUM_CHAR = 120
_C.SEQUENCE.KANNADA.EMBED_SIZE = 30
_C.SEQUENCE.KANNADA.HIDDEN_SIZE = 192

_C.SEQUENCE.KHMER = CN()
_C.SEQUENCE.KHMER.ARCH = "seq2seq_a"
_C.SEQUENCE.KHMER.NUM_CHAR = 140
_C.SEQUENCE.KHMER.EMBED_SIZE = 30
_C.SEQUENCE.KHMER.HIDDEN_SIZE = 192

_C.SEQUENCE.LATIN = CN()
_C.SEQUENCE.LATIN.ARCH = "seq2seq_a"
_C.SEQUENCE.LATIN.NUM_CHAR = 230
_C.SEQUENCE.LATIN.EMBED_SIZE = 50
_C.SEQUENCE.LATIN.HIDDEN_SIZE = 256

_C.SEQUENCE.MALAY = CN()
_C.SEQUENCE.MALAY.ARCH = "seq2seq_a"
_C.SEQUENCE.MALAY.NUM_CHAR = 110
_C.SEQUENCE.MALAY.EMBED_SIZE = 30
_C.SEQUENCE.MALAY.HIDDEN_SIZE = 192

_C.SEQUENCE.MALAYALAM = CN()
_C.SEQUENCE.MALAYALAM.ARCH = "seq2seq_a"
_C.SEQUENCE.MALAYALAM.NUM_CHAR = 130
_C.SEQUENCE.MALAYALAM.EMBED_SIZE = 30
_C.SEQUENCE.MALAYALAM.HIDDEN_SIZE = 192

_C.SEQUENCE.MARATHI = CN()
_C.SEQUENCE.MARATHI.ARCH = "seq2seq_a"
_C.SEQUENCE.MARATHI.NUM_CHAR = 130
_C.SEQUENCE.MARATHI.EMBED_SIZE = 30
_C.SEQUENCE.MARATHI.HIDDEN_SIZE = 192

_C.SEQUENCE.NUMBER = CN()
_C.SEQUENCE.NUMBER.ARCH = "seq2seq_a"
_C.SEQUENCE.NUMBER.NUM_CHAR = 20
_C.SEQUENCE.NUMBER.EMBED_SIZE = 10
_C.SEQUENCE.NUMBER.HIDDEN_SIZE = 64

_C.SEQUENCE.PERSIAN = CN()
_C.SEQUENCE.PERSIAN.ARCH = "seq2seq_a"
_C.SEQUENCE.PERSIAN.NUM_CHAR = 160
_C.SEQUENCE.PERSIAN.EMBED_SIZE = 50
_C.SEQUENCE.PERSIAN.HIDDEN_SIZE = 256

_C.SEQUENCE.POLISH = CN()
_C.SEQUENCE.POLISH.ARCH = "seq2seq_a"
_C.SEQUENCE.POLISH.NUM_CHAR = 120
_C.SEQUENCE.POLISH.EMBED_SIZE = 30
_C.SEQUENCE.POLISH.HIDDEN_SIZE = 192

_C.SEQUENCE.PORTUGUESE = CN()
_C.SEQUENCE.PORTUGUESE.ARCH = "seq2seq_a"
_C.SEQUENCE.PORTUGUESE.NUM_CHAR = 110
_C.SEQUENCE.PORTUGUESE.EMBED_SIZE = 30
_C.SEQUENCE.PORTUGUESE.HIDDEN_SIZE = 192

_C.SEQUENCE.PUNJABI = CN()
_C.SEQUENCE.PUNJABI.ARCH = "seq2seq_a"
_C.SEQUENCE.PUNJABI.NUM_CHAR = 120
_C.SEQUENCE.PUNJABI.EMBED_SIZE = 30
_C.SEQUENCE.PUNJABI.HIDDEN_SIZE = 192

_C.SEQUENCE.ROMANIAN = CN()
_C.SEQUENCE.ROMANIAN.ARCH = "seq2seq_a"
_C.SEQUENCE.ROMANIAN.NUM_CHAR = 110
_C.SEQUENCE.ROMANIAN.EMBED_SIZE = 30
_C.SEQUENCE.ROMANIAN.HIDDEN_SIZE = 192

_C.SEQUENCE.RUSSIAN = CN()
_C.SEQUENCE.RUSSIAN.ARCH = "seq2seq_a"
_C.SEQUENCE.RUSSIAN.NUM_CHAR = 140
_C.SEQUENCE.RUSSIAN.EMBED_SIZE = 30
_C.SEQUENCE.RUSSIAN.HIDDEN_SIZE = 256

_C.SEQUENCE.SINHALA = CN()
_C.SEQUENCE.SINHALA.ARCH = "seq2seq_a"
_C.SEQUENCE.SINHALA.NUM_CHAR = 120
_C.SEQUENCE.SINHALA.EMBED_SIZE = 30
_C.SEQUENCE.SINHALA.HIDDEN_SIZE = 256

_C.SEQUENCE.SPANISH = CN()
_C.SEQUENCE.SPANISH.ARCH = "seq2seq_a"
_C.SEQUENCE.SPANISH.NUM_CHAR = 120
_C.SEQUENCE.SPANISH.EMBED_SIZE = 30
_C.SEQUENCE.SPANISH.HIDDEN_SIZE = 192

_C.SEQUENCE.SYMBOL = CN()
_C.SEQUENCE.SYMBOL.ARCH = "seq2seq_a"
_C.SEQUENCE.SYMBOL.NUM_CHAR = 60
_C.SEQUENCE.SYMBOL.EMBED_SIZE = 30
_C.SEQUENCE.SYMBOL.HIDDEN_SIZE = 64

_C.SEQUENCE.TAGALOG = CN()
_C.SEQUENCE.TAGALOG.ARCH = "seq2seq_a"
_C.SEQUENCE.TAGALOG.NUM_CHAR = 110
_C.SEQUENCE.TAGALOG.EMBED_SIZE = 30
_C.SEQUENCE.TAGALOG.HIDDEN_SIZE = 192

_C.SEQUENCE.TAMIL = CN()
_C.SEQUENCE.TAMIL.ARCH = "seq2seq_a"
_C.SEQUENCE.TAMIL.NUM_CHAR = 110
_C.SEQUENCE.TAMIL.EMBED_SIZE = 30
_C.SEQUENCE.TAMIL.HIDDEN_SIZE = 256

_C.SEQUENCE.TELUGU = CN()
_C.SEQUENCE.TELUGU.ARCH = "seq2seq_a"
_C.SEQUENCE.TELUGU.NUM_CHAR = 130
_C.SEQUENCE.TELUGU.EMBED_SIZE = 30
_C.SEQUENCE.TELUGU.HIDDEN_SIZE = 192

_C.SEQUENCE.THAI = CN()
_C.SEQUENCE.THAI.ARCH = "seq2seq_a"
_C.SEQUENCE.THAI.NUM_CHAR = 210
_C.SEQUENCE.THAI.EMBED_SIZE = 30
_C.SEQUENCE.THAI.HIDDEN_SIZE = 192

_C.SEQUENCE.TURKISH = CN()
_C.SEQUENCE.TURKISH.ARCH = "seq2seq_a"
_C.SEQUENCE.TURKISH.NUM_CHAR = 120
_C.SEQUENCE.TURKISH.EMBED_SIZE = 30
_C.SEQUENCE.TURKISH.HIDDEN_SIZE = 192

_C.SEQUENCE.UNIFIEDAPU = CN()
_C.SEQUENCE.UNIFIEDAPU.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDAPU.NUM_CHAR = 250
_C.SEQUENCE.UNIFIEDAPU.EMBED_SIZE = 200
_C.SEQUENCE.UNIFIEDAPU.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDAPUE = CN()
_C.SEQUENCE.UNIFIEDAPUE.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDAPUE.NUM_CHAR = 300
_C.SEQUENCE.UNIFIEDAPUE.EMBED_SIZE = 250
_C.SEQUENCE.UNIFIEDAPUE.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDBGHMP = CN()
_C.SEQUENCE.UNIFIEDBGHMP.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDBGHMP.NUM_CHAR = 400
_C.SEQUENCE.UNIFIEDBGHMP.EMBED_SIZE = 270
_C.SEQUENCE.UNIFIEDBGHMP.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDBKT = CN()
_C.SEQUENCE.UNIFIEDBKT.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDBKT.NUM_CHAR = 400
_C.SEQUENCE.UNIFIEDBKT.EMBED_SIZE = 250
_C.SEQUENCE.UNIFIEDBKT.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDCG = CN()
_C.SEQUENCE.UNIFIEDCG.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDCG.NUM_CHAR = 260
_C.SEQUENCE.UNIFIEDCG.EMBED_SIZE = 220
_C.SEQUENCE.UNIFIEDCG.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDCGE = CN()
_C.SEQUENCE.UNIFIEDCGE.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDCGE.NUM_CHAR = 280
_C.SEQUENCE.UNIFIEDCGE.EMBED_SIZE = 220
_C.SEQUENCE.UNIFIEDCGE.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDCJ = CN()
_C.SEQUENCE.UNIFIEDCJ.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDCJ.NUM_CHAR = 9000
_C.SEQUENCE.UNIFIEDCJ.EMBED_SIZE = 2000
_C.SEQUENCE.UNIFIEDCJ.HIDDEN_SIZE = 320

_C.SEQUENCE.UNIFIEDCJE = CN()
_C.SEQUENCE.UNIFIEDCJE.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDCJE.NUM_CHAR = 9000
_C.SEQUENCE.UNIFIEDCJE.EMBED_SIZE = 2000
_C.SEQUENCE.UNIFIEDCJE.HIDDEN_SIZE = 320

_C.SEQUENCE.UNIFIEDCYRILLIC = CN()
_C.SEQUENCE.UNIFIEDCYRILLIC.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDCYRILLIC.NUM_CHAR = 180
_C.SEQUENCE.UNIFIEDCYRILLIC.EMBED_SIZE = 180
_C.SEQUENCE.UNIFIEDCYRILLIC.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDDEVANAGARI = CN()
_C.SEQUENCE.UNIFIEDDEVANAGARI.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDDEVANAGARI.NUM_CHAR = 180
_C.SEQUENCE.UNIFIEDDEVANAGARI.EMBED_SIZE = 180
_C.SEQUENCE.UNIFIEDDEVANAGARI.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDKE = CN()
_C.SEQUENCE.UNIFIEDKE.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDKE.NUM_CHAR = 2500
_C.SEQUENCE.UNIFIEDKE.EMBED_SIZE = 1500
_C.SEQUENCE.UNIFIEDKE.HIDDEN_SIZE = 320

_C.SEQUENCE.UNIFIEDKT = CN()
_C.SEQUENCE.UNIFIEDKT.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDKT.NUM_CHAR = 220
_C.SEQUENCE.UNIFIEDKT.EMBED_SIZE = 200
_C.SEQUENCE.UNIFIEDKT.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDLATIN1 = CN()
_C.SEQUENCE.UNIFIEDLATIN1.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDLATIN1.NUM_CHAR = 500
_C.SEQUENCE.UNIFIEDLATIN1.EMBED_SIZE = 270
_C.SEQUENCE.UNIFIEDLATIN1.HIDDEN_SIZE = 256

_C.SEQUENCE.UNIFIEDMST = CN()
_C.SEQUENCE.UNIFIEDMST.ARCH = "seq2seq_a"
_C.SEQUENCE.UNIFIEDMST.NUM_CHAR = 300
_C.SEQUENCE.UNIFIEDMST.EMBED_SIZE = 250
_C.SEQUENCE.UNIFIEDMST.HIDDEN_SIZE = 256

_C.SEQUENCE.URDU = CN()
_C.SEQUENCE.URDU.ARCH = "seq2seq_a"
_C.SEQUENCE.URDU.NUM_CHAR = 160
_C.SEQUENCE.URDU.EMBED_SIZE = 30
_C.SEQUENCE.URDU.HIDDEN_SIZE = 192

_C.SEQUENCE.VIETNAMESE = CN()
_C.SEQUENCE.VIETNAMESE.ARCH = "seq2seq_a"
_C.SEQUENCE.VIETNAMESE.NUM_CHAR = 260
_C.SEQUENCE.VIETNAMESE.EMBED_SIZE = 30
_C.SEQUENCE.VIETNAMESE.HIDDEN_SIZE = 256

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Used in SquareResize, s.t.
# min_size <= min(w,h) <= sqr_size
# sqr_size <= max(w,h) <= max_size
_C.INPUT.SQR_SIZE_TEST = 600
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1.0, 1.0, 1.0]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True
_C.INPUT.STRICT_RESIZE = False


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

_C.DATASETS.RATIOS = []

_C.DATASETS.AUG = False
_C.DATASETS.RANDOM_CROP_PROB = 0.0
_C.DATASETS.RANDOM_ROTATE_PROB = 0.5
_C.DATASETS.IGNORE_DIFFICULT = False
_C.DATASETS.FIX_CROP = False
_C.DATASETS.CROP_SIZE = (512, 512)
_C.DATASETS.MAX_ROTATE_THETA = 30
_C.DATASETS.FIX_ROTATE = False

_C.DATASETS.AUGMENTER = CN()
# Options for DATASETS.AUGMENTER.NAME and DATASETS.AUGMENTER.TEST
# ResizerV0: equivalent to DATASETS.AUG = False
# CropperV0: Fixed cropper, equivalent to DATASETS.FIX_ROTATE = True
# CropperV1: Crop first, then rotate (used in SPN)
# CropperV2: Rotate first, then crop (support keeping a subset of words)
# SquareResizerV0: used during test to simulate production transforms
_C.DATASETS.AUGMENTER.NAME = "ResizerV0"
_C.DATASETS.AUGMENTER.TEST = "ResizerV0"
# MIN_WIDTH_RATIO is used by CropperV2 only for now
_C.DATASETS.AUGMENTER.MIN_WIDTH_RATIO = 0.5
# MIN_HEIGHT_RATIO is used by CropperV2 only for now
_C.DATASETS.AUGMENTER.MIN_HEIGHT_RATIO = 0.5
# MIN_BOX_NUM_RATIO is used by CropperV2 only for now
_C.DATASETS.AUGMENTER.MIN_BOX_NUM_RATIO = 0.5


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# options: build_resnet_backbone, build_resnet_fpn_backbone
# here we use build_resnet_fpn_backbone as default for backward compatibility
_C.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
_C.MODEL.BACKBONE.FROZEN = False
_C.MODEL.BACKBONE.OUT_CHANNELS = 256 * 4

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1
_C.MODEL.RESNETS.LAYERS = (3, 4, 6, 3)

# ---------------------------------------------------------------------------- #
# FBNET options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.NUM_GROUPS = 32  # for gn usage only
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""

# number of channels input to trunk
_C.MODEL.FBNET.STEM_IN_CHANNELS = 3

# ---------------------------------------------------------------------------- #
# FBNET_V2 options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET_V2 = CN()

_C.MODEL.FBNET_V2.ARCH = "default"
_C.MODEL.FBNET_V2.ARCH_DEF = []
# number of channels input to trunk
_C.MODEL.FBNET_V2.STEM_IN_CHANNELS = 3
_C.MODEL.FBNET_V2.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET_V2.WIDTH_DIVISOR = 1

# normalization configs
# name of norm such as "bn", "sync_bn", "gn"
_C.MODEL.FBNET_V2.NORM = "bn"
# for advanced use case that requries extra arguments, passing a list of
# dict such as [{"num_groups": 8}, {"momentum": 0.1}] (merged in given order).
# Note that string written it in .yaml will be evaluated by yacs, thus this
# node will become normal python object.
# https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L410
_C.MODEL.FBNET_V2.NORM_ARGS = []

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False

# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
# for FBNet: ["trunk0", "trunk1", "trunk2", "trunk3"]
_C.MODEL.FPN.IN_FEATURES = []
_C.MODEL.FPN.OUT_CHANNELS = 256
_C.MODEL.FPN.USE_PRETRAINED = False

# Options: "" (no norm), "GN"
_C.MODEL.FPN.NORM = ""

# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.FPN.FUSE_TYPE = "sum"

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000

_C.MODEL.SEG = CN()
# Segmentation module name
_C.MODEL.SEG.NAME = "SEGModule"
# Segmentation loss. Options: Dice, BCE
_C.MODEL.SEG.LOSS = "Dice"
# Post processor name
_C.MODEL.SEG.POST_PROCESSOR = "SEGPostProcessor"
_C.MODEL.SEG.FROZEN = False
_C.MODEL.SEG.BN_FROZEN = False
_C.MODEL.SEG.USE_FPN = False
_C.MODEL.SEG.USE_FUSE_FEATURE = False
# Total number of SEG examples per image
_C.MODEL.SEG.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per SEG minibatch
_C.MODEL.SEG.POSITIVE_FRACTION = 0.5
# NMS threshold used on SEG proposals
_C.MODEL.SEG.BINARY_THRESH = 0.5
_C.MODEL.SEG.USE_MULTIPLE_THRESH = False
_C.MODEL.SEG.MULTIPLE_THRESH = (0.2, 0.3, 0.5, 0.7)
_C.MODEL.SEG.BOX_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.SEG.MIN_SIZE = 0
_C.MODEL.SEG.SHRINK_RATIO = 0.5
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.SEG.TOP_N_TRAIN = 1000
_C.MODEL.SEG.TOP_N_TEST = 1000
_C.MODEL.SEG.AUG_PROPOSALS = False
_C.MODEL.SEG.IGNORE_DIFFICULT = True
_C.MODEL.SEG.EXPAND_RATIO = 1.6
# Options: constant, log_a
_C.MODEL.SEG.EXPAND_METHOD = "constant"
_C.MODEL.SEG.BOX_EXPAND_RATIO = 1.5
_C.MODEL.SEG.USE_SEG_POLY = False
_C.MODEL.SEG.USE_PPM = False


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.NAME = "CombinedROIHead"
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
# _C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.0
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.NAME = "ROIBoxHead"
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POST_PROCESSOR = "BaseBoxPostProcessor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_BOX_HEAD.USE_REGRESSION = True
_C.MODEL.ROI_BOX_HEAD.INFERENCE_USE_BOX = True
_C.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE = False
_C.MODEL.ROI_BOX_HEAD.SOFT_MASKED_FEATURE_RATIO = 0.0
_C.MODEL.ROI_BOX_HEAD.MIX_OPTION = ""
_C.MODEL.ROI_BOX_HEAD.FROZEN = False


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.NAME = "BaseROIMaskHead"
# Options: RotatedROICropper, HorizontalROICropper
_C.MODEL.ROI_MASK_HEAD.ROI_CROPPER = "RotatedROICropper"
_C.MODEL.ROI_MASK_HEAD.CROPPER_RESOLUTION_H = 80
# CROPPER_RESOLUTION_W means max width for dynamic croppers and fixed width for static croppers
_C.MODEL.ROI_MASK_HEAD.CROPPER_RESOLUTION_W = 160
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR_FROZEN = False
_C.MODEL.ROI_MASK_HEAD.PREDICTOR_TRUNK_FROZEN = False
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POST_PROCESSOR = "MultiSeq1CharMaskPostProcessor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_H = 32
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_W = 128
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.CONV5_ARCH = "transpose_a"
_C.MODEL.ROI_MASK_HEAD.MASK_FCN_INPUT_DIM = 256
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.RESOLUTION_H = 32
_C.MODEL.ROI_MASK_HEAD.RESOLUTION_W = 128
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
_C.MODEL.ROI_MASK_HEAD.CHAR_NUM_CLASSES = 38
_C.MODEL.ROI_MASK_HEAD.USE_WEIGHTED_CHAR_MASK = False
_C.MODEL.ROI_MASK_HEAD.MASK_BATCH_SIZE_PER_IM = 64
_C.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE = False
_C.MODEL.ROI_MASK_HEAD.SOFT_MASKED_FEATURE_RATIO = 0.0
_C.MODEL.ROI_MASK_HEAD.MIX_OPTION = ""

_C.MODEL.LANGUAGE_HEAD = CN()
_C.MODEL.LANGUAGE_HEAD.FROZEN = False
_C.MODEL.LANGUAGE_HEAD.NUM_CLASSES = 2
_C.MODEL.LANGUAGE_HEAD.PREDICTOR = "V1LanguagePredictor"
_C.MODEL.LANGUAGE_HEAD.LOSS_WEIGHT = 1.0
_C.MODEL.LANGUAGE_HEAD.INPUT_C = 256
_C.MODEL.LANGUAGE_HEAD.INPUT_H = 40
_C.MODEL.LANGUAGE_HEAD.INPUT_W = 40
_C.MODEL.LANGUAGE_HEAD.CONV1_C = 64
_C.MODEL.LANGUAGE_HEAD.CONV2_C = 32

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

_C.SOLVER.RESUME = True

_C.SOLVER.USE_ADAM = False

_C.SOLVER.POW_SCHEDULE = False

_C.SOLVER.DISPLAY_FREQ = 20

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.VIS = False
# from 0 to 255
_C.TEST.CHAR_THRESH = 128

# Test-time augmentation, or multi-scale testing
# Run inference at different sizes and merge at eval time with NMS
_C.TEST.BBOX_AUG = CN()
# Enable test-time augmentation if True
_C.TEST.BBOX_AUG.ENABLED = False
# Min size of different input sizes to test
_C.TEST.BBOX_AUG.MIN_SIZE = (800, 1000, 1200, 1400)

# Options: python/cpp, default to python for backward compatibility
_C.TEST.MASK2POLYGON_OP = "python"

# Torchscript-related options at test time
_C.TEST.TORCHSCRIPT = CN()
# When TEST.TORCHSCRIPT is True, the
_C.TEST.TORCHSCRIPT.ENABLED = False
# When TEST.TORCHSCRIPT.ENABLED is True,
# use TEST.TORCHSCRIPT.WEIGHT instead of MODEL.WEIGHT for inferencing
_C.TEST.TORCHSCRIPT.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")


# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #

_C.OUTPUT = CN()
_C.OUTPUT.TMP_FOLDER = "/tmp/multiplexer"
_C.OUTPUT.ON_THE_FLY = True
_C.OUTPUT.FB_COCO = CN()
_C.OUTPUT.FB_COCO.EVAL = False
_C.OUTPUT.FB_COCO.DET_THRESH = 0.2
_C.OUTPUT.FB_COCO.SEQ_THRESH = 0.0
_C.OUTPUT.ICDAR15 = CN()
_C.OUTPUT.ICDAR15.TASK1 = False
_C.OUTPUT.ICDAR15.TASK4 = False
_C.OUTPUT.ICDAR15.INTERMEDIATE = False
_C.OUTPUT.MLT17 = CN()
_C.OUTPUT.MLT17.TASK1 = False
_C.OUTPUT.MLT17.TASK3 = False
_C.OUTPUT.MLT19 = CN()
_C.OUTPUT.MLT19.TASK1 = False
_C.OUTPUT.MLT19.TASK3 = False
_C.OUTPUT.MLT19.TASK4 = False
_C.OUTPUT.MLT19.INTERMEDIATE = False
# when MLT19.INTERMEDIATE_WITH_PKL is True, pkl files will be saved,
# which can be used in weighted ed-dist when using lexicon
_C.OUTPUT.MLT19.INTERMEDIATE_WITH_PKL = False
_C.OUTPUT.MLT19.DET_THRESH = CN()
_C.OUTPUT.MLT19.DET_THRESH.TASK4 = 0.2
_C.OUTPUT.MLT19.SEQ_THRESH = CN()
_C.OUTPUT.MLT19.SEQ_THRESH.TASK4 = 0.8
_C.OUTPUT.MLT19.LEXICON = CN()
_C.OUTPUT.MLT19.LEXICON.NAME = "none"
_C.OUTPUT.MLT19.LEXICON.EDIT_DIST_THRESH = 0.5
# when MLT19.VALIDATION_EVAL is True, perform evaluation on validation set;
# for test set, set this to be False since we don't have ground truth for MLT19 test set
_C.OUTPUT.MLT19.VALIDATION_EVAL = True
_C.OUTPUT.TOTAL_TEXT = CN()
_C.OUTPUT.TOTAL_TEXT.DET_EVAL = False
_C.OUTPUT.TOTAL_TEXT.E2E_EVAL = False
_C.OUTPUT.TOTAL_TEXT.INTERMEDIATE = False

_C.OUTPUT.SEG_VIS = False
# whether we create a zip file per GPU (useful for intermediate modes with pkl files)
_C.OUTPUT.ZIP_PER_GPU = False
