# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .icdar15.task4.script import icdar15_eval_task4
from .mlt19 import mlt19_eval_task1, mlt19_eval_task3, mlt19_eval_task4
from .output import (
    output_fb_coco_class_format,
    output_icdar15,
    output_icdar15_intermediate,
    output_mlt17,
    output_mlt19,
    output_mlt19_intermediate,
    output_total_text_det,
    output_total_text_e2e,
    output_total_text_intermediate,
)
from .total_text import total_text_eval_det, total_text_eval_e2e
