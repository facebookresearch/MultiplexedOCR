import getpass
import json
import logging
import math
import zipfile
from functools import partial

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from multiplexer.data import transforms as T
from multiplexer.engine.deprecated import creat_color_map, render_char_mask
from multiplexer.evaluation import (
    icdar15_eval_task4,
    mlt19_eval_task1,
    mlt19_eval_task3,
    mlt19_eval_task4,
    output_icdar15,
    output_icdar15_intermediate,
    output_mlt17,
    output_mlt19,
    output_mlt19_intermediate,
    output_total_text_det,
    output_total_text_e2e,
    output_total_text_intermediate,
    total_text_eval_det,
    total_text_eval_e2e,
)
from multiplexer.structures.image_list import to_image_list
from multiplexer.structures.word_result import WordResult
from multiplexer.utils.comm import get_rank, is_main_process, scatter_gather, synchronize
from multiplexer.utils.languages import cyrillic_greek_to_latin, lang_code_to_char_map_class
from multiplexer.utils.vocabulary import Vocabulary
from virtual_fs import virtual_os as os
from virtual_fs import virtual_shutil as shutil

logger = logging.getLogger(__name__)


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    return predictions


def _accumulate_fb_coco_json_from_multiple_gpus(pred_json_per_gpu):
    all_pred_jsons = scatter_gather(pred_json_per_gpu)
    if not is_main_process():
        return
    # merge the list of pred jsons, each has 3 main dicts: imgs, imgToAnns, anns
    merged_pred_json = None
    for pred_json in all_pred_jsons:
        if merged_pred_json is None:
            merged_pred_json = pred_json
        else:
            merged_pred_json["imgs"].update(pred_json["imgs"])
            merged_pred_json["imgToAnns"].update(pred_json["imgToAnns"])
            merged_pred_json["anns"].update(pred_json["anns"])
    return merged_pred_json


def _accumulate_txt_lists_from_multiple_gpus(txt_lists_per_gpu):
    all_txt_lists = scatter_gather(txt_lists_per_gpu)
    if not is_main_process():
        return
    # merge the list of text lists
    txt_lists = []
    for lst in all_txt_lists:
        txt_lists.extend(lst)
    return txt_lists


def append_txt_to_zip(zip_file, txt_file):
    with zipfile.ZipFile(zip_file, "a") as zipf:
        zipf.write(txt_file, os.path.basename(txt_file))


# For each acive task, create paths for txt results, zip file and file list
# and specify functions to output and evaluate
def build_task_utils(cfg, output_root, model_name="model"):
    # set up active task keys and descriptions based on config
    active_tasks = {}
    if cfg.OUTPUT.ICDAR15.TASK1:
        active_tasks["icdar15_task1"] = "ICDAR15 Task 1"
    if cfg.OUTPUT.ICDAR15.TASK4:
        active_tasks["icdar15_task4_generic"] = "ICDAR15 Task 4 Generic Lexicon"
        active_tasks["icdar15_task4_weak"] = "ICDAR15 Task 4 Weak Lexicon"
        active_tasks["icdar15_task4_strong"] = "ICDAR15 Task 4 Strong Lexicon"
    if cfg.OUTPUT.ICDAR15.INTERMEDIATE:
        active_tasks["icdar15_intermediate"] = "ICDAR15 Intermediate"
    if cfg.OUTPUT.MLT17.TASK1:
        active_tasks["mlt17_task1"] = "MLT17 Task 1"
    if cfg.OUTPUT.MLT17.TASK3:
        active_tasks["mlt17_task3"] = "MLT17 Task 3"
    if cfg.OUTPUT.MLT19.TASK1:
        active_tasks["mlt19_task1"] = "MLT19 Task 1"
    if cfg.OUTPUT.MLT19.TASK3:
        active_tasks["mlt19_task3"] = "MLT19 Task 3"
    if cfg.OUTPUT.MLT19.TASK4:
        active_tasks["mlt19_task4"] = "MLT19 Task 4"
    if cfg.OUTPUT.MLT19.INTERMEDIATE:
        active_tasks["mlt19_intermediate"] = "MLT19 Intermediate"
    if cfg.OUTPUT.TOTAL_TEXT.DET_EVAL:
        active_tasks["total_text_det"] = "Total Text Detection"
    if cfg.OUTPUT.TOTAL_TEXT.E2E_EVAL:
        active_tasks["total_text_e2e"] = "Total Text End-to-End"
    if cfg.OUTPUT.TOTAL_TEXT.INTERMEDIATE:
        active_tasks["total_text_intermediate"] = "Total Text Intermediate"

    # set up basic paths for each active tasks
    task_utils = {}
    for task_key, task_name in active_tasks.items():
        task_utils[task_key] = {}
        task_utils[task_key]["name"] = task_name
        task_utils[task_key]["dir"] = os.path.join(
            output_root, model_name + "_{}_results".format(task_key)
        )
        os.makedirs(task_utils[task_key]["dir"], exist_ok=True)

        if not cfg.OUTPUT.ZIP_PER_GPU:
            task_utils[task_key]["zip"] = os.path.join(
                output_root, "{}_{}.zip".format(model_name, task_key)
            )
        else:
            # if zip separately for each gpu
            task_utils[task_key]["zip"] = os.path.join(
                output_root, "{}_{}_part{}.zip".format(model_name, task_key, get_rank())
            )
        task_utils[task_key]["file_list"] = []
        task_utils[task_key]["output_func"] = None
        task_utils[task_key]["eval_func"] = None

    # customize output and eval functions for each active tasks
    if cfg.OUTPUT.ICDAR15.TASK1:
        task_utils["icdar15_task1"]["output_func"] = partial(
            output_icdar15, task=1, vocabulary=None, det_conf_thresh=0.01
        )
    # NOTE: icdar15 task4 output and eval are both dealt with differently
    if cfg.OUTPUT.ICDAR15.INTERMEDIATE:
        task_utils["icdar15_intermediate"]["output_func"] = partial(
            output_icdar15_intermediate, det_conf_thresh=0, seq_conf_thresh=0
        )
    if cfg.OUTPUT.MLT17.TASK1:
        task_utils["mlt17_task1"]["output_func"] = partial(
            output_mlt17,
            task=1,
            det_conf_thresh=0.3,
            seq_conf_thresh=0,
        )
    if cfg.OUTPUT.MLT17.TASK3:
        task_utils["mlt17_task3"]["output_func"] = partial(
            output_mlt17, task=3, det_conf_thresh=0.3, seq_conf_thresh=0
        )
    if cfg.OUTPUT.MLT19.TASK1:
        task_utils["mlt19_task1"]["output_func"] = partial(
            output_mlt19, task=1, det_conf_thresh=0.3, seq_conf_thresh=0
        )
        # Note: for MLT19 test set, set VALIDATION_EVAL=False
        # since we don't have gt annotations.
        if cfg.OUTPUT.MLT19.VALIDATION_EVAL:
            task_utils["mlt19_task1"]["eval_func"] = mlt19_eval_task1
    if cfg.OUTPUT.MLT19.TASK3:
        task_utils["mlt19_task3"]["output_func"] = partial(
            output_mlt19, task=3, det_conf_thresh=0.3, seq_conf_thresh=0
        )
        if cfg.OUTPUT.MLT19.VALIDATION_EVAL:
            task_utils["mlt19_task3"]["eval_func"] = mlt19_eval_task3
    if cfg.OUTPUT.MLT19.TASK4:
        if cfg.OUTPUT.MLT19.LEXICON.NAME == "none":
            task_utils["mlt19_task4"]["output_func"] = partial(
                output_mlt19,
                task=4,
                det_conf_thresh=cfg.OUTPUT.MLT19.DET_THRESH.TASK4,
                seq_conf_thresh=cfg.OUTPUT.MLT19.SEQ_THRESH.TASK4,
            )
        else:
            # lang_list = {"ar", "bn", "hi", "ja", "ko", "la", "zh", "symbol"}
            lang_list = {"la"}
            vocabularies_list = {}
            char_map_class_list = {}
            lexicon = cfg.OUTPUT.MLT19.LEXICON.NAME
            for language in lang_list:
                lexicon_path = (
                    f"/checkpoint/{getpass.getuser()}/"
                    f"datasets/MLT19/eval/lexicons/{lexicon}/{language}.txt"
                )

                with open(lexicon_path, "r") as lexicon_fid:
                    vocabularies_list[language] = [voc.strip() for voc in lexicon_fid.readlines()]

                print(
                    "[Info] Loaded {} words for lexicon {}".format(
                        len(vocabularies_list[language]), language
                    )
                )

                char_map_class_list[language] = lang_code_to_char_map_class[language]
                char_map_class_list[language].init(char_map_path=cfg.CHAR_MAP.DIR)

            if "ar" in vocabularies_list:
                # reverse words in Arabic
                for i in range(len(vocabularies_list["ar"])):
                    vocabularies_list["ar"][i] = vocabularies_list["ar"][i][::-1]

            task_utils["mlt19_task4"]["output_func"] = partial(
                output_mlt19,
                task=4,
                det_conf_thresh=cfg.OUTPUT.MLT19.DET_THRESH.TASK4,
                seq_conf_thresh=cfg.OUTPUT.MLT19.SEQ_THRESH.TASK4,
                lexicon=lexicon,
                vocabularies_list=vocabularies_list,
                char_map_class_list=char_map_class_list,
                edit_dist_thresh=cfg.OUTPUT.MLT19.LEXICON.EDIT_DIST_THRESH,
            )
        if cfg.OUTPUT.MLT19.VALIDATION_EVAL:
            task_utils["mlt19_task4"]["eval_func"] = mlt19_eval_task4
    if cfg.OUTPUT.MLT19.INTERMEDIATE:
        task_utils["mlt19_intermediate"]["output_func"] = partial(
            output_mlt19_intermediate,
            det_conf_thresh=0.001,
            seq_conf_thresh=0,
            save_pkl=cfg.OUTPUT.MLT19.INTERMEDIATE_WITH_PKL,
        )
    if cfg.OUTPUT.TOTAL_TEXT.DET_EVAL:
        task_utils["total_text_det"]["output_func"] = partial(
            output_total_text_det, det_conf_thresh=0.05, seq_conf_thresh=0.5
        )
    if cfg.OUTPUT.TOTAL_TEXT.E2E_EVAL:
        task_utils["total_text_e2e"]["output_func"] = partial(
            output_total_text_e2e, det_conf_thresh=0.05, seq_conf_thresh=0.9
        )
        task_utils["total_text_e2e"]["eval_func"] = total_text_eval_e2e
    if cfg.OUTPUT.TOTAL_TEXT.INTERMEDIATE:
        task_utils["total_text_intermediate"]["output_func"] = partial(
            output_total_text_intermediate, det_conf_thresh=0, seq_conf_thresh=0
        )

    return task_utils


def compute_result_logs(
    prediction_dict,
    cfg,
    img,
    polygon_format,
    use_seg_poly,
):
    # polygons = []

    width, height = img.size

    word_result_list = prediction_dict["word_result_list"]
    scores = prediction_dict["scores"]
    rotated_boxes_5d = prediction_dict["rotated_boxes_5d"]
    final_word_result_list = []

    for k, box in enumerate(prediction_dict["boxes"]):
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue

        box = list(map(int, box))
        if not use_seg_poly:
            mask = prediction_dict["masks"][k, 0, :, :]
            m2p = mask2polygon
            # m2p = (
            #     mask2polygon_cpp_op
            #     if cfg.TEST.MASK2POLYGON_OP == "cpp"
            #     else mask2polygon
            # )
            polygon = m2p(
                mask,
                box,
                img.size,
                threshold=0.5,
                polygon_format=polygon_format,
            )
        else:
            polygon = list(prediction_dict["masks"][k].get_polygons()[0].cpu().numpy())
            if polygon_format == "rect":
                polygon = polygon2rbox(polygon, height, width)

        if polygon is None:
            continue

        # polygons.append(polygon)

        if cfg.MODEL.ROI_BOX_HEAD.INFERENCE_USE_BOX:
            word_result_list[k].det_score = scores[k]
        else:
            word_result_list[k].det_score = 1.0

        word_result_list[k].box = [int(x * 1.0) for x in box[:4]]
        if len(polygon) != 8:
            if len(polygon) < 8:
                logger.warning("Polygon {} has fewer than 4 points!".format(polygon))
            rbox = polygon2rbox(polygon, height, width)
            word_result_list[k].rotated_box = rbox
            msg = "Polygon {} is also saved in rbox: {}".format(polygon, rbox)
            logger.info(msg)
        word_result_list[k].polygon = polygon

        if rotated_boxes_5d is None:
            poly = np.array(polygon).reshape((-1, 2))
            rect = cv2.minAreaRect(poly)
            x, y, w, h, a = rect[0][0], rect[0][1], rect[1][0], rect[1][1], -rect[2]
            word_result_list[k].rotated_box_5d = [x, y, w, h, a]
        else:
            word_result_list[k].rotated_box_5d = rotated_boxes_5d[k].tolist()

        final_word_result_list.append(word_result_list[k])

    return {
        "result_logs": final_word_result_list,
    }


def compute_result_logs_tensor(results, cfg, img, polygon_format, input_size):
    if len(results) == 13:
        # GeneralizedRCNN - torchscript
        (
            boxes,
            polygon_pts,
            polygon_len,
            box_scores,
            final_boxes,
            final_box_scores,
            seq_words_val,
            seq_words_len,
            seq_scores,
            language_ids,
            language_scores,
            head_ids,
            head_scores,
        ) = results
    elif len(results) == 10:
        # CroppedRCNN - torchscript
        (
            boxes,
            box_scores,
            rotated_boxes,
            seq_words_val,
            seq_words_len,
            seq_scores,
            language_ids,
            language_scores,
            head_ids,
            head_scores,
        ) = results
        has_angle_info = True
        # compute polygon_pts and polygon_len from the rotated_boxes for accurate rescaling
        if len(rotated_boxes) > 0:
            cnt_x = rotated_boxes[..., 0]
            cnt_y = rotated_boxes[..., 1]
            half_w = rotated_boxes[..., 2] / 2.0
            half_h = rotated_boxes[..., 3] / 2.0
            theta = rotated_boxes[..., 4] * math.pi / 180.0
            c = torch.cos(theta)
            s = torch.sin(theta)

            # calculate the polygons (TODO: remove when not used)
            sxh = s * half_h
            sxw = s * half_w
            cxh = c * half_h
            cxw = c * half_w

            # n x 8
            polygon_pts = torch.vstack(
                [
                    cnt_x + sxh - cxw,
                    cnt_y + cxh + sxw,
                    cnt_x - sxh - cxw,
                    cnt_y - cxh + sxw,
                    cnt_x - sxh + cxw,
                    cnt_y - cxh - sxw,
                    cnt_x + sxh + cxw,
                    cnt_y + cxh - sxw,
                ]
            ).transpose(0, 1)

            polygon_len = [1 for _ in range(len(rotated_boxes))]
        else:
            polygon_pts = []
            polygon_len = []
    else:
        raise ValueError(f"Unknown results format (len(results) = {len(results)}): {results}")

    n = len(polygon_len)
    assert boxes.shape[0] == n, f"number of polygon = {n} != {boxes.shape[0]} = number of boxes!"

    width, height = img.size
    input_width, input_height = input_size
    scale_width = (width * 1.0) / input_width
    scale_height = (height * 1.0) / input_height

    # word_result_list = prediction_dict["word_result_list"]
    # scores = prediction_dict["scores"]
    # rotated_boxes_5d = prediction_dict["rotated_boxes_5d"]
    final_word_result_list = []

    polygon_pts_start = 0
    polygon_pts_end = 0
    seq_words_val_start = 0
    seq_words_val_end = 0

    latin_count = 0
    cg_count = 0
    confident_count = 0

    for i in range(n):
        box = boxes[i]

        polygon_pts_start = polygon_pts_end
        polygon_pts_end += int(polygon_len[i])
        seq_words_val_start = seq_words_val_end
        seq_words_val_end += int(seq_words_len[i])

        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue

        box = list(map(int, box))

        # if not use_seg_poly:
        #     mask = prediction_dict["masks"][k, 0, :, :]
        #     polygon = mask2polygon_cpp_op(
        #         mask,
        #         box,
        #         img.size,
        #         threshold=0.5,
        #         polygon_format=polygon_format,
        #     )
        # else:
        #     polygon = list(prediction_dict["masks"][k].get_polygons()[0].cpu().numpy())
        #     if polygon_format == "rect":
        #         polygon = polygon2rbox(polygon, height, width)

        polygon = polygon_pts[polygon_pts_start:polygon_pts_end].flatten().numpy()
        word_code = seq_words_val[seq_words_val_start:seq_words_val_end]

        assert polygon is not None

        word_result = WordResult()
        word_result.language_id = int(language_ids[i])
        word_result.language_prob = language_scores[i].item()
        word_result.language = cfg.SEQUENCE.LANGUAGES[word_result.language_id]
        word_result.language_id_enabled = int(head_ids[i])
        word_result.language_enabled = cfg.SEQUENCE.LANGUAGES_ENABLED[
            word_result.language_id_enabled
        ]
        word_result.language_prob_enabled = head_scores[i].item()

        word_result.seq_word = decode_word(word_code, word_result.language_enabled, cfg)
        # convert CTC loss to confidence - should only apply to the CTC-based heads!
        word_result.seq_score = min(max(1.0 - seq_scores[i].item() / 10.0, 0.0), 1.0)
        word_result.det_score = box_scores[i].item()

        if word_result.det_score > 0.5 and word_result.seq_score > 0.7:
            confident_count += 1
            if word_result.language_enabled == "u_la1":
                latin_count += 1
            elif word_result.language_enabled == "u_cg":
                cg_count += 1

        word_result.box = []
        for j, x in enumerate(box):
            scale = scale_width if j % 2 == 0 else scale_height
            word_result.box.append(int(x * scale))

        word_result.polygon = []
        for j, x in enumerate(polygon):
            scale = scale_width if j % 2 == 0 else scale_height
            word_result.polygon.append(int(x * scale))

        if len(polygon) != 8:
            if len(polygon) < 8:
                print("Polygon {} has fewer than 4 points!".format(polygon))
            rbox = polygon2rbox(word_result.polygon, height, width)
            word_result.rotated_box = rbox
        # print(f"Polygon {polygon} is also saved in rbox: {rbox}")

        poly = np.array(word_result.polygon).reshape((-1, 2))
        rect = cv2.minAreaRect(poly)

        x, y, w, h, a = rect[0][0], rect[0][1], rect[1][0], rect[1][1], -rect[2]

        if has_angle_info:
            # for CroppedRCNN, the final angle could be more accurately estimated based on rotated_boxes[..., 4]
            # angle_diff intevals
            # a' - a                          : [-45, 45], [45, 135], [135, 225], [225, 315]
            # (a' - a + 45) % 360             : [0, 90], [90, 180], [180, 270], [270, 360]
            # ((a' - a + 45) % 360 // 90) % 4 : 0, 1, 2, 3
            # 0: correct, 1: a += 90, swap w and h, 2: a += 180, 3: a += 270, swap w and h
            angle_diff_index = (((rotated_boxes[i][4].item() - a + 45) % 360) // 90) % 4
            # transform and make sure the range is [-90, 90]
            a = (a + angle_diff_index * 90 + 180) % 360 - 180
            if angle_diff_index % 2 == 1:
                w, h = h, w

        word_result.rotated_box_5d = [x, y, w, h, a]

        final_word_result_list.append(word_result)

    if confident_count >= 10:
        dominating_thresh = 0.7 * confident_count
        if latin_count > dominating_thresh:
            for word_result in final_word_result_list:
                if word_result.language_enabled == "u_cg":
                    word_result.original_word = word_result.seq_word
                    word_result.seq_word = cyrillic_greek_to_latin(word_result.seq_word)
                    if word_result.seq_word != word_result.original_word:
                        print(
                            f"[Warning] Auto-corrected {word_result.original_word} to {word_result.seq_word}"
                        )
                    else:
                        print(f"[Warning] Unable to correct {word_result.original_word} to Latin")

    return {
        "result_logs": final_word_result_list,
    }


def decode_word(word_code, lang_code, cfg):
    if lang_code in char_map_class_dict:
        char_map_class = char_map_class_dict[lang_code]
    else:
        char_map_class = lang_code_to_char_map_class[lang_code]
        char_map_class.init(char_map_path=cfg.CHAR_MAP.DIR)
        char_map_class_dict[lang_code] = char_map_class

    word = ""
    for k in word_code:
        word += char_map_class.num2char(int(k))
    return word


def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y
    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]


def inference(
    model,
    data_loader,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    model_name=None,
    cfg=None,
):
    assert cfg.OUTPUT.ON_THE_FLY
    inference_on_the_fly(
        model=model,
        data_loader=data_loader,
        device=device,
        output_folder=output_folder,
        model_name=model_name,
        cfg=cfg,
    )


def inference_on_the_fly(
    model, data_loader, device="cuda", output_folder=None, model_name=None, cfg=None
):
    if cfg.TEST.TORCHSCRIPT.ENABLED:
        full_model_output_path = cfg.TEST.TORCHSCRIPT.WEIGHT

        logger.info(f"Using torchscript model {full_model_output_path} for inferencing!")
        with open(full_model_output_path, "rb") as buffer:
            model = torch.jit.load(buffer)

        device = "cpu"

    model_name = model_name.split(".")[0]
    device = torch.device(device)

    # all text results will be saved to temp folder, then zip and copy to output_folder
    if cfg.OUTPUT.TMP_FOLDER is None or cfg.OUTPUT.TMP_FOLDER == "":
        tmp_folder = "/tmp/multiplexer"
    else:
        tmp_folder = cfg.OUTPUT.TMP_FOLDER
    if not os.path.isdir(tmp_folder):
        logger.info("Creating temp folder to save txt results: {}".format(tmp_folder))
        os.makedirs(tmp_folder, exist_ok=True)

    task_utils = build_task_utils(cfg, output_root=tmp_folder, model_name=model_name)

    # special setup needed for icdar15 task4 lexicons
    # later we're adding lexicons for other tasks we could merge them in task_utils
    if cfg.OUTPUT.ICDAR15.TASK4:
        voc_path = f"/checkpoint/{getpass.getuser()}/datasets/ICDAR15/test/vocabulary/"
        vocabularies = {
            "weak": Vocabulary(os.path.join(voc_path, "weak", "ch4_test_vocabulary.txt")),
            "generic": Vocabulary(os.path.join(voc_path, "generic", "generic_vocabulary.txt")),
        }
        icdar15_seq_conf_thresh_list = {"generic": 0.8, "weak": 0.7, "strong": 0.7}

    if cfg.OUTPUT.FB_COCO.EVAL:
        fb_coco_dir = os.path.join(output_folder, model_name + "_fb_coco_results")
        os.makedirs(fb_coco_dir, exist_ok=True)

        # assert isinstance(data_loader.dataset, IcdarCocoDataset)

        ann_file = data_loader.dataset.ann_file

        with open(ann_file, "r", encoding="utf-8") as f:
            fb_coco_json_ann_gt = json.load(f)

        # compute everstore-filename-to-image-id map
        # note that one everstore handle could correspond to multiple image ids
        # due to multi-annotation!
        everstore_to_id_map = {}
        for img_id in fb_coco_json_ann_gt["imgs"]:
            if fb_coco_json_ann_gt["imgs"][img_id]["file_name"] not in everstore_to_id_map:
                everstore_to_id_map[fb_coco_json_ann_gt["imgs"][img_id]["file_name"]] = []
            everstore_to_id_map[fb_coco_json_ann_gt["imgs"][img_id]["file_name"]].append(img_id)

        # make a duplication of gt json, but delete the gt annotation info and keep the static info
        fb_coco_json_ann_pred = fb_coco_json_ann_gt.copy()
        fb_coco_json_ann_pred["imgs"] = {}
        fb_coco_json_ann_pred["anns"] = {}
        fb_coco_json_ann_pred["imgToAnns"] = {}

    # results_dir = os.path.join(output_folder, model_name + "_results")
    # os.makedirs(results_dir, exist_ok=True)
    # seg_results_dir = os.path.join(output_folder, model_name + "_seg_results")
    # os.makedirs(seg_results_dir, exist_ok=True)

    if cfg.OUTPUT.SEG_VIS:
        seg_visu_dir = os.path.join(output_folder, model_name + "_seg_visu")
        os.makedirs(seg_visu_dir, exist_ok=True)

    if cfg.TEST.VIS:
        if cfg.MODEL.CHAR_MASK_ON:
            vis_char_mask_dir = os.path.join(output_folder, model_name + "_char_mask")
            os.makedirs(vis_char_mask_dir, exist_ok=True)

        vis_box_text_dir = os.path.join(output_folder, model_name + "_box_text")
        os.makedirs(vis_box_text_dir, exist_ok=True)

    if cfg.SEQUENCE.LANGUAGES_ENABLED == cfg.SEQUENCE.LANGUAGES:
        enabled_all_rec_heads = True
        rec_head_map = None
    else:
        # Note LANGUAGES_ENABLED must be a subset of LANGUAGES
        enabled_all_rec_heads = False
        rec_head_map = {}
        for rec_id, language_rec in enumerate(cfg.SEQUENCE.LANGUAGES_ENABLED):
            for pred_id, language_pred in enumerate(cfg.SEQUENCE.LANGUAGES):
                if language_rec == language_pred:
                    rec_head_map[rec_id] = pred_id
                    break

    # Enter eval mode
    model.eval()

    if cfg.OUTPUT.FB_COCO.EVAL:
        allow_image_loading_failure = True
    else:
        allow_image_loading_failure = False

    num_images = len(data_loader)

    rank = get_rank()

    for i, batch in tqdm(enumerate(data_loader)):
        raw_images, raw_targets, image_paths = batch

        image_path = image_paths[0]

        if len(raw_images) == 1:
            # logger can only print logs from main process (rank = 0), therefore we use print for now
            print("[{}/{}][rank: {}] Inferencing on {}".format(i + 1, num_images, rank, image_path))
        else:
            # Not supported actually
            logger.info("Computing batch {} with {} images".format(i + 1, len(raw_images)))

        if cfg.TEST.BBOX_AUG.ENABLED:
            test_sizes = list(cfg.TEST.BBOX_AUG.MIN_SIZE)
        else:
            test_sizes = [cfg.INPUT.MIN_SIZE_TEST]

        im_name = image_path.split("/")[-1]

        all_result_logs = []
        image_not_loaded = False
        for min_size in test_sizes:
            if cfg.TEST.BBOX_AUG.ENABLED:
                logger.info("Processing min size {}...".format(min_size))

                # if bbox aug is enabled, resize images to different min size
                # targets is not needed in test, but resized it anyway in case needed in future
                transform = T.Compose(
                    [
                        T.Resize(min_size, cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.STRICT_RESIZE),
                        T.ToTensor(),
                        T.Normalize(
                            mean=cfg.INPUT.PIXEL_MEAN,
                            std=cfg.INPUT.PIXEL_STD,
                            to_bgr255=cfg.INPUT.TO_BGR255,
                        ),
                    ]
                )

                transformed = [
                    transform(image, target) for image, target in zip(raw_images, raw_targets)
                ]
                images, targets = list(zip(*transformed))
                images = to_image_list(images, cfg.DATALOADER.SIZE_DIVISIBILITY)
            else:
                images, targets = raw_images, raw_targets

            images = images.to(device)

            img = load_image(image_path, allow_failure=allow_image_loading_failure)
            if img is None:
                image_not_loaded = True
                break
            width, height = img.size

            if cfg.TEST.TORCHSCRIPT.ENABLED:
                with torch.no_grad():
                    results = model(images.tensors)

                # Note: images.get_sizes()[0] can be different from
                # (images.tensors.shape[3], images.tensors.shape[2])!
                test_image_height, test_image_width = images.get_sizes()[0]

                result_logs_dict = compute_result_logs_tensor(
                    results,
                    cfg,
                    img,
                    polygon_format="polygon",
                    input_size=(test_image_width, test_image_height),
                )

                resize_ratio = float(height) / test_image_height
                prediction_dict = {}  # dummy
            else:
                prediction_dict = model(images)

                global_prediction = prediction_dict["global_prediction"][0]
                test_image_width, test_image_height = global_prediction.size

                resize_ratio = float(height) / test_image_height
                global_prediction = global_prediction.resize((width, height))
                if cfg.MODEL.SEG.POST_PROCESSOR == "RotatedSEGPostProcessor":
                    assert global_prediction.has_field("rotated_boxes_5d"), "{}".format(
                        global_prediction.fields()
                    )
                if global_prediction.has_field("rotated_boxes_5d"):
                    rotated_boxes_5d = global_prediction.get_field("rotated_boxes_5d")
                    rotated_boxes_5d = rotated_boxes_5d.scale(
                        float(width) / test_image_width,
                        float(height) / test_image_height,
                    )
                    prediction_dict["rotated_boxes_5d"] = rotated_boxes_5d
                else:
                    prediction_dict["rotated_boxes_5d"] = None

                prediction_dict["boxes"] = global_prediction.bbox.tolist()

                if cfg.MODEL.TRAIN_DETECTION_ONLY:
                    use_seg_poly = True
                else:
                    if cfg.MODEL.ROI_BOX_HEAD.INFERENCE_USE_BOX:
                        prediction_dict["scores"] = global_prediction.get_field("scores").tolist()
                    use_seg_poly = cfg.MODEL.SEG.USE_SEG_POLY

                if not use_seg_poly:
                    prediction_dict["masks"] = global_prediction.get_field("mask").cpu().numpy()
                else:
                    prediction_dict["masks"] = global_prediction.get_field("masks").get_polygons()

                polygon_format = (
                    "polygon"
                    if ("total_text" in output_folder or "cute80" in output_folder)
                    else "rect"
                )
                if cfg.TEST.VIS:
                    polygon_format = "polygon"
                result_logs_dict = compute_result_logs(
                    prediction_dict=prediction_dict,
                    cfg=cfg,
                    img=img,
                    polygon_format=polygon_format,
                    use_seg_poly=use_seg_poly,
                )

            all_result_logs.extend(result_logs_dict["result_logs"])

            if cfg.TEST.VIS:
                if cfg.MODEL.CHAR_MASK_ON:
                    logger.info("Rendering char_mask_{} ...".format(im_name))
                    colors = creat_color_map(37, 255)
                    img_char_mask = img.copy()
                    render_char_mask(
                        img_char_mask,
                        result_logs_dict["polygons"],
                        resize_ratio,
                        colors,
                        prediction_dict["char_mask_result"]["char_polygons"],
                        prediction_dict["char_mask_result"]["words"],
                    )
                    img_char_mask.save(os.path.join(vis_char_mask_dir, "char_mask_" + im_name))

                vis_img_name = os.path.join(
                    vis_box_text_dir, f"box_text_minsize{min_size}_{im_name}"
                )

                logger.info(f"Rendering {vis_img_name}")

                render_box_multi_text(
                    cfg=cfg,
                    image=img,
                    result_logs_dict=result_logs_dict,
                    resize_ratio=resize_ratio,
                    language_probs=prediction_dict["language_probs"]
                    if "language_probs" in prediction_dict
                    else None,
                )

                with open(vis_img_name, "wb") as buffer:
                    img.save(buffer, format="PNG")

        if image_not_loaded:
            print(f"[{i+1}/{num_images}][rank: {rank}] Image not loaded")
            continue
        result_logs = all_result_logs

        # save inference per image to txt files for each active task
        for task in task_utils.values():
            if task["output_func"] is not None:
                task["output_func"](
                    task["dir"], result_logs, im_name, txt_file_list=task["file_list"]
                )

        if cfg.OUTPUT.ICDAR15.TASK4:
            vocabularies["strong"] = Vocabulary(
                os.path.join(voc_path, "strong", "voc_" + im_name.split(".")[0] + ".txt")
            )
            for lexicon in ["generic", "weak", "strong"]:
                task_key = "icdar15_task4_{}".format(lexicon)
                output_icdar15(
                    task_utils[task_key]["dir"],
                    result_logs,
                    im_name,
                    task=4,
                    txt_file_list=task_utils[task_key]["file_list"],
                    vocabulary=vocabularies[lexicon],
                    det_conf_thresh=0.01,
                    seq_conf_thresh=icdar15_seq_conf_thresh_list[lexicon],
                )

        if cfg.OUTPUT.FB_COCO.EVAL:
            output_fb_coco_class_format(
                fb_coco_dir,
                result_logs,
                image_path,
                everstore_to_id_map,
                fb_coco_json_ann_pred,
                det_conf_thresh=cfg.OUTPUT.FB_COCO.DET_THRESH,
                seq_conf_thresh=cfg.OUTPUT.FB_COCO.SEQ_THRESH,
            )

        print(f"[{i+1}/{num_images}][rank: {rank}] Completed")

    # if zip separately for each gpu and copy (then disable the accumulate and zip below)
    if cfg.OUTPUT.ZIP_PER_GPU:
        zip_txt_files_and_copy_to_output(task["file_list"], task["zip"], output_folder)

    if is_main_process():
        print(f"[Debug] Before synchronize() - main process (rank: {rank})")
    else:
        print(f"[Debug] Before synchronize() - rank: {rank}")
    synchronize()
    if is_main_process():
        print("[Debug] After synchronize() - main process")
    else:
        print("[Debug] After synchronize()")
    if not cfg.OUTPUT.ZIP_PER_GPU:
        # merge text lists for each task
        for task in task_utils.values():
            task["file_list"] = _accumulate_txt_lists_from_multiple_gpus(task["file_list"])

    # merge fb coco format json from multi-gpus
    if cfg.OUTPUT.FB_COCO.EVAL:
        fb_coco_json_ann_pred = _accumulate_fb_coco_json_from_multiple_gpus(fb_coco_json_ann_pred)

    # predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        logger.info("Not main process, returned")
        return
    logger.info("Main process, continued")

    # add files to zip and copy to main output folder
    if not cfg.OUTPUT.ZIP_PER_GPU:
        for task in task_utils.values():
            zip_txt_files_and_copy_to_output(task["file_list"], task["zip"], output_folder)

    # run eval if needed (after all zip files have been saved in the previous
    # step to allow re-running evaluation later in case there's an exception)
    for task in task_utils.values():
        if task["eval_func"] is not None:
            logger.info("Evaluating {} ...".format(task["name"]))
            task["eval_func"](pred_zip_file=task["zip"])

    # icdar15 task4 is special in that it needs to gather and print results
    # for all lexicon in the end, if that's not needed then it can be merged above
    if cfg.OUTPUT.ICDAR15.TASK4:
        eval_results = {}
        for lexicon in ["strong", "weak", "generic"]:
            task_key = "icdar15_task4_{}".format(lexicon)
            logger.info("Evaluating {} ...".format(task_utils[task_key]["name"]))
            eval_results[lexicon] = icdar15_eval_task4(pred_zip_file=task_utils[task_key]["zip"])
        logger.info(eval_results)

    # TODO: total text det eval is processed differently from others, can be unified later
    if cfg.OUTPUT.TOTAL_TEXT.DET_EVAL:
        logger.info("Evaluating total text detection ...")
        eval_default = total_text_eval_det(
            pred_dir=task_utils["total_text_det"]["dir"], tp=0.4, tr=0.8
        )
        eval_new = total_text_eval_det(pred_dir=task_utils["total_text_det"]["dir"], tp=0.6, tr=0.7)
        logger.info(
            "[Default] Precision: {:.2f} / Recall: {:.2f} / F-Score: {:.2f}".format(
                eval_default["Precision"] * 100,
                eval_default["Recall"] * 100,
                eval_default["F-Score"] * 100,
            )
        )
        logger.info(
            "[New] Precision: {:.2f} / Recall: {:.2f} / F-Score: {:.2f}".format(
                eval_new["Precision"] * 100,
                eval_new["Recall"] * 100,
                eval_new["F-Score"] * 100,
            )
        )

    if cfg.OUTPUT.FB_COCO.EVAL:
        # TODO: directly call evaluation functionality
        pred_json = os.path.join(fb_coco_dir, "pred.json")
        logger.info(f"Saving prediction in fb coco format in {pred_json}...")

        with open(pred_json, "w", encoding="utf-8") as f:
            json.dump(fb_coco_json_ann_pred, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {pred_json}")

    # purge temp folder with text files and zip
    if os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)


def load_image(image_path, allow_failure=False):
    with open(image_path, "rb") as f:
        img = Image.open(f).convert("RGB")
    return img


def mask2polygon(mask, box, im_size, threshold=0.5, polygon_format="polygon"):
    # mask 32*128
    image_width, image_height = im_size[0], im_size[1]
    box_h = box[3] - box[1]
    box_w = box[2] - box[0]
    cls_polys = (mask * 255).astype(np.uint8)
    poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
    poly_map = poly_map.astype(np.float32) / 255
    poly_map = cv2.GaussianBlur(poly_map, (3, 3), sigmaX=3)
    ret, poly_map = cv2.threshold(poly_map, threshold, 1, cv2.THRESH_BINARY)
    if polygon_format == "polygon":
        SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        poly_map = cv2.erode(poly_map, SE1)
        poly_map = cv2.dilate(poly_map, SE1)
        poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
        try:
            _, contours, _ = cv2.findContours(
                (poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        except Exception:
            contours, _ = cv2.findContours(
                (poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        if len(contours) == 0:
            # print(contours)
            # print(len(contours))
            return None
        max_area = 0
        max_cnt = contours[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt
        # perimeter = cv2.arcLength(max_cnt, True)
        epsilon = 0.01 * cv2.arcLength(max_cnt, True)
        approx = cv2.approxPolyDP(max_cnt, epsilon, True)
        pts = approx.reshape((-1, 2))
        pts[:, 0] = pts[:, 0] + box[0]
        pts[:, 1] = pts[:, 1] + box[1]
        polygon = list(pts.reshape((-1,)))
        polygon = list(map(int, polygon))
        if len(polygon) < 6:
            return None
    else:
        assert polygon_format == "rect", f"Unknown polygon format: {polygon_format}"
        SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        poly_map = cv2.erode(poly_map, SE1)
        poly_map = cv2.dilate(poly_map, SE1)
        poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
        idy, idx = np.where(poly_map == 1)
        xy = np.vstack((idx, idy))
        xy = np.transpose(xy)
        hull = cv2.convexHull(xy, clockwise=True)
        # reverse order of points.
        if hull is None:
            return None
        hull = hull[::-1]
        # find minimum area bounding box.
        rect = cv2.minAreaRect(hull)
        corners = cv2.boxPoints(rect)
        corners = np.array(corners, dtype="int")
        pts = get_tight_rect(corners, box[0], box[1], image_height, image_width, 1)
        polygon = [x * 1.0 for x in pts]
        polygon = list(map(int, polygon))
    return polygon


def polygon2rbox(polygon, image_height, image_width):
    poly = np.array(polygon).reshape((-1, 2))
    try:
        rect = cv2.minAreaRect(poly)
    except Exception:
        print(f"cv2.minAreaRect failed for polygon {polygon}")
        return None
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = list(map(int, pts))
    return pts


def render_box_multi_text(cfg, image, result_logs_dict, resize_ratio, det_thresh=0.2):
    word_result_list = result_logs_dict["result_logs"]

    draw = ImageDraw.Draw(image, "RGBA")

    color_green = (0, 255, 0, 192)

    renderer = render_box_multi_text

    if not hasattr(renderer, "fonts"):
        renderer.fonts = {0: ImageFont.load_default()}
        print("[Info] fonts initiated.")

    font_path = "/checkpoint/jinghuang/fonts/Arial-Unicode-Regular.ttf"
    if not os.path.exists(font_path):
        use_default_font = True
        logger.warning("[Warning] Font {} doesn't exist, using default.".format(font_path))
    else:
        use_default_font = False

    for word_result in word_result_list:
        # Use line width to illustrate boxes with low confidence
        score_det = word_result.det_score
        score_rec_seq = word_result.seq_score

        if score_det < det_thresh:
            continue

        line_width = 6

        if score_det < 0.05:
            line_width -= 2

        if score_rec_seq < 0.9:
            line_width -= 1

        polygon = word_result.polygon

        min_x = min(polygon[0::2])
        min_y = min(polygon[1::2])
        height = min(word_result.rotated_box_5d[2], word_result.rotated_box_5d[3])

        if use_default_font:
            font_size = 0
        else:
            font_size = max(2, min(10, int(round(0.3 * height / 5)))) * 5

        if font_size in renderer.fonts:
            font = renderer.fonts[font_size]
        else:
            with open(font_path, "rb") as f:
                font = ImageFont.truetype(f, font_size)
            renderer.fonts[font_size] = font

        # r = randint(0, 255)
        # g = randint(0, 255)
        # b = randint(0, 255)

        # draw.polygon(polygon, outline=(0,0,255,128), fill=(0,255,0,64))
        # draw.polygon(polygon, outline=(0,0,255,128), fill=(r, g, b, 96))
        vertices = polygon + [polygon[0], polygon[1]]  # append the 1st vertex
        draw.line(vertices, width=line_width, fill=(255, 0, 0, 255))

        word = f"{word_result.seq_word} [{int(round(score_det*100))}%,{word_result.language}"

        if word_result.language_enabled != word_result.language:
            word += f",{word_result.language_enabled}"

        word += "]"

        draw_loc = (min_x, min_y - font_size)
        text_size = draw.textsize(word, font=font)
        rec_pad = 0
        rec_min = (draw_loc[0] - rec_pad, draw_loc[1] - rec_pad)
        rec_max = (
            draw_loc[0] + text_size[0] + rec_pad,
            draw_loc[1] + text_size[1] + rec_pad,
        )
        draw.rectangle([rec_min, rec_max], fill=color_green)
        draw.text((min_x, min_y - font_size), word, font=font, fill=(0, 0, 0, 255))


# given a list of text files, add them to a zip file, then copy to the output folder
# this is to deal with the slow zipping on Gluster
# later we can update this to copy to manifold too
def zip_txt_files_and_copy_to_output(txt_list, zip_path, output_folder):
    logger.info("Adding txt files to {}".format(zip_path))
    for txt_file in txt_list:
        append_txt_to_zip(zip_file=zip_path, txt_file=txt_file)
    final_zip = os.path.join(output_folder, os.path.basename(zip_path))
    logger.info(f"Copying zip file to {final_zip}")
    shutil.copy2(zip_path, final_zip)
