# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import getpass
import math
import pickle
import random
from functools import partial
from multiprocessing import Pool

import editdistance
import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon

from multiplexer.evaluation.mlt19.utils import MLT19Utils as utils
from multiplexer.evaluation.utils.weighted_editdistance import weighted_edit_distance
from multiplexer.utils.languages import code_to_name, lang_code_to_char_map_class, name_to_code
from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open


def result_from_line(line):
    parts = line.split(",")
    # # polygon[0:8], detection_score, seq_score, language_score, language, seq_word
    return {
        "polygon": [float(a) for a in parts[0:8]],
        "det_score": float(parts[8]),
        "seq_score": float(parts[9]),
        "lang_score": float(parts[10]),
        "language": parts[11],
        "word": ",".join(parts[12:]),
    }


def polygon_from_list(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    polygon_points = np.array(line).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            iou = float(inter_area) / (union_area + 1e-6)
        except shapely.geos.TopologicalError:
            print("shapely.geos.TopologicalError occured, iou set to 0")
            iou = 0
    return iou


def nms(boxes, overlap):
    rec_scores = [b["seq_score"] for b in boxes]
    indices = sorted(range(len(rec_scores)), key=lambda k: -rec_scores[k])
    box_num = len(boxes)
    nms_flag = [True] * box_num
    for i in range(box_num):
        ii = indices[i]
        if not nms_flag[ii]:
            continue
        for j in range(box_num):
            jj = indices[j]
            if j == i:
                continue
            if not nms_flag[jj]:
                continue
            box1 = boxes[ii]
            box2 = boxes[jj]
            box1_score = rec_scores[ii]
            box2_score = rec_scores[jj]
            poly1 = polygon_from_list(box1["polygon"])
            poly2 = polygon_from_list(box2["polygon"])
            iou = polygon_iou(box1["polygon"], box2["polygon"])
            thresh = overlap

            if iou > thresh:
                if box1_score > box2_score:
                    nms_flag[jj] = False
                if box1_score == box2_score and poly1.area > poly2.area:
                    nms_flag[jj] = False
                if box1_score == box2_score and poly1.area <= poly2.area:
                    nms_flag[ii] = False
                    break

    return nms_flag


def packing(save_dir, cache_dir, pack_name):
    # files = os.listdir(save_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    command = "zip -r -q -j " + os.path.join(cache_dir, pack_name) + " " + save_dir + "/*"
    os.system(command)


def filter_with_confidence(results, score_det, score_rec_seq, filter_heuristic="equal"):
    filtered_results = []
    count_det_filter = 0
    count_rec_filter = 0
    if filter_heuristic == "more_latin":
        for result in results:
            if result["det_score"] < score_det:
                if result["language"] == "Latin":
                    if result["det_score"] > 0.05:
                        print(
                            "[Debug] Kept Latin word {} with det_score = {}".format(
                                result["word"], result["det_score"]
                            )
                        )
                else:
                    count_det_filter += 1
                    continue
            if result["seq_score"] < score_rec_seq:
                print(
                    "[Debug] Filtered {} with seq_score {} < {}".format(
                        result["word"], result["seq_score"], score_rec_seq
                    )
                )
                count_rec_filter += 1
                continue
            filtered_results.append(result)
    else:
        assert filter_heuristic == "equal", f"Unknown filter heuristic: {filter_heuristic}"
        for result in results:
            if result["det_score"] < score_det:
                count_det_filter += 1
                continue
            if result["seq_score"] < score_rec_seq:
                print(
                    "[Debug] Filtered {} with seq_score {} < {}".format(
                        result["word"], result["seq_score"], score_rec_seq
                    )
                )
                count_rec_filter += 1
                continue
            filtered_results.append(result)

    print(
        "[Debug] Filtered ({}, {}) with (det, seq) criteria among {} instances".format(
            count_det_filter, count_rec_filter, len(results)
        )
    )
    return filtered_results


# NOTE: score_rec_charmask, use_rec_seq, use_rec_charmask is not really used
def process_res_files(
    all_files,
    results_dir,
    nms_dir,
    vocabularies_list,
    char_map_class_list,
    task="task4",
    score_det=0.5,
    score_rec_seq=0.5,
    score_rec_charmask=0.5,
    overlap=0.2,
    use_rec_seq=False,
    use_rec_charmask=False,
    confidence_type="det",
    split="val",
    filter_heuristic="equal",
    lexicon=None,
    weighted_ed=True,
):
    count = 0
    total_count = len(all_files)
    for res_file in all_files:
        count += 1
        print("[{}/{}] Processing {}".format(count, total_count, res_file))
        result_path = os.path.join(results_dir, res_file)
        assert os.path.isfile(result_path)

        with open(result_path, "r") as f:
            lines = [a.strip() for a in f.readlines()]
        try:
            results = [result_from_line(line) for line in lines]
        except ValueError:
            print("[WARNING] bad result format, return empty")
            results = []

        if lexicon is not None:
            for i in range(len(results)):
                results[i]["pkl_name"] = res_file.split(".")[0] + "_" + str(i) + ".pkl"

        # Filter results based on confidence thresholds
        filtered_results = filter_with_confidence(
            results, score_det, score_rec_seq, filter_heuristic
        )

        nms_flag = nms(filtered_results, overlap)
        boxes = []
        for k in range(len(filtered_results)):
            dt = filtered_results[k]
            if nms_flag[k]:
                if dt not in boxes:
                    boxes.append(dt)

        if res_file.startswith("ts_"):
            # MLT19 test set image name looks like ts_img_09983.jpg, remove the ts_ prefix here
            res_file = res_file[3:]
        final_res_path = os.path.join(nms_dir, os.path.basename(res_file))

        with open(final_res_path, "w") as f:
            for g in boxes:
                assert not use_rec_charmask, "not supported yet"
                gt_coor_strs = [str(int(x)) for x in g["polygon"]]

                if lexicon is not None and len(g["word"]) > 2:
                    matched = False
                    pkl_file = os.path.join(results_dir, g["pkl_name"])

                    with open(pkl_file, "rb") as input_file:
                        dict_scores = pickle.load(input_file)

                    scores = dict_scores["seq_char_scores"][:, 1:-1].swapaxes(0, 1)

                    lang_code = name_to_code(g["language"])

                    word = g["word"]

                    if lang_code == "ar":
                        word = word[::-1]

                    match_word, match_dist = find_match_word(
                        word,
                        scores,
                        weighted_ed,
                        vocabularies_list[lang_code],
                        char_map_class_list[lang_code],
                    )

                    if lang_code == "ar":
                        match_word = match_word[::-1]

                    if match_dist < 0.5:
                        if g["word"] != match_word:
                            if match_dist < 0.1 or g["seq_score"] >= 0.8:
                                matched = True
                            print(
                                (
                                    "[lexicon: {}-{}][seq_conf:{:.3f}]{}"
                                    + " Corrected word {} to {} with edit dist {:.4f}"
                                ).format(
                                    lexicon,
                                    lang_code,
                                    g["seq_score"],
                                    "" if matched else "[filtered]",
                                    g["word"],
                                    match_word,
                                    match_dist,
                                )
                            )
                            g["word"] = match_word
                        else:
                            print(
                                "[matched][seq_conf:{:.3f}] Matched word {}".format(
                                    g["seq_score"], g["word"]
                                )
                            )
                            matched = True
                    else:
                        print(
                            (
                                "[big-dist][seq_conf:{:.3f}]"
                                + " Keep word {} from {} with edit dist {:.4f}"
                            ).format(g["seq_score"], g["word"], match_word, match_dist)
                        )

                    if not matched and g["seq_score"] < 0.8:
                        continue

                f.write(",".join(gt_coor_strs))
                if confidence_type == "det":
                    confidence = g["det_score"]
                elif confidence_type == "seq":
                    confidence = g["seq_score"]
                elif confidence_type == "lang":
                    confidence = g["lang_score"]
                else:
                    raise Exception("Unknown confidence type: {}".format(confidence_type))
                f.write(",{:.4f}".format(confidence))

                if task == "task3":
                    if g["language"] == "Any":
                        max_count = 0
                        best_lang_code = "la"
                        num_max = 1
                        for lang_code in char_map_class_list:
                            count = 0
                            for ch in g["word"]:
                                if char_map_class_list[lang_code].contain_char_exclusive(ch):
                                    count += 1
                            if count > 0:
                                if count > max_count:
                                    best_lang_code = lang_code
                                    max_count = count
                                    num_max = 1
                                elif count == max_count:
                                    # when equal, do reservoir sampling
                                    num_max += 1
                                    if random.random() < 1.0 / num_max:
                                        best_lang_code = lang_code

                        g["language"] = code_to_name(best_lang_code)
                        print(
                            "[Info] Auto-inferred language for {}: {}".format(
                                g["word"], g["language"]
                            )
                        )

                    f.write("," + g["language"])
                elif task == "task4":
                    f.write("," + g["word"])

                f.write("\n")

            print("[Debug] Saved {}".format(final_res_path))
    return


# split input data into chunks for multiprocessing
def _chunks(input_list, n):
    n = max(1, n)
    return [input_list[i : i + n] for i in range(0, len(input_list), n)]


def prepare_results_for_evaluation(
    results_dir,
    task="task4",
    cache_dir=None,
    score_det=0.5,
    score_rec_seq=0.5,
    score_rec_charmask=0.5,
    overlap=0.2,
    use_rec_seq=False,
    use_rec_charmask=False,
    confidence_type="det",
    languages=None,
    split="val",
    filter_heuristic="equal",
    lexicon=None,
    char_map_version="v3",
    weighted_ed=True,
):
    """
    split: dataset split, "val" - validation dataset, "test" - testing dataset
    results_dir: result directory
    task: task name. Options: task1, task3, task4
    score_det: score of detection bounding box
    score_rec_seq: score of the sequence recognition branch
    score_rec_charmask: score of the character mask recognition branch
    overlap: overlap threshold used for nms
    use_rec_seq: use the recognition result of sequence branch
    use_rec_charmask: use the recognition result of the mask
    confidence_type: the type of confidence to be used in the final output,
        options: det, seq
    languages: List of selected languages to be processed;
        If it's None, all languages are processed

    Note: when use_rec_seq and use_rec_charmask are both enabled,
        use both the recognition result of the mask and
        the sequence branches, and choose the one with higher score.
    """
    print(
        "split:",
        split,
        "task:",
        task,
        "score_det:",
        score_det,
        "score_rec_seq:",
        score_rec_seq,
        "score_rec_charmask:",
        score_rec_charmask,
        "overlap:",
        overlap,
        "use_rec_seq:",
        use_rec_seq,
        "use_rec_charmask:",
        use_rec_charmask,
        "confidence_type:",
        confidence_type,
        "languages:",
        languages,
        "filter_heuristic:",
        filter_heuristic,
        "lexicon:",
        lexicon,
        "char_map_version:",
        char_map_version,
        "weighted_ed:",
        weighted_ed,
    )
    languages_str = "" if languages is None else ("_" + ":".join(languages))

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    prefix = ""
    if use_rec_seq:
        prefix += "seq"
    if use_rec_charmask:
        prefix += "charmask"

    suffix = ""
    if lexicon is not None:
        suffix += f"_lex-{lexicon}_{char_map_version}"

    nms_dir = os.path.join(
        cache_dir,
        "{}_det{}_charmask{}_seq{}_iou{}_conf-{}_filter-{}_{}{}{}".format(
            prefix,
            score_det,
            score_rec_charmask,
            score_rec_seq,
            overlap,
            confidence_type,
            filter_heuristic,
            task,
            languages_str,
            suffix,
        ),
    )

    os.makedirs(nms_dir, exist_ok=True)

    lang_list = {"ar", "bn", "hi", "ja", "ko", "la", "zh", "symbol"}
    vocabularies_list = {}
    char_map_class_list = {}

    if char_map_version != "none":
        char_map_dir = (
            f"/checkpoint/{getpass.getuser()}/multiplexer/charmap/public/{char_map_version}/"
        )

        for language in lang_list:
            char_map_class_list[language] = lang_code_to_char_map_class[language]
            char_map_class_list[language].init(char_map_path=char_map_dir)

    if lexicon is not None:
        for language in lang_list:
            lexicon_path = (
                f"/checkpoint/{getpass.getuser()}/"
                + f"datasets/MLT19/eval/lexicons/{lexicon}/{language}.txt"
            )

            with open(lexicon_path, "r") as lexicon_fid:
                vocabularies_list[language] = [voc.strip() for voc in lexicon_fid.readlines()]

        if "ar" in vocabularies_list:
            # reverse words in Arabic
            for i in range(len(vocabularies_list["ar"])):
                vocabularies_list["ar"][i] = vocabularies_list["ar"][i][::-1]

    all_files = utils.get_result_file_list(results_dir, split, languages)

    # TODO: add num_workers to args
    num_workers = 50
    chunk_size = math.ceil(len(all_files) / num_workers)

    process_res_files_partial = partial(
        process_res_files,
        results_dir=results_dir,
        nms_dir=nms_dir,
        vocabularies_list=vocabularies_list,
        char_map_class_list=char_map_class_list,
        task=task,
        score_det=score_det,
        score_rec_seq=score_rec_seq,
        score_rec_charmask=score_rec_charmask,
        overlap=overlap,
        use_rec_seq=use_rec_seq,
        use_rec_charmask=use_rec_charmask,
        confidence_type=confidence_type,
        split=split,
        filter_heuristic=filter_heuristic,
        lexicon=lexicon,
        weighted_ed=weighted_ed,
    )

    # split input data into chunks for multiprocessing
    file_chunks = _chunks(all_files, chunk_size)
    pool = Pool(min(len(all_files), num_workers))
    pool.map(process_res_files_partial, file_chunks)
    pool.close()

    zip_name = "{}_det{}_charmask{}_seq{}_iou{}_conf-{}_filter-{}_{}{}{}.zip".format(
        prefix,
        score_det,
        score_rec_charmask,
        score_rec_seq,
        overlap,
        confidence_type,
        filter_heuristic,
        task,
        languages_str,
        suffix,
    )

    packing(nms_dir, cache_dir, zip_name)
    submit_file_path = os.path.join(cache_dir, zip_name)
    print("[Info] Final submission file: {}".format(submit_file_path))
    return submit_file_path


def find_match_word(
    rec_str, scores_numpy, weighted_ed=False, vocabularies=None, char_map_class=None
):
    # rec_str = rec_str.upper()
    dist_min = 100
    # dist_min_pre = 100
    dist_min_pre = 4
    match_word = ""
    match_dist = 100
    if not weighted_ed:
        for word in vocabularies:
            # word = word.upper()
            ed = editdistance.eval(rec_str, word)
            # length_dist = abs(len(word) - len(rec_str))
            # dist = ed + length_dist
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = word
                match_dist = dist
        return match_word, match_dist
    else:
        small_lexicon_dict = {}
        for word in vocabularies:
            # word = word.upper()
            ed = editdistance.eval(rec_str, word)
            small_lexicon_dict[word] = ed
            dist = ed
            if dist < dist_min_pre:
                dist_min_pre = dist
        small_lexicon = []
        for word in small_lexicon_dict:
            if small_lexicon_dict[word] <= dist_min_pre + 2:
                small_lexicon.append(word)

        for word in small_lexicon:
            # word = word.upper()
            ed = weighted_edit_distance(rec_str, word, scores_numpy, char_map_class)
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = word
                match_dist = dist
        return match_word, match_dist
