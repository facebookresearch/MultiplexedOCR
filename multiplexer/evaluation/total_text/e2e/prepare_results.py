#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import pickle
import zipfile

import editdistance
import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon

from multiplexer.evaluation.utils.weighted_editdistance import weighted_edit_distance

# from tqdm import tqdm


def append_txt_to_zip(zip_file, txt_file):
    with zipfile.ZipFile(zip_file, "a") as zipf:
        zipf.write(txt_file, os.path.basename(txt_file))


def list_from_str(st):
    line = st.split(";")
    segms = line[1].split(",")
    scores = line[2].split(",")
    new_line = (
        [float(a) for a in segms]
        + [float(scores[-4])]
        + [scores[-5]]
        + [scores[-6]]
        + [float(scores[-3])]
        + [float(scores[-2])]
        + [scores[-1]]
    )
    return new_line


def polygon_from_list(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    polygon_points = np.array(line).reshape(-1, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(-1, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(-1, 2)
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
    rec_scores = [b[-6] for b in boxes]
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
            poly1 = polygon_from_list(box1[0:-6])
            poly2 = polygon_from_list(box2[0:-6])
            iou = polygon_iou(box1[0:-6], box2[0:-6])
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
    os.system("zip -r -q -j " + os.path.join(cache_dir, pack_name) + " " + save_dir + "/*")


def load_lexicon_pairs(pair_path):
    pairs = {}
    with open(pair_path, "r") as pair_list:
        for line in pair_list.readlines():
            line = line.strip()
            word_pair = line.split(" ")
            assert len(word_pair) == 2, "More than two words found in a pair in {}: {}".format(
                pair_path, line
            )
            word = word_pair[0].strip().upper()
            if word_pair[1].lstrip() != word_pair[1]:
                print("[Warning] Strange character found in {}: {}", line)
            word_gt = word_pair[1].strip()
            pairs[word] = word_gt

    return pairs


def prepare_results_for_evaluation(
    results_dir,
    lexicon=None,
    cache_dir=None,
    score_det=0.5,
    score_rec_seq=0.5,
    score_rec_charmask=0.5,
    overlap=0.2,
    weighted_ed=True,
    use_rec_seq=False,
    use_rec_charmask=False,
    lexicon_path=None,
    lexicon_pair_path=None,
):
    """
    results_dir: result directory
    score_det: score of detection bounding box
    score_rec_seq: score of the sequence recognition branch
    score_rec_charmask: score of the character mask recognition branch
    overlap: overlap threshold used for nms
    use_rec_seq: use the recognition result of sequence branch
    use_rec_charmask: use the recognition result of the mask

    Note: when use_rec_seq and use_rec_charmask are both enabled,
        use both the recognition result of the mask and
        the sequence branches, and choose the one with higher score.
    """
    print(
        "score_det:",
        score_det,
        "score_rec_seq:",
        score_rec_seq,
        "score_rec_charmask:",
        score_rec_charmask,
        "overlap:",
        overlap,
        "lexicon:",
        lexicon,
        "weighted_ed:",
        weighted_ed,
        "use_rec_seq:",
        use_rec_seq,
        "use_rec_charmask:",
        use_rec_charmask,
    )
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    prefix = ""
    if use_rec_seq:
        prefix += "seq"
    if use_rec_charmask:
        prefix += "charmask"

    nms_dir = os.path.join(
        cache_dir,
        "{}_det{}_charmask{}_seq{}_iou{}_lex-{}".format(
            prefix, score_det, score_rec_charmask, score_rec_seq, overlap, lexicon
        ),
    )
    if not os.path.exists(nms_dir):
        os.mkdir(nms_dir)
    if lexicon == "weak":
        with open(lexicon_path, "r") as lexicon_fid:
            vocabularies = [voc.strip() for voc in lexicon_fid.readlines()]

        pairs = load_lexicon_pairs(lexicon_pair_path)

    count = 0
    for res_file in glob.glob("{}/*.txt".format(results_dir)):
        count += 1
        print("[{}/300] Processing {}".format(count, res_file))
        result_path = os.path.join(results_dir, res_file)
        assert os.path.isfile(result_path), "{}/{}".format(results_dir, res_file)
        if os.path.isfile(result_path):
            with open(result_path, "r") as f:
                dt_lines = [a.strip() for a in f.readlines()]
            dt_lines = [list_from_str(dt) for dt in dt_lines]
        else:
            dt_lines = []
        dt_lines = [
            dt
            for dt in dt_lines
            if dt[-2] > score_rec_seq and dt[-3] > score_rec_charmask and dt[-6] > score_det
        ]
        nms_flag = nms(dt_lines, overlap)
        boxes = []
        for k in range(len(dt_lines)):
            dt = dt_lines[k]
            if nms_flag[k]:
                if dt not in boxes:
                    boxes.append(dt)

        final_res_path = os.path.join(nms_dir, os.path.basename(res_file))

        with open(final_res_path, "w") as f:
            for g in boxes:
                gt_coors = [int(b) for b in g[0:-6]]
                pkl_name = g[-1]
                if not os.path.isfile(pkl_name):
                    pkl_name = os.path.join(results_dir, os.path.basename(pkl_name))
                with open(pkl_name, "rb") as input_file:
                    dict_scores = pickle.load(input_file)
                # print("[Debug] Loaded {}".format(pkl_name))
                # print("[Debug] dict_scores: {}".format(dict_scores))

                if use_rec_charmask and use_rec_seq:
                    if g[-2] > g[-3]:
                        word = g[-5]
                        scores = dict_scores["seq_char_scores"][:, 1:-1].swapaxes(0, 1)
                    else:
                        word = g[-4]
                        scores = dict_scores["seg_char_scores"]
                elif use_rec_seq:
                    word = g[-5]
                    scores = dict_scores["seq_char_scores"][:, 1:-1].swapaxes(0, 1)
                else:
                    word = g[-4]
                    scores = dict_scores["seg_char_scores"]

                if lexicon == "none":
                    match_word = word
                    match_dist = 0.0
                else:
                    match_word, match_dist = find_match_word(
                        word, pairs, scores, weighted_ed, vocabularies
                    )

                if match_dist < 1.5 or lexicon == "none":
                    if match_dist > 0:
                        print(
                            "[Info] Corrected word {} to {} with edit dist {:.2f}".format(
                                word, match_word, match_dist
                            )
                        )
                    gt_coor_strs = [str(a) for a in gt_coors]
                    f.write(",".join(gt_coor_strs) + "\n")
                    f.write(match_word + "\n")
                else:
                    print(
                        "[Info] Filtered word {} ({} with edit dist {:.2f})".format(
                            word, match_word, match_dist
                        )
                    )

            print("[Debug] Saved {}".format(final_res_path))

    zip_name = "{}_det{}_charmask{}_seq{}_iou{}_lex-{}.zip".format(
        prefix, score_det, score_rec_charmask, score_rec_seq, overlap, lexicon
    )

    packing(nms_dir, cache_dir, zip_name)
    submit_file_path = os.path.join(cache_dir, zip_name)
    print("[Info] Final submission file: {}".format(submit_file_path))
    return submit_file_path


def find_match_word(rec_str, pairs, scores_numpy, weighted_ed=False, vocabularies=None):
    rec_str = rec_str.upper()
    dist_min = 100
    dist_min_pre = 100
    match_word = ""
    match_dist = 100
    if not weighted_ed:
        for word in vocabularies:
            word = word.upper()
            ed = editdistance.eval(rec_str, word)
            # length_dist = abs(len(word) - len(rec_str))
            # dist = ed + length_dist
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = pairs[word]
                match_dist = dist
        return match_word, match_dist
    else:
        small_lexicon_dict = {}
        for word in vocabularies:
            word = word.upper()
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
            word = word.upper()
            ed = weighted_edit_distance(rec_str, word, scores_numpy)
            dist = ed
            if dist < dist_min:
                dist_min = dist
                match_word = pairs[word]
                match_dist = dist
        return match_word, match_dist
