# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from multiplexer.evaluation.mlt19.prepare_results import find_match_word
from multiplexer.utils.languages import (
    ArabicCharMap,
    HebrewCharMap,
    UrduCharMap,
    code_to_name,
    cyrillic_greek_to_latin,
    lang_code_to_char_map_class,
)
from virtual_fs import virtual_os as os


def output_fb_coco_class_format(
    out_dir,
    ocr_results,
    img_name,
    everstore_to_id_map,
    pred_ann,
    det_conf_thresh=0,
    seq_conf_thresh=0,
):
    assert img_name in everstore_to_id_map
    img_id_list = everstore_to_id_map[img_name]
    count_anns = len(pred_ann["anns"])
    for img_id in img_id_list:
        if img_id in pred_ann["imgs"]:
            # This everstore handle has already been processed with a different img_id
            return
        pred_ann["imgs"][img_id] = {
            "id": img_id,
            # "languages": xx
            # "file_name": xx
        }
        pred_ann["imgToAnns"][img_id] = []

    language_sum = {}
    language_count = {}

    for word_result in ocr_results:
        if word_result.should_be_filtered(
            name=img_name,
            det_conf_thresh=det_conf_thresh,
            seq_conf_thresh=seq_conf_thresh,
        ):
            continue

        word = word_result.seq_word
        # handling right-to-left languages (e.g., Arabic)
        is_right_to_left = False
        for i in range(len(word)):
            if (
                ArabicCharMap.contain_char_exclusive(word[i])
                or HebrewCharMap.contain_char_exclusive(word[i])
                or UrduCharMap.contain_char_exclusive(word[i])
            ):
                is_right_to_left = True
                break
        if is_right_to_left:
            word = word[::-1]

        # accumulate language
        lang = word_result.language
        if lang in language_count:
            language_count[lang] += 1
            language_sum[lang] += word_result.language_prob
        else:
            language_count[lang] = 1
            language_sum[lang] = word_result.language_prob

        # code_to_name(result_log["language"])

        for img_id in img_id_list:
            ann_key = "{}_{}".format(img_id, count_anns)
            pred_ann["anns"][ann_key] = {
                "id": ann_key,
                "image_id": img_id,
                "bbox": word_result.rotated_box_5d,
                "utf8_string": word,
                "detection_score": word_result.det_score,
                "recognition_score": word_result.seq_score,
                "language": word_result.language,
                "language_enabled": word_result.language_enabled,
            }

            pred_ann["imgToAnns"][img_id].append(ann_key)

        count_anns += 1

    language_sum_thresh = 5
    max_language_result_below_sum_thresh = 1.0 - math.exp(-language_sum_thresh)
    language_result_per_img = {}
    for lang in language_sum:
        if language_sum[lang] > language_sum_thresh:
            # minus math.exp(-language_sum[lang]) here as a heuristic for tie-breaker
            # in case there are more than one dominating languages
            language_result_per_img[lang] = 1.0 - math.exp(-language_sum[lang])
        else:
            language_result_per_img[lang] = (
                language_sum[lang] / language_count[lang]
            ) * max_language_result_below_sum_thresh

    sorted_language_result_per_img = [
        {"language": lang, "confidence": language_result_per_img[lang]}
        for lang in sorted(language_result_per_img, key=lambda k: -language_result_per_img[k])
    ]

    for img_id in img_id_list:
        pred_ann["imgs"][img_id]["languages"] = sorted_language_result_per_img


def output_icdar15(
    out_dir,
    ocr_results,
    img_name,
    task,
    txt_file_list,
    vocabulary=None,
    det_conf_thresh=0,
    seq_conf_thresh=0,
):
    assert task in [1, 4], "Task {} not supported!".format(task)
    txt_file = os.path.join(out_dir, "res_" + img_name.split(".")[0] + ".txt")
    txt_file_list.append(txt_file)
    with open(txt_file, "wt") as res:
        for result_log in ocr_results:
            if should_be_filtered(
                benchmark="icdar15_task{}".format(task),
                result_log=result_log,
                det_conf_thresh=det_conf_thresh,
                seq_conf_thresh=seq_conf_thresh,
            ):
                continue

            polygon = result_log["polygon"]
            if len(polygon) != 8:
                polygon = result_log["rotated_box"]
            output = "{}".format(polygon[0])
            for i in range(1, len(polygon)):
                output += ",{}".format(polygon[i])

            # output += "{:.4f}".format(result_log["score"])  # confidence

            if task == 1:
                output += "\n"
            elif task == 4:
                word = result_log["seq_word"]
                # handling right-to-left languages (e.g., Arabic)
                is_right_to_left = False
                for i in range(len(word)):
                    if ArabicCharMap.contain_char_exclusive(word[i]):
                        is_right_to_left = True
                        break
                if is_right_to_left:
                    word = word[::-1]

                word = vocabulary.auto_correct(word, max_edit_dist=4, max_ratio=0.35, unique=True)

                output += ",{}\n".format(word)  # predicted word

            res.write(output)


def output_icdar15_intermediate(
    out_dir, ocr_results, img_name, txt_file_list, det_conf_thresh=0, seq_conf_thresh=0
):
    txt_file = os.path.join(out_dir, "res_" + img_name.split(".")[0] + ".txt")
    txt_file_list.append(txt_file)

    with open(txt_file, "wt") as res:
        for i, result_log in enumerate(ocr_results):
            if should_be_filtered(
                benchmark="icdar15_intermediate",
                result_log=result_log,
                det_conf_thresh=det_conf_thresh,
                seq_conf_thresh=seq_conf_thresh,
            ):
                continue

            pkl_file = os.path.join(
                out_dir, "res_" + img_name.split(".")[0] + "_" + str(i) + ".pkl"
            )
            txt_file_list.append(pkl_file)
            save_dict = {}

            horizontal_box = result_log["box"]
            polygon = result_log["polygon"]
            if len(polygon) != 8:
                polygon = result_log["rotated_box"]

            save_dict["seg_char_scores"] = result_log["char_score"]
            save_dict["seq_char_scores"] = result_log["detailed_seq_score"]
            output = "{},{},{},{},{},{},{},{}\n".format(
                ",".join([str(x) for x in horizontal_box]),
                ",".join([str(x) for x in polygon]),
                result_log["word"],
                result_log["seq_word"],
                result_log["score"],
                result_log["rec_score"],
                result_log["seq_score"],
                pkl_file,
            )

            # TODO: no need to dump a file per box
            with open(pkl_file, "wb") as f:
                pickle.dump(save_dict, f, protocol=2)

            res.write(output)


def output_mlt17(
    out_dir,
    ocr_results,
    img_name,
    task,
    txt_file_list,
    det_conf_thresh=0,
    seq_conf_thresh=0,
    det_conf_thresh2=None,
    seq_conf_thresh2=None,
):
    assert task in [1, 3], "Task {} not supported!".format(task)
    if img_name.startswith("ts_"):
        # MLT17 test set image name looks like ts_img_00002.jpg, remove the ts_ prefix here
        img_name = img_name[3:]
    txt_file = os.path.join(out_dir, "res_" + img_name.split(".")[0] + ".txt")
    txt_file_list.append(txt_file)
    with open(txt_file, "wt") as res:
        for result_log in ocr_results:
            if should_be_filtered(
                benchmark=f"mlt17_task{task}][{img_name}",
                result_log=result_log,
                det_conf_thresh=det_conf_thresh,
                seq_conf_thresh=seq_conf_thresh,
                det_conf_thresh2=det_conf_thresh2,
                seq_conf_thresh2=seq_conf_thresh2,
            ):
                continue

            polygon = result_log["polygon"]
            if len(polygon) != 8:
                polygon = result_log["rotated_box"]
            output = ""
            for i in range(0, len(polygon)):
                output += "{},".format(polygon[i])

            output += "{:.4f}".format(result_log["score"])  # confidence

            if task == 1:
                output += "\n"
            elif task == 3:
                language = code_to_name(result_log["language"])
                if language == "Hindi":
                    # MLT17 benchmark doesn't recognize Hindi
                    print("[Warning] Converting Hindi to Bangla for MLT17 Benchmark")
                    language = "Bangla"
                output += ",{}\n".format(language)

            res.write(output)


def output_mlt19(
    out_dir,
    ocr_results,
    img_name,
    task,
    txt_file_list,
    det_conf_thresh=0,
    seq_conf_thresh=0,
    lexicon=None,
    vocabularies_list=None,
    char_map_class_list=None,
    edit_dist_thresh=0.5,
):
    assert task in [1, 3, 4], "Task {} not supported!".format(task)
    if img_name.startswith("ts_"):
        # MLT19 test set image name looks like ts_img_09983.jpg, remove the ts_ prefix here
        img_name = img_name[3:]
    elif img_name.startswith("val_"):
        # MLT19 validation set image name looks like val_img_01716.jpg, remove the val_ prefix here
        img_name = img_name[4:]
    txt_file = os.path.join(out_dir, "res_" + img_name.split(".")[0] + ".txt")
    txt_file_list.append(txt_file)
    with open(txt_file, "wt") as res:
        for i, word_result in enumerate(ocr_results):
            if word_result.should_be_filtered(
                name=img_name,
                det_conf_thresh=det_conf_thresh,
                seq_conf_thresh=seq_conf_thresh,
            ):
                continue

            if lexicon is not None:
                assert vocabularies_list is not None
                assert char_map_class_list is not None
                lang_code = word_result.language
                if lang_code in vocabularies_list and len(word_result.seq_word) > 2:
                    match_word, match_dist = find_match_word(
                        rec_str=word_result.seq_word,
                        scores_numpy=word_result.detailed_seq_score[:, 1:-1].swapaxes(0, 1),
                        weighted_ed=True,
                        vocabularies=vocabularies_list[lang_code],
                        char_map_class=char_map_class_list[lang_code],
                    )

                    if match_dist < edit_dist_thresh:
                        if word_result.seq_word != match_word:
                            print(
                                "[lexicon: {}-{}][seq_conf:{:.3f}] Corrected word {} to {} with edit dist {:.4f}".format(
                                    lexicon,
                                    lang_code,
                                    word_result.seq_score,
                                    word_result.seq_word,
                                    match_word,
                                    match_dist,
                                )
                            )
                            word_result.seq_word = match_word
                        else:
                            print(
                                "[matched][seq_conf:{:.3f}] Matched word {}".format(
                                    word_result.seq_score, word_result.seq_score
                                )
                            )
                    else:
                        print(
                            "[big-dist][seq_conf:{:.3f}] Kept word {} from {} with edit dist {:.4f}".format(
                                word_result.seq_score,
                                word_result.seq_word,
                                match_word,
                                match_dist,
                            )
                        )

            polygon = word_result.polygon
            if len(polygon) != 8:
                polygon = word_result.rotated_box
            output = ""
            for i in range(0, len(polygon)):
                output += "{},".format(polygon[i])

            output += "{:.4f}".format(word_result.det_score)  # confidence

            if task == 1:
                output += "\n"
            elif task == 3:
                output += ",{}\n".format(code_to_name(word_result.language))
            elif task == 4:
                word = word_result.seq_word
                # handling right-to-left languages (e.g., Arabic)
                is_right_to_left = False
                for i in range(len(word)):
                    if ArabicCharMap.contain_char_exclusive(word[i]):
                        is_right_to_left = True
                        break
                if is_right_to_left:
                    word = word[::-1]

                output += ",{}\n".format(word)  # predicted word

            res.write(output)


def output_mlt19_intermediate(
    out_dir,
    ocr_results,
    img_name,
    txt_file_list,
    det_conf_thresh=0,
    seq_conf_thresh=0,
    save_pkl=False,
):
    if img_name.startswith("ts_"):
        # MLT19 test set image name looks like ts_img_09983.jpg, remove the ts_ prefix here
        img_name = img_name[3:]
    elif img_name.startswith("val_"):
        # MLT19 validation set image name looks like val_img_01716.jpg, remove the val_ prefix here
        img_name = img_name[4:]
    txt_file = os.path.join(out_dir, "res_" + img_name.split(".")[0] + ".txt")
    txt_file_list.append(txt_file)

    with open(txt_file, "wt") as res:
        for i, word_result in enumerate(ocr_results):
            # for i, result_log in enumerate(ocr_results):
            if word_result.should_be_filtered(
                name=img_name,
                det_conf_thresh=det_conf_thresh,
                seq_conf_thresh=seq_conf_thresh,
            ):
                continue

            if save_pkl:
                pkl_file = os.path.join(
                    out_dir, "res_" + img_name.split(".")[0] + "_" + str(i) + ".pkl"
                )
                txt_file_list.append(pkl_file)

                save_dict = {
                    # "seg_char_scores": result_log["char_score"],
                    "seq_char_scores": word_result.detailed_seq_scores
                }

                # TODO: no need to dump a file per box
                with open(pkl_file, "wb") as f:
                    pickle.dump(save_dict, f, protocol=2)

            polygon = word_result.polygon
            if len(polygon) != 8:
                polygon = word_result.rotated_box
            output = ""
            for i in range(0, len(polygon)):
                output += "{},".format(polygon[i])

            output += "{:.4f},".format(word_result.det_score)  # detection confidence
            output += "{:.4f},".format(word_result.seq_score)  # sequence recogntion confidence
            output += "{:.4f},".format(word_result.language_prob)  # language probabilities

            output += "{},".format(code_to_name(word_result.language))

            word = word_result.seq_word
            # handling right-to-left languages (e.g., Arabic)
            is_right_to_left = False
            for i in range(len(word)):
                if ArabicCharMap.contain_char_exclusive(word[i]):
                    is_right_to_left = True
                    break
            if is_right_to_left:
                word = word[::-1]

            output += "{}\n".format(word)  # predicted word

            res.write(output)


def output_total_text_det(
    out_dir,
    ocr_results,
    img_name,
    txt_file_list,
    det_conf_thresh=0.05,
    seq_conf_thresh=0.5,
):
    txt_file = os.path.join(out_dir, img_name.split(".")[0] + ".txt")
    txt_file_list.append(txt_file)
    with open(txt_file, "wt") as res:
        for result_log in ocr_results:
            if should_be_filtered(
                benchmark="total_text_det",
                result_log=result_log,
                det_conf_thresh=det_conf_thresh,
                seq_conf_thresh=seq_conf_thresh,
            ):
                continue

            polygon = result_log["polygon"]

            # total text result format assumes (y, x) pair for each point
            output = "{},{}".format(polygon[1], polygon[0])
            for i in range(2, len(polygon), 2):
                output += ",{},{}".format(polygon[i + 1], polygon[i])

            output += "\n"

            res.write(output)


def output_total_text_e2e(
    out_dir,
    ocr_results,
    img_name,
    txt_file_list,
    det_conf_thresh=0.05,
    seq_conf_thresh=0.9,
):
    # txt_file = os.path.join(out_dir, img_name.split(".")[0] + ".txt")
    txt_file = os.path.join(out_dir, "res_" + img_name.split(".")[0] + ".txt")
    txt_file_list.append(txt_file)
    with open(txt_file, "wt") as res:
        for result_log in ocr_results:
            if should_be_filtered(
                benchmark="total_text_e2e",
                result_log=result_log,
                det_conf_thresh=det_conf_thresh,
                seq_conf_thresh=seq_conf_thresh,
            ):
                continue

            word = result_log["seq_word"]
            polygon = result_log["polygon"]
            output = "{}".format(polygon[0])
            for i in range(1, len(polygon)):
                output += ",{}".format(polygon[i])

            # handling right-to-left languages (e.g., Arabic)
            # is_right_to_left = False
            # for i in range(len(word)):
            #     if ArabicCharMap.contain_char_exclusive(word[i]):
            #         is_right_to_left = True
            #         break
            # if is_right_to_left:
            #     word = word[::-1]

            # word = vocabulary.auto_correct(
            #     word, max_edit_dist=4, max_ratio=0.35, unique=True
            # )

            # Write predicted word in a new line (to avoid confusion of character ',')
            output += "\n{}\n".format(word)

            res.write(output)


def output_total_text_intermediate(
    out_dir, ocr_results, img_name, txt_file_list, det_conf_thresh=0, seq_conf_thresh=0
):
    txt_file = os.path.join(out_dir, "res_" + img_name.split(".")[0] + ".txt")
    txt_file_list.append(txt_file)

    with open(txt_file, "wt") as res:
        for i, result_log in enumerate(ocr_results):
            if should_be_filtered(
                benchmark="total_text_intermediate",
                result_log=result_log,
                det_conf_thresh=det_conf_thresh,
                seq_conf_thresh=seq_conf_thresh,
            ):
                continue

            pkl_file = os.path.join(
                out_dir, "res_" + img_name.split(".")[0] + "_" + str(i) + ".pkl"
            )
            txt_file_list.append(pkl_file)
            save_dict = {}

            horizontal_box = result_log["box"]
            polygon = result_log["polygon"]

            save_dict["seg_char_scores"] = result_log["char_score"]
            save_dict["seq_char_scores"] = result_log["detailed_seq_score"]
            output = "{};{};{},{},{},{},{},{}\n".format(
                ",".join([str(x) for x in horizontal_box]),
                ",".join([str(x) for x in polygon]),
                result_log["word"],
                result_log["seq_word"],
                result_log["score"],
                result_log["rec_score"],
                result_log["seq_score"],
                pkl_file,
            )

            # TODO: no need to dump a file per box
            with open(pkl_file, "wb") as f:
                pickle.dump(save_dict, f, protocol=2)

            res.write(output)


def should_be_filtered(
    benchmark,
    result_log,
    det_conf_thresh=0,
    seq_conf_thresh=0,
    det_conf_thresh2=None,
    seq_conf_thresh2=None,
):
    score_det = result_log["score"]
    score_rec_seq = result_log["seq_score"]
    word = result_log["seq_word"]

    filtered = False

    if score_det < det_conf_thresh or score_rec_seq < seq_conf_thresh:
        filtered = True

    if filtered:
        if det_conf_thresh2 is not None:
            assert seq_conf_thresh2 is not None
            if score_det > det_conf_thresh2 and score_rec_seq > seq_conf_thresh2:
                filtered = False

    status = "filtered" if filtered else "kept"
    lang = result_log["language"]  # predicted language
    rec_lang = result_log["language_enabled"]  # actual rec head
    score_str = "[det={:.3f}, seq={:.3f}]".format(score_det, score_rec_seq)

    print(f"[{benchmark}][{status}][{lang}][{rec_lang}][{score_str}] {word}")

    return filtered
