#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf8

# File: MLT_task_4_1_0.py
# Version: 1.1
# Date: 2019-03-28
# Description: Evaluation script that computes End to End Recognition.
#   For Text Localization it's used Intersection over Union criteria.
# Average Precision is also calcuted when 'CONFIDENCES' parameter is True
# 1-NED is also calculated
# A detection is correct if it's area overlaps with a GT item more than IOU_CONSTRAINT
# and the detected Transcriptio is EQUAL (case insensitive) than the detected GT
# Changes: Added 'Hindi' script

import getpass
import re

import editdistance
import numpy as np
from shapely import geometry as plg
from shapely.geos import TopologicalError
from tqdm import tqdm

from multiplexer.evaluation.mlt19.task4 import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
from multiplexer.evaluation.mlt19.utils import MLT19Utils as utils


def default_evaluation_params():
    """
    default_evaluation_params:
        Default parameters to use for the validation and evaluation.
    """
    return {
        "IOU_CONSTRAINT": 0.5,
        "AREA_PRECISION_CONSTRAINT": 0.5,
        "GT_SAMPLE_NAME_2_ID": "gt_img_([0-9]+).txt",
        "DET_SAMPLE_NAME_2_ID": "res_img_([0-9]+).txt",
        "CRLF": False,  # Lines are delimited by Windows CRLF format
        "CONFIDENCES": True,
    }


def validate_data(gtFilePath, submFilePath, evaluationParams, languages=None):
    """
    Method validate_data:
        validates that all files in the results folder are correct
        (have the correct name contents).
        Validates also that there are no missing files in the folder.
        If some error detected, the method raises the error
    """
    gt = utils.load_zip_file(
        gtFilePath, evaluationParams["GT_SAMPLE_NAME_2_ID"], languages=languages
    )

    subm = utils.load_zip_file(
        submFilePath,
        evaluationParams["DET_SAMPLE_NAME_2_ID"],
        True,
        languages=languages,
    )

    # Validate format of GroundTruth
    for k in gt:
        validate_lines_in_file(k, gt[k], evaluationParams["CRLF"], True, True, False)

    # Validate format of results
    for k in subm:
        if k not in gt:
            raise Exception(
                "The sample %s not present in GT - num gt=%s, num det= %s" % (k, len(gt), len(subm))
            )

        validate_lines_in_file(
            k,
            subm[k],
            evaluationParams["CRLF"],
            False,
            True,
            evaluationParams["CONFIDENCES"],
        )


def validate_lines_in_file(
    fileName, file_contents, CRLF, withScript, withTranscription, withConfidence
):
    """
    This function validates that all lines of the file calling
    the Line validation function for each line
    """
    utf8File = rrc_evaluation_funcs.decode_utf8(file_contents)
    if utf8File is None:
        raise Exception("The file %s is not UTF-8" % fileName)

    lines = utf8File.split("\r\n" if CRLF else "\n")
    for line in lines:
        line = line.replace("\r", "").replace("\n", "")
        if line != "":
            try:
                validate_line_with_script(line, withScript, withTranscription, withConfidence)
            except Exception as e:
                raise Exception(
                    (
                        "Line in sample not valid. Sample: %s Line: %s Error: %s"
                        % (fileName, line, str(e))
                    ).encode("utf-8", "replace")
                )


def validate_line_with_script(line, withScript, withTranscription, withConfidence):
    get_tl_line_values(line, withScript, withTranscription, withConfidence)


def get_tl_line_values(line, withScript=False, withTranscription=False, withConfidence=False):
    """
    Validate the format of the line.
        If the line is not valid an exception will be raised.
        If maxWidth and maxHeight are specified,
        all points must be inside the imgage bounds.
    Posible values are:
    x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,script][,transcription]
    Returns values from a textline.
    Points, [Confidences] , [Script], [Transcriptions]
    """
    script = ""
    transcription = ""
    points = []
    confidence = 0.0

    numPoints = 8

    if withTranscription and withConfidence:
        m = re.match(
            r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$",
            line,
        )
        if m is None:
            raise Exception(
                "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence,transcription"
            )

    elif withTranscription and withScript:
        m = re.match(
            r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(Arabic|Latin|Chinese|Japanese|Korean|Bangla|Hindi|Symbols|Mixed|None)\s*,(.*)$",
            line,
        )
        if m is None:
            raise Exception(
                "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,script,transcription (and script must be: Arabic|Latin|Chinese|Japanese|Korean|Bangla|Hindi|Symbols)"
            )

        transcription = m.group(numPoints + 2)
    elif withScript and withConfidence:
        m = re.match(
            r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,\s*(Arabic|Latin|Chinese|Japanese|Korean|Bangla|Hindi|Symbols|Mixed|None)\s*$",
            line,
        )
        if m is None:
            raise Exception(
                "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence,script (and script must be: Arabic|Latin|Chinese|Japanese|Korean|Bangla|Hindi|Symbols)"
            )

    elif withConfidence:
        m = re.match(
            r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$",
            line,
        )
        if m is None:
            raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence )")
    elif withTranscription:
        m = re.match(
            r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(.+)$",
            line,
        )
        if m is None:
            raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,transcription )")
    else:
        m = re.match(
            r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*$",
            line,
        )
        if m is None:
            raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4 )")

    points = [m.group(i) for i in range(1, (numPoints + 1))]
    rrc_evaluation_funcs.validate_clockwise_points(points)

    if withScript:
        script = m.group(numPoints + (2 if withConfidence else 1))

    if withTranscription:
        transcription = m.group(numPoints + (2 if withScript or withConfidence else 1))

    if withConfidence:
        try:
            confidence = float(m.group(numPoints + 1))
        except ValueError:
            raise Exception("Confidence value must be a float")

    return points, confidence, script, transcription


def get_tl_line_values_from_file_contents(
    content, CRLF=True, withScript=False, withTranscription=False, withConfidence=False
):
    """
    Returns all points, scripts and transcriptions of a file in lists. Valid line format:
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[script],[transcription]
    """
    pointsList = []
    transcriptionsList = []
    scriptsList = []
    confidencesList = []

    lines = content.split("\r\n" if CRLF else "\n")
    for line in lines:
        line = line.replace("\r", "").replace("\n", "")
        if line != "":
            points, confidence, script, transcription = get_tl_line_values(
                line, withScript, withTranscription, withConfidence
            )
            pointsList.append(points)
            transcriptionsList.append(transcription)
            scriptsList.append(script)
            confidencesList.append(confidence)

    return pointsList, confidencesList, scriptsList, transcriptionsList


def evaluate_method(gtFilePath, submFilePath, evaluationParams, languages=None):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics.
            Ex: { 'Precision': 0.8, 'Recall': 0.9 }
        - samples (optional) Per sample metrics.
            Ex: {
                    'sample1': {'Precision': 0.8, 'Recall': 0.9},
                    'sample2': {'Precision': 0.8, 'Recall': 0.9}
                }
    """

    def polygon_from_points(points, correctOffset=False):
        """
        Returns a Polygon object to use with the Polygon2 class
        from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """

        if correctOffset:
            # this will substract 1 from the coordinates
            # that correspond to the xmax and ymax
            points[2] -= 1
            points[4] -= 1
            points[5] -= 1
            points[7] -= 1

        resBoxes = np.empty([1, 8], dtype="int32")
        resBoxes[0, 0] = int(points[0])
        resBoxes[0, 4] = int(points[1])
        resBoxes[0, 1] = int(points[2])
        resBoxes[0, 5] = int(points[3])
        resBoxes[0, 2] = int(points[4])
        resBoxes[0, 6] = int(points[5])
        resBoxes[0, 3] = int(points[6])
        resBoxes[0, 7] = int(points[7])
        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg.Polygon(pointMat)

    def get_union(pD, pG):
        areaA = pD.area
        areaB = pG.area
        return areaA + areaB - get_intersection(pD, pG)

    def get_intersection_over_union(pD, pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except Exception:
            return 0

    def get_intersection(pD, pG):
        pInt = pD & pG
        if pInt.length == 0:
            return 0
        return pInt.area

    def compute_ap(confList, matchList, numGtCare):
        correct = 0
        AP = 0
        if len(confList) > 0:

            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct) / (n + 1)

            if numGtCare > 0:
                AP /= numGtCare

        return AP

    perSampleMetrics = {}

    matchedSum = 0
    matchedSumUpper = 0
    globalNedSum = 0
    globalNedElements = 0
    globalNedSumUpper = 0
    globalNedElementsUpper = 0

    gt = utils.load_zip_file(
        gtFilePath, evaluationParams["GT_SAMPLE_NAME_2_ID"], languages=languages
    )
    subm = utils.load_zip_file(
        submFilePath, evaluationParams["DET_SAMPLE_NAME_2_ID"], languages=languages
    )

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    arrGlobalConfidences = []
    arrGlobalMatches = []

    for resFile in tqdm(gt):

        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        if gtFile is None:
            raise Exception("The file %s is not UTF-8" % resFile)

        recall = 0
        precision = 0
        hmean = 0
        recallUpper = 0
        precisionUpper = 0
        hmeanUpper = 0
        detCorrect = 0
        detCorrectUpper = 0
        iouMat = np.empty([1, 1])
        gtPols = []
        detPols = []
        gtTrans = []
        detTrans = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCarePolsNum = []  # Array of Ground Truth Polygons' keys marked as don't Care
        detDontCarePolsNum = []  # Array of Detected Polygons' matched with a don't Care GT
        detMatchedNums = []
        pairs = []

        arrSampleConfidences = []
        arrSampleMatch = []
        sampleAP = 0

        ned = 0
        nedSum = 0
        nedElements = 0

        nedUpper = 0
        nedSumUpper = 0
        nedElementsUpper = 0

        evaluationLog = ""

        pointsList, _, _, transcriptionsList = get_tl_line_values_from_file_contents(
            gtFile, evaluationParams["CRLF"], True, True, False
        )
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)
            gtTrans.append(transcription)

        evaluationLog += (
            "GT polygons: "
            + str(len(gtPols))
            + (
                " (" + str(len(gtDontCarePolsNum)) + " don't care)\n"
                if len(gtDontCarePolsNum) > 0
                else "\n"
            )
        )

        if resFile in subm:

            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile])

            (
                pointsList,
                confidencesList,
                _,
                transcriptionsList,
            ) = get_tl_line_values_from_file_contents(
                detFile,
                evaluationParams["CRLF"],
                False,
                True,
                evaluationParams["CONFIDENCES"],
            )

            print("[Debug] Comparing {} ..".format(resFile))

            for n in range(len(pointsList)):
                points = pointsList[n]
                detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                transcription = transcriptionsList[n]
                detTrans.append(transcription)

                if len(gtDontCarePolsNum) > 0:
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        try:
                            intersected_area = get_intersection(dontCarePol, detPol)
                        except TopologicalError:
                            # some gt annotation has self-intersection, e.g.,
                            # POLYGON ((1905 16, 1905 14, 1866 29, 1863 13, 1905 16))
                            # in MLT19/val/val_gts/gt_img_01436.txt
                            # see n375134 for an visualization
                            assert (
                                not dontCarePol.is_valid
                            ), f"dontCarePol={dontCarePol}, detPol={detPol}"
                            print(f"[Warning] Found invalid gt polygon: {dontCarePol} in {resFile}")
                            intersected_area = 0

                        pdDimensions = detPol.area
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if precision > evaluationParams["AREA_PRECISION_CONSTRAINT"]:
                            detDontCarePolsNum.append(len(detPols) - 1)
                            break

            evaluationLog += (
                "DET polygons: "
                + str(len(detPols))
                + (
                    " (" + str(len(detDontCarePolsNum)) + " don't care)\n"
                    if len(detDontCarePolsNum) > 0
                    else "\n"
                )
            )

            if len(gtPols) > 0 and len(detPols) > 0:
                # Calculate IoU and precision matrixs
                outputShape = [len(gtPols), len(detPols)]
                iouMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        try:
                            iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
                        except TopologicalError:
                            assert not pG.is_valid, f"pD={pD}, pG={pG}"
                            print(f"[Warning] Found invalid gt polygon: {pG} in {resFile}")
                            iouMat[gtNum, detNum] = 0

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if (
                            gtRectMat[gtNum] == 0
                            and detRectMat[detNum] == 0
                            and gtNum not in gtDontCarePolsNum
                            and detNum not in detDontCarePolsNum
                        ):
                            if iouMat[gtNum, detNum] > evaluationParams["IOU_CONSTRAINT"]:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                # detection matched only if transcription is equal
                                # (case insensitive)
                                correct = gtTrans[gtNum] == detTrans[detNum]

                                correctUpper = gtTrans[gtNum].upper() == detTrans[detNum].upper()

                                if not correctUpper:
                                    # 'not correctUpper' implies 'not correct'
                                    print(
                                        "Mismatch (gt vs pred): {} vs {}".format(
                                            gtTrans[gtNum], detTrans[detNum]
                                        )
                                    )
                                elif not correct:
                                    # upper case matches, but original words mismatch
                                    print(
                                        "Case Mismatch (gt vs pred): {} vs {}".format(
                                            gtTrans[gtNum], detTrans[detNum]
                                        )
                                    )

                                detCorrect += 1 if correct else 0
                                detCorrectUpper += 1 if correctUpper else 0
                                if correctUpper:
                                    detMatchedNums.append(detNum)
                                pairs.append(
                                    {
                                        "gt": gtNum,
                                        "det": detNum,
                                        "correct": correctUpper,
                                    }
                                )
                                evaluationLog += (
                                    "Match GT #"
                                    + str(gtNum)
                                    + " with Det #"
                                    + str(detNum)
                                    + " trancription correct: "
                                    + str(correctUpper)
                                    + "\n"
                                )

                                # Calculate also 1-NED
                                distance = editdistance.eval(gtTrans[gtNum], detTrans[detNum])
                                nedSum += (
                                    0
                                    if len(gtTrans[gtNum]) == 0 and len(detTrans[detNum]) == 0
                                    else float(distance)
                                    / max(len(gtTrans[gtNum]), len(detTrans[detNum]))
                                )
                                nedElements += 1
                                distanceUpper = editdistance.eval(
                                    gtTrans[gtNum].upper(), detTrans[detNum].upper()
                                )
                                nedSumUpper += (
                                    0
                                    if len(gtTrans[gtNum]) == 0 and len(detTrans[detNum]) == 0
                                    else float(distanceUpper)
                                    / max(len(gtTrans[gtNum]), len(detTrans[detNum]))
                                )
                                nedElementsUpper += 1

                # Add unmatched GT and DET Elements to NED summatory to penalize
                for gtNum in range(len(gtPols)):
                    if gtRectMat[gtNum] == 0 and gtNum not in gtDontCarePolsNum:
                        nedSum += 1
                        nedElements += 1
                        nedSumUpper += 1
                        nedElementsUpper += 1

                for detNum in range(len(detPols)):
                    if detRectMat[detNum] == 0 and detNum not in detDontCarePolsNum:
                        nedSum += 1
                        nedElements += 1
                        nedSumUpper += 1
                        nedElementsUpper += 1

            if evaluationParams["CONFIDENCES"]:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum:
                        # we exclude the don't care detections
                        match = detNum in detMatchedNums

                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)

                        arrGlobalConfidences.append(confidencesList[detNum])
                        arrGlobalMatches.append(match)

        numGtCare = len(gtPols) - len(gtDontCarePolsNum)
        numDetCare = len(detPols) - len(detDontCarePolsNum)
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
            recallUpper = float(1)
            precisionUpper = float(0) if numDetCare > 0 else float(1)
            sampleAP = precisionUpper
            ned = float(0) if numDetCare > 0 else float(1)
            nedUpper = float(0) if numDetCare > 0 else float(1)
            nedSum = numDetCare
            nedSumUpper = numDetCare
            nedElements = numDetCare
            nedElementsUpper = numDetCare
        else:
            recall = float(detCorrect) / numGtCare
            precision = 0 if numDetCare == 0 else float(detCorrect) / numDetCare
            recallUpper = float(detCorrectUpper) / numGtCare
            precisionUpper = 0 if numDetCare == 0 else float(detCorrectUpper) / numDetCare
            if evaluationParams["CONFIDENCES"]:
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare)
            if numDetCare == 0:
                nedSum = numGtCare
                nedSumUpper = numGtCare
                nedElements = numGtCare
                nedElementsUpper = numGtCare
                ned = 0
                nedUpper = 0
            else:
                ned = 0 if nedElements == 0 else 1 - float(nedSum) / nedElements
                nedUpper = 0 if nedElementsUpper == 0 else 1 - float(nedSumUpper) / nedElementsUpper

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
        hmeanUpper = (
            0
            if (precisionUpper + recallUpper) == 0
            else 2.0 * precisionUpper * recallUpper / (precisionUpper + recallUpper)
        )

        matchedSum += detCorrect
        matchedSumUpper += detCorrectUpper
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        globalNedSum += nedSum
        globalNedElements += nedElements
        globalNedSumUpper += nedSumUpper
        globalNedElementsUpper += nedElementsUpper

        perSampleMetrics[resFile] = {
            "precision": precision,
            "recall": recall,
            "hmean": hmean,
            "precisionUpper": precisionUpper,
            "recallUpper": recallUpper,
            "hmeanUpper": hmeanUpper,
            "pairs": pairs,
            "AP": sampleAP,
            "ned": ned,
            "nedUpper": nedUpper,
            "iouMat": [] if len(detPols) > 100 else iouMat.tolist(),
            "gtPolPoints": gtPolPoints,
            "detPolPoints": detPolPoints,
            "gtTrans": gtTrans,
            "detTrans": detTrans,
            "gtDontCare": gtDontCarePolsNum,
            "detDontCare": detDontCarePolsNum,
            "evaluationParams": evaluationParams,
            "evaluationLog": evaluationLog,
        }

    # Compute AP
    AP = 0
    if evaluationParams["CONFIDENCES"]:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    methodHmean = (
        0
        if methodRecall + methodPrecision == 0
        else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
    )

    methodRecallUpper = 0 if numGlobalCareGt == 0 else float(matchedSumUpper) / numGlobalCareGt
    methodPrecisionUpper = 0 if numGlobalCareDet == 0 else float(matchedSumUpper) / numGlobalCareDet
    methodHmeanUpper = (
        0
        if methodRecallUpper + methodPrecisionUpper == 0
        else 2
        * methodRecallUpper
        * methodPrecisionUpper
        / (methodRecallUpper + methodPrecisionUpper)
    )

    methodNed = 0 if globalNedElements == 0 else 1 - float(globalNedSum) / globalNedElements
    methodNedUpper = (
        0 if globalNedElementsUpper == 0 else 1 - float(globalNedSumUpper) / globalNedElementsUpper
    )

    methodMetrics = {
        "precision": methodPrecision,
        "recall": methodRecall,
        "hmean": methodHmean,
        "precisionUpper": methodPrecisionUpper,
        "recallUpper": methodRecallUpper,
        "hmeanUpper": methodHmeanUpper,
        "AP": AP,
        "ned": methodNed,
        "nedUpper": methodNedUpper,
    }

    resDict = {
        "calculated": True,
        "Message": "",
        "method": methodMetrics,
        "per_sample": perSampleMetrics,
    }

    return resDict


def mlt19_eval_task4(
    pred_zip_file,
    gt_zip_file=f"/checkpoint/{getpass.getuser()}/datasets/MLT19/eval/val/task4/gt.zip",
    output_dir=None,
    languages=None,
):
    p = {"g": gt_zip_file, "s": pred_zip_file}
    if output_dir is not None:
        p["o"] = output_dir
    return rrc_evaluation_funcs.main_evaluation(
        p,
        default_evaluation_params,
        validate_data,
        evaluate_method,
        languages=languages,
    )
