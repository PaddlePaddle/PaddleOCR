#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
from . import rrc_evaluation_funcs
import Polygon as plg
import numpy as np


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        "IOU_CONSTRAINT": 0.5,
        "AREA_PRECISION_CONSTRAINT": 0.5,
        "GT_SAMPLE_NAME_2_ID": "gt_img_([0-9]+).txt",
        "DET_SAMPLE_NAME_2_ID": "res_img_([0-9]+).txt",
        "LTRB": False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        "CRLF": False,  # Lines are delimited by Windows CRLF format
        "CONFIDENCES": False,  # Detections must include confidence value. AP will be calculated
        "PER_SAMPLE_RESULTS": True,  # Generate per sample results and produce data for visualization
    }


def validate_data(gtFilePath, submFilePath, evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    gt = rrc_evaluation_funcs.load_folder_file(
        gtFilePath, evaluationParams["GT_SAMPLE_NAME_2_ID"]
    )

    subm = rrc_evaluation_funcs.load_folder_file(
        submFilePath, evaluationParams["DET_SAMPLE_NAME_2_ID"], True
    )

    # Validate format of GroundTruth
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file(
            k, gt[k], evaluationParams["CRLF"], evaluationParams["LTRB"], True
        )

    # Validate format of results
    for k in subm:
        if (k in gt) == False:
            raise Exception("The sample %s not present in GT" % k)

        rrc_evaluation_funcs.validate_lines_in_file(
            k,
            subm[k],
            evaluationParams["CRLF"],
            evaluationParams["LTRB"],
            False,
            evaluationParams["CONFIDENCES"],
        )


def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
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

    def rectangle_to_polygon(rect):
        resBoxes = np.empty([1, 8], dtype="int32")
        resBoxes[0, 0] = int(rect.xmin)
        resBoxes[0, 4] = int(rect.ymax)
        resBoxes[0, 1] = int(rect.xmin)
        resBoxes[0, 5] = int(rect.ymin)
        resBoxes[0, 2] = int(rect.xmax)
        resBoxes[0, 6] = int(rect.ymin)
        resBoxes[0, 3] = int(rect.xmax)
        resBoxes[0, 7] = int(rect.ymax)

        pointMat = resBoxes[0].reshape([2, 4]).T

        return plg.Polygon(pointMat)

    def rectangle_to_points(rect):
        points = [
            int(rect.xmin),
            int(rect.ymax),
            int(rect.xmax),
            int(rect.ymax),
            int(rect.xmax),
            int(rect.ymin),
            int(rect.xmin),
            int(rect.ymin),
        ]
        return points

    def get_union(pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)

    def get_intersection_over_union(pD, pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def get_intersection(pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

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

    Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

    gt = rrc_evaluation_funcs.load_folder_file(
        gtFilePath, evaluationParams["GT_SAMPLE_NAME_2_ID"]
    )
    subm = rrc_evaluation_funcs.load_folder_file(
        submFilePath, evaluationParams["DET_SAMPLE_NAME_2_ID"], True
    )

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    arrGlobalConfidences = []
    arrGlobalMatches = []

    for resFile in gt:
        gtFile = gt[resFile]  # rrc_evaluation_funcs.decode_utf8(gt[resFile])
        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []
        sampleAP = 0

        evaluationLog = ""

        (
            pointsList,
            _,
            transcriptionsList,
        ) = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
            gtFile, evaluationParams["CRLF"], evaluationParams["LTRB"], True, False
        )
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams["LTRB"]:
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

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
            detFile = subm[resFile]  # rrc_evaluation_funcs.decode_utf8(subm[resFile])

            (
                pointsList,
                confidencesList,
                _,
            ) = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
                detFile,
                evaluationParams["CRLF"],
                evaluationParams["LTRB"],
                False,
                evaluationParams["CONFIDENCES"],
            )
            for n in range(len(pointsList)):
                points = pointsList[n]

                if evaluationParams["LTRB"]:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                if len(gtDontCarePolsNum) > 0:
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol, detPol)
                        pdDimensions = detPol.area()
                        precision = (
                            0 if pdDimensions == 0 else intersected_area / pdDimensions
                        )
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
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if (
                            gtRectMat[gtNum] == 0
                            and detRectMat[detNum] == 0
                            and gtNum not in gtDontCarePolsNum
                            and detNum not in detDontCarePolsNum
                        ):
                            if (
                                iouMat[gtNum, detNum]
                                > evaluationParams["IOU_CONSTRAINT"]
                            ):
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                detMatched += 1
                                pairs.append({"gt": gtNum, "det": detNum})
                                detMatchedNums.append(detNum)
                                evaluationLog += (
                                    "Match GT #"
                                    + str(gtNum)
                                    + " with Det #"
                                    + str(detNum)
                                    + "\n"
                                )

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
            sampleAP = precision
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare
            if (
                evaluationParams["CONFIDENCES"]
                and evaluationParams["PER_SAMPLE_RESULTS"]
            ):
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare)

        hmean = (
            0
            if (precision + recall) == 0
            else 2.0 * precision * recall / (precision + recall)
        )

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        if evaluationParams["PER_SAMPLE_RESULTS"]:
            perSampleMetrics[resFile] = {
                "precision": precision,
                "recall": recall,
                "hmean": hmean,
                "pairs": pairs,
                "AP": sampleAP,
                "iouMat": [] if len(detPols) > 100 else iouMat.tolist(),
                "gtPolPoints": gtPolPoints,
                "detPolPoints": detPolPoints,
                "gtDontCare": gtDontCarePolsNum,
                "detDontCare": detDontCarePolsNum,
                "evaluationParams": evaluationParams,
                "evaluationLog": evaluationLog,
            }

    # Compute MAP and MAR
    AP = 0
    if evaluationParams["CONFIDENCES"]:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = (
        0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    )
    methodHmean = (
        0
        if methodRecall + methodPrecision == 0
        else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
    )

    methodMetrics = {
        "precision": methodPrecision,
        "recall": methodRecall,
        "hmean": methodHmean,
        "AP": AP,
    }

    resDict = {
        "calculated": True,
        "Message": "",
        "method": methodMetrics,
        "per_sample": perSampleMetrics,
    }

    return resDict


def cal_recall_precison_f1(gt_path, result_path, show_result=False):
    p = {"g": gt_path, "s": result_path}
    result = rrc_evaluation_funcs.main_evaluation(
        p, default_evaluation_params, validate_data, evaluate_method, show_result
    )
    return result["method"]
