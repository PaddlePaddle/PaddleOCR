#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon


class DetectionICDAR2013Evaluator(object):
    def __init__(
        self,
        area_recall_constraint=0.8,
        area_precision_constraint=0.4,
        ev_param_ind_center_diff_thr=1,
        mtype_oo_o=1.0,
        mtype_om_o=0.8,
        mtype_om_m=1.0,
    ):
        self.area_recall_constraint = area_recall_constraint
        self.area_precision_constraint = area_precision_constraint
        self.ev_param_ind_center_diff_thr = ev_param_ind_center_diff_thr
        self.mtype_oo_o = mtype_oo_o
        self.mtype_om_o = mtype_om_o
        self.mtype_om_m = mtype_om_m

    def evaluate_image(self, gt, pred):
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def one_to_one_match(row, col):
            cont = 0
            for j in range(len(recallMat[0])):
                if (
                    recallMat[row, j] >= self.area_recall_constraint
                    and precisionMat[row, j] >= self.area_precision_constraint
                ):
                    cont = cont + 1
            if cont != 1:
                return False
            cont = 0
            for i in range(len(recallMat)):
                if (
                    recallMat[i, col] >= self.area_recall_constraint
                    and precisionMat[i, col] >= self.area_precision_constraint
                ):
                    cont = cont + 1
            if cont != 1:
                return False

            if (
                recallMat[row, col] >= self.area_recall_constraint
                and precisionMat[row, col] >= self.area_precision_constraint
            ):
                return True
            return False

        def one_to_many_match(gtNum):
            many_sum = 0
            detRects = []
            for detNum in range(len(recallMat[0])):
                if (
                    gtRectMat[gtNum] == 0
                    and detRectMat[detNum] == 0
                    and detNum not in detDontCareRectsNum
                ):
                    if precisionMat[gtNum, detNum] >= self.area_precision_constraint:
                        many_sum += recallMat[gtNum, detNum]
                        detRects.append(detNum)
            if round(many_sum, 4) >= self.area_recall_constraint:
                return True, detRects
            else:
                return False, []

        def many_to_one_match(detNum):
            many_sum = 0
            gtRects = []
            for gtNum in range(len(recallMat)):
                if (
                    gtRectMat[gtNum] == 0
                    and detRectMat[detNum] == 0
                    and gtNum not in gtDontCareRectsNum
                ):
                    if recallMat[gtNum, detNum] >= self.area_recall_constraint:
                        many_sum += precisionMat[gtNum, detNum]
                        gtRects.append(gtNum)
            if round(many_sum, 4) >= self.area_precision_constraint:
                return True, gtRects
            else:
                return False, []

        def center_distance(r1, r2):
            return ((np.mean(r1, axis=0) - np.mean(r2, axis=0)) ** 2).sum() ** 0.5

        def diag(r):
            r = np.array(r)
            return (
                (r[:, 0].max() - r[:, 0].min()) ** 2
                + (r[:, 1].max() - r[:, 1].min()) ** 2
            ) ** 0.5

        perSampleMetrics = {}

        recall = 0
        precision = 0
        hmean = 0
        recallAccum = 0.0
        precisionAccum = 0.0
        gtRects = []
        detRects = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCareRectsNum = (
            []
        )  # Array of Ground Truth Rectangles' keys marked as don't Care
        detDontCareRectsNum = (
            []
        )  # Array of Detected Rectangles' matched with a don't Care GT
        pairs = []
        evaluationLog = ""

        recallMat = np.empty([1, 1])
        precisionMat = np.empty([1, 1])

        for n in range(len(gt)):
            points = gt[n]["points"]
            # transcription = gt[n]['text']
            dontCare = gt[n]["ignore"]

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtRects.append(points)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCareRectsNum.append(len(gtRects) - 1)

        evaluationLog += (
            "GT rectangles: "
            + str(len(gtRects))
            + (
                " (" + str(len(gtDontCareRectsNum)) + " don't care)\n"
                if len(gtDontCareRectsNum) > 0
                else "\n"
            )
        )

        for n in range(len(pred)):
            points = pred[n]["points"]

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detRect = points
            detRects.append(detRect)
            detPolPoints.append(points)
            if len(gtDontCareRectsNum) > 0:
                for dontCareRectNum in gtDontCareRectsNum:
                    dontCareRect = gtRects[dontCareRectNum]
                    intersected_area = get_intersection(dontCareRect, detRect)
                    rdDimensions = Polygon(detRect).area
                    if rdDimensions == 0:
                        precision = 0
                    else:
                        precision = intersected_area / rdDimensions
                    if precision > self.area_precision_constraint:
                        detDontCareRectsNum.append(len(detRects) - 1)
                        break

        evaluationLog += (
            "DET rectangles: "
            + str(len(detRects))
            + (
                " (" + str(len(detDontCareRectsNum)) + " don't care)\n"
                if len(detDontCareRectsNum) > 0
                else "\n"
            )
        )

        if len(gtRects) == 0:
            recall = 1
            precision = 0 if len(detRects) > 0 else 1

        if len(detRects) > 0:
            # Calculate recall and precision matrixs
            outputShape = [len(gtRects), len(detRects)]
            recallMat = np.empty(outputShape)
            precisionMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtRects), np.int8)
            detRectMat = np.zeros(len(detRects), np.int8)
            for gtNum in range(len(gtRects)):
                for detNum in range(len(detRects)):
                    rG = gtRects[gtNum]
                    rD = detRects[detNum]
                    intersected_area = get_intersection(rG, rD)
                    rgDimensions = Polygon(rG).area
                    rdDimensions = Polygon(rD).area
                    recallMat[gtNum, detNum] = (
                        0 if rgDimensions == 0 else intersected_area / rgDimensions
                    )
                    precisionMat[gtNum, detNum] = (
                        0 if rdDimensions == 0 else intersected_area / rdDimensions
                    )

            # Find one-to-one matches
            evaluationLog += "Find one-to-one matches\n"
            for gtNum in range(len(gtRects)):
                for detNum in range(len(detRects)):
                    if (
                        gtRectMat[gtNum] == 0
                        and detRectMat[detNum] == 0
                        and gtNum not in gtDontCareRectsNum
                        and detNum not in detDontCareRectsNum
                    ):
                        match = one_to_one_match(gtNum, detNum)
                        if match is True:
                            # in deteval we have to make other validation before mark as one-to-one
                            rG = gtRects[gtNum]
                            rD = detRects[detNum]
                            normDist = center_distance(rG, rD)
                            normDist /= diag(rG) + diag(rD)
                            normDist *= 2.0
                            if normDist < self.ev_param_ind_center_diff_thr:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                recallAccum += self.mtype_oo_o
                                precisionAccum += self.mtype_oo_o
                                pairs.append({"gt": gtNum, "det": detNum, "type": "OO"})
                                evaluationLog += (
                                    "Match GT #"
                                    + str(gtNum)
                                    + " with Det #"
                                    + str(detNum)
                                    + "\n"
                                )
                            else:
                                evaluationLog += (
                                    "Match Discarded GT #"
                                    + str(gtNum)
                                    + " with Det #"
                                    + str(detNum)
                                    + " normDist: "
                                    + str(normDist)
                                    + " \n"
                                )
            # Find one-to-many matches
            evaluationLog += "Find one-to-many matches\n"
            for gtNum in range(len(gtRects)):
                if gtNum not in gtDontCareRectsNum:
                    match, matchesDet = one_to_many_match(gtNum)
                    if match is True:
                        evaluationLog += "num_overlaps_gt=" + str(
                            num_overlaps_gt(gtNum)
                        )
                        gtRectMat[gtNum] = 1
                        recallAccum += (
                            self.mtype_oo_o if len(matchesDet) == 1 else self.mtype_om_o
                        )
                        precisionAccum += (
                            self.mtype_oo_o
                            if len(matchesDet) == 1
                            else self.mtype_om_o * len(matchesDet)
                        )
                        pairs.append(
                            {
                                "gt": gtNum,
                                "det": matchesDet,
                                "type": "OO" if len(matchesDet) == 1 else "OM",
                            }
                        )
                        for detNum in matchesDet:
                            detRectMat[detNum] = 1
                        evaluationLog += (
                            "Match GT #"
                            + str(gtNum)
                            + " with Det #"
                            + str(matchesDet)
                            + "\n"
                        )

            # Find many-to-one matches
            evaluationLog += "Find many-to-one matches\n"
            for detNum in range(len(detRects)):
                if detNum not in detDontCareRectsNum:
                    match, matchesGt = many_to_one_match(detNum)
                    if match is True:
                        detRectMat[detNum] = 1
                        recallAccum += (
                            self.mtype_oo_o
                            if len(matchesGt) == 1
                            else self.mtype_om_m * len(matchesGt)
                        )
                        precisionAccum += (
                            self.mtype_oo_o if len(matchesGt) == 1 else self.mtype_om_m
                        )
                        pairs.append(
                            {
                                "gt": matchesGt,
                                "det": detNum,
                                "type": "OO" if len(matchesGt) == 1 else "MO",
                            }
                        )
                        for gtNum in matchesGt:
                            gtRectMat[gtNum] = 1
                        evaluationLog += (
                            "Match GT #"
                            + str(matchesGt)
                            + " with Det #"
                            + str(detNum)
                            + "\n"
                        )

            numGtCare = len(gtRects) - len(gtDontCareRectsNum)
            if numGtCare == 0:
                recall = float(1)
                precision = float(0) if len(detRects) > 0 else float(1)
            else:
                recall = float(recallAccum) / numGtCare
                precision = (
                    float(0)
                    if (len(detRects) - len(detDontCareRectsNum)) == 0
                    else float(precisionAccum)
                    / (len(detRects) - len(detDontCareRectsNum))
                )
            hmean = (
                0
                if (precision + recall) == 0
                else 2.0 * precision * recall / (precision + recall)
            )

        numGtCare = len(gtRects) - len(gtDontCareRectsNum)
        numDetCare = len(detRects) - len(detDontCareRectsNum)

        perSampleMetrics = {
            "precision": precision,
            "recall": recall,
            "hmean": hmean,
            "pairs": pairs,
            "recallMat": [] if len(detRects) > 100 else recallMat.tolist(),
            "precisionMat": [] if len(detRects) > 100 else precisionMat.tolist(),
            "gtPolPoints": gtPolPoints,
            "detPolPoints": detPolPoints,
            "gtCare": numGtCare,
            "detCare": numDetCare,
            "gtDontCare": gtDontCareRectsNum,
            "detDontCare": detDontCareRectsNum,
            "recallAccum": recallAccum,
            "precisionAccum": precisionAccum,
            "evaluationLog": evaluationLog,
        }

        return perSampleMetrics

    def combine_results(self, results):
        numGt = 0
        numDet = 0
        methodRecallSum = 0
        methodPrecisionSum = 0

        for result in results:
            numGt += result["gtCare"]
            numDet += result["detCare"]
            methodRecallSum += result["recallAccum"]
            methodPrecisionSum += result["precisionAccum"]

        methodRecall = 0 if numGt == 0 else methodRecallSum / numGt
        methodPrecision = 0 if numDet == 0 else methodPrecisionSum / numDet
        methodHmean = (
            0
            if methodRecall + methodPrecision == 0
            else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
        )

        methodMetrics = {
            "precision": methodPrecision,
            "recall": methodRecall,
            "hmean": methodHmean,
        }

        return methodMetrics


if __name__ == "__main__":
    evaluator = DetectionICDAR2013Evaluator()
    gts = [
        [
            {
                "points": [(0, 0), (1, 0), (1, 1), (0, 1)],
                "text": 1234,
                "ignore": False,
            },
            {
                "points": [(2, 2), (3, 2), (3, 3), (2, 3)],
                "text": 5678,
                "ignore": True,
            },
        ]
    ]
    preds = [
        [
            {
                "points": [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
                "text": 123,
                "ignore": False,
            }
        ]
    ]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
