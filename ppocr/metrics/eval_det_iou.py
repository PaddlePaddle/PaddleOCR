#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from shapely.geometry import Polygon

"""
reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        care_gt, ignored_gt = [], []
        for item in gt:
            polygon = Polygon(item["points"])
            if not polygon.is_valid:
                continue
            if item["ignore"]:
                ignored_gt.append(polygon)
            else:
                care_gt.append(polygon)

        detections = []
        for item in pred:
            polygon = Polygon(item["points"])
            if polygon.is_valid:
                detections.append(polygon)

        # A positive intersection requires overlapping bounds on both axes.
        # Keep candidates in input order for the evaluator's greedy matching.
        bounds = np.asarray(
            [polygon.bounds or (np.nan,) * 4 for polygon in detections]
        ).reshape(-1, 4)
        available = np.ones(len(detections), dtype=bool)

        def candidates(polygon, threshold):
            if threshold < 0:
                # Disjoint pairs also pass a negative threshold.
                return np.flatnonzero(available)
            if polygon.is_empty:
                return ()
            xmin, ymin, xmax, ymax = polygon.bounds
            return np.flatnonzero(
                available
                & (bounds[:, 0] < xmax)
                & (bounds[:, 2] > xmin)
                & (bounds[:, 1] < ymax)
                & (bounds[:, 3] > ymin)
            )

        for polygon in ignored_gt:
            for idx in candidates(polygon, self.area_precision_constraint):
                detection = detections[idx]
                area = detection.area
                coverage = polygon.intersection(detection).area / area if area else 0
                if coverage > self.area_precision_constraint:
                    available[idx] = False

        det_care = int(available.sum())
        matched = 0
        for polygon in care_gt:
            for idx in candidates(polygon, self.iou_constraint):
                detection = detections[idx]
                intersection = detection.intersection(polygon).area
                iou = intersection / detection.union(polygon).area
                if iou > self.iou_constraint:
                    available[idx] = False
                    matched += 1
                    break

        return {"gtCare": len(care_gt), "detCare": det_care, "detMatched": matched}

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result["gtCare"]
            numGlobalCareDet += result["detCare"]
            matchedSum += result["detMatched"]

        methodRecall = (
            0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        )
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
        }

        return methodMetrics


if __name__ == "__main__":
    evaluator = DetectionIoUEvaluator()
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
                "ignore": False,
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
