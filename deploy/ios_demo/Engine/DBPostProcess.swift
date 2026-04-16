// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import CoreGraphics

// MARK: - Detection Result

/// A detected text region with its bounding quadrilateral and confidence score.
struct DetectionBox {
    /// Four corner points of the bounding quadrilateral, each as [x, y].
    /// Order: top-left, top-right, bottom-right, bottom-left (`getMiniBoxes` / min-area rect).
    let points: [[Int32]]

    /// Confidence score from `boxScoreFast` (mean probability in the box region).
    let score: Float
}

// MARK: - DBPostProcessor

/// DB (Differentiable Binarization) text detection postprocessor.
///
/// Maps raw model output probability map to
/// bounding polygons (threshold → contours → min-area quads → score filter → unclip → scale).
///
/// Steps: threshold → contours → min-area quads → box score → polygon offset (unclip) → scale to original size.
///
struct DBPostProcessor {

    /// Binary threshold for the probability map (pixels above this are foreground).
    let thresh: Float

    /// Minimum box confidence score to keep a detection.
    let boxThresh: Float

    /// Maximum number of contours to evaluate.
    let maxCandidates: Int

    /// Expansion ratio for polygon offset (unclip step).
    let unclipRatio: Float

    /// Minimum side length of a bounding box to be kept (pixels in probability map space).
    let minSize: Float = 3.0

    /// Score mode: only `"fast"` is implemented (`"slow"` is not supported).
    let scoreMode: String = "fast"

    // MARK: - Initializers

    /// Initialize with explicit parameters.
    init(thresh: Float = 0.3, boxThresh: Float = 0.6, maxCandidates: Int = 1000, unclipRatio: Float = 1.5) {
        self.thresh = thresh
        self.boxThresh = boxThresh
        self.maxCandidates = maxCandidates
        self.unclipRatio = unclipRatio
    }

    /// Initialize from a parsed ``PostProcessConfig`.
    init(config: PostProcessConfig) {
        self.thresh = config.thresh
        self.boxThresh = config.boxThresh
        self.maxCandidates = config.maxCandidates
        self.unclipRatio = config.unclipRatio
    }

    // MARK: - Public API

    /// Process raw ORT detection output into bounding polygons.
    ///
    /// - Parameters:
    ///   - outputTensor: Raw float output from ONNX Runtime, shape [1, 1, H, W].
    ///   - tensorHeight: Height of the output tensor (H).
    ///   - tensorWidth: Width of the output tensor (W).
    ///   - originalWidth: Width of the original input image.
    ///   - originalHeight: Height of the original input image.
    /// - Returns: Array of detected text boxes with confidence scores.
    func process(
        outputTensor: [Float],
        tensorHeight: Int,
        tensorWidth: Int,
        originalWidth: Int,
        originalHeight: Int
    ) -> [DetectionBox] {
        // 1. Extract probability map: pred[0, 0, :, :] from shape [1, 1, H, W]
        let mapSize = tensorHeight * tensorWidth
        let pred: [Float]
        if outputTensor.count >= mapSize {
            pred = Array(outputTensor.prefix(mapSize))
        } else {
            return []
        }

        // 2. Binary threshold: create mask where prob > thresh
        var binaryMask = [UInt8](repeating: 0, count: mapSize)
        for i in 0..<mapSize {
            binaryMask[i] = pred[i] > thresh ? 1 : 0
        }

        // 3. Find contours on the binary mask
        let contours = findContours(
            binaryMask: binaryMask,
            width: tensorWidth,
            height: tensorHeight
        )

        // 4. Process each contour
        let numContours = min(contours.count, maxCandidates)
        var results: [DetectionBox] = []

        for i in 0..<numContours {
            let contour = contours[i]

            // 4a. Get minimum area bounding rectangle
            guard let (miniBox, sside) = getMiniBoxes(contour: contour) else {
                continue
            }
            if sside < minSize {
                continue
            }

            // 4b. Score the box using fast mode
            let points = miniBox.map { [$0.x, $0.y] }
            let score = boxScoreFast(
                pred: pred,
                predWidth: tensorWidth,
                predHeight: tensorHeight,
                box: points
            )
            if boxThresh > score {
                continue
            }

            // 4c. Expand polygon (unclip) via integer-grid offset.
            let nsMini = miniBox.map { NSValue(cgPoint: CGPoint(x: CGFloat($0.x), y: CGFloat($0.y))) }
            let area = Double(PDBOpenCVDBBridge.contourArea(fromPoints: nsMini))
            let length = Double(PDBOpenCVDBBridge.arcLength(fromPoints: nsMini, closed: true))
            guard length > 0 else { continue }
            let distance = area * Double(unclipRatio) / length

            let rings = PDBPolygonOffsetBridge.inflateClosedPolygon(with: nsMini, distance: distance) as [[NSValue]]
            guard rings.count == 1 else {
                continue
            }
            let firstRing = rings[0]

            // 4d. Second getMiniBoxes on the expanded polygon
            let expandedFloats = firstRing.map { FloatPoint(x: Float($0.cgPointValue.x), y: Float($0.cgPointValue.y)) }
            guard let (finalBox, finalSside) = getMiniBoxes(contour: expandedFloats) else {
                continue
            }
            if finalSside < minSize + 2 {
                continue
            }

            // 4e. Scale to original image dimensions
            let w = Float(tensorWidth)
            let h = Float(tensorHeight)
            let dw = Float(originalWidth)
            let dh = Float(originalHeight)

            var scaledPoints: [[Int32]] = []
            for pt in finalBox {
                let sx = Int32(min(max((pt.x / w * dw).rounded(), 0), dw))
                let sy = Int32(min(max((pt.y / h * dh).rounded(), 0), dh))
                scaledPoints.append([sx, sy])
            }

            results.append(DetectionBox(points: scaledPoints, score: score))
        }

        return results
    }
}

// MARK: - Internal Point Type

/// Float 2D point for `DBPostProcessor` geometry helpers.
internal struct FloatPoint {
    var x: Float
    var y: Float
}

// MARK: - OpenCV (CocoaPods OpenCV 4.3.x)

extension DBPostProcessor {

    /// Foreground pixels as 255 on an 8-bit mask; runs `findContours` with `RETR_LIST` and `CHAIN_APPROX_SIMPLE`.
    func findContours(binaryMask: [UInt8], width: Int, height: Int) -> [[FloatPoint]] {
        let data = Data(binaryMask)
        let raw = PDBOpenCVDBBridge.findContours(
            binaryMask: data,
            width: width,
            height: height,
            maxCandidates: maxCandidates
        )
        return raw.map { contour in
            contour.map { v in
                let p = v.cgPointValue
                return FloatPoint(x: Float(p.x), y: Float(p.y))
            }
        }
    }

    /// Minimum-area rectangle: `minAreaRect` on contour points, `boxPoints` for corners, then TL/TR/BR/BL ordering.
    func getMiniBoxes(contour: [FloatPoint]) -> (box: [FloatPoint], minSide: Float)? {
        guard contour.count >= 2 else { return nil }
        let values = contour.map { NSValue(cgPoint: CGPoint(x: CGFloat($0.x), y: CGFloat($0.y))) }
        guard let result = PDBOpenCVDBBridge.miniBox(fromPoints: values) else { return nil }
        let corners = result.corners.map { v in
            let p = v.cgPointValue
            return FloatPoint(x: Float(p.x), y: Float(p.y))
        }
        return (corners, result.minSide)
    }
}

// MARK: - Box Scoring

extension DBPostProcessor {

    /// Compute the mean probability score within a box region of the probability map.
    ///
    /// Mean probability inside the quad: `fillPoly` mask on the score map, then `mean` over the masked region.
    func boxScoreFast(
        pred: [Float],
        predWidth: Int,
        predHeight: Int,
        box: [[Float]]
    ) -> Float {
        let predData = pred.withUnsafeBufferPointer { Data(buffer: $0) }
        let boxNS: [[NSNumber]] = box.map { [NSNumber(value: $0[0]), NSNumber(value: $0[1])] }
        return PDBOpenCVImageBridge.meanPredInQuad(
            predData,
            width: predWidth,
            height: predHeight,
            box: boxNS
        )
    }
}
