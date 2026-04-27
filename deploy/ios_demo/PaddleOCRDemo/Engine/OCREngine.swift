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

import CoreGraphics
import Foundation

// MARK: - OCR run result types

/// A single OCR result: one detected text region with its recognized text.
struct OCRResult {
    /// Four corner points of the bounding polygon [x, y] in original image coordinates.
    /// Order: top-left, top-right, bottom-right, bottom-left.
    let polygon: [[Int32]]
    /// The recognized text string from CTC decoding.
    let text: String
    /// Recognition confidence score (0.0 to 1.0).
    let confidence: Float
}

/// Result of one full OCR run on an image, with per-stage timing.
struct OCRRunResult {
    /// All detected and recognized text regions, in reading order.
    let results: [OCRResult]
    /// Total time spent in the detection stage (preprocess + inference + postprocess).
    let detectionTime: TimeInterval
    /// Seconds in detection preprocessing (resize, normalize, HWC-to-CHW).
    let detectionPreprocessTime: TimeInterval
    /// Seconds in detection ONNX inference.
    let detectionInferenceTime: TimeInterval
    /// Seconds in detection postprocessing (DB map, contours, polygons).
    let detectionPostprocessTime: TimeInterval
    /// Total time spent recognizing all text regions (sum of per-line totals).
    let recognitionTime: TimeInterval
    /// Sum of recognition preprocess time across all lines (batched work split per line).
    let recognitionPreprocessTime: TimeInterval
    /// Sum of recognition inference time across all lines.
    let recognitionInferenceTime: TimeInterval
    /// Sum of recognition postprocess (CTC decode) time across all lines.
    let recognitionPostprocessTime: TimeInterval
    /// Wall-clock time for the entire run (detect + sort + crop + recognize).
    let totalTime: TimeInterval
    /// Time not attributed to detection or recognition totals.
    let pipelineOverheadTime: TimeInterval
    /// Number of text lines from detection sent through recognition.
    let recognitionLineCount: Int
    /// Detection model input tensor shape for this OCR run, e.g. [1, 3, H, W].
    let detectionInputTensorShape: [Int]
    /// Recognition model input tensor shapes for each non-empty recognition batch.
    let recognitionInputTensorShapes: [[Int]]
    /// Per-line recognition ONNX inference times in **seconds**.
    let lineRecognitionInferenceTimes: [TimeInterval]
    /// Per-line recognition preprocess times in **seconds**.
    let lineRecognitionPreprocessTimes: [TimeInterval]
    /// Per-line recognition postprocess times in **seconds**.
    let lineRecognitionPostprocessTimes: [TimeInterval]
}

// MARK: - OCR Engine Errors

enum OCREngineError: LocalizedError {
    case quadTextCropFailed(boxIndex: Int, underlying: Error)
    case recognitionBatchSizeMismatch(expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .quadTextCropFailed(let idx, let err):
            return "Quad text crop failed for box \(idx): \(err.localizedDescription)"
        case .recognitionBatchSizeMismatch(let expected, let actual):
            return "Recognition returned \(actual) results but \(expected) crops were sent"
        }
    }
}

// MARK: - OCREngine

/// End-to-end OCR: detect → sort → crop → recognize.
///
/// Composes `DetectionEngine`, `BoxSorter`, `QuadTextCrop`, and `RecognitionEngine`
/// into a single `run(CGImage)` call.
///
/// Runs entirely via async/await. Since `DetectionEngine` and
/// `RecognitionEngine` delegate to `ORTSessionManager` (a Swift actor), all ORT
/// calls are off the main thread.
///
/// Usage:
/// ```swift
/// let manager = ORTSessionManager()
/// try await manager.loadModels()
/// let engine = try OCREngine(sessionManager: manager)
/// let result = try await engine.run(cgImage, params: .noOverrides)
/// for item in result.results {
///     print("\(item.text) (\(item.confidence))")
/// }
/// ```
class OCREngine {
    private let detectionEngine: DetectionEngine
    private let recognitionEngine: RecognitionEngine

    /// Initialize with an existing ORTSessionManager (models must already be loaded).
    ///
    /// Creates both DetectionEngine and RecognitionEngine.
    ///
    /// - Parameter sessionManager: A loaded ORTSessionManager.
    /// - Throws: If either engine's model config cannot be loaded.
    init(sessionManager: ORTSessionManager) throws {
        self.detectionEngine = try DetectionEngine(sessionManager: sessionManager)
        self.recognitionEngine = try RecognitionEngine(sessionManager: sessionManager)
    }

    func baselineRuntimeDefaults() -> ResolvedOCRRuntimeParams {
        ResolvedOCRRuntimeParams.fromModelConfig(detectionEngine.modelConfig)
    }

    /// Run full OCR on an image.
    ///
    /// End-to-end OCR flow (detect → sort → crop → recognize):
    /// 1. **Detect**: Run detection to get bounding polygons
    /// 2. **Sort**: Sort boxes in reading order (top-to-bottom, left-to-right)
    /// 3. **Crop + Recognize**: Crop each region, order crops by ascending width/height for batched
    ///    recognition, then map results back to reading order
    ///
    /// - Parameters:
    ///   - image: The input `CGImage` to process.
    ///   - params: Optional runtime parameter overrides (see ``OCRRuntimeParams``).
    /// - Returns: `OCRRunResult` with all line results and timing.
    func run(_ image: CGImage, params: OCRRuntimeParams = .noOverrides) async throws -> OCRRunResult {
        let runStart = CFAbsoluteTimeGetCurrent()

        let resolved = params.resolved(detectionEngine.modelConfig)

        let detResult = try await detectionEngine.detect(image, runtimeParams: params)
        let sortedBoxes = BoxSorter.sortInReadingOrder(detResult.boxes)
        let (ocrResults, recTiming, perLine) = try await recognizeSortedBoxes(
            sortedBoxes,
            sourceImage: image,
            resolved: resolved
        )

        let totalTime = CFAbsoluteTimeGetCurrent() - runStart
        let overhead = totalTime - detResult.totalTime - recTiming.total

        return OCRRunResult(
            results: ocrResults,
            detectionTime: detResult.totalTime,
            detectionPreprocessTime: detResult.preprocessTime,
            detectionInferenceTime: detResult.inferenceTime,
            detectionPostprocessTime: detResult.postprocessTime,
            recognitionTime: recTiming.total,
            recognitionPreprocessTime: recTiming.preprocess,
            recognitionInferenceTime: recTiming.inference,
            recognitionPostprocessTime: recTiming.postprocess,
            totalTime: totalTime,
            pipelineOverheadTime: max(0, overhead),
            recognitionLineCount: perLine.count,
            detectionInputTensorShape: detResult.inputTensorShape,
            recognitionInputTensorShapes: perLine.inputTensorShapes,
            lineRecognitionInferenceTimes: perLine.inference,
            lineRecognitionPreprocessTimes: perLine.preprocess,
            lineRecognitionPostprocessTimes: perLine.postprocess
        )
    }

    /// Aggregated recognition timing across all lines (sums of per-line split batch times).
    private struct RecognitionTimingAggregate {
        let preprocess: TimeInterval
        let inference: TimeInterval
        let postprocess: TimeInterval
        let total: TimeInterval
    }

    /// Per-line rec timings
    private struct PerLineRecognitionTimes {
        let count: Int
        let inputTensorShapes: [[Int]]
        let inference: [TimeInterval]
        let preprocess: [TimeInterval]
        let postprocess: [TimeInterval]
    }

    private func recognizeSortedBoxes(
        _ sortedBoxes: [DetectionBox],
        sourceImage: CGImage,
        resolved: ResolvedOCRRuntimeParams
    ) async throws -> ([OCRResult], RecognitionTimingAggregate, PerLineRecognitionTimes) {
        struct LineCrop {
            let lineIndex: Int
            let aspectRatio: Float
            let crop: CGImage
            let polygon: [[Int32]]
        }

        var lines: [LineCrop] = []
        lines.reserveCapacity(sortedBoxes.count)

        for (index, box) in sortedBoxes.enumerated() {
            let croppedImage: CGImage
            do {
                croppedImage = try QuadTextCrop.crop(sourceImage, polygon: box.points)
            } catch {
                throw OCREngineError.quadTextCropFailed(boxIndex: index, underlying: error)
            }
            let h = max(croppedImage.height, 1)
            let aspect = Float(croppedImage.width) / Float(h)
            lines.append(LineCrop(lineIndex: index, aspectRatio: aspect, crop: croppedImage, polygon: box.points))
        }

        let sortedForInference = lines.sorted { a, b in
            if a.aspectRatio != b.aspectRatio {
                return a.aspectRatio < b.aspectRatio
            }
            return a.lineIndex < b.lineIndex
        }

        let batchSize = max(1, resolved.textRecBatchSize)
        var perLine: [RecognitionEngineResult?] = Array(repeating: nil, count: lines.count)
        var batchInputShapes: [[Int]] = []

        var chunkStart = 0
        while chunkStart < sortedForInference.count {
            let chunkEnd = min(chunkStart + batchSize, sortedForInference.count)
            let chunk = Array(sortedForInference[chunkStart..<chunkEnd])
            let crops = chunk.map(\.crop)
            let recBatch = try await recognitionEngine.recognizeBatch(crops)
            guard recBatch.count == chunk.count else {
                throw OCREngineError.recognitionBatchSizeMismatch(expected: chunk.count, actual: recBatch.count)
            }
            if let first = recBatch.first {
                batchInputShapes.append(first.inputTensorShape)
            }
            for (i, item) in chunk.enumerated() {
                perLine[item.lineIndex] = recBatch[i]
            }
            chunkStart = chunkEnd
        }

        var ocrResults: [OCRResult] = []
        var totalRecTime: TimeInterval = 0
        var totalRecPre: TimeInterval = 0
        var totalRecInf: TimeInterval = 0
        var totalRecPost: TimeInterval = 0
        var lineInf: [TimeInterval] = []
        var linePre: [TimeInterval] = []
        var linePost: [TimeInterval] = []
        lineInf.reserveCapacity(lines.count)
        linePre.reserveCapacity(lines.count)
        linePost.reserveCapacity(lines.count)

        for line in lines {
            guard let recResult = perLine[line.lineIndex] else { continue }
            linePre.append(recResult.preprocessTime)
            lineInf.append(recResult.inferenceTime)
            linePost.append(recResult.postprocessTime)
            totalRecTime += recResult.totalTime
            totalRecPre += recResult.preprocessTime
            totalRecInf += recResult.inferenceTime
            totalRecPost += recResult.postprocessTime
            if recResult.confidence < resolved.textRecScoreThresh {
                continue
            }
            ocrResults.append(OCRResult(
                polygon: line.polygon,
                text: recResult.text,
                confidence: recResult.confidence
            ))
        }

        let aggregate = RecognitionTimingAggregate(
            preprocess: totalRecPre,
            inference: totalRecInf,
            postprocess: totalRecPost,
            total: totalRecTime
        )
        let perLineOut = PerLineRecognitionTimes(
            count: lines.count,
            inputTensorShapes: batchInputShapes,
            inference: lineInf,
            preprocess: linePre,
            postprocess: linePost
        )
        return (ocrResults, aggregate, perLineOut)
    }
}
