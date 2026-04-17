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
    /// Total time spent recognizing all text regions (sum of all recognition calls).
    let recognitionTime: TimeInterval
    /// Wall-clock time for the entire run (detect + sort + crop + recognize).
    let totalTime: TimeInterval
}

// MARK: - OCR Engine Errors

enum OCREngineError: LocalizedError {
    case quadTextCropFailed(boxIndex: Int, underlying: Error)

    var errorDescription: String? {
        switch self {
        case .quadTextCropFailed(let idx, let err):
            return "Quad text crop failed for box \(idx): \(err.localizedDescription)"
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
    /// 3. **Crop + Recognize**: For each sorted box, crop the text region (quad path)
    ///    from the original image, then run recognition on the crop
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
        let (ocrResults, totalRecTime) = try await recognizeSortedBoxes(
            sortedBoxes,
            sourceImage: image,
            resolved: resolved
        )

        let totalTime = CFAbsoluteTimeGetCurrent() - runStart

        return OCRRunResult(
            results: ocrResults,
            detectionTime: detResult.totalTime,
            recognitionTime: totalRecTime,
            totalTime: totalTime
        )
    }

    private func recognizeSortedBoxes(
        _ sortedBoxes: [DetectionBox],
        sourceImage: CGImage,
        resolved: ResolvedOCRRuntimeParams
    ) async throws -> ([OCRResult], TimeInterval) {
        var ocrResults: [OCRResult] = []
        var totalRecTime: TimeInterval = 0

        for (index, box) in sortedBoxes.enumerated() {
            let croppedImage: CGImage
            do {
                croppedImage = try QuadTextCrop.crop(sourceImage, polygon: box.points)
            } catch {
                throw OCREngineError.quadTextCropFailed(boxIndex: index, underlying: error)
            }

            let recResult = try await recognitionEngine.recognize(croppedImage)
            totalRecTime += recResult.totalTime

            if recResult.confidence < resolved.textRecScoreThresh {
                continue
            }

            ocrResults.append(OCRResult(
                polygon: box.points,
                text: recResult.text,
                confidence: recResult.confidence
            ))
        }

        return (ocrResults, totalRecTime)
    }
}
