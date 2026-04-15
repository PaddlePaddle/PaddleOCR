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

// MARK: - OCR Pipeline Result Types

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

/// Complete pipeline result with per-stage timing.
struct OCRPipelineResult {
    /// All detected and recognized text regions, in reading order.
    let results: [OCRResult]
    /// Total time spent in the detection stage (preprocess + inference + postprocess).
    let detectionTime: TimeInterval
    /// Total time spent recognizing all text regions (sum of all recognition calls).
    let recognitionTime: TimeInterval
    /// Wall-clock time for the entire pipeline (detect + sort + crop + recognize).
    let totalTime: TimeInterval
}

// MARK: - OCR Engine Errors

enum OCREngineError: LocalizedError {
    case perspectiveCropFailed(boxIndex: Int, underlying: Error)

    var errorDescription: String? {
        switch self {
        case .perspectiveCropFailed(let idx, let err):
            return "Perspective crop failed for box \(idx): \(err.localizedDescription)"
        }
    }
}

// MARK: - OCREngine

/// End-to-end OCR pipeline: detect -> sort -> crop -> recognize.
///
/// Composes `DetectionEngine`, `BoxSorter`, `PerspectiveCrop`, and `RecognitionEngine`
/// into a single `run(CGImage)` call. All preprocessing and postprocessing parameters
/// are config-driven via each engine's model config file.
///
/// The pipeline runs entirely via async/await. Since `DetectionEngine` and
/// `RecognitionEngine` delegate to `ORTSessionManager` (a Swift actor), all ORT
/// calls are off the main thread.
///
/// Usage:
/// ```swift
/// let manager = ORTSessionManager()
/// try await manager.loadModels()
/// let engine = try OCREngine(sessionManager: manager)
/// let result = try await engine.run(cgImage)
/// for item in result.results {
///     print("\(item.text) (\(item.confidence))")
/// }
/// ```
class OCREngine {
    private let detectionEngine: DetectionEngine
    private let recognitionEngine: RecognitionEngine

    /// Initialize with an existing ORTSessionManager (models must already be loaded).
    ///
    /// Creates both DetectionEngine and RecognitionEngine, each loading their
    /// own model config files for config-driven preprocessing and postprocessing.
    ///
    /// - Parameter sessionManager: A loaded ORTSessionManager.
    /// - Throws: If either engine's model config cannot be loaded.
    init(sessionManager: ORTSessionManager) throws {
        self.detectionEngine = try DetectionEngine(sessionManager: sessionManager)
        self.recognitionEngine = try RecognitionEngine(sessionManager: sessionManager)
    }

    /// Run the complete OCR pipeline on an image.
    ///
    /// End-to-end OCR flow (detect → sort → crop → recognize):
    /// 1. **Detect**: Run detection to get bounding polygons
    /// 2. **Sort**: Sort boxes in reading order (top-to-bottom, left-to-right)
    /// 3. **Crop + Recognize**: For each sorted box, perspective-crop the region
    ///    from the original image, then run recognition on the crop
    ///
    /// - Parameter image: The input CGImage to process.
    /// - Returns: `OCRPipelineResult` with all results and timing.
    func run(_ image: CGImage) async throws -> OCRPipelineResult {
        let pipelineStart = CFAbsoluteTimeGetCurrent()

        // Step 1: Detect text regions
        let detResult = try await detectionEngine.detect(image)

        // Step 2: Sort boxes in reading order
        let sortedBoxes = BoxSorter.sortInReadingOrder(detResult.boxes)

        // Step 3: Crop and recognize each box
        var ocrResults: [OCRResult] = []
        var totalRecTime: TimeInterval = 0

        for (index, box) in sortedBoxes.enumerated() {
            // Perspective crop from original image
            let croppedImage: CGImage
            do {
                croppedImage = try PerspectiveCrop.crop(image, polygon: box.points)
            } catch {
                throw OCREngineError.perspectiveCropFailed(boxIndex: index, underlying: error)
            }

            // Recognize text in cropped image
            let recResult = try await recognitionEngine.recognize(croppedImage)
            totalRecTime += recResult.totalTime

            ocrResults.append(OCRResult(
                polygon: box.points,
                text: recResult.text,
                confidence: recResult.confidence
            ))
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - pipelineStart

        return OCRPipelineResult(
            results: ocrResults,
            detectionTime: detResult.totalTime,
            recognitionTime: totalRecTime,
            totalTime: totalTime
        )
    }
}
