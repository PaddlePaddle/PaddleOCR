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

// MARK: - Recognition Engine Result

/// The result of running text recognition on a cropped text image.
///
/// Contains the decoded text, confidence score, and per-stage timing metrics.
struct RecognitionEngineResult {
    /// The decoded text string from CTC decoding.
    let text: String
    /// Average confidence across all decoded characters (0.0 to 1.0).
    let confidence: Float
    /// Seconds spent in preprocessing (resize, normalize, HWC-to-CHW, pad).
    let preprocessTime: TimeInterval
    /// Seconds spent in ONNX Runtime inference.
    let inferenceTime: TimeInterval
    /// Seconds spent in CTC decoding (best-class pick, dedup, char mapping).
    let postprocessTime: TimeInterval
    /// Total recognition time (preprocess + inference + postprocess).
    let totalTime: TimeInterval
    /// Recognition model input tensor shape for the batch this line belonged to.
    let inputTensorShape: [Int]
}

// MARK: - Recognition Engine Errors

enum RecognitionEngineError: LocalizedError {
    case noOutputTensor
    case unexpectedOutputShape([Int])
    case batchOutputCountMismatch(expectedBatch: Int, shape: [Int])

    var errorDescription: String? {
        switch self {
        case .noOutputTensor:
            return "Recognition model produced no output tensor"
        case .unexpectedOutputShape(let shape):
            return "Unexpected recognition output shape: \(shape), expected [1, T, C] or [N, T, C]"
        case .batchOutputCountMismatch(let expectedBatch, let shape):
            return "Recognition batch size mismatch: expected batch dimension \(expectedBatch), got shape \(shape)"
        }
    }
}

// MARK: - RecognitionEngine

/// Orchestrates text recognition:
/// CGImage → RecPreprocessor → ORT inference → CTCDecoder.
///
/// This is the integration layer that composes `RecPreprocessor`, `ORTSessionManager`,
/// and `CTCDecoder` into a single callable unit. All preprocessing parameters are read
/// from the recognition model config file at initialization time.
///
/// Usage:
/// ```swift
/// let engine = try RecognitionEngine(sessionManager: manager)
/// let result = try await engine.recognize(croppedTextImage)
/// print("Text: \(result.text) (confidence: \(result.confidence))")
/// ```
class RecognitionEngine {
    private let sessionManager: ORTSessionManager
    private let preprocessor: RecPreprocessor
    private let decoder: CTCDecoder
    private(set) var modelConfig: InferenceConfig

    /// Initialize with an existing ORTSessionManager (models must already be loaded).
    ///
    /// Loads the recognition model config file from the bundle to configure
    /// the preprocessor and CTC decoder.
    ///
    /// - Parameter sessionManager: A loaded ORTSessionManager with recognition model ready.
    /// - Throws: If the model config cannot be loaded or required fields are missing.
    init(sessionManager: ORTSessionManager) throws {
        self.sessionManager = sessionManager

        let recPaths = try ModelConfig.recognition()
        let cfg = try InferenceConfig.load(from: recPaths.configPath)
        self.modelConfig = cfg

        self.preprocessor = try RecPreprocessor(config: cfg)
        self.decoder = try CTCDecoder(config: cfg)
    }

    /// Run recognition on a cropped text region CGImage, returning decoded text with confidence.
    ///
    /// Steps: RecPreprocessor → ORT inference → CTCDecoder
    ///
    /// - Parameter image: A cropped text region image.
    /// - Returns: A `RecognitionEngineResult` with text, confidence, and per-stage timing.
    func recognize(_ image: CGImage) async throws -> RecognitionEngineResult {
        let batch = try await recognizeBatch([image])
        guard let first = batch.first else {
            throw RecognitionEngineError.noOutputTensor
        }
        return first
    }

    /// Runs recognition on multiple cropped lines in one batched inference call.
    ///
    /// - Parameter images: Cropped text regions in inference order (caller controls ordering).
    /// - Returns: One result per input image, same order.
    func recognizeBatch(_ images: [CGImage]) async throws -> [RecognitionEngineResult] {
        guard !images.isEmpty else { return [] }

        let preprocessStart = CFAbsoluteTimeGetCurrent()
        let batchPreprocessed = try preprocessor.preprocessBatch(images)
        let preprocessTime = CFAbsoluteTimeGetCurrent() - preprocessStart

        let inferenceStart = CFAbsoluteTimeGetCurrent()
        let outputs = try await sessionManager.runRecognition(
            inputData: batchPreprocessed.tensorData,
            shape: batchPreprocessed.tensorShape
        )
        let inferenceTime = CFAbsoluteTimeGetCurrent() - inferenceStart

        guard let firstOutput = outputs.values.first else {
            throw RecognitionEngineError.noOutputTensor
        }
        let outputData = firstOutput.data
        let outputShape = firstOutput.shape

        guard outputShape.count == 3, outputShape[0] == images.count else {
            if outputShape.count == 3 {
                throw RecognitionEngineError.batchOutputCountMismatch(expectedBatch: images.count, shape: outputShape)
            }
            throw RecognitionEngineError.unexpectedOutputShape(outputShape)
        }

        let postprocessStart = CFAbsoluteTimeGetCurrent()
        let decoded = try decoder.decodeBatch(outputData: outputData, outputShape: outputShape)
        let postprocessTime = CFAbsoluteTimeGetCurrent() - postprocessStart

        let n = Double(images.count)
        let preprocessPer = preprocessTime / n
        let inferencePer = inferenceTime / n
        let postPer = postprocessTime / n
        let totalPer = (preprocessTime + inferenceTime + postprocessTime) / n

        return decoded.map { d in
            RecognitionEngineResult(
                text: d.text,
                confidence: d.confidence,
                preprocessTime: preprocessPer,
                inferenceTime: inferencePer,
                postprocessTime: postPer,
                totalTime: totalPer,
                inputTensorShape: batchPreprocessed.tensorShape
            )
        }
    }
}
