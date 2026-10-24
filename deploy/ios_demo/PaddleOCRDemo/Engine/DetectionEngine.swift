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

// MARK: - Detection Result

/// The result of running text detection on an image.
///
/// Contains the detected text boxes and per-stage timing metrics for performance analysis.
struct DetectionResult {
    /// Detected text region bounding polygons with confidence scores.
    let boxes: [DetectionBox]
    /// Seconds spent in preprocessing (resize, normalize, HWC-to-CHW).
    let preprocessTime: TimeInterval
    /// Seconds spent in ONNX Runtime inference.
    let inferenceTime: TimeInterval
    /// Seconds spent in DB postprocessing (threshold, contours, polygon offset / unclip).
    let postprocessTime: TimeInterval
    /// Total detection time (preprocess + inference + postprocess).
    let totalTime: TimeInterval
    /// Detection model input tensor shape, e.g. [1, 3, H, W].
    let inputTensorShape: [Int]
}

// MARK: - Detection Engine Errors

enum DetectionEngineError: LocalizedError {
    case noOutputTensor
    case unexpectedOutputShape([Int])

    var errorDescription: String? {
        switch self {
        case .noOutputTensor:
            return "Detection model produced no output tensor"
        case .unexpectedOutputShape(let shape):
            return "Unexpected detection output shape: \(shape), expected [1, 1, H, W]"
        }
    }
}

// MARK: - DetectionEngine

/// Orchestrates text detection: CGImage → preprocess → ORT inference → postprocess.
///
/// This is the integration layer that composes `DetPreprocessor`, `ORTSessionManager`, and
/// `DBPostProcessor` into a single callable unit.
///
/// Usage:
/// ```swift
/// let engine = try DetectionEngine(sessionManager: manager)
/// let result = try await engine.detect(cgImage)
/// print("Found \(result.boxes.count) text regions in \(result.totalTime)s")
/// ```
class DetectionEngine {
    private let sessionManager: ORTSessionManager
    private let preprocessor: DetPreprocessor
    private(set) var modelConfig: InferenceConfig

    /// Initialize with an existing ORTSessionManager (models must already be loaded).
    ///
    /// Loads the detection model config file from the bundle to configure
    /// the preprocessor and postprocessor.
    ///
    /// - Parameter sessionManager: A loaded ORTSessionManager with detection model ready.
    /// - Throws: If the model config cannot be loaded or required fields are missing.
    init(sessionManager: ORTSessionManager) throws {
        self.sessionManager = sessionManager

        // Load detection model config
        let detPaths = try ModelConfig.detection()
        let cfg = try InferenceConfig.load(from: detPaths.configPath)
        self.modelConfig = cfg
        self.preprocessor = try DetPreprocessor(config: cfg)
    }

    private func makePostProcessor(effective: EffectiveOCRParams) -> DBPostProcessor {
        DBPostProcessor(
            thresh: effective.detThresh,
            boxThresh: effective.detBoxThresh,
            maxCandidates: modelConfig.postProcess.maxCandidates,
            unclipRatio: effective.detUnclipRatio
        )
    }

    /// Run detection on a CGImage, returning bounding polygons with confidence scores.
    ///
    /// Steps: DetResizeForTest → NormalizeImage → ToCHW → ORT inference → DBPostProcess
    ///
    /// - Parameters:
    ///   - image: The input image to detect text regions in.
    ///   - runtimeParams: Detection postprocess / resize overrides (see ``OCRRuntimeParams``).
    /// - Returns: A `DetectionResult` with boxes and per-stage timing metrics.
    func detect(_ image: CGImage, runtimeParams: OCRRuntimeParams = .noOverrides) async throws -> DetectionResult {
        let eff = OCRParameterResolver.effective(modelConfig: modelConfig, runtime: runtimeParams)
        return try await detect(effective: eff) {
            try preprocessor.preprocess(image, detResize: eff.detResize)
        }
    }

    private func detect(
        effective: EffectiveOCRParams,
        preprocess: () throws -> PreprocessResult
    ) async throws -> DetectionResult {
        let postprocessor = makePostProcessor(effective: effective)
        // Step 1: Preprocess
        let preprocessStart = CFAbsoluteTimeGetCurrent()
        let preprocessed = try preprocess()
        let preprocessTime = CFAbsoluteTimeGetCurrent() - preprocessStart

        // Step 2: Run ORT inference
        let inferenceStart = CFAbsoluteTimeGetCurrent()
        let outputs = try await sessionManager.runDetection(
            inputData: preprocessed.tensorData,
            shape: preprocessed.tensorShape
        )
        let inferenceTime = CFAbsoluteTimeGetCurrent() - inferenceStart

        // Step 3: Extract the probability map from ORT output
        // The det model outputs a single tensor with shape [1, 1, H, W]
        guard let firstOutput = outputs.values.first else {
            throw DetectionEngineError.noOutputTensor
        }
        let outputData = firstOutput.data
        let outputShape = firstOutput.shape

        // Output shape should be [1, 1, H, W]
        guard outputShape.count == 4 else {
            throw DetectionEngineError.unexpectedOutputShape(outputShape)
        }
        let tensorHeight = outputShape[2]
        let tensorWidth = outputShape[3]

        // Step 4: Postprocess
        let postprocessStart = CFAbsoluteTimeGetCurrent()
        let boxes = postprocessor.process(
            outputTensor: outputData,
            tensorHeight: tensorHeight,
            tensorWidth: tensorWidth,
            originalWidth: preprocessed.originalSize.width,
            originalHeight: preprocessed.originalSize.height
        )
        let postprocessTime = CFAbsoluteTimeGetCurrent() - postprocessStart

        let totalTime = preprocessTime + inferenceTime + postprocessTime

        return DetectionResult(
            boxes: boxes,
            preprocessTime: preprocessTime,
            inferenceTime: inferenceTime,
            postprocessTime: postprocessTime,
            totalTime: totalTime,
            inputTensorShape: preprocessed.tensorShape
        )
    }
}
