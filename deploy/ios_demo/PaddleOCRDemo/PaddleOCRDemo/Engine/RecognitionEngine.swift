import CoreGraphics
import Foundation

// MARK: - Recognition Engine Result

/// The result of running the full recognition pipeline on a cropped text image.
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
    /// Seconds spent in CTC decoding (argmax, dedup, char mapping).
    let postprocessTime: TimeInterval
    /// Total recognition time (preprocess + inference + postprocess).
    let totalTime: TimeInterval
}

// MARK: - Recognition Engine Errors

enum RecognitionEngineError: LocalizedError {
    case noOutputTensor
    case unexpectedOutputShape([Int])

    var errorDescription: String? {
        switch self {
        case .noOutputTensor:
            return "Recognition model produced no output tensor"
        case .unexpectedOutputShape(let shape):
            return "Unexpected recognition output shape: \(shape), expected [1, T, C]"
        }
    }
}

// MARK: - RecognitionEngine

/// Orchestrates the complete text recognition pipeline:
/// CGImage -> RecPreprocessor -> ORT inference -> CTCDecoder.
///
/// This is the integration layer that composes `RecPreprocessor`, `ORTSessionManager`,
/// and `CTCDecoder` into a single callable unit. All preprocessing parameters are read
/// from the recognition model's `inference.yml` at initialization time.
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

    /// Initialize with an existing ORTSessionManager (models must already be loaded).
    ///
    /// Loads `inference.yml` from the recognition model bundle path to configure
    /// the preprocessor and CTC decoder.
    ///
    /// - Parameter sessionManager: A loaded ORTSessionManager with recognition model ready.
    /// - Throws: If inference.yml cannot be loaded or required config fields are missing.
    init(sessionManager: ORTSessionManager) throws {
        self.sessionManager = sessionManager

        // Load inference.yml for recognition model
        let modelConfig = try ModelConfig.recognition()
        let config = try InferenceConfig.load(from: modelConfig.configPath)

        // Initialize preprocessor and decoder from config
        self.preprocessor = try RecPreprocessor(config: config)
        self.decoder = try CTCDecoder(config: config)
    }

    /// Run recognition on a cropped text region CGImage, returning decoded text with confidence.
    ///
    /// Pipeline: RecPreprocessor (OCRResizeNormImg) -> ORT inference -> CTCDecoder
    ///
    /// - Parameter image: A cropped text region image.
    /// - Returns: A `RecognitionEngineResult` with text, confidence, and per-stage timing.
    func recognize(_ image: CGImage) async throws -> RecognitionEngineResult {
        // Step 1: Preprocess
        let preprocessStart = CFAbsoluteTimeGetCurrent()
        let preprocessed = try preprocessor.preprocess(image)
        let preprocessTime = CFAbsoluteTimeGetCurrent() - preprocessStart

        // Step 2: Run ORT inference
        let inferenceStart = CFAbsoluteTimeGetCurrent()
        let outputs = try await sessionManager.runRecognition(
            inputData: preprocessed.tensorData,
            shape: preprocessed.tensorShape
        )
        let inferenceTime = CFAbsoluteTimeGetCurrent() - inferenceStart

        // Step 3: Extract output tensor
        // The rec model outputs a single tensor with shape [1, T, C]
        // where T = timesteps (sequence length), C = vocabulary size
        guard let firstOutput = outputs.values.first else {
            throw RecognitionEngineError.noOutputTensor
        }
        let outputData = firstOutput.data
        let outputShape = firstOutput.shape

        guard outputShape.count == 3 else {
            throw RecognitionEngineError.unexpectedOutputShape(outputShape)
        }

        // Step 4: CTC Decode
        let postprocessStart = CFAbsoluteTimeGetCurrent()
        let decoded = try decoder.decode(outputData: outputData, outputShape: outputShape)
        let postprocessTime = CFAbsoluteTimeGetCurrent() - postprocessStart

        let totalTime = preprocessTime + inferenceTime + postprocessTime

        return RecognitionEngineResult(
            text: decoded.text,
            confidence: decoded.confidence,
            preprocessTime: preprocessTime,
            inferenceTime: inferenceTime,
            postprocessTime: postprocessTime,
            totalTime: totalTime
        )
    }
}
