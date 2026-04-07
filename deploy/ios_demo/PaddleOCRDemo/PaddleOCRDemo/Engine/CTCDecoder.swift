import Foundation

// MARK: - Errors

enum CTCDecoderError: LocalizedError {
    case emptyDictionary
    case invalidOutputShape([Int])

    var errorDescription: String? {
        switch self {
        case .emptyDictionary:
            return "Character dictionary is empty"
        case .invalidOutputShape(let shape):
            return "Invalid recognition output shape: \(shape), expected [1, T, C]"
        }
    }
}

// MARK: - Recognition Result

/// A single text recognition result from CTC decoding.
struct RecognitionResult {
    /// The decoded text string.
    let text: String
    /// Average confidence across all decoded characters (0.0 to 1.0).
    let confidence: Float
}

// MARK: - CTCDecoder

/// Implements CTCLabelDecode, ported from `ppocr/postprocess/rec_postprocess.py`.
///
/// The CTC (Connectionist Temporal Classification) decoding algorithm:
/// 1. Argmax across the class dimension to get per-timestep predicted indices
/// 2. Max across the class dimension to get per-timestep prediction probabilities
/// 3. Remove consecutive duplicate indices (CTC merge rule)
/// 4. Remove blank tokens (index 0)
/// 5. Map remaining indices to characters via the dictionary
/// 6. Confidence = mean of probabilities at selected positions
///
/// The character dictionary is loaded from inference.yml's PostProcess.character_dict.
/// A "blank" token is prepended at index 0 (matching PaddleOCR's `add_special_char`).
struct CTCDecoder {
    /// Full character list with blank at index 0.
    /// character[0] = "blank", character[1..] = dict characters.
    private let characters: [String]

    /// The blank token index (always 0 for CTC).
    private let blankIndex: Int = 0

    /// Creates a CTCDecoder from a recognition model's InferenceConfig.
    ///
    /// Reads the character_dict from PostProcess config and prepends the blank token.
    ///
    /// - Parameter config: A parsed InferenceConfig from the recognition model's inference.yml.
    /// - Throws: `CTCDecoderError.emptyDictionary` if no character dict is found.
    init(config: InferenceConfig) throws {
        guard let dict = config.postProcess.characterDict, !dict.isEmpty else {
            throw CTCDecoderError.emptyDictionary
        }

        // Prepend blank token at index 0, matching Python's add_special_char
        var chars = ["blank"]
        chars.append(contentsOf: dict)
        self.characters = chars
    }

    /// Decodes the raw recognition model output into text with confidence.
    ///
    /// The output tensor has shape `[1, T, C]` where:
    /// - T = number of timesteps (sequence length)
    /// - C = number of classes (character vocabulary size + 1 for blank)
    ///
    /// - Parameters:
    ///   - outputData: Flat float32 array of the model output.
    ///   - outputShape: Shape of the output tensor, expected `[1, T, C]`.
    /// - Returns: A `RecognitionResult` with decoded text and confidence score.
    func decode(outputData: [Float], outputShape: [Int]) throws -> RecognitionResult {
        guard outputShape.count == 3, outputShape[0] == 1 else {
            throw CTCDecoderError.invalidOutputShape(outputShape)
        }

        let timesteps = outputShape[1]
        let numClasses = outputShape[2]

        // Step 1 & 2: Argmax and max across the class dimension for each timestep
        var predIndices = [Int](repeating: 0, count: timesteps)
        var predProbs = [Float](repeating: 0, count: timesteps)

        for t in 0..<timesteps {
            let offset = t * numClasses
            var maxVal: Float = outputData[offset]
            var maxIdx = 0

            for c in 1..<numClasses {
                let val = outputData[offset + c]
                if val > maxVal {
                    maxVal = val
                    maxIdx = c
                }
            }

            predIndices[t] = maxIdx
            predProbs[t] = maxVal
        }

        // Step 3: Build selection mask
        // - Remove consecutive duplicates (keep first of each run)
        // - Remove blank tokens (index 0)
        //
        // Matches Python: selection[1:] = indices[1:] != indices[:-1]
        //                 selection &= indices != blank
        var selection = [Bool](repeating: true, count: timesteps)

        // Remove consecutive duplicates
        for t in 1..<timesteps {
            if predIndices[t] == predIndices[t - 1] {
                selection[t] = false
            }
        }

        // Remove blank tokens
        for t in 0..<timesteps {
            if predIndices[t] == blankIndex {
                selection[t] = false
            }
        }

        // Step 4: Collect selected characters and their probabilities
        var charList: [String] = []
        var confList: [Float] = []

        for t in 0..<timesteps {
            if selection[t] {
                let idx = predIndices[t]
                if idx < characters.count {
                    charList.append(characters[idx])
                }
                confList.append(predProbs[t])
            }
        }

        // Step 5: Join characters into text
        let text = charList.joined()

        // Step 6: Compute mean confidence
        // Matches Python: np.mean(conf_list) or [0] if empty
        let confidence: Float
        if confList.isEmpty {
            confidence = 0
        } else {
            confidence = confList.reduce(0, +) / Float(confList.count)
        }

        return RecognitionResult(text: text, confidence: confidence)
    }
}
