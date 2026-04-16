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

enum ORTSessionManagerError: LocalizedError {
    case modelNotFound(String)
    case sessionCreationFailed(String)
    case inferenceFailed(String)
    case outputContainsNaN(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name): return "Model not found: \(name)"
        case .sessionCreationFailed(let detail): return "Session creation failed: \(detail)"
        case .inferenceFailed(let detail): return "Inference failed: \(detail)"
        case .outputContainsNaN(let name): return "Output tensor '\(name)' contains NaN values"
        }
    }
}

actor ORTSessionManager {
    private var env: ORTEnv?
    private var detSession: ORTSession?
    private var recSession: ORTSession?

    /// Load both models. Creates one ORTEnv (per ORT docs: one per process),
    /// configures CoreML EP -> XNNPACK EP fallback chain,
    /// and creates sessions for detection and recognition models.
    func loadModels() async throws {
        // 1. Create environment (one per process)
        let env = try ORTEnv(loggingLevel: .warning)
        self.env = env

        // 2. Configure session options
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)

        // 3. CoreML EP (device default)
        let coremlOptions = ORTCoreMLExecutionProviderOptions()
        try options.appendCoreMLExecutionProvider(with: coremlOptions)

        // 4. XNNPACK EP (CPU fallback)
        let xnnpackOptions = ORTXnnpackExecutionProviderOptions()
        try options.appendXnnpackExecutionProvider(with: xnnpackOptions)

        // 5. Load detection model
        let detConfig = try ModelConfig.detection()
        detSession = try ORTSession(env: env, modelPath: detConfig.modelPath, sessionOptions: options)

        // 6. Load recognition model (reuse same options)
        let recConfig = try ModelConfig.recognition()
        recSession = try ORTSession(env: env, modelPath: recConfig.modelPath, sessionOptions: options)
    }

    /// Run detection inference with real preprocessed input data.
    ///
    /// - Parameters:
    ///   - inputData: Float32 array in CHW layout; length must equal the product of `shape`.
    ///   - shape: Tensor shape, e.g. [1, 3, 896, 960].
    /// - Returns: Dictionary mapping output name to (data as [Float], shape as [Int]).
    func runDetection(inputData: [Float], shape: [Int]) async throws -> [String: (data: [Float], shape: [Int])] {
        guard let session = detSession else {
            throw ORTSessionManagerError.sessionCreationFailed("Detection session not loaded")
        }
        return try runInference(session: session, modelName: "det", inputData: inputData, shape: shape)
    }

    /// Run recognition inference with preprocessed input data.
    ///
    /// The recognition model accepts dynamic-width input: shape `[1, 3, 48, W]`
    /// where W varies per image depending on the aspect ratio of the cropped text region.
    ///
    /// - Parameters:
    ///   - inputData: Float32 array in CHW layout; length must equal the product of `shape`.
    ///   - shape: Tensor shape, e.g. [1, 3, 48, 320].
    /// - Returns: Dictionary mapping output name to (data as [Float], shape as [Int]).
    func runRecognition(inputData: [Float], shape: [Int]) async throws -> [String: (data: [Float], shape: [Int])] {
        guard let session = recSession else {
            throw ORTSessionManagerError.sessionCreationFailed("Recognition session not loaded")
        }
        return try runInference(session: session, modelName: "rec", inputData: inputData, shape: shape)
    }

    /// Shared inference logic used by both detection and recognition.
    ///
    /// Discovers input/output names at runtime, creates a float32 tensor from the
    /// provided data and shape, runs the session, extracts output arrays, and
    /// validates that no output contains NaN values.
    private func runInference(
        session: ORTSession,
        modelName: String,
        inputData: [Float],
        shape: [Int]
    ) throws -> [String: (data: [Float], shape: [Int])] {
        let inputNames = try session.inputNames()
        let outputNamesList = try session.outputNames()
        let outputNamesSet = Set(outputNamesList)

        guard let firstInputName = inputNames.first else {
            throw ORTSessionManagerError.inferenceFailed("\(modelName): no input names found")
        }

        // Create ORT tensor from input data
        let nsShape = shape.map { NSNumber(value: $0) }
        var data = inputData
        let tensorData = NSMutableData(
            bytes: &data,
            length: inputData.count * MemoryLayout<Float>.stride
        )
        let inputTensor = try ORTValue(
            tensorData: tensorData,
            elementType: .float,
            shape: nsShape
        )

        let inputs: [String: ORTValue] = [firstInputName: inputTensor]
        let outputs = try session.run(
            withInputs: inputs,
            outputNames: outputNamesSet,
            runOptions: nil
        )

        // Extract output tensors as Float arrays
        var result: [String: (data: [Float], shape: [Int])] = [:]
        for (name, value) in outputs {
            let info = try value.tensorTypeAndShapeInfo()
            let outputShape = info.shape.map { $0.intValue }
            let outputData = try value.tensorData() as Data
            let floats: [Float] = outputData.withUnsafeBytes { buffer in
                Array(buffer.bindMemory(to: Float.self))
            }

            if floats.contains(where: \.isNaN) {
                throw ORTSessionManagerError.outputContainsNaN(name)
            }

            result[name] = (data: floats, shape: outputShape)
        }

        return result
    }
}
