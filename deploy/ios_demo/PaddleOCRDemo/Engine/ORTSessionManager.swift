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

/// ONNX Runtime execution provider selection for this demo.
enum ORTInferenceBackend: String, CaseIterable, Identifiable, Sendable {
    /// Core ML execution provider only (ANE/GPU). Default on Apple devices.
    case coreMLOnly
    /// XNNPACK (CPU) only.
    case xnnpackOnly
    /// Built-in CPU execution provider only (no Core ML, no XNNPACK).
    case cpuOnly

    var id: String { rawValue }

    var displayTitle: String {
        switch self {
        case .coreMLOnly: return "Core ML"
        case .xnnpackOnly: return "XNNPACK"
        case .cpuOnly: return "CPU"
        }
    }
}

enum ORTSessionManagerError: LocalizedError {
    case modelNotFound(String)
    case sessionCreationFailed(String)
    case inferenceFailed(String)
    case inputTensorElementCountMismatch(expected: Int, actual: Int, modelName: String)
    case outputContainsNaN(String)
    case ortProfilingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name): return "Model not found: \(name)"
        case .sessionCreationFailed(let detail): return "Session creation failed: \(detail)"
        case .inferenceFailed(let detail): return "Inference failed: \(detail)"
        case .inputTensorElementCountMismatch(let expected, let actual, let modelName):
            return "\(modelName): input tensor size mismatch (expected \(expected) floats, got \(actual))"
        case .outputContainsNaN(let name): return "Output tensor '\(name)' contains NaN values"
        case .ortProfilingFailed(let detail): return "ONNX Runtime profiling: \(detail)"
        }
    }
}

/// File URLs of ONNX Runtime JSON profiles after ``ORTSessionManager/finalizeORTProfiling()`` (separate file per sub-model).
struct ORTProfilingOutput: Sendable {
    var detectionProfileJSON: URL
    var recognitionProfileJSON: URL
}

actor ORTSessionManager {
    private struct SessionIONames {
        let inputName: String
        let outputNames: Set<String>
    }

    private var env: ORTEnv?
    private var detSession: ORTSession?
    private var recSession: ORTSession?
    private var detIO: SessionIONames?
    private var recIO: SessionIONames?
    private var ortProfilingPendingFinalize: Bool = false

    /// Load both models. Creates one ORTEnv (per ORT docs: one per process),
    /// configures execution providers per ``ORTInferenceBackend``,
    /// and creates sessions for detection and recognition models.
    ///
    /// - Parameter ortProfiling: When `true`, enables ONNX Runtime session profiling (written as JSON on finalize).
    ///   Profiling adds overhead; do not treat wall-clock times from the same run as a clean latency benchmark.
    func loadModels(backend: ORTInferenceBackend = .coreMLOnly, ortProfiling: Bool = false) async throws {
        let env = try ORTEnv(loggingLevel: .warning)
        self.env = env
        ortProfilingPendingFinalize = ortProfiling

        let detConfig = try ModelConfig.detection()
        let recConfig = try ModelConfig.recognition()

        if ortProfiling {
            let baseDir = try ortProfilingDirectoryURL()
            let detPrefix = baseDir.appendingPathComponent("paddle_ort_det", isDirectory: false).path
            let recPrefix = baseDir.appendingPathComponent("paddle_ort_rec", isDirectory: false).path
            let detOptions = try makeSessionOptions(backend: backend, ortProfilePathPrefix: detPrefix)
            let recOptions = try makeSessionOptions(backend: backend, ortProfilePathPrefix: recPrefix)
            detSession = try ORTSession(env: env, modelPath: detConfig.modelPath, sessionOptions: detOptions)
            recSession = try ORTSession(env: env, modelPath: recConfig.modelPath, sessionOptions: recOptions)
        } else {
            let options = try makeSessionOptions(backend: backend, ortProfilePathPrefix: nil)
            detSession = try ORTSession(env: env, modelPath: detConfig.modelPath, sessionOptions: options)
            recSession = try ORTSession(env: env, modelPath: recConfig.modelPath, sessionOptions: options)
        }
        if let detSession {
            detIO = try makeSessionIONames(session: detSession, modelName: "det")
        }
        if let recSession {
            recIO = try makeSessionIONames(session: recSession, modelName: "rec")
        }
    }

    /// Flushes ORT profiling and returns the JSON file paths. Call once after the last inference when
    /// ``loadModels(ortProfiling:)`` was `true`. Safe to call when profiling was disabled (returns `nil`).
    func finalizeORTProfiling() throws -> ORTProfilingOutput? {
        guard ortProfilingPendingFinalize else { return nil }
        ortProfilingPendingFinalize = false
        guard let det = detSession, let rec = recSession else {
            throw ORTSessionManagerError.sessionCreationFailed("Cannot finalize ORT profiling: session missing")
        }
        let detPath = try ORTProfilingBridge.endProfiling(session: det)
        let recPath = try ORTProfilingBridge.endProfiling(session: rec)
        if detPath.isEmpty || recPath.isEmpty {
            throw ORTSessionManagerError.ortProfilingFailed("EndProfiling returned an empty path")
        }
        return ORTProfilingOutput(
            detectionProfileJSON: URL(fileURLWithPath: detPath),
            recognitionProfileJSON: URL(fileURLWithPath: recPath)
        )
    }

    private func ortProfilingDirectoryURL() throws -> URL {
        let base = try FileManager.default.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let dir = base.appendingPathComponent("PaddleOCRORTProfiling", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func makeSessionOptions(backend: ORTInferenceBackend, ortProfilePathPrefix: String?) throws -> ORTSessionOptions {
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        if let prefix = ortProfilePathPrefix {
            try ORTProfilingBridge.enableProfiling(sessionOptions: options, pathPrefix: prefix)
        }
        switch backend {
        case .coreMLOnly:
            let coremlOptions = ORTCoreMLExecutionProviderOptions()
            try options.appendCoreMLExecutionProvider(with: coremlOptions)
        case .xnnpackOnly:
            let xnnpackOptions = ORTXnnpackExecutionProviderOptions()
            try options.appendXnnpackExecutionProvider(with: xnnpackOptions)
        case .cpuOnly:
            break
        }
        return options
    }

    private func makeSessionIONames(session: ORTSession, modelName: String) throws -> SessionIONames {
        let inputNames = try session.inputNames()
        let outputNamesList = try session.outputNames()
        guard let firstInputName = inputNames.first else {
            throw ORTSessionManagerError.inferenceFailed("\(modelName): no input names found")
        }
        return SessionIONames(inputName: firstInputName, outputNames: Set(outputNamesList))
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
        guard let io = detIO else {
            throw ORTSessionManagerError.sessionCreationFailed("Detection session IO names not loaded")
        }
        return try runInference(session: session, io: io, modelName: "det", inputData: inputData, shape: shape)
    }

    /// Run recognition inference with preprocessed input data.
    ///
    /// The recognition model accepts dynamic batch and width: shape `[N, 3, H, W]` with `N ≥ 1`,
    /// fixed `H` (e.g. 48), and `W` shared after padding within each batch. A single line uses `N = 1`.
    ///
    /// - Parameters:
    ///   - inputData: Float32 row-major tensor; length must equal the product of `shape`.
    ///   - shape: e.g. `[1, 3, 48, 320]` or `[6, 3, 48, 640]` for a batch of six lines.
    /// - Returns: Dictionary mapping output name to (data as [Float], shape as [Int]).
    func runRecognition(inputData: [Float], shape: [Int]) async throws -> [String: (data: [Float], shape: [Int])] {
        guard let session = recSession else {
            throw ORTSessionManagerError.sessionCreationFailed("Recognition session not loaded")
        }
        guard let io = recIO else {
            throw ORTSessionManagerError.sessionCreationFailed("Recognition session IO names not loaded")
        }
        return try runInference(session: session, io: io, modelName: "rec", inputData: inputData, shape: shape)
    }

    /// Shared inference logic used by both detection and recognition.
    ///
    /// Discovers input/output names at runtime, creates a float32 tensor from the
    /// provided data and shape, runs the session, extracts output arrays, and
    /// validates that no output contains NaN values.
    private func runInference(
        session: ORTSession,
        io: SessionIONames,
        modelName: String,
        inputData: [Float],
        shape: [Int]
    ) throws -> [String: (data: [Float], shape: [Int])] {
        let expectedElements = shape.reduce(1, *)
        guard expectedElements == inputData.count else {
            throw ORTSessionManagerError.inputTensorElementCountMismatch(
                expected: expectedElements,
                actual: inputData.count,
                modelName: modelName
            )
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

        let inputs: [String: ORTValue] = [io.inputName: inputTensor]
        let outputs = try session.run(
            withInputs: inputs,
            outputNames: io.outputNames,
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
