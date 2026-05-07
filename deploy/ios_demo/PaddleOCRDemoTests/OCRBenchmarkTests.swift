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

import Darwin
import Foundation

import XCTest

@testable import PaddleOCRDemo

import UIKit

// MARK: - Tests

final class OCRBenchmarkTests: XCTestCase {

    /// `PADDLEOCR_BENCHMARK_IMAGE_NAME`: bundled image stem or `stem.ext` (optional).
    private static let imageNameEnvKey = "PADDLEOCR_BENCHMARK_IMAGE_NAME"

    /// Optional non-negative int; default `3`. Used only by `testOnDevicePerformanceMetrics`.
    private static let warmupIterationsEnvKey = "PADDLEOCR_BENCHMARK_WARMUP_ITERATIONS"

    /// Optional non-negative int; default `10`. Used only by `testOnDevicePerformanceMetrics`.
    private static let measuredIterationsEnvKey = "PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS"

    /// `CPU` (default), `XNNPACK`, or `CORE_ML` — ONNX Runtime execution provider preset for benchmarks.
    private static let ortExecutionProviderEnvKey = "PADDLEOCR_BENCHMARK_ORT_EXECUTION_PROVIDER"

    /// `1` / `true` / `yes` / `on` enables ONNX Runtime JSON profiling (see ``ORTSessionManager/finalizeORTProfiling()``).
    /// When set, on-device performance timings are not representative of a clean latency benchmark.
    private static let ortProfilingEnvKey = "PADDLEOCR_BENCHMARK_ORT_PROFILING"

    private static let buildConfigurationEnvKey = "PADDLEOCR_BENCHMARK_BUILD_CONFIGURATION"

    private static let intraOpThreadsEnvKey = "PADDLEOCR_BENCHMARK_INTRA_OP_THREADS"
    private static let xnnpackThreadsEnvKey = "PADDLEOCR_BENCHMARK_XNNPACK_THREADS"
    private static let coremlOptionsEnvKey = "PADDLEOCR_BENCHMARK_COREML_OPTIONS"
    private static let recBatchSizeEnvKey = "PADDLEOCR_BENCHMARK_REC_BATCH_SIZE"

    func testOCRExportJSONSchema() async throws {
        let cgImage = try resolveBenchmarkImage()
        let executionProvider = try resolveBenchmarkExecutionProvider()
        let manager = ORTSessionManager()
        try await manager.loadModels(executionProvider: executionProvider, ortProfiling: resolveORTProfilingEnabled(), tuning: resolveSessionTuningOptions())
        let engine = try OCREngine(sessionManager: manager)
        let run = try await engine.run(cgImage, params: .noOverrides)

        let payload = OCRExportPayload(
            source: "ios_ocr_demo",
            items: run.results.map { r in
                OCRExportItem(
                    polygon: r.polygon.map { $0.map { Int($0) } },
                    text: r.text,
                    score: Double(r.confidence)
                )
            }
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(payload)
        _ = try JSONDecoder().decode(OCRExportPayload.self, from: data)
        attachJSON(data, artifact: .iOSExport)

        if resolveORTProfilingEnabled(), let out = try await manager.finalizeORTProfiling() {
            attachProfileFile(out.detectionProfileJSON, name: "ort_profile_detection")
            attachProfileFile(out.recognitionProfileJSON, name: "ort_profile_recognition")
        }
    }

    func testOnDeviceLatencyBenchmark() async throws {
        let executionProvider = try resolveBenchmarkExecutionProvider()
        let ortProfiling = resolveORTProfilingEnabled()
        let context = try await makeBenchmarkContext(executionProvider: executionProvider, ortProfiling: ortProfiling)
        let warmup = try parseNonNegativeIntEnv(Self.warmupIterationsEnvKey, defaultValue: 3)
        let iterations = max(try parseNonNegativeIntEnv(Self.measuredIterationsEnvKey, defaultValue: 10), 1)
        try await runWarmup(engine: context.engine, image: context.cgImage, iterations: warmup)

        var totals: [Double] = []
        var dets: [Double] = []
        var detPre: [Double] = []
        var detInf: [Double] = []
        var detPost: [Double] = []
        var recs: [Double] = []
        var recPre: [Double] = []
        var recInf: [Double] = []
        var recPost: [Double] = []
        var overheads: [Double] = []
        totals.reserveCapacity(iterations)
        dets.reserveCapacity(iterations)
        detPre.reserveCapacity(iterations)
        detInf.reserveCapacity(iterations)
        detPost.reserveCapacity(iterations)
        recs.reserveCapacity(iterations)
        recPre.reserveCapacity(iterations)
        recInf.reserveCapacity(iterations)
        recPost.reserveCapacity(iterations)
        overheads.reserveCapacity(iterations)

        var lineInferenceMsPooled: [Double] = []
        var linePreprocessMsPooled: [Double] = []
        var linePostprocessMsPooled: [Double] = []
        var lineTotalMsPooled: [Double] = []
        var firstMeasuredLineCount = 0
        var detectionInputShapes: [[Int]] = []
        var recognitionInputShapes: [[Int]] = []

        let benchmarkParams = resolveBenchmarkRuntimeParams()

        for index in 0..<iterations {
            let run = try await context.engine.run(context.cgImage, params: benchmarkParams)
            if index == 0 {
                firstMeasuredLineCount = run.recognitionLineCount
            }
            detectionInputShapes.append(run.detectionInputTensorShape)
            recognitionInputShapes.append(contentsOf: run.recognitionInputTensorShapes)
            appendRun(
                run,
                totals: &totals,
                dets: &dets,
                detPre: &detPre,
                detInf: &detInf,
                detPost: &detPost,
                recs: &recs,
                recPre: &recPre,
                recInf: &recInf,
                recPost: &recPost,
                overheads: &overheads,
                lineInferenceMsPooled: &lineInferenceMsPooled,
                linePreprocessMsPooled: &linePreprocessMsPooled,
                linePostprocessMsPooled: &linePostprocessMsPooled,
                lineTotalMsPooled: &lineTotalMsPooled
            )
        }

        let perLine: RecognitionPerLineBlock? = lineInferenceMsPooled.isEmpty
            ? nil
            : RecognitionPerLineBlock(
                pooled: RecognitionPerLinePooledMs(
                    count: lineInferenceMsPooled.count,
                    inferenceMs: summarizeMs(lineInferenceMsPooled),
                    preprocessMs: summarizeMs(linePreprocessMsPooled),
                    postprocessMs: summarizeMs(linePostprocessMsPooled),
                    totalMs: summarizeMs(lineTotalMsPooled)
                )
            )

        let stats = OCRDeviceBenchmarkPayload(
            schemaVersion: 1,
            buildConfiguration: resolveBuildConfiguration(),
            deviceModel: deviceModelName(),
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            isSimulator: isRunningOnSimulator(),
            ortExecutionProvider: executionProvider.rawValue,
            ortProfilingEnabled: ortProfiling,
            warmupIterations: warmup,
            measuredIterations: iterations,
            inputShapeDistribution: BenchmarkInputShapeDistribution(
                detection: summarizeShapes(detectionInputShapes),
                recognition: summarizeShapes(recognitionInputShapes)
            ),
            firstMeasuredLineCount: firstMeasuredLineCount,
            coldLoadTimeMs: context.coldLoadTime * 1000,
            totalTimeMs: summarizeMs(totals),
            detectionTimeMs: summarizeMs(dets),
            detectionPreprocessTimeMs: summarizeMs(detPre),
            detectionInferenceTimeMs: summarizeMs(detInf),
            detectionPostprocessTimeMs: summarizeMs(detPost),
            recognitionTimeMs: summarizeMs(recs),
            recognitionPreprocessTimeMs: summarizeMs(recPre),
            recognitionInferenceTimeMs: summarizeMs(recInf),
            recognitionPostprocessTimeMs: summarizeMs(recPost),
            recognitionPerLine: perLine,
            pipelineOverheadTimeMs: summarizeMs(overheads),
            memoryFootprintBeforeLoadBytes: context.memoryBeforeLoad,
            memoryFootprintAfterLoadBytes: context.memoryAfterLoad,
            thermalState: String(describing: ProcessInfo.processInfo.thermalState)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let encoded = try encoder.encode(stats)
        let json = String(data: encoded, encoding: .utf8) ?? ""
        XCTAssertFalse(json.isEmpty, "on-device performance stats should encode")

        attachJSON(encoded, artifact: .onDevicePerformance)

        if ortProfiling, let out = try await context.manager.finalizeORTProfiling() {
            attachProfileFile(out.detectionProfileJSON, name: "ort_profile_detection")
            attachProfileFile(out.recognitionProfileJSON, name: "ort_profile_recognition")
        }
    }

    func testOnDeviceMemoryBenchmark() async throws {
        let executionProvider = try resolveBenchmarkExecutionProvider()
        let context = try await makeBenchmarkContext(executionProvider: executionProvider, ortProfiling: false)
        let warmup = try parseNonNegativeIntEnv(Self.warmupIterationsEnvKey, defaultValue: 3)
        let iterations = max(try parseNonNegativeIntEnv(Self.measuredIterationsEnvKey, defaultValue: 10), 1)
        try await runWarmup(engine: context.engine, image: context.cgImage, iterations: warmup)

        var measuredError: Error?
        let options = XCTMeasureOptions()
        options.iterationCount = iterations
        measure(metrics: [XCTMemoryMetric()], options: options) {
            guard measuredError == nil else { return }
            do {
                _ = try waitForAsync {
                    try await context.engine.run(context.cgImage, params: self.resolveBenchmarkRuntimeParams())
                }
            } catch {
                measuredError = error
            }
        }
        if let measuredError {
            throw measuredError
        }
    }

    // MARK: - Helpers

    private struct BenchmarkContext {
        let cgImage: CGImage
        let manager: ORTSessionManager
        let engine: OCREngine
        let memoryBeforeLoad: UInt64
        let memoryAfterLoad: UInt64
        let coldLoadTime: TimeInterval
    }

    private func makeBenchmarkContext(executionProvider: ORTPrimaryExecutionProvider, ortProfiling: Bool) async throws -> BenchmarkContext {
        let cgImage = try resolveBenchmarkImage()
        let memoryBeforeLoad = physicalFootprintBytes()
        let manager = ORTSessionManager()
        let loadStart = CFAbsoluteTimeGetCurrent()
        try await manager.loadModels(executionProvider: executionProvider, ortProfiling: ortProfiling, tuning: resolveSessionTuningOptions())
        let coldLoadTime = CFAbsoluteTimeGetCurrent() - loadStart
        let memoryAfterLoad = physicalFootprintBytes()
        let engine = try OCREngine(sessionManager: manager)
        return BenchmarkContext(
            cgImage: cgImage,
            manager: manager,
            engine: engine,
            memoryBeforeLoad: memoryBeforeLoad,
            memoryAfterLoad: memoryAfterLoad,
            coldLoadTime: coldLoadTime
        )
    }

    private func summarizeShapes(_ shapes: [[Int]]) -> [BenchmarkTensorShapeSample] {
        let counts = shapes.reduce(into: [[Int]: Int]()) { partial, shape in
            partial[shape, default: 0] += 1
        }
        return counts
            .map { BenchmarkTensorShapeSample(shape: $0.key, count: $0.value) }
            .sorted { lhs, rhs in
                if lhs.shape != rhs.shape {
                    return lhs.shape.lexicographicallyPrecedes(rhs.shape)
                }
                return lhs.count > rhs.count
            }
    }

    private func runWarmup(engine: OCREngine, image: CGImage, iterations: Int) async throws {
        let params = resolveBenchmarkRuntimeParams()
        for _ in 0..<iterations {
            _ = try await engine.run(image, params: params)
        }
    }

    private func appendRun(
        _ run: OCRRunResult,
        totals: inout [Double],
        dets: inout [Double],
        detPre: inout [Double],
        detInf: inout [Double],
        detPost: inout [Double],
        recs: inout [Double],
        recPre: inout [Double],
        recInf: inout [Double],
        recPost: inout [Double],
        overheads: inout [Double],
        lineInferenceMsPooled: inout [Double],
        linePreprocessMsPooled: inout [Double],
        linePostprocessMsPooled: inout [Double],
        lineTotalMsPooled: inout [Double]
    ) {
        totals.append(run.totalTime * 1000)
        dets.append(run.detectionTime * 1000)
        detPre.append(run.detectionPreprocessTime * 1000)
        detInf.append(run.detectionInferenceTime * 1000)
        detPost.append(run.detectionPostprocessTime * 1000)
        recs.append(run.recognitionTime * 1000)
        recPre.append(run.recognitionPreprocessTime * 1000)
        recInf.append(run.recognitionInferenceTime * 1000)
        recPost.append(run.recognitionPostprocessTime * 1000)
        overheads.append(run.pipelineOverheadTime * 1000)

        let n = run.recognitionLineCount
        if n > 0 {
            let infMs = run.lineRecognitionInferenceTimes.map { $0 * 1000.0 }
            let preMs = run.lineRecognitionPreprocessTimes.map { $0 * 1000.0 }
            let postMs = run.lineRecognitionPostprocessTimes.map { $0 * 1000.0 }
            lineInferenceMsPooled.append(contentsOf: infMs)
            linePreprocessMsPooled.append(contentsOf: preMs)
            linePostprocessMsPooled.append(contentsOf: postMs)
            for i in 0..<n {
                lineTotalMsPooled.append(preMs[i] + infMs[i] + postMs[i])
            }
        }
    }

    private func resolveBuildConfiguration() -> String {
        let raw =
            ProcessInfo.processInfo.environment[Self.buildConfigurationEnvKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return raw.isEmpty ? "Release" : raw
    }

    private func isRunningOnSimulator() -> Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }

    private func deviceModelName() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let mirror = Mirror(reflecting: systemInfo.machine)
        return mirror.children.reduce(into: "") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            identifier.append(String(UnicodeScalar(UInt8(value))))
        }
    }

    private func resolveORTProfilingEnabled() -> Bool {
        let raw =
            ProcessInfo.processInfo.environment[Self.ortProfilingEnvKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if raw.isEmpty { return false }
        switch raw.lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    private func resolveSessionTuningOptions() -> ORTSessionTuningOptions {
        let env = ProcessInfo.processInfo.environment
        var opts = ORTSessionTuningOptions()
        if let v = env[Self.intraOpThreadsEnvKey], let n = Int(v), n > 0 {
            opts.intraOpThreads = n
        }
        if let v = env[Self.xnnpackThreadsEnvKey], let n = Int(v), n > 0 {
            opts.xnnpackThreads = n
        }
        if let v = env[Self.coremlOptionsEnvKey], !v.isEmpty {
            opts.coreMLFlags = Set(v.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) })
        }
        return opts
    }

    private func attachProfileFile(_ url: URL, name: String) {
        let attachment = XCTAttachment(contentsOfFile: url)
        attachment.name = name
        attachment.lifetime = XCTAttachment.Lifetime.keepAlways
        add(attachment)
    }

    private func resolveBenchmarkExecutionProvider() throws -> ORTPrimaryExecutionProvider {
        let raw =
            ProcessInfo.processInfo.environment[Self.ortExecutionProviderEnvKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if raw.isEmpty {
            return .cpu
        }
        if let exact = ORTPrimaryExecutionProvider(rawValue: raw) {
            return exact
        }
        switch raw.lowercased() {
        case "core_ml":
            return .coreML
        case "xnnpack":
            return .xnnpack
        case "cpu":
            return .cpu
        default:
            throw NSError(
                domain: "OCRBenchmarkTests",
                code: 5,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Invalid \(Self.ortExecutionProviderEnvKey): \"\(raw)\". "
                        + "Use CPU (default), XNNPACK, CORE_ML, or Swift raw values "
                        + "\(ORTPrimaryExecutionProvider.cpu.rawValue) / "
                        + "\(ORTPrimaryExecutionProvider.xnnpack.rawValue) / "
                        + "\(ORTPrimaryExecutionProvider.coreML.rawValue).",
                ]
            )
        }
    }

    private func resolveBenchmarkImage() throws -> CGImage {
        let envRaw =
            ProcessInfo.processInfo.environment[Self.imageNameEnvKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let raw = envRaw.isEmpty ? BenchmarkFixtures.defaultReferenceImageStem : envRaw
        let bundle = Bundle(for: Self.self)
        guard let path = bundle.path(forBundledImageNamed: raw, subdirectory: "Fixtures") else {
            throw NSError(
                domain: "OCRBenchmarkTests",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Image \"\(raw)\" not found in the test bundle. Check the name and that the file is included in the test target resources (e.g. \"\(BenchmarkFixtures.defaultReferenceImageStem)\").",
                ]
            )
        }
        let url = URL(fileURLWithPath: path)
        guard let ui = UIImage(contentsOfFile: url.path),
              let cg = normalizeOrientation(ui).cgImage
        else {
            throw NSError(
                domain: "OCRBenchmarkTests",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Could not decode image at \(url.path)."]
            )
        }
        return cg
    }

    private func attachJSON(_ data: Data, artifact: BenchmarkArtifact) {
        let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.json")
        attachment.name = artifact.rawValue
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func waitForAsync<T>(_ operation: @escaping () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let lock = NSLock()
        var capturedResult: Result<T, Error>?
        Task {
            do {
                let value = try await operation()
                lock.lock()
                capturedResult = .success(value)
                lock.unlock()
            } catch {
                lock.lock()
                capturedResult = .failure(error)
                lock.unlock()
            }
            semaphore.signal()
        }
        semaphore.wait()
        lock.lock()
        let result = capturedResult
        lock.unlock()
        switch result {
        case .success(let value):
            return value
        case .failure(let error):
            throw error
        case .none:
            throw NSError(
                domain: "OCRBenchmarkTests",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Async operation completed without a result."]
            )
        }
    }

    /// Parses a non-negative integer from the environment, or returns `defaultValue` when unset/blank.
    private func parseNonNegativeIntEnv(_ key: String, defaultValue: Int) throws -> Int {
        let raw =
            ProcessInfo.processInfo.environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if raw.isEmpty {
            return defaultValue
        }
        guard let n = Int(raw), n >= 0 else {
            throw NSError(
                domain: "OCRBenchmarkTests",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Invalid \(key): expected a non-negative integer (default \(defaultValue) if unset), got \(raw).",
                ]
            )
        }
        return n
    }

    private func resolveBenchmarkRuntimeParams() -> OCRRuntimeParams {
        let env = ProcessInfo.processInfo.environment
        if let v = env[Self.recBatchSizeEnvKey], let n = Int(v), n >= 1 {
            return OCRRuntimeParams(
                textDetLimitSideLen: nil,
                textDetLimitType: nil,
                textDetMaxSideLimit: nil,
                textDetThresh: nil,
                textDetBoxThresh: nil,
                textDetUnclipRatio: nil,
                textRecBatchSize: n,
                textRecScoreThresh: nil
            )
        }
        return .noOverrides
    }

    /// Physical memory footprint (bytes) — matches Xcode Memory gauge more closely than RSS alone.
    /// Uses `task_vm_info_data_t.phys_footprint` via `task_info(TASK_VM_INFO, ...)`.
    private func physicalFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return 0 }
        return UInt64(info.phys_footprint)
    }

    private func summarizeMs(_ samples: [Double]) -> TimingSummary {
        let sorted = samples.sorted()
        let mean = samples.reduce(0, +) / Double(max(samples.count, 1))
        let variance =
            samples.isEmpty
                ? 0
                : samples.map { pow($0 - mean, 2) }.reduce(0, +) / Double(samples.count)
        let stdev = sqrt(variance)
        let p90Idx = max(0, min(sorted.count - 1, Int(floor(0.9 * Double(max(sorted.count - 1, 0))))))
        let p90 = sorted.isEmpty ? 0 : sorted[p90Idx]
        return TimingSummary(mean: mean, stdev: stdev, p90: p90)
    }
}

// MARK: - Orientation (match OCRViewModel)

private func normalizeOrientation(_ image: UIImage) -> UIImage {
    guard image.imageOrientation != .up else { return image }
    let format = UIGraphicsImageRendererFormat()
    format.scale = image.scale
    let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
    return renderer.image { _ in
        image.draw(in: CGRect(origin: .zero, size: image.size))
    }
}
