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

final class OCRValidationTests: XCTestCase {

    /// `PADDLEOCR_VALIDATION_IMAGE_NAME`: bundled image stem or `stem.ext` (optional).
    private static let imageNameEnvKey = "PADDLEOCR_VALIDATION_IMAGE_NAME"

    /// Optional non-negative int; default `3`. Used only by `testOnDevicePerformanceMetrics`.
    private static let warmupIterationsEnvKey = "PADDLEOCR_VALIDATION_WARMUP_ITERATIONS"

    /// Optional non-negative int; default `10`. Used only by `testOnDevicePerformanceMetrics`.
    private static let measuredIterationsEnvKey = "PADDLEOCR_VALIDATION_MEASURED_ITERATIONS"

    /// `CORE_ML` (default) or `XNNPACK` — ONNX Runtime EP for validation runs.
    private static let inferenceBackendEnvKey = "PADDLEOCR_VALIDATION_INFERENCE_BACKEND"

    func testOCRExportJSONSchema() async throws {
        let cgImage = try resolveValidationImage()
        let backend = try resolveInferenceBackend()
        let manager = ORTSessionManager()
        try await manager.loadModels(backend: backend)
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
    }

    func testOnDevicePerformanceMetrics() async throws {
        let cgImage = try resolveValidationImage()
        let memoryBeforeLoad = physicalFootprintBytes()
        let backend = try resolveInferenceBackend()
        let manager = ORTSessionManager()
        try await manager.loadModels(backend: backend)
        let memoryAfterLoad = physicalFootprintBytes()
        let engine = try OCREngine(sessionManager: manager)

        let warmup = try parseNonNegativeIntEnv(Self.warmupIterationsEnvKey, defaultValue: 3)
        let iterations = max(try parseNonNegativeIntEnv(Self.measuredIterationsEnvKey, defaultValue: 10), 1)

        for _ in 0..<warmup {
            _ = try await engine.run(cgImage, params: .noOverrides)
        }
        var measuredError: Error?
        var observedRuns: [OCRRunResult] = []
        observedRuns.reserveCapacity(iterations)

        let options = XCTMeasureOptions()
        options.iterationCount = iterations
        measure(metrics: [XCTMemoryMetric()], options: options) {
            guard measuredError == nil else { return }
            do {
                let run = try waitForAsync {
                    try await engine.run(cgImage, params: .noOverrides)
                }
                observedRuns.append(run)
            } catch {
                measuredError = error
            }
        }
        if let measuredError {
            throw measuredError
        }
        let measuredRuns = Array(observedRuns.suffix(iterations))
        XCTAssertEqual(measuredRuns.count, iterations, "XCTest should execute at least the configured measured iterations")

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

        for run in measuredRuns {
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
            warmupIterations: warmup,
            measuredIterations: iterations,
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
            memoryFootprintBeforeLoadBytes: memoryBeforeLoad,
            memoryFootprintAfterLoadBytes: memoryAfterLoad,
            thermalState: String(describing: ProcessInfo.processInfo.thermalState)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let encoded = try encoder.encode(stats)
        let json = String(data: encoded, encoding: .utf8) ?? ""
        XCTAssertFalse(json.isEmpty, "on-device performance stats should encode")

        attachJSON(encoded, artifact: .onDevicePerformance)
    }

    // MARK: - Helpers

    private func resolveInferenceBackend() throws -> ORTInferenceBackend {
        let raw =
            ProcessInfo.processInfo.environment[Self.inferenceBackendEnvKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if raw.isEmpty {
            return .coreMLOnly
        }
        if let exact = ORTInferenceBackend(rawValue: raw) {
            return exact
        }
        switch raw.lowercased() {
        case "core_ml":
            return .coreMLOnly
        case "xnnpack":
            return .xnnpackOnly
        case "cpu":
            return .cpuOnly
        default:
            throw NSError(
                domain: "OCRValidationTests",
                code: 5,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Invalid \(Self.inferenceBackendEnvKey): \"\(raw)\". "
                        + "Use CORE_ML (default), XNNPACK, CPU, or Swift raw values "
                        + "\(ORTInferenceBackend.coreMLOnly.rawValue) / "
                        + "\(ORTInferenceBackend.xnnpackOnly.rawValue) / "
                        + "\(ORTInferenceBackend.cpuOnly.rawValue). See README.",
                ]
            )
        }
    }

    private func resolveValidationImage() throws -> CGImage {
        let envRaw =
            ProcessInfo.processInfo.environment[Self.imageNameEnvKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let raw = envRaw.isEmpty ? ValidationFixtures.defaultReferenceImageStem : envRaw
        let bundle = Bundle(for: Self.self)
        guard let path = bundle.path(forBundledImageNamed: raw, subdirectory: "Fixtures") else {
            throw NSError(
                domain: "OCRValidationTests",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Image \"\(raw)\" not found in the test bundle. Check the name and that the file is included in the test target resources (e.g. \"\(ValidationFixtures.defaultReferenceImageStem)\"). See README.",
                ]
            )
        }
        let url = URL(fileURLWithPath: path)
        guard let ui = UIImage(contentsOfFile: url.path),
              let cg = normalizeOrientation(ui).cgImage
        else {
            throw NSError(
                domain: "OCRValidationTests",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Could not decode image at \(url.path)."]
            )
        }
        return cg
    }

    private func attachJSON(_ data: Data, artifact: ValidationArtifact) {
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
                domain: "OCRValidationTests",
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
                domain: "OCRValidationTests",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Invalid \(key): expected a non-negative integer (default \(defaultValue) if unset), got \(raw).",
                ]
            )
        }
        return n
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
