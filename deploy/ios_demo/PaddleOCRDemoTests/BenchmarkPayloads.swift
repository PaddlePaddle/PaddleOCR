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

import Foundation

@testable import PaddleOCRDemo

/// Names for JSON written as `XCTAttachment`s; keep in sync with whatever exports attachments from test results.
enum BenchmarkArtifact: String {
    case iOSExport = "ios-ocr-export.json"
    case onDevicePerformance = "on-device-performance.json"
}

// MARK: - Accuracy export payload

struct OCRExportPayload: Codable {
    var schemaVersion: Int = 1
    var source: String
    var items: [OCRExportItem]
}

struct OCRExportItem: Codable {
    var polygon: [[Int]]
    var text: String
    var score: Double?
}

// MARK: - On-device performance payload

struct TimingSummary: Codable {
    var mean: Double
    var stdev: Double
    var p90: Double
}

/// Pooled: one sample per det line per measured run (all lines concatenated, ms).
struct RecognitionPerLinePooledMs: Codable {
    var count: Int
    var inferenceMs: TimingSummary
    var preprocessMs: TimingSummary
    var postprocessMs: TimingSummary
    var totalMs: TimingSummary
}

struct RecognitionPerLineBlock: Codable {
    var pooled: RecognitionPerLinePooledMs
}

struct BenchmarkTensorShapeSample: Codable {
    var shape: [Int]
    var count: Int
}

struct BenchmarkInputShapeDistribution: Codable {
    var detection: [BenchmarkTensorShapeSample]
    var recognition: [BenchmarkTensorShapeSample]
}

struct OCRDeviceBenchmarkPayload: Codable {
    var schemaVersion: Int = 1
    var buildConfiguration: String
    var deviceModel: String
    var osVersion: String
    var isSimulator: Bool
    var ortExecutionProvider: String
    var ortProfilingEnabled: Bool
    var warmupIterations: Int
    var measuredIterations: Int
    var inputShapeDistribution: BenchmarkInputShapeDistribution
    var firstMeasuredLineCount: Int
    var coldLoadTimeMs: Double
    var totalTimeMs: TimingSummary
    var detectionTimeMs: TimingSummary
    var detectionPreprocessTimeMs: TimingSummary
    var detectionInferenceTimeMs: TimingSummary
    var detectionPostprocessTimeMs: TimingSummary
    var recognitionTimeMs: TimingSummary
    var recognitionPreprocessTimeMs: TimingSummary
    var recognitionInferenceTimeMs: TimingSummary
    var recognitionPostprocessTimeMs: TimingSummary
    var recognitionPerLine: RecognitionPerLineBlock?
    var pipelineOverheadTimeMs: TimingSummary
    var memoryFootprintBeforeLoadBytes: UInt64?
    var memoryFootprintAfterLoadBytes: UInt64
    var thermalState: String?
}
