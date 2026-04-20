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

/// Named JSON attachments emitted by validation tests.
///
/// `rawValue` is the file name used as the `XCTAttachment.name`; `run_validation.sh`
/// passes these same strings to `extract_xcresult_attachments.py` to pull the
/// payload back out of the `.xcresult`.
enum ValidationArtifact: String {
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

struct OCRDeviceBenchmarkPayload: Codable {
    /// Bumped when the JSON shape changes; pre-release builds keep this at `1`.
    var schemaVersion: Int = 1
    var warmupIterations: Int
    var measuredIterations: Int
    var totalTimeMs: TimingSummary
    var detectionTimeMs: TimingSummary
    var detectionPreprocessTimeMs: TimingSummary
    var detectionInferenceTimeMs: TimingSummary
    var detectionPostprocessTimeMs: TimingSummary
    var recognitionTimeMs: TimingSummary
    var recognitionPreprocessTimeMs: TimingSummary
    var recognitionInferenceTimeMs: TimingSummary
    var recognitionPostprocessTimeMs: TimingSummary
    var pipelineOverheadTimeMs: TimingSummary
    var memoryFootprintBeforeLoadBytes: UInt64?
    var memoryFootprintAfterLoadBytes: UInt64
    var memoryInferencePeakBytes: UInt64
    var memoryInferenceMeanBytes: UInt64
    var memoryInferenceSampleCount: Int
    var thermalState: String?
}
