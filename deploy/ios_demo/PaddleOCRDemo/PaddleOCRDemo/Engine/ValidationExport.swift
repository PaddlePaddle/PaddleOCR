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

/// Exports OCR pipeline results as JSON for validation against the Python reference.
///
/// The JSON schema matches `deploy/ios_demo/Validation/validate.py`:
/// ```json
/// {
///   "image": "filename.jpg",
///   "box_count": 5,
///   "boxes": [
///     {
///       "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
///       "text": "recognized text",
///       "confidence": 0.9876
///     }
///   ]
/// }
/// ```
struct ValidationExport {

    /// Serialize an OCRPipelineResult to JSON Data matching the validation schema.
    ///
    /// - Parameters:
    ///   - result: The pipeline result to serialize.
    ///   - imageName: The filename of the source image (e.g., "test_001.jpg").
    /// - Returns: JSON data as UTF-8 encoded Data.
    static func toJSON(result: OCRPipelineResult, imageName: String) throws -> Data {
        let boxes: [[String: Any]] = result.results.map { ocrResult in
            [
                "polygon": ocrResult.polygon.map { [$0[0], $0[1]] },
                "text": ocrResult.text,
                "confidence": ocrResult.confidence,
            ]
        }

        let root: [String: Any] = [
            "image": imageName,
            "box_count": result.results.count,
            "boxes": boxes,
        ]

        return try JSONSerialization.data(
            withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys]
        )
    }

    /// Write an OCRPipelineResult to a JSON file in the app's Documents directory.
    ///
    /// - Parameters:
    ///   - result: The pipeline result to export.
    ///   - imageName: The filename of the source image.
    ///   - directory: The directory to write to (defaults to Documents/validation_output).
    /// - Returns: The URL of the written file.
    @discardableResult
    static func writeJSON(
        result: OCRPipelineResult,
        imageName: String,
        directory: URL? = nil
    ) throws -> URL {
        let data = try toJSON(result: result, imageName: imageName)

        let outputDir = directory ?? FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask
        ).first!.appendingPathComponent("validation_output")

        try FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true
        )

        let stem = (imageName as NSString).deletingPathExtension
        let fileURL = outputDir.appendingPathComponent("\(stem).json")
        try data.write(to: fileURL)

        return fileURL
    }
}
