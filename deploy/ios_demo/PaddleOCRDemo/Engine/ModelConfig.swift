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

/// Resolves exported ONNX or ORT-format models and their **model config files** in the app bundle (`Models/det`, `Models/rec`).
/// Resource lookup uses the conventional shared basename for the weight file and the model config file in each folder.
/// Weights: prefers `inference*.ort` in the model directory (see ``bundledOrtPath``), otherwise `inference.onnx`.
struct ModelConfig {
    let modelPath: String
    let configPath: String
    /// From `Global.model_name` in the model config file (same directory as the ONNX weights).
    let name: String

    /// Human-readable label from `Global.model_name` in the model config file; `fallback` if the file cannot be read.
    private static func displayName(fromModelConfigPath configPath: String, fallback: String) -> String {
        guard let cfg = try? InferenceConfig.load(from: configPath) else {
            return fallback
        }
        return cfg.modelName
    }

    /// Bundled model weights: any `inference*.ort` in the subdirectory (``ortBundledFileMatchOrder``), else `inference.onnx`.
    ///
    /// ONNX Runtime’s `convert_onnx_models_to_ort` may emit `inference.with_runtime_opt.ort` instead of `inference.ort`
    /// when using Runtime optimization style; the bundle cannot resolve that with `forResource: "inference"`.
    private static let ortResourceStems: [String] = [
        "inference",
        "inference.with_runtime_opt",
    ]

    private static func bundledOrtPath(inDirectory directory: String) -> String? {
        for stem in ortResourceStems {
            if let p = Bundle.main.path(forResource: stem, ofType: "ort", inDirectory: directory) {
                return p
            }
        }
        // Other ORT export suffixes (e.g. non-default optim levels): one `inference*.ort` in the folder.
        guard let urls = Bundle.main.urls(forResourcesWithExtension: "ort", subdirectory: directory) else {
            return nil
        }
        let matches = urls.filter { url in
            let c = url.lastPathComponent
            return c.hasPrefix("inference") && c.hasSuffix(".ort")
        }
        guard !matches.isEmpty else { return nil }
        if matches.count == 1 { return matches[0].path }
        // If multiple, prefer lexicographic first (deterministic); avoid shipping several exports in one dir.
        return matches.sorted { $0.lastPathComponent < $1.lastPathComponent }.first?.path
    }

    private static func bundledWeightsPath(inDirectory directory: String, notFoundMessage: String) throws -> String {
        if let ort = bundledOrtPath(inDirectory: directory) {
            return ort
        }
        if let onnx = Bundle.main.path(forResource: "inference", ofType: "onnx", inDirectory: directory) {
            return onnx
        }
        throw ModelConfigError.modelNotFound(notFoundMessage)
    }

    static func detection() throws -> ModelConfig {
        let modelPath = try Self.bundledWeightsPath(
            inDirectory: "Models/det",
            notFoundMessage: "det/inference.ort or det/inference.onnx"
        )
        guard let configPath = Bundle.main.path(forResource: "inference", ofType: "yml", inDirectory: "Models/det") else {
            throw ModelConfigError.modelNotFound("det model config file")
        }
        let name = displayName(fromModelConfigPath: configPath, fallback: "text_detection")
        return ModelConfig(modelPath: modelPath, configPath: configPath, name: name)
    }

    static func recognition() throws -> ModelConfig {
        let modelPath = try Self.bundledWeightsPath(
            inDirectory: "Models/rec",
            notFoundMessage: "rec/inference.ort or rec/inference.onnx"
        )
        guard let configPath = Bundle.main.path(forResource: "inference", ofType: "yml", inDirectory: "Models/rec") else {
            throw ModelConfigError.modelNotFound("rec model config file")
        }
        let name = displayName(fromModelConfigPath: configPath, fallback: "text_recognition")
        return ModelConfig(modelPath: modelPath, configPath: configPath, name: name)
    }
}

enum ModelConfigError: LocalizedError {
    case modelNotFound(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model file not found in bundle: \(path)"
        }
    }
}
