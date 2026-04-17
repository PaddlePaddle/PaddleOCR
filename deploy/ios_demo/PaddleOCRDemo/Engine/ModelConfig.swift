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

/// Resolves exported ONNX models and their **model config files** in the app bundle (`Models/det`, `Models/rec`).
/// Resource lookup uses the conventional shared basename for the weight file and the model config file in each folder.
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

    static func detection() throws -> ModelConfig {
        guard let modelPath = Bundle.main.path(forResource: "inference", ofType: "onnx", inDirectory: "Models/det") else {
            throw ModelConfigError.modelNotFound("det/inference.onnx")
        }
        guard let configPath = Bundle.main.path(forResource: "inference", ofType: "yml", inDirectory: "Models/det") else {
            throw ModelConfigError.modelNotFound("det model config file")
        }
        let name = displayName(fromModelConfigPath: configPath, fallback: "text_detection")
        return ModelConfig(modelPath: modelPath, configPath: configPath, name: name)
    }

    static func recognition() throws -> ModelConfig {
        guard let modelPath = Bundle.main.path(forResource: "inference", ofType: "onnx", inDirectory: "Models/rec") else {
            throw ModelConfigError.modelNotFound("rec/inference.onnx")
        }
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
