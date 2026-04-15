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

struct ModelConfig {
    let modelPath: String
    let configPath: String
    let name: String

    static func detection() throws -> ModelConfig {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx", inDirectory: "Models/det") else {
            throw ModelConfigError.modelNotFound("det/model.onnx")
        }
        let configPath = Bundle.main.path(forResource: "inference", ofType: "yml", inDirectory: "Models/det")
        return ModelConfig(modelPath: modelPath, configPath: configPath ?? "", name: "PP-OCRv5 Mobile Det")
    }

    static func recognition() throws -> ModelConfig {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx", inDirectory: "Models/rec") else {
            throw ModelConfigError.modelNotFound("rec/model.onnx")
        }
        let configPath = Bundle.main.path(forResource: "inference", ofType: "yml", inDirectory: "Models/rec")
        return ModelConfig(modelPath: modelPath, configPath: configPath ?? "", name: "PP-OCRv5 Mobile Rec")
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
