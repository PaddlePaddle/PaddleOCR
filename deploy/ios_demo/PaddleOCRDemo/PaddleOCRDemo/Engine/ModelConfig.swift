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
