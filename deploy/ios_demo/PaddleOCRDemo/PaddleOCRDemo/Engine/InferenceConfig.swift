import Foundation
import Yams

// MARK: - Errors

enum InferenceConfigError: LocalizedError {
    case fileNotFound(String)
    case parseError(String)
    case missingField(String)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Inference config file not found: \(path)"
        case .parseError(let detail):
            return "Failed to parse inference config: \(detail)"
        case .missingField(let field):
            return "Missing required field in inference config: \(field)"
        }
    }
}

// MARK: - Transform Operations

/// Represents a single preprocessing transform operation parsed from inference.yml.
/// Each case carries associated parameters read from the YAML config.
enum TransformOp {
    case detResizeForTest(resizeLong: Int)
    case normalizeImage(scale: Float, mean: [Float], std: [Float], order: String)
    case toCHWImage
    case recResizeImg(imageShape: [Int])
    case unknown(name: String)
}

// MARK: - Config Structures

struct PreProcessConfig {
    let transformOps: [TransformOp]
}

struct PostProcessConfig {
    let name: String
    let thresh: Float?
    let boxThresh: Float?
    let maxCandidates: Int?
    let unclipRatio: Float?
    let characterDictPath: String?
    let characterDict: [String]?
}

struct InferenceConfig {
    let modelName: String
    let preProcess: PreProcessConfig
    let postProcess: PostProcessConfig

    // MARK: - Loading

    /// Loads and parses an inference.yml file into a typed InferenceConfig.
    ///
    /// - Parameter yamlPath: Absolute filesystem path to the inference.yml file.
    /// - Returns: A fully parsed InferenceConfig with typed transform operations.
    static func load(from yamlPath: String) throws -> InferenceConfig {
        guard FileManager.default.fileExists(atPath: yamlPath) else {
            throw InferenceConfigError.fileNotFound(yamlPath)
        }

        let yamlString: String
        do {
            yamlString = try String(contentsOfFile: yamlPath, encoding: .utf8)
        } catch {
            throw InferenceConfigError.parseError("Cannot read file: \(error.localizedDescription)")
        }

        guard let root = try Yams.load(yaml: yamlString) as? [String: Any] else {
            throw InferenceConfigError.parseError("Root element is not a dictionary")
        }

        // Parse Global.model_name
        guard let global = root["Global"] as? [String: Any],
              let modelName = global["model_name"] as? String else {
            throw InferenceConfigError.missingField("Global.model_name")
        }

        // Parse PreProcess
        guard let preProcessDict = root["PreProcess"] as? [String: Any],
              let transformOpsRaw = preProcessDict["transform_ops"] as? [[String: Any?]] else {
            throw InferenceConfigError.missingField("PreProcess.transform_ops")
        }

        let transformOps = transformOpsRaw.map { parseTransformOp($0) }
        let preProcess = PreProcessConfig(transformOps: transformOps)

        // Parse PostProcess
        guard let postProcessDict = root["PostProcess"] as? [String: Any] else {
            throw InferenceConfigError.missingField("PostProcess")
        }
        let postProcess = parsePostProcess(postProcessDict)

        return InferenceConfig(
            modelName: modelName,
            preProcess: preProcess,
            postProcess: postProcess
        )
    }

    // MARK: - Private Parsing Helpers

    /// Parses a single transform operation dictionary (one key = op name, value = params or null).
    private static func parseTransformOp(_ dict: [String: Any?]) -> TransformOp {
        guard let opName = dict.keys.first else {
            return .unknown(name: "empty")
        }

        switch opName {
        case "DetResizeForTest":
            let params = dict[opName] as? [String: Any] ?? [:]
            let resizeLong = params["resize_long"] as? Int ?? 960
            return .detResizeForTest(resizeLong: resizeLong)

        case "NormalizeImage":
            let params = dict[opName] as? [String: Any] ?? [:]
            let scale = parseScale(params["scale"])
            let mean = parseFloatArray(params["mean"]) ?? [0.485, 0.456, 0.406]
            let std = parseFloatArray(params["std"]) ?? [0.229, 0.224, 0.225]
            let order = params["order"] as? String ?? "hwc"
            return .normalizeImage(scale: scale, mean: mean, std: std, order: order)

        case "ToCHWImage":
            return .toCHWImage

        case "RecResizeImg":
            let params = dict[opName] as? [String: Any] ?? [:]
            let imageShape = (params["image_shape"] as? [Any])?.compactMap { toInt($0) } ?? [3, 48, 320]
            return .recResizeImg(imageShape: imageShape)

        default:
            return .unknown(name: opName)
        }
    }

    /// Parses the `scale` field which may be a numeric value or a Python-style string like "1./255.".
    /// Handles string division expressions by splitting on "/" and computing the result.
    private static func parseScale(_ value: Any?) -> Float {
        if let floatVal = value as? Double {
            return Float(floatVal)
        }
        if let stringVal = value as? String {
            return parseScaleString(stringVal)
        }
        // Default scale: 1/255
        return 1.0 / 255.0
    }

    /// Evaluates a Python-style scale string like "1./255." by splitting on "/" and dividing.
    private static func parseScaleString(_ s: String) -> Float {
        if s.contains("/") {
            let parts = s.split(separator: "/")
            if parts.count == 2,
               let numerator = Double(parts[0].trimmingCharacters(in: .init(charactersIn: "."))),
               let denominator = Double(parts[1].trimmingCharacters(in: .init(charactersIn: "."))) {
                // "1." -> 1.0, "255." -> 255.0
                // Handle edge cases: "1./255." -> numerator=1.0, denominator=255.0
                return Float(numerator / denominator)
            }
            // Fallback: try parsing the parts as-is (e.g. "1.0/255.0")
            let rawParts = s.split(separator: "/")
            if rawParts.count == 2,
               let num = Double(rawParts[0]),
               let den = Double(rawParts[1]) {
                return Float(num / den)
            }
        }
        // Try direct parse
        if let val = Double(s) {
            return Float(val)
        }
        return 1.0 / 255.0
    }

    private static func parseFloatArray(_ value: Any?) -> [Float]? {
        guard let array = value as? [Any] else { return nil }
        return array.compactMap { element -> Float? in
            if let d = element as? Double { return Float(d) }
            if let i = element as? Int { return Float(i) }
            return nil
        }
    }

    private static func parsePostProcess(_ dict: [String: Any]) -> PostProcessConfig {
        let name = dict["name"] as? String ?? "Unknown"
        let thresh = (dict["thresh"] as? Double).map { Float($0) }
        let boxThresh = (dict["box_thresh"] as? Double).map { Float($0) }
        let maxCandidates = dict["max_candidates"] as? Int
        let unclipRatio = (dict["unclip_ratio"] as? Double).map { Float($0) }
        let characterDictPath = dict["character_dict_path"] as? String
        let characterDict = (dict["character_dict"] as? [Any])?.compactMap { $0 as? String }

        return PostProcessConfig(
            name: name,
            thresh: thresh,
            boxThresh: boxThresh,
            maxCandidates: maxCandidates,
            unclipRatio: unclipRatio,
            characterDictPath: characterDictPath,
            characterDict: characterDict
        )
    }

    private static func toInt(_ value: Any) -> Int? {
        if let i = value as? Int { return i }
        if let d = value as? Double { return Int(d) }
        return nil
    }
}
