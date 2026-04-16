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
import Yams

// MARK: - Errors

enum InferenceConfigError: LocalizedError {
    case fileNotFound(String)
    case parseError(String)
    case missingField(String)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Model config file not found: \(path)"
        case .parseError(let detail):
            return "Failed to parse model config: \(detail)"
        case .missingField(let field):
            return "Missing required field in model config: \(field)"
        }
    }
}

// MARK: - Transform Operations

/// Channel order implied by `DecodeImage` in the model config file (`img_mode`: BGR or RGB).
/// When `DecodeImage` is absent, defaults to **BGR** (typical decode path for three-channel inputs).
enum InferenceImageChannelOrder: Equatable, Sendable {
    case bgr
    case rgb

    /// Parses `DecodeImage.img_mode`; empty/missing defaults to **BGR**.
    static func fromDecodeImage(imgMode: String?) -> InferenceImageChannelOrder {
        guard let raw = imgMode?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return .bgr
        }
        switch raw.lowercased() {
        case "rgb": return .rgb
        case "bgr": return .bgr
        default:
            return .bgr
        }
    }
}

/// Parameters for `DetResizeForTest` from the model config file.
struct DetResizeParams: Equatable {
    var limitSideLen: Int
    var limitType: String
    var maxSideLimit: Int
}

/// Represents a single preprocessing transform operation parsed from the model config file.
/// Each case carries associated parameters read from the model config file.
enum TransformOp {
    /// `DecodeImage` in the model config file — selects BGR vs RGB for decoded pixels.
    case decodeImage(channelOrder: InferenceImageChannelOrder)
    case detResizeForTest(DetResizeParams)
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
    let thresh: Float
    let boxThresh: Float
    let maxCandidates: Int
    let unclipRatio: Float
    let characterDict: [String]?
}

struct InferenceConfig {
    let modelName: String
    let preProcess: PreProcessConfig
    let postProcess: PostProcessConfig

    // MARK: - Loading

    /// Loads and parses a model config file into a typed `InferenceConfig`.
    ///
    /// - Parameter path: Absolute filesystem path to the model config file.
    /// - Returns: A fully parsed `InferenceConfig` with typed transform operations.
    static func load(from path: String) throws -> InferenceConfig {
        guard FileManager.default.fileExists(atPath: path) else {
            throw InferenceConfigError.fileNotFound(path)
        }

        let configFileText: String
        do {
            configFileText = try String(contentsOfFile: path, encoding: .utf8)
        } catch {
            throw InferenceConfigError.parseError("Cannot read file: \(error.localizedDescription)")
        }

        guard let root = try Yams.load(yaml: configFileText) as? [String: Any] else {
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
}

extension InferenceConfig {
    // MARK: - Private Parsing Helpers

    private enum ParseDefaults {
        static let postThresh: Float = 0.3
        static let postBoxThresh: Float = 0.6
        static let postMaxCandidates: Int = 1000
        static let postUnclipRatio: Float = 2.0
        static let detResizeLimitSideLen: Int = 960
        static let detResizeLimitType: String = "max"
        static let detResizeMaxSideLimit: Int = 4000
    }

    private static let defaultDetResizeParams = DetResizeParams(
        limitSideLen: ParseDefaults.detResizeLimitSideLen,
        limitType: ParseDefaults.detResizeLimitType,
        maxSideLimit: ParseDefaults.detResizeMaxSideLimit
    )

    /// Parses a single transform operation dictionary (one key = op name, value = params or null).
    private static func parseTransformOp(_ dict: [String: Any?]) -> TransformOp {
        guard let opName = dict.keys.first else {
            return .unknown(name: "empty")
        }

        switch opName {
        case "DecodeImage":
            let params = dict[opName] as? [String: Any] ?? [:]
            let imgMode = params["img_mode"] as? String
            return .decodeImage(channelOrder: InferenceImageChannelOrder.fromDecodeImage(imgMode: imgMode))

        case "DetResizeForTest":
            let params = dict[opName] as? [String: Any] ?? [:]
            let limitSideLen =
                optionalIntValue(params["resize_long"])
                ?? optionalIntValue(params["limit_side_len"])
                ?? ParseDefaults.detResizeLimitSideLen
            let limitType = params["limit_type"] as? String ?? ParseDefaults.detResizeLimitType
            let maxSideLimit =
                intValueForKeyIfPresent(params, key: "max_side_limit")
                ?? ParseDefaults.detResizeMaxSideLimit
            return .detResizeForTest(
                DetResizeParams(
                    limitSideLen: limitSideLen,
                    limitType: limitType,
                    maxSideLimit: maxSideLimit
                )
            )

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

    /// Parses the `scale` field which may be a numeric value or a fraction string like `"1./255."`.
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

    /// Evaluates a fraction string by splitting on `"/"` and dividing numerator by denominator.
    private static func parseScaleString(_ s: String) -> Float {
        if s.contains("/") {
            let parts = s.split(separator: "/")
            if parts.count == 2,
               let numerator = Double(parts[0].trimmingCharacters(in: .init(charactersIn: "."))),
               let denominator = Double(parts[1].trimmingCharacters(in: .init(charactersIn: "."))) {
                return Float(numerator / denominator)
            }
            let rawParts = s.split(separator: "/")
            if rawParts.count == 2,
               let num = Double(rawParts[0]),
               let den = Double(rawParts[1]) {
                return Float(num / den)
            }
        }
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
        let thresh = (dict["thresh"] as? Double).map { Float($0) } ?? ParseDefaults.postThresh
        let boxThresh = (dict["box_thresh"] as? Double).map { Float($0) } ?? ParseDefaults.postBoxThresh
        let maxCandidates = dict["max_candidates"] as? Int ?? ParseDefaults.postMaxCandidates
        let unclipRatio = (dict["unclip_ratio"] as? Double).map { Float($0) } ?? ParseDefaults.postUnclipRatio
        let characterDict = (dict["character_dict"] as? [Any])?.compactMap { $0 as? String }

        return PostProcessConfig(
            name: name,
            thresh: thresh,
            boxThresh: boxThresh,
            maxCandidates: maxCandidates,
            unclipRatio: unclipRatio,
            characterDict: characterDict
        )
    }

    private static func toInt(_ value: Any) -> Int? {
        if let i = value as? Int { return i }
        if let d = value as? Double { return Int(d) }
        return nil
    }

    private static func optionalIntValue(_ value: Any?) -> Int? {
        guard let value else { return nil }
        return toInt(value)
    }

    /// Only returns a value when the key exists in `params` (a missing key yields `nil`).
    private static func stringValueForKeyIfPresent(_ params: [String: Any], key: String) -> String? {
        guard params.keys.contains(key) else { return nil }
        return params[key] as? String
    }

    private static func intValueForKeyIfPresent(_ params: [String: Any], key: String) -> Int? {
        guard params.keys.contains(key) else { return nil }
        return optionalIntValue(params[key])
    }

    /// First `DetResizeForTest` block in the model config file next to the ONNX weights, if present.
    var detResizeFromModel: DetResizeParams? {
        for op in preProcess.transformOps {
            if case .detResizeForTest(let p) = op { return p }
        }
        return nil
    }

    var detResizeForMerge: DetResizeParams {
        detResizeFromModel ?? Self.defaultDetResizeParams
    }

    /// First `DecodeImage` entry in `transform_ops` and its `img_mode` (BGR or RGB). If absent, **BGR**.
    var decodeImageChannelOrder: InferenceImageChannelOrder {
        for op in preProcess.transformOps {
            if case .decodeImage(let order) = op { return order }
        }
        return .bgr
    }
}
