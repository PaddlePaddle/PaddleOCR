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
// WITHOUT WARRANTIES OR ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

enum OCRPipelineDefaults {

    // MARK: Text detection (DB) — shipped pipeline-tier defaults for the end-to-end OCR path

    /// Shipped defaults for the end-to-end OCR path; the user can override in the UI, then the model config file,
    /// then final-tier defaults when keys are absent.
    static let textDetThresh: Float = 0.3
    static let textDetBoxThresh: Float = 0.6
    static let textDetUnclipRatio: Float = 1.5

    /// Limit-side resize (`limit_side_len` + `limit_type` + `max_side_limit`).
    static let textDetLimitSideLen: Int = 64
    static let textDetLimitType: String = "min"
    static let textDetMaxSideLimit: Int = 4000

    /// Long-edge resize when the UI sets `textDetResizeLong` (typically `nil` so limit-side merge applies).
    static let textDetResizeLong: Int? = nil

    /// Hint for the parameters UI when long-edge resize is enabled (typical export value; the model config may differ).
    static let modelFileResizeLongHint: Int = 960

    /// DB `max_candidates` when neither runtime nor the model config sets it.
    static let textDetMaxCandidates: Int = 1000

    // MARK: Text recognition (end-to-end filter)

    static let textRecScoreThresh: Float = 0.0

    // MARK: Merge helpers

    /// Merge order: **runtime (UI)** → **`OCRPipelineDefaults`** → **detection PostProcess from the model config file** → **`OCRDefaultThresholds`** (when the model file omits keys).
    static func effectiveDetPostprocess(inference: InferenceConfig, runtime: OCRRuntimeParams) -> (
        thresh: Float, boxThresh: Float, maxCandidates: Int, unclipRatio: Float
    ) {
        let pp = inference.postProcess
        let thresh =
            runtime.textDetThresh ?? Optional(textDetThresh) ?? pp.thresh ?? OCRDefaultThresholds.textDetThresh
        let boxThresh =
            runtime.textDetBoxThresh ?? Optional(textDetBoxThresh) ?? pp.boxThresh
            ?? OCRDefaultThresholds.textDetBoxThresh
        let unclipRatio =
            runtime.textDetUnclipRatio ?? Optional(textDetUnclipRatio) ?? pp.unclipRatio
            ?? OCRDefaultThresholds.textDetUnclipRatio
        let maxCandidates =
            runtime.textDetMaxCandidates ?? Optional(textDetMaxCandidates) ?? pp.maxCandidates ?? 1000
        return (thresh, boxThresh, maxCandidates, unclipRatio)
    }

    /// Merge order for detection resize: **runtime** → **`OCRPipelineDefaults`** (limit side / type / max side) → **model config `DetResizeForTest`** (explicit keys only) → **heuristic / final-tier defaults**.
    static func effectiveDetResize(inference: InferenceConfig, runtime: OCRRuntimeParams) -> DetResizeParams {
        let m = inference.detResizeFromModel

        let resolvedLimitSide: Int =
            runtime.textDetLimitSideLen
            ?? Optional(textDetLimitSideLen)
            ?? m?.limitSideLen
            ?? runtime.textDetResizeLong
            ?? textDetResizeLong
            ?? m?.resizeLong
            ?? 960

        let resolvedLimitType: String =
            runtime.textDetLimitType
            ?? Optional(textDetLimitType)
            ?? m?.limitType
            ?? "max"

        let resolvedMaxSide: Int =
            runtime.textDetMaxSideLimit
            ?? Optional(textDetMaxSideLimit)
            ?? m?.maxSideLimit
            ?? 4000

        return DetResizeParams(
            limitSideLen: resolvedLimitSide,
            limitType: resolvedLimitType,
            maxSideLimit: resolvedMaxSide,
            resizeLong: nil
        )
    }

    /// Recognition score floor — UI overrides the shipped default.
    static func effectiveRecScoreThresh(runtime: OCRRuntimeParams) -> Float {
        runtime.textRecScoreThresh ?? textRecScoreThresh ?? 0.0
    }
}
