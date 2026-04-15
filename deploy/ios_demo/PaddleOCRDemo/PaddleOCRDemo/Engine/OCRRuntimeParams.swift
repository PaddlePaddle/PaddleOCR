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

// MARK: - Built-in defaults

/// Fallback numeric thresholds when a value is absent from the detection model config file (`PostProcess`), or for the end-to-end recognition score floor (not stored in the recognition model config).
enum OCRDefaultThresholds {
    static let textDetThresh: Float = 0.3
    static let textDetBoxThresh: Float = 0.6
    static let textDetUnclipRatio: Float = 1.5
    /// Minimum recognition line score to keep a result after decode; applied in `OCREngine`, not inside the recognition model.
    static let textRecScoreThresh: Float = 0
}

// MARK: - Runtime overrides (optional)

/// Optional overrides for DB postprocess and the post-recognition score filter.
struct OCRRuntimeParams: Equatable, Sendable {
    /// Binarization threshold on the detection probability map (`thresh`).
    var textDetThresh: Float?
    /// Minimum mean score inside a quad to keep a detection (`box_thresh`).
    var textDetBoxThresh: Float?
    /// Polygon expansion ratio (`unclip_ratio`).
    var textDetUnclipRatio: Float?
    /// End-to-end minimum line score; **`nil` uses `OCRDefaultThresholds.textRecScoreThresh`** (not read from the recognition model config file).
    var textRecScoreThresh: Float?

    static let noOverrides = OCRRuntimeParams(
        textDetThresh: nil,
        textDetBoxThresh: nil,
        textDetUnclipRatio: nil,
        textRecScoreThresh: nil
    )
}

// MARK: - Resolved values (effective thresholds)

struct ResolvedOCRRuntimeParams: Equatable, Sendable {
    var textDetThresh: Float
    var textDetBoxThresh: Float
    var textDetUnclipRatio: Float
    var textRecScoreThresh: Float

    init(textDetThresh: Float, textDetBoxThresh: Float, textDetUnclipRatio: Float, textRecScoreThresh: Float) {
        self.textDetThresh = textDetThresh
        self.textDetBoxThresh = textDetBoxThresh
        self.textDetUnclipRatio = textDetUnclipRatio
        self.textRecScoreThresh = textRecScoreThresh
    }

    /// Shown before models finish loading.
    static let fallbackForUI = ResolvedOCRRuntimeParams(
        textDetThresh: OCRDefaultThresholds.textDetThresh,
        textDetBoxThresh: OCRDefaultThresholds.textDetBoxThresh,
        textDetUnclipRatio: OCRDefaultThresholds.textDetUnclipRatio,
        textRecScoreThresh: OCRDefaultThresholds.textRecScoreThresh
    )
}

extension OCRRuntimeParams {
    func resolved(det: InferenceConfig) -> ResolvedOCRRuntimeParams {
        let base = ResolvedOCRRuntimeParams.fromDetectionModelConfig(det)
        return ResolvedOCRRuntimeParams(
            textDetThresh: textDetThresh ?? base.textDetThresh,
            textDetBoxThresh: textDetBoxThresh ?? base.textDetBoxThresh,
            textDetUnclipRatio: textDetUnclipRatio ?? base.textDetUnclipRatio,
            textRecScoreThresh: textRecScoreThresh ?? base.textRecScoreThresh
        )
    }
}

extension ResolvedOCRRuntimeParams {
    /// Baseline for sliders: detection fields from the detection model config file; line score from `OCRDefaultThresholds` only.
    static func fromDetectionModelConfig(_ det: InferenceConfig) -> ResolvedOCRRuntimeParams {
        let d = det.postProcess
        return ResolvedOCRRuntimeParams(
            textDetThresh: d.thresh ?? OCRDefaultThresholds.textDetThresh,
            textDetBoxThresh: d.boxThresh ?? OCRDefaultThresholds.textDetBoxThresh,
            textDetUnclipRatio: d.unclipRatio ?? OCRDefaultThresholds.textDetUnclipRatio,
            textRecScoreThresh: OCRDefaultThresholds.textRecScoreThresh
        )
    }
}
