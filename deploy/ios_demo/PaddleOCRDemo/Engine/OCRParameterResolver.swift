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

// MARK: - Runtime (UI) parameters

struct OCRRuntimeParams: Equatable, Sendable {
    // Detection — preprocess
    var textDetLimitSideLen: Int?
    var textDetLimitType: String?
    var textDetMaxSideLimit: Int?
    // Detection — postprocess
    var textDetThresh: Float?
    var textDetBoxThresh: Float?
    var textDetUnclipRatio: Float?
    // Recognition
    var textRecBatchSize: Int?
    var textRecScoreThresh: Float?

    static let noOverrides = OCRRuntimeParams(
        textDetLimitSideLen: nil,
        textDetLimitType: nil,
        textDetMaxSideLimit: nil,
        textDetThresh: nil,
        textDetBoxThresh: nil,
        textDetUnclipRatio: nil,
        textRecBatchSize: nil,
        textRecScoreThresh: nil
    )
}

// MARK: - Resolved parameters (for UI baseline & filtering)

struct ResolvedOCRRuntimeParams: Equatable, Sendable {
    // Detection — preprocess
    var textDetLimitSideLen: Int
    var textDetLimitType: String
    var textDetMaxSideLimit: Int
    // Detection — postprocess
    var textDetThresh: Float
    var textDetBoxThresh: Float
    var textDetUnclipRatio: Float
    // Recognition
    var textRecBatchSize: Int
    var textRecScoreThresh: Float

    init(
        textDetLimitSideLen: Int,
        textDetLimitType: String,
        textDetMaxSideLimit: Int,
        textDetThresh: Float,
        textDetBoxThresh: Float,
        textDetUnclipRatio: Float,
        textRecBatchSize: Int,
        textRecScoreThresh: Float,
    ) {
        self.textDetLimitSideLen = textDetLimitSideLen
        self.textDetLimitType = textDetLimitType
        self.textDetMaxSideLimit = textDetMaxSideLimit
        self.textDetThresh = textDetThresh
        self.textDetBoxThresh = textDetBoxThresh
        self.textDetUnclipRatio = textDetUnclipRatio
        self.textRecBatchSize = textRecBatchSize
        self.textRecScoreThresh = textRecScoreThresh
    }

    static let fallbackForUI = ResolvedOCRRuntimeParams(
        textDetLimitSideLen: OCRParameterResolver.textDetLimitSideLenAppDefault,
        textDetLimitType: OCRParameterResolver.textDetLimitTypeAppDefault,
        textDetMaxSideLimit: OCRParameterResolver.textDetMaxSideLimitAppDefault,
        textDetThresh: OCRParameterResolver.textDetThreshAppDefault,
        textDetBoxThresh: OCRParameterResolver.textDetBoxThreshAppDefault,
        textDetUnclipRatio: OCRParameterResolver.textDetUnclipRatioAppDefault,
        textRecBatchSize: OCRParameterResolver.textRecBatchSizeAppDefault,
        textRecScoreThresh: OCRParameterResolver.textRecScoreThreshAppDefault
    )
}

extension OCRRuntimeParams {
    func resolved(_ modelConfig: InferenceConfig) -> ResolvedOCRRuntimeParams {
        let base = ResolvedOCRRuntimeParams.fromModelConfig(modelConfig)
        return ResolvedOCRRuntimeParams(
            textDetLimitSideLen: textDetLimitSideLen ?? base.textDetLimitSideLen,
            textDetLimitType: textDetLimitType ?? base.textDetLimitType,
            textDetMaxSideLimit: textDetMaxSideLimit ?? base.textDetMaxSideLimit,
            textDetThresh: textDetThresh ?? base.textDetThresh,
            textDetBoxThresh: textDetBoxThresh ?? base.textDetBoxThresh,
            textDetUnclipRatio: textDetUnclipRatio ?? base.textDetUnclipRatio,
            textRecBatchSize: textRecBatchSize ?? base.textRecBatchSize,
            textRecScoreThresh: textRecScoreThresh ?? base.textRecScoreThresh
        )
    }
}

extension ResolvedOCRRuntimeParams {
    /// Baseline with no UI overrides, using ``OCRParameterResolver`` app-tier defaults.
    static func fromModelConfig(_ modelConfig: InferenceConfig) -> ResolvedOCRRuntimeParams {
        let o = OCRParameterResolver.effective(modelConfig: modelConfig, runtime: .noOverrides)
        return ResolvedOCRRuntimeParams(
            textDetLimitSideLen: o.mergedLimitSideLen,
            textDetLimitType: o.mergedLimitType,
            textDetMaxSideLimit: o.mergedMaxSideLimit,
            textDetThresh: o.detThresh,
            textDetBoxThresh: o.detBoxThresh,
            textDetUnclipRatio: o.detUnclipRatio,
            textRecBatchSize: o.textRecBatchSize,
            textRecScoreThresh: o.recScoreThresh
        )
    }
}

// MARK: - Effective parameters (single merge pass)

struct EffectiveOCRParams: Equatable, Sendable {
    // Detection — preprocess
    var detResize: DetResizeParams
    var mergedLimitSideLen: Int
    var mergedLimitType: String
    var mergedMaxSideLimit: Int
    // Detection — postprocess
    var detThresh: Float
    var detBoxThresh: Float
    var detUnclipRatio: Float
    // Recognition
    var textRecBatchSize: Int
    var recScoreThresh: Float
}

// MARK: - Resolver

enum OCRParameterResolver {

    // MARK: App tier

    // Detection — preprocess
    static let textDetLimitSideLenAppDefault: Int = 64
    static let textDetLimitTypeAppDefault: String = "min"
    static let textDetMaxSideLimitAppDefault: Int = 4000
    // Detection — postprocess
    static let textDetThreshAppDefault: Float = 0.3
    static let textDetBoxThreshAppDefault: Float = 0.6
    static let textDetUnclipRatioAppDefault: Float = 1.5
    // Recognition
    static let textRecBatchSizeAppDefault: Int = 1
    static let textRecScoreThreshAppDefault: Float = 0.0

    // MARK: Merge helpers

    /// UI override if set; otherwise the app-tier default.
    private static func mergeFloat(ui: Float?, app: Float) -> Float {
        ui ?? app
    }

    private static func mergeInt(ui: Int?, app: Int) -> Int {
        ui ?? app
    }

    private static func mergeString(ui: String?, app: String) -> String {
        ui ?? app
    }

    private static func mergedLimitSideFields(runtime: OCRRuntimeParams) -> (
        sideLen: Int, limitType: String, maxSide: Int
    ) {
        let sideLen = mergeInt(
            ui: runtime.textDetLimitSideLen,
            app: Self.textDetLimitSideLenAppDefault
        )
        let limitType = mergeString(
            ui: runtime.textDetLimitType,
            app: Self.textDetLimitTypeAppDefault
        )
        let maxSide = mergeInt(
            ui: runtime.textDetMaxSideLimit,
            app: Self.textDetMaxSideLimitAppDefault
        )
        return (sideLen, limitType, maxSide)
    }

    private static func limitSideParams(runtime: OCRRuntimeParams) -> DetResizeParams {
        let m = mergedLimitSideFields(runtime: runtime)
        return DetResizeParams(
            limitSideLen: m.sideLen,
            limitType: m.limitType,
            maxSideLimit: m.maxSide
        )
    }

    /// Merges ``OCRRuntimeParams`` with app-tier defaults (UI → app). The `modelConfig` argument is
    /// unused at this moment.
    static func effective(modelConfig _: InferenceConfig, runtime: OCRRuntimeParams) -> EffectiveOCRParams {
        let detResize = limitSideParams(runtime: runtime)
        let merged = mergedLimitSideFields(runtime: runtime)

        let detThresh = mergeFloat(
            ui: runtime.textDetThresh,
            app: Self.textDetThreshAppDefault
        )
        let detBoxThresh = mergeFloat(
            ui: runtime.textDetBoxThresh,
            app: Self.textDetBoxThreshAppDefault
        )
        let detUnclipRatio = mergeFloat(
            ui: runtime.textDetUnclipRatio,
            app: Self.textDetUnclipRatioAppDefault
        )

        let textRecBatchSize = mergeInt(
            ui: runtime.textRecBatchSize,
            app: Self.textRecBatchSizeAppDefault
        )
        let recScoreThresh = runtime.textRecScoreThresh ?? Self.textRecScoreThreshAppDefault

        return EffectiveOCRParams(
            detResize: detResize,
            mergedLimitSideLen: merged.sideLen,
            mergedLimitType: merged.limitType,
            mergedMaxSideLimit: merged.maxSide,
            detThresh: detThresh,
            detBoxThresh: detBoxThresh,
            detUnclipRatio: detUnclipRatio,
            textRecBatchSize: textRecBatchSize,
            recScoreThresh: recScoreThresh
        )
    }
}
