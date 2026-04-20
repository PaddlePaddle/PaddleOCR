/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DetModelConfig, DetRuntimeOverrides, LimitType } from "../../models/det";
import type { RecModelConfig } from "../../models/rec";

export type { LimitType };

export interface OcrModelConfig {
  det: DetModelConfig;
  rec: RecModelConfig;
}

export interface ResolvedOcrParams {
  det: DetRuntimeOverrides;
  pipeline: { scoreThresh: number };
}

export interface OcrRuntimeParamsInput {
  text_det_limit_side_len?: number;
  textDetLimitSideLen?: number;
  text_det_limit_type?: LimitType;
  textDetLimitType?: LimitType;
  text_det_max_side_limit?: number;
  textDetMaxSideLimit?: number;
  text_det_thresh?: number;
  textDetThresh?: number;
  text_det_box_thresh?: number;
  textDetBoxThresh?: number;
  text_det_unclip_ratio?: number;
  textDetUnclipRatio?: number;
  text_rec_score_thresh?: number;
  textRecScoreThresh?: number;
}

function firstDefined<T>(...values: Array<T | undefined | null>): T | undefined {
  for (const value of values) {
    if (value !== undefined && value !== null) {
      return value;
    }
  }
  return undefined;
}

function toNumberOrUndefined(value: unknown): number | undefined {
  if (value === undefined || value === null) return undefined;
  const num = Number(value);
  return Number.isFinite(num) ? num : undefined;
}

export function getOcrRuntimeParams(
  config: OcrModelConfig,
  defaults: Partial<OcrRuntimeParamsInput> = {},
  params: OcrRuntimeParamsInput = {}
): ResolvedOcrParams {
  return {
    det: {
      limitSideLen: toNumberOrUndefined(
        firstDefined(
          params.text_det_limit_side_len,
          params.textDetLimitSideLen,
          defaults.text_det_limit_side_len,
          defaults.textDetLimitSideLen,
          config.det.resizeLong
        )
      ),
      limitType: firstDefined(
        params.text_det_limit_type,
        params.textDetLimitType,
        defaults.text_det_limit_type,
        defaults.textDetLimitType,
        config.det.limitType
      ),
      maxSideLimit: toNumberOrUndefined(
        firstDefined(
          params.text_det_max_side_limit,
          params.textDetMaxSideLimit,
          defaults.text_det_max_side_limit,
          defaults.textDetMaxSideLimit,
          config.det.maxSideLimit
        )
      ),
      thresh: toNumberOrUndefined(
        firstDefined(
          params.text_det_thresh,
          params.textDetThresh,
          defaults.text_det_thresh,
          defaults.textDetThresh,
          config.det.postprocess.thresh
        )
      ),
      boxThresh: toNumberOrUndefined(
        firstDefined(
          params.text_det_box_thresh,
          params.textDetBoxThresh,
          defaults.text_det_box_thresh,
          defaults.textDetBoxThresh,
          config.det.postprocess.boxThresh
        )
      ),
      unclipRatio: toNumberOrUndefined(
        firstDefined(
          params.text_det_unclip_ratio,
          params.textDetUnclipRatio,
          defaults.text_det_unclip_ratio,
          defaults.textDetUnclipRatio,
          config.det.postprocess.unclipRatio
        )
      )
    },
    pipeline: {
      scoreThresh: Number(
        firstDefined(
          params.text_rec_score_thresh,
          params.textRecScoreThresh,
          defaults.text_rec_score_thresh,
          defaults.textRecScoreThresh,
          0
        )
      )
    }
  };
}
