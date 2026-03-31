import type { DetModelConfig } from "../../models/det";
import type { RecModelConfig } from "../../models/rec";

export interface OcrModelConfig {
  det: DetModelConfig;
  rec: RecModelConfig;
}

export type LimitType = "min" | "max";

export interface OcrRuntimeParams {
  text_det_limit_side_len: number;
  text_det_limit_type: LimitType;
  text_det_max_side_limit: number;
  text_det_thresh: number;
  text_det_box_thresh: number;
  text_det_unclip_ratio: number;
  text_rec_score_thresh: number;
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

export function getOcrRuntimeParams(
  config: OcrModelConfig,
  defaults: Partial<OcrRuntimeParams> = {},
  params: OcrRuntimeParamsInput = {},
): OcrRuntimeParams {
  return {
    text_det_limit_side_len: Number(
      firstDefined(
        params.text_det_limit_side_len,
        params.textDetLimitSideLen,
        defaults.text_det_limit_side_len,
        config.det.resizeLong,
      ),
    ),
    text_det_limit_type:
      firstDefined(
        params.text_det_limit_type,
        params.textDetLimitType,
        defaults.text_det_limit_type,
      ) ?? "max",
    text_det_max_side_limit: Number(
      firstDefined(
        params.text_det_max_side_limit,
        params.textDetMaxSideLimit,
        defaults.text_det_max_side_limit,
        config.det.maxSideLimit,
      ),
    ),
    text_det_thresh: Number(
      firstDefined(
        params.text_det_thresh,
        params.textDetThresh,
        defaults.text_det_thresh,
        config.det.postprocess.thresh,
      ),
    ),
    text_det_box_thresh: Number(
      firstDefined(
        params.text_det_box_thresh,
        params.textDetBoxThresh,
        defaults.text_det_box_thresh,
        config.det.postprocess.boxThresh,
      ),
    ),
    text_det_unclip_ratio: Number(
      firstDefined(
        params.text_det_unclip_ratio,
        params.textDetUnclipRatio,
        defaults.text_det_unclip_ratio,
        config.det.postprocess.unclipRatio,
      ),
    ),
    text_rec_score_thresh: Number(
      firstDefined(
        params.text_rec_score_thresh,
        params.textRecScoreThresh,
        defaults.text_rec_score_thresh,
        config.rec.scoreThresh,
      ),
    ),
  };
}
