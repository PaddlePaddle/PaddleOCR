import { describe, expect, it } from "vitest";

import { DEFAULT_DET_MODEL_CONFIG, DEFAULT_REC_MODEL_CONFIG } from "../src/models/index";
import type { OcrModelConfig } from "../src/pipelines/ocr/runtime-params";
import { getOcrRuntimeParams } from "../src/pipelines/ocr/runtime-params";

/** Fixture: full `OcrModelConfig` shapes (same as pipeline `getModelConfig()` after init). */
const SAMPLE_MODEL_CONFIG: OcrModelConfig = {
  det: {
    ...DEFAULT_DET_MODEL_CONFIG,
    resizeLong: 960,
    maxSideLimit: 3200
  },
  rec: {
    ...DEFAULT_REC_MODEL_CONFIG
  }
};

describe("OCR runtime params", () => {
  it("falls back from params to defaults and config values", () => {
    expect(
      getOcrRuntimeParams(
        SAMPLE_MODEL_CONFIG,
        {
          text_det_limit_side_len: 736,
          text_det_limit_type: "min",
          text_det_max_side_limit: 4096,
          text_det_thresh: 0.25,
          text_det_box_thresh: 0.55,
          text_det_unclip_ratio: 1.8,
          text_rec_score_thresh: 0.4
        },
        {}
      )
    ).toEqual({
      det: {
        limitSideLen: 736,
        limitType: "min",
        maxSideLimit: 4096,
        thresh: 0.25,
        boxThresh: 0.55,
        unclipRatio: 1.8
      },
      pipeline: { scoreThresh: 0.4 }
    });
  });

  it("prefers camelCase params; limit type falls back to model config when not specified", () => {
    expect(
      getOcrRuntimeParams(
        SAMPLE_MODEL_CONFIG,
        {},
        {
          textDetLimitSideLen: 512,
          textDetMaxSideLimit: 2048,
          textDetThresh: 0.22,
          textDetBoxThresh: 0.44,
          textDetUnclipRatio: 2.2,
          textRecScoreThresh: 0.9
        }
      )
    ).toEqual({
      det: {
        limitSideLen: 512,
        limitType: DEFAULT_DET_MODEL_CONFIG.limitType,
        maxSideLimit: 2048,
        thresh: 0.22,
        boxThresh: 0.44,
        unclipRatio: 2.2
      },
      pipeline: { scoreThresh: 0.9 }
    });
  });

  it("treats nulls as missing values when resolving fallbacks", () => {
    expect(
      getOcrRuntimeParams(
        SAMPLE_MODEL_CONFIG,
        {
          text_det_limit_type: null
        },
        {
          text_det_limit_side_len: null,
          text_det_max_side_limit: null,
          text_det_thresh: null,
          text_det_box_thresh: null,
          text_det_unclip_ratio: null,
          text_rec_score_thresh: null
        }
      )
    ).toEqual({
      det: {
        limitSideLen: SAMPLE_MODEL_CONFIG.det.resizeLong,
        limitType: SAMPLE_MODEL_CONFIG.det.limitType,
        maxSideLimit: SAMPLE_MODEL_CONFIG.det.maxSideLimit,
        thresh: SAMPLE_MODEL_CONFIG.det.postprocess.thresh,
        boxThresh: SAMPLE_MODEL_CONFIG.det.postprocess.boxThresh,
        unclipRatio: SAMPLE_MODEL_CONFIG.det.postprocess.unclipRatio
      },
      pipeline: { scoreThresh: 0 }
    });
  });
});
