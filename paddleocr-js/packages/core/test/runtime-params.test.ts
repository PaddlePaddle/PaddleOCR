import { describe, expect, it } from "vitest";

import { getOcrRuntimeParams } from "../src/pipelines/ocr/runtime-params";

/** Fixture: det/rec model config shapes used only by these tests. */
const SAMPLE_MODEL_CONFIG = {
  det: {
    resizeLong: 960,
    maxSideLimit: 3200,
    postprocess: {
      thresh: 0.3,
      boxThresh: 0.6,
      unclipRatio: 1.5
    }
  },
  rec: {
    scoreThresh: 0.2
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
      text_det_limit_side_len: 736,
      text_det_limit_type: "min",
      text_det_max_side_limit: 4096,
      text_det_thresh: 0.25,
      text_det_box_thresh: 0.55,
      text_det_unclip_ratio: 1.8,
      text_rec_score_thresh: 0.4
    });
  });

  it("prefers camelCase params and falls back to max when limit type is missing", () => {
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
      text_det_limit_side_len: 512,
      text_det_limit_type: "max",
      text_det_max_side_limit: 2048,
      text_det_thresh: 0.22,
      text_det_box_thresh: 0.44,
      text_det_unclip_ratio: 2.2,
      text_rec_score_thresh: 0.9
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
      text_det_limit_side_len: 960,
      text_det_limit_type: "max",
      text_det_max_side_limit: 3200,
      text_det_thresh: 0.3,
      text_det_box_thresh: 0.6,
      text_det_unclip_ratio: 1.5,
      text_rec_score_thresh: 0.2
    });
  });
});
