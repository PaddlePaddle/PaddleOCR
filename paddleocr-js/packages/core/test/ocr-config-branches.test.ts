import { describe, expect, it } from "vitest";

import {
  normalizeOcrPipelineConfig,
  parseOcrPipelineConfigText
} from "../src/pipelines/ocr/config";

describe("OCR pipeline config branches", () => {
  it("rejects non-object YAML payloads and invalid input types", () => {
    expect(() => parseOcrPipelineConfigText("- item")).toThrow(/must decode to an object/i);
    expect(() => normalizeOcrPipelineConfig(123)).toThrow(/must be an object or YAML text/i);
  });

  it("rejects unsupported pipeline names", () => {
    expect(() =>
      normalizeOcrPipelineConfig({
        pipeline_name: "DET_ONLY",
        SubModules: {
          TextDetection: { model_name: "PP-OCRv5_mobile_det" },
          TextRecognition: { model_name: "PP-OCRv5_mobile_rec" }
        }
      })
    ).toThrow(/Unsupported pipeline_name/i);
  });

  it("accepts default pipeline names and suppresses general text_type warnings", () => {
    const normalized = normalizeOcrPipelineConfig({
      SubModules: {
        TextDetection: { model_name: "PP-OCRv5_mobile_det" },
        TextRecognition: { model_name: "PP-OCRv5_mobile_rec" }
      },
      text_type: "general"
    });

    expect(normalized.pipelineName).toBe("OCR");
    expect(normalized.warnings).toEqual([]);
    expect(normalized.unsupportedFeatures).toEqual([]);
  });

  it("supports object model_dir assets and warns on unsupported text_type values", () => {
    const normalized = normalizeOcrPipelineConfig({
      SubModules: {
        TextDetection: {
          model_name: "custom_det",
          model_dir: {
            url: "/det.tar"
          },
          limit_side_len: "",
          max_side_limit: null
        },
        TextRecognition: {
          model_name: "custom_rec",
          score_thresh: "0.7"
        }
      },
      text_type: "seal"
    });

    expect(normalized.assets.det).toMatchObject({
      url: "/det.tar"
    });
    expect(normalized.runtimeDefaults.text_det_limit_side_len).toBeUndefined();
    expect(normalized.runtimeDefaults.text_det_max_side_limit).toBeUndefined();
    expect(normalized.runtimeDefaults.text_rec_score_thresh).toBe(0.7);
    expect(normalized.warnings).toContain('text_type "seal" is not used by PaddleOCR.js yet.');
  });

  it("rejects invalid model_dir values and missing model names", () => {
    expect(() =>
      normalizeOcrPipelineConfig({
        SubModules: {
          TextDetection: {
            model_dir: {
              url: "/det.tar"
            }
          },
          TextRecognition: {
            model_name: "PP-OCRv5_mobile_rec"
          }
        }
      })
    ).toThrow(/model_name must be provided/i);

    expect(() =>
      normalizeOcrPipelineConfig({
        SubModules: {
          TextDetection: {
            model_name: "PP-OCRv5_mobile_det",
            model_dir: "/det"
          },
          TextRecognition: {
            model_name: "PP-OCRv5_mobile_rec"
          }
        }
      })
    ).toThrow(/model_dir must be null or an asset descriptor object/i);
  });
});
