import { describe, expect, it, vi } from "vitest";

vi.mock("@techstark/opencv-js", () => ({
  default: {
    Mat() {}
  }
}));

import * as publicPipelines from "../src/pipelines/index";
import * as ocrPipeline from "../src/pipelines/ocr/index";

describe("pipeline entrypoints", () => {
  it("re-exports OCR pipeline APIs from the pipelines index", () => {
    expect(publicPipelines.PaddleOCR).toBe(ocrPipeline.PaddleOCR);
    expect(publicPipelines.parseOcrPipelineConfigText).toBe(ocrPipeline.parseOcrPipelineConfigText);
    expect(publicPipelines.normalizeOcrPipelineConfig).toBe(ocrPipeline.normalizeOcrPipelineConfig);
  });

  it("exposes only the barrel surface re-exported from pipelines/ocr (no drift)", () => {
    expect(Object.keys(publicPipelines).sort()).toEqual(
      ["normalizeOcrPipelineConfig", "parseOcrPipelineConfigText", "PaddleOCR"].sort()
    );
  });
});
