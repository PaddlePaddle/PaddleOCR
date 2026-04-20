import { describe, expect, it, vi } from "vitest";

vi.mock("@techstark/opencv-js", () => ({
  default: {
    Mat() {}
  }
}));

import { PaddleOCR, normalizeOcrPipelineConfig, parseOcrPipelineConfigText } from "../src/index";

describe("public pipeline exports", () => {
  it("exports PaddleOCR", () => {
    expect(typeof PaddleOCR).toBe("function");
  });

  it("exports OCR pipeline config helpers", () => {
    expect(typeof parseOcrPipelineConfigText).toBe("function");
    expect(typeof normalizeOcrPipelineConfig).toBe("function");
  });

  it("supports worker mode through PaddleOCR.create()", async () => {
    const ocr = await PaddleOCR.create({
      worker: true,
      initialize: false
    });

    expect(typeof ocr.initialize).toBe("function");
    expect(typeof ocr.predict).toBe("function");
    expect(typeof ocr.dispose).toBe("function");
  });

  it("resolves ortOptions on the instance (create-time ORT settings)", async () => {
    const ocr = await PaddleOCR.create({
      lang: "ch",
      ocrVersion: "PP-OCRv5",
      initialize: false,
      ortOptions: { backend: "wasm", wasmPaths: "/dist/" }
    });

    expect(ocr.options.ortOptions.backend).toBe("wasm");
    expect(ocr.options.ortOptions.wasmPaths).toBe("/dist/");
  });
});
