import { describe, expect, it } from "vitest";

import {
  cloneDefaultOcrConfig,
  normalizeOrtOptions,
  resolvePaddleOCROptions,
  resolveWorkerOptions,
  validateLoadedModelName
} from "../src/pipelines/ocr/shared";

describe("OCR shared option resolution", () => {
  it("normalizes ORT options and reuses the same backend fallback", () => {
    const defaultOrt = normalizeOrtOptions();

    expect(defaultOrt).toMatchObject({
      backend: expect.any(String)
    });
    expect(normalizeOrtOptions({ backend: "invalid", proxy: true })).toEqual({
      backend: defaultOrt.backend,
      proxy: true
    });
    expect(
      normalizeOrtOptions({
        backend: "wasm",
        wasmPaths: "/wasm/",
        numThreads: 2,
        simd: true,
        proxy: false
      })
    ).toEqual({
      backend: "wasm",
      wasmPaths: "/wasm/",
      numThreads: 2,
      simd: true,
      proxy: false
    });
  });

  it("resolves worker options from booleans and custom factories", () => {
    const createWorker = () => ({});

    expect(resolveWorkerOptions(false)).toEqual({
      enabled: false,
      createWorker: null
    });
    expect(resolveWorkerOptions(true)).toEqual({
      enabled: true,
      createWorker: null
    });
    expect(resolveWorkerOptions({ createWorker })).toEqual({
      enabled: true,
      createWorker
    });
    expect(resolveWorkerOptions({})).toEqual({
      enabled: true,
      createWorker: null
    });
  });

  it("rejects unsupported worker option types", () => {
    expect(() => resolveWorkerOptions("yes")).toThrow(
      /worker must be a boolean or an options object/i
    );
  });

  it("returns ortOptions, assets, and model selection for explicit model names", () => {
    const options = resolvePaddleOCROptions({
      text_detection_model_name: "PP-OCRv5_mobile_det",
      text_recognition_model_name: "PP-OCRv5_mobile_rec",
      ortOptions: {
        backend: "webgpu",
        proxy: true
      }
    });

    expect(options.ortOptions).toEqual({
      backend: "webgpu",
      proxy: true
    });
    expect(options.pipelineConfig.assets.det?.url).toMatch(/PP-OCRv5_mobile_det.*\.tar$/);
    expect(options.pipelineConfig.assets.rec?.url).toMatch(/PP-OCRv5_mobile_rec.*\.tar$/);
    expect(options.pipelineConfig.modelSelection).toEqual({
      textDetectionModelName: "PP-OCRv5_mobile_det",
      textRecognitionModelName: "PP-OCRv5_mobile_rec"
    });
  });

  it("rejects incomplete pipeline model selection", () => {
    expect(() =>
      resolvePaddleOCROptions({
        pipelineConfig: {
          pipeline_name: "OCR",
          SubModules: {
            TextDetection: {
              model_name: "PP-OCRv5_mobile_det"
            }
          }
        }
      })
    ).toThrow(/must define both "SubModules.TextDetection" and "SubModules.TextRecognition"/i);
  });

  it("clones the default OCR config deeply", () => {
    const cloned = cloneDefaultOcrConfig();
    cloned.det.postprocess.thresh = 0.99;

    expect(cloned.det.postprocess.thresh).toBe(0.99);

    const freshClone = cloneDefaultOcrConfig();
    expect(freshClone.det.postprocess.thresh).not.toBe(0.99);
  });

  it("validates loaded model names against inference.yml (roles match pipeline initialize)", () => {
    expect(() =>
      validateLoadedModelName(
        "TextDetection",
        "PP-OCRv5_mobile_det",
        "Global:\n  model_name: PP-OCRv5_mobile_det"
      )
    ).not.toThrow();

    expect(() =>
      validateLoadedModelName(
        "TextDetection",
        "PP-OCRv5_mobile_det",
        "Global:\n  model_name: other"
      )
    ).toThrow(/requested model_name is "PP-OCRv5_mobile_det"/i);
  });
});
