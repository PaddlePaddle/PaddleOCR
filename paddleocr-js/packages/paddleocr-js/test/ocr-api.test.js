import { describe, expect, it, vi } from "vitest";

vi.mock("@techstark/opencv-js", () => ({
  default: {
    Mat() {}
  }
}));

import { PaddleOCR, normalizeOcrPipelineConfig, parseOcrPipelineConfigText } from "../src/index.js";
import { extractInferenceModelName } from "../src/models/common.js";
import { DEFAULT_OCR_PIPELINE_CONFIG_TEXT } from "../src/pipelines/ocr/default-config.js";
import { normalizeRuntimeOptions } from "../src/pipelines/ocr/shared.js";
import { getOcrRuntimeParams } from "../src/pipelines/ocr/runtime-params.js";
import { DEFAULT_MODEL_ASSETS } from "../src/resources/registry.js";

const CREATE_WITHOUT_INIT = Object.freeze({
  initialize: false
});

const IGNORE_UNSUPPORTED = Object.freeze({
  initialize: false,
  unsupportedBehavior: "ignore"
});

const DEFAULT_DET_ASSET_ID = DEFAULT_MODEL_ASSETS["PP-OCRv5_mobile_det"].id;
const DEFAULT_REC_ASSET_ID = DEFAULT_MODEL_ASSETS["PP-OCRv5_mobile_rec"].id;

const pipelineConfigText = `
pipeline_name: OCR
use_doc_preprocessor: true
use_textline_orientation: true
SubModules:
  TextDetection:
    model_name: PP-OCRv5_mobile_det
    limit_side_len: 64
    limit_type: min
    max_side_limit: 4000
    thresh: 0.3
    box_thresh: 0.6
    unclip_ratio: 1.5
  TextLineOrientation:
    model_name: PP-LCNet_x1_0_textline_ori
  TextRecognition:
    model_name: PP-OCRv5_mobile_rec
    score_thresh: 0.0
`;

const customDetAsset = {
  id: "custom-det",
  kind: "tar",
  url: "https://example.com/custom-det.tar",
  entries: {
    model: "inference.onnx",
    config: "inference.yml"
  }
};

const customRecAsset = {
  id: "custom-rec",
  kind: "tar",
  url: "https://example.com/custom-rec.tar",
  entries: {
    model: "inference.onnx",
    config: "inference.yml"
  }
};

class MockWorker {
  constructor(responder) {
    this.responder = responder;
    this.messages = [];
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
  }

  postMessage(message) {
    this.messages.push(message);
    queueMicrotask(() => {
      if (this.terminated) return;
      const response = this.responder(message);
      if (!response) return;
      this.onmessage?.({ data: response });
    });
  }

  terminate() {
    this.terminated = true;
  }
}

function expectDefaultModelAssets(ocr) {
  expect(ocr.options.assets.det.id).toBe(DEFAULT_DET_ASSET_ID);
  expect(ocr.options.assets.rec.id).toBe(DEFAULT_REC_ASSET_ID);
}

describe("PaddleOCR high-level API", () => {
  it("parses and normalizes OCR pipeline configs", () => {
    const parsed = parseOcrPipelineConfigText(pipelineConfigText);
    const normalized = normalizeOcrPipelineConfig(parsed);

    expect(normalized.pipelineName).toBe("OCR");
    expect(normalized.modelSelection.textDetectionModelName).toBe("PP-OCRv5_mobile_det");
    expect(normalized.modelSelection.textRecognitionModelName).toBe("PP-OCRv5_mobile_rec");
    expect(normalized.modelSelection).not.toHaveProperty("detAsset");
    expect(normalized.modelSelection).not.toHaveProperty("recAsset");
    expect(normalized.runtimeDefaults.text_det_limit_side_len).toBe(64);
    expect(normalized.runtimeDefaults.text_det_limit_type).toBe("min");
    expect(normalized.warnings).toHaveLength(2);
  });

  it("keeps pipeline-declared custom assets separate from model selection", () => {
    const normalized = normalizeOcrPipelineConfig({
      pipeline_name: "OCR",
      SubModules: {
        TextDetection: {
          model_name: "custom_det",
          model_dir: customDetAsset
        },
        TextRecognition: {
          model_name: "custom_rec",
          model_dir: customRecAsset
        }
      }
    });

    expect(normalized.modelSelection.textDetectionModelName).toBe("custom_det");
    expect(normalized.modelSelection.textRecognitionModelName).toBe("custom_rec");
    expect(normalized.assets.det.id).toBe("custom-det");
    expect(normalized.assets.rec.id).toBe("custom-rec");
  });

  it("creates an OCR instance from lang and ocrVersion", async () => {
    const ocr = await PaddleOCR.create({
      lang: "ch",
      ocrVersion: "PP-OCRv5",
      ...CREATE_WITHOUT_INIT
    });

    expect(ocr).toBeInstanceOf(PaddleOCR);
    expectDefaultModelAssets(ocr);
  });

  it("keeps the same create API when worker mode is enabled", async () => {
    const defaultRuntime = normalizeRuntimeOptions();
    const ocr = await PaddleOCR.create({
      lang: "ch",
      ocrVersion: "PP-OCRv5",
      worker: true,
      ...CREATE_WITHOUT_INIT
    });

    expect(typeof ocr.initialize).toBe("function");
    expect(typeof ocr.predict).toBe("function");
    expect(typeof ocr.dispose).toBe("function");
    expect(ocr.options.assets.det.id).toBe(DEFAULT_DET_ASSET_ID);
    expect(ocr.options.runtime.backend).toBe(defaultRuntime.backend);
  });

  it("uses the package OCR.yaml defaults when no pipeline config is passed", async () => {
    const defaultPipeline = normalizeOcrPipelineConfig(DEFAULT_OCR_PIPELINE_CONFIG_TEXT);
    const ocr = await PaddleOCR.create(CREATE_WITHOUT_INIT);

    expect(ocr.options.assets.det.id).toBe(
      DEFAULT_MODEL_ASSETS[defaultPipeline.modelSelection.textDetectionModelName].id
    );
    expect(ocr.options.assets.rec.id).toBe(
      DEFAULT_MODEL_ASSETS[defaultPipeline.modelSelection.textRecognitionModelName].id
    );
    expect(ocr.runtimeDefaults).toMatchObject(defaultPipeline.runtimeDefaults);
  });

  it("maps English PP-OCRv5 selection to the mobile model set", async () => {
    const ocr = await PaddleOCR.create({
      lang: "en",
      ocrVersion: "PP-OCRv5",
      ...CREATE_WITHOUT_INIT
    });

    expectDefaultModelAssets(ocr);
  });

  it("allows overriding model selection via model_name options", async () => {
    const ocr = await PaddleOCR.create({
      text_detection_model_name: "PP-OCRv5_mobile_det",
      text_recognition_model_name: "PP-OCRv5_mobile_rec",
      ...CREATE_WITHOUT_INIT
    });

    expectDefaultModelAssets(ocr);
  });

  it("creates an OCR instance from pipeline config model names", async () => {
    const ocr = await PaddleOCR.fromPipelineConfig(pipelineConfigText, IGNORE_UNSUPPORTED);

    expectDefaultModelAssets(ocr);
    expect(ocr.runtimeDefaults.text_det_limit_type).toBe("min");
    expect(ocr.runtimeDefaults.text_rec_score_thresh).toBe(0);
  });

  it("lets explicit model assets override pipeline config model names", async () => {
    const ocr = await PaddleOCR.fromPipelineConfig(pipelineConfigText, {
      text_detection_model_name: "custom_det",
      textDetectionModelAsset: customDetAsset,
      text_recognition_model_name: "custom_rec",
      textRecognitionModelAsset: customRecAsset,
      ...IGNORE_UNSUPPORTED
    });

    expect(ocr.options.assets.det.id).toBe("custom-det");
    expect(ocr.options.assets.rec.id).toBe("custom-rec");
  });

  it("allows overriding only one side with a custom model asset", async () => {
    const ocr = await PaddleOCR.fromPipelineConfig(pipelineConfigText, {
      text_detection_model_name: "custom_det",
      textDetectionModelAsset: customDetAsset,
      ...IGNORE_UNSUPPORTED
    });

    expect(ocr.options.assets.det.id).toBe("custom-det");
    expect(ocr.options.assets.rec.id).toBe(DEFAULT_REC_ASSET_ID);
  });

  it("lets explicit model names override pipeline config model assets", async () => {
    const ocr = await PaddleOCR.fromPipelineConfig(
      {
        pipeline_name: "OCR",
        SubModules: {
          TextDetection: {
            model_name: "custom_det",
            model_dir: customDetAsset
          },
          TextRecognition: {
            model_name: "custom_rec",
            model_dir: customRecAsset
          }
        }
      },
      {
        text_detection_model_name: "PP-OCRv5_mobile_det",
        text_recognition_model_name: "PP-OCRv5_mobile_rec",
        ...CREATE_WITHOUT_INIT
      }
    );

    expectDefaultModelAssets(ocr);
  });

  it("rejects unsupported lang/ocrVersion combinations", async () => {
    await expect(
      PaddleOCR.create({
        lang: "kl",
        ocrVersion: "PP-OCRv5",
        ...CREATE_WITHOUT_INIT
      })
    ).rejects.toThrow(/Unsupported lang\/ocrVersion combination/i);
  });

  it("warns about unsupported pipeline features by default", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    await PaddleOCR.fromPipelineConfig(pipelineConfigText, { ...CREATE_WITHOUT_INIT });

    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("can turn unsupported pipeline warnings into errors", async () => {
    await expect(
      PaddleOCR.fromPipelineConfig(pipelineConfigText, {
        ...CREATE_WITHOUT_INIT,
        unsupportedBehavior: "error"
      })
    ).rejects.toThrow(/not yet supported/i);
  });

  it("parses the generated default OCR pipeline config text", () => {
    const parsed = parseOcrPipelineConfigText(DEFAULT_OCR_PIPELINE_CONFIG_TEXT);
    const normalized = normalizeOcrPipelineConfig(DEFAULT_OCR_PIPELINE_CONFIG_TEXT);

    expect(normalized).toMatchObject(normalizeOcrPipelineConfig(parsed));
  });

  it("rejects custom model_dir without matching model_name", async () => {
    await expect(
      PaddleOCR.create({
        textDetectionModelAsset: customDetAsset,
        ...CREATE_WITHOUT_INIT
      })
    ).rejects.toThrow(/text_detection_model_dir requires text_detection_model_name/i);
  });

  it("rejects conflicting model asset aliases", async () => {
    await expect(
      PaddleOCR.create({
        text_detection_model_name: "custom_det",
        textDetectionModelAsset: customDetAsset,
        text_detection_model_dir: customRecAsset,
        ...CREATE_WITHOUT_INIT
      })
    ).rejects.toThrow(/Conflicting values provided for text detection model asset/i);
  });

  it("ignores unsupported asset metadata before inference.yml is loaded", async () => {
    const ocr = await PaddleOCR.create({
      text_detection_model_name: "custom_det",
      textDetectionModelAsset: {
        ...customDetAsset,
        model_name: "other_det"
      },
      text_recognition_model_name: "custom_rec",
      textRecognitionModelAsset: customRecAsset,
      ...CREATE_WITHOUT_INIT
    });

    expect(ocr.options.assets.det.id).toBe("custom-det");
  });

  it("initializes worker mode through the same API surface", async () => {
    const worker = new MockWorker((message) => {
      if (message.type === "init") {
        return {
          kind: "worker-transport-response",
          status: "success",
          requestId: message.requestId,
          payload: {
            summary: {
              backend: "wasm",
              webgpuAvailable: false,
              detProvider: "wasm",
              recProvider: "wasm",
              assets: [],
              elapsedMs: 12,
              cacheHits: 0,
              cacheMisses: 2,
              pipelineConfigWarnings: []
            },
            modelConfig: {
              det: { resizeLong: 960 },
              rec: { imageShape: [3, 48, 320] }
            }
          }
        };
      }
      if (message.type === "dispose") {
        return {
          kind: "worker-transport-response",
          status: "success",
          requestId: message.requestId,
          payload: {}
        };
      }
      return null;
    });
    const ocr = await PaddleOCR.create({
      worker: {
        createWorker: () => worker
      },
      runtime: {
        backend: "wasm",
        proxy: true
      },
      ...CREATE_WITHOUT_INIT
    });

    const summary = await ocr.initialize();

    expect(summary.backend).toBe("wasm");
    expect(worker.messages[0].type).toBe("init");
    expect(worker.messages[0].payload.options.runtime.backend).toBe("wasm");
    expect(worker.messages[0].payload.options.runtime.disableWasmProxy).toBe(true);
    expect(ocr.getModelConfig().det.resizeLong).toBe(960);

    await ocr.dispose();
    expect(worker.messages[1].type).toBe("dispose");
    expect(worker.terminated).toBe(true);
  });

  it("surfaces worker initialization failures", async () => {
    const worker = new MockWorker((message) => {
      if (message.type === "init") {
        return {
          kind: "worker-transport-response",
          status: "error",
          requestId: message.requestId,
          error: {
            name: "Error",
            message: "worker init failed"
          }
        };
      }
      return null;
    });
    const ocr = await PaddleOCR.create({
      worker: {
        createWorker: () => worker
      },
      ...CREATE_WITHOUT_INIT
    });

    await expect(ocr.initialize()).rejects.toThrow(/worker init failed/i);
    expect(worker.terminated).toBe(true);
  });
});

describe("OCR runtime parameter normalization", () => {
  it("accepts camelCase aliases while preserving PaddleOCR names", () => {
    const params = getOcrRuntimeParams(
      {
        det: {
          resizeLong: 960,
          maxSideLimit: 4000,
          postprocess: {
            thresh: 0.3,
            boxThresh: 0.6,
            unclipRatio: 1.5
          }
        },
        rec: {
          scoreThresh: 0.1
        }
      },
      {
        text_det_limit_type: "min"
      },
      {
        textDetThresh: 0.4,
        textDetBoxThresh: 0.7,
        textDetUnclipRatio: 2,
        textRecScoreThresh: 0.2
      }
    );

    expect(params.text_det_limit_type).toBe("min");
    expect(params.text_det_thresh).toBe(0.4);
    expect(params.text_det_box_thresh).toBe(0.7);
    expect(params.text_det_unclip_ratio).toBe(2);
    expect(params.text_rec_score_thresh).toBe(0.2);
  });
});

describe("inference.yml model_name extraction", () => {
  it("prefers Global.model_name when present", () => {
    expect(
      extractInferenceModelName(`
Global:
  model_name: custom_det
PostProcess:
  name: DBPostProcess
`)
    ).toBe("custom_det");
  });

  it("falls back to nested model_name fields", () => {
    expect(
      extractInferenceModelName(`
Deploy:
  metadata:
    model_name: custom_rec
`)
    ).toBe("custom_rec");
  });

  it("returns null when inference.yml has no model_name", () => {
    expect(
      extractInferenceModelName(`
PostProcess:
  name: DBPostProcess
`)
    ).toBeNull();
  });
});
