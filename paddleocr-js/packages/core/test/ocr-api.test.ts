import { describe, expect, it, vi } from "vitest";

vi.mock("@techstark/opencv-js", () => ({
  default: {
    Mat() {}
  }
}));

import { PaddleOCR, normalizeOcrPipelineConfig, parseOcrPipelineConfigText } from "../src/index";
import { extractInferenceModelName } from "../src/models/common";
import { DEFAULT_OCR_PIPELINE_CONFIG_TEXT } from "../src/pipelines/ocr/default-config";
import { normalizeOrtOptions } from "../src/pipelines/ocr/shared";
import { getOcrRuntimeParams } from "../src/pipelines/ocr/runtime-params";

const CREATE_WITHOUT_INIT = Object.freeze({
  initialize: false
});

const IGNORE_UNSUPPORTED = Object.freeze({
  initialize: false,
  unsupportedBehavior: "ignore"
});

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
  url: "https://example.com/custom-det.tar"
};

const customRecAsset = {
  url: "https://example.com/custom-rec.tar"
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
  expect(ocr.options.pipelineConfig.assets.det?.url).toMatch(/PP-OCRv5_mobile_det/);
  expect(ocr.options.pipelineConfig.assets.rec?.url).toMatch(/PP-OCRv5_mobile_rec/);
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
    expect(normalized.assets.det.url).toBe("https://example.com/custom-det.tar");
    expect(normalized.assets.rec.url).toBe("https://example.com/custom-rec.tar");
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
    const defaultOrt = normalizeOrtOptions();
    const ocr = await PaddleOCR.create({
      lang: "ch",
      ocrVersion: "PP-OCRv5",
      worker: true,
      ...CREATE_WITHOUT_INIT
    });

    expect(typeof ocr.initialize).toBe("function");
    expect(typeof ocr.predict).toBe("function");
    expect(typeof ocr.dispose).toBe("function");
    expect(ocr.options.pipelineConfig.assets.det?.url).toMatch(/PP-OCRv5_mobile_det/);
    expect(ocr.options.ortOptions.backend).toBe(defaultOrt.backend);
  });

  it("uses the package OCR.yaml defaults when no pipeline config is passed", async () => {
    const defaultPipeline = normalizeOcrPipelineConfig(DEFAULT_OCR_PIPELINE_CONFIG_TEXT);
    const ocr = await PaddleOCR.create(CREATE_WITHOUT_INIT);

    expect(ocr.options.pipelineConfig.assets.det?.url).toMatch(
      new RegExp(defaultPipeline.modelSelection.textDetectionModelName)
    );
    expect(ocr.options.pipelineConfig.assets.rec?.url).toMatch(
      new RegExp(defaultPipeline.modelSelection.textRecognitionModelName)
    );
    expect(ocr.pipelineConfig.runtimeDefaults).toMatchObject(defaultPipeline.runtimeDefaults);
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
    const ocr = await PaddleOCR.create({
      pipelineConfig: pipelineConfigText,
      ...IGNORE_UNSUPPORTED
    });

    expectDefaultModelAssets(ocr);
    expect(ocr.pipelineConfig.runtimeDefaults.text_det_limit_type).toBe("min");
    expect(ocr.pipelineConfig.runtimeDefaults.text_rec_score_thresh).toBe(0);
  });

  it("lets explicit model assets override pipeline config model names", async () => {
    const ocr = await PaddleOCR.create({
      pipelineConfig: pipelineConfigText,
      text_detection_model_name: "custom_det",
      textDetectionModelAsset: customDetAsset,
      text_recognition_model_name: "custom_rec",
      textRecognitionModelAsset: customRecAsset,
      ...IGNORE_UNSUPPORTED
    });

    expect(ocr.options.pipelineConfig.assets.det?.url).toBe("https://example.com/custom-det.tar");
    expect(ocr.options.pipelineConfig.assets.rec?.url).toBe("https://example.com/custom-rec.tar");
  });

  it("allows overriding only one side with a custom model asset", async () => {
    const ocr = await PaddleOCR.create({
      pipelineConfig: pipelineConfigText,
      text_detection_model_name: "custom_det",
      textDetectionModelAsset: customDetAsset,
      ...IGNORE_UNSUPPORTED
    });

    expect(ocr.options.pipelineConfig.assets.det?.url).toBe("https://example.com/custom-det.tar");
    expect(ocr.options.pipelineConfig.assets.rec?.url).toMatch(/PP-OCRv5_mobile_rec/);
  });

  it("lets explicit model names override pipeline config model assets", async () => {
    const ocr = await PaddleOCR.create({
      pipelineConfig: {
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
      text_detection_model_name: "PP-OCRv5_mobile_det",
      text_recognition_model_name: "PP-OCRv5_mobile_rec",
      ...CREATE_WITHOUT_INIT
    });

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
    await PaddleOCR.create({ pipelineConfig: pipelineConfigText, ...CREATE_WITHOUT_INIT });

    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("lets direct pipeline runtime parameters override pipelineConfig defaults", async () => {
    const ocr = await PaddleOCR.create({
      pipelineConfig: pipelineConfigText,
      text_det_limit_side_len: 200,
      text_rec_score_thresh: 0.42,
      ...IGNORE_UNSUPPORTED
    });

    expect(ocr.pipelineConfig.runtimeDefaults.text_det_limit_side_len).toBe(200);
    expect(ocr.pipelineConfig.runtimeDefaults.text_rec_score_thresh).toBe(0.42);
  });

  it("lets direct batch sizes override pipelineConfig", async () => {
    const ocr = await PaddleOCR.create({
      pipelineConfig: pipelineConfigText,
      textRecognitionBatchSize: 2,
      text_detection_batch_size: 3,
      ...IGNORE_UNSUPPORTED
    });

    expect(ocr.pipelineConfig.textRecognitionBatchSize).toBe(2);
    expect(ocr.pipelineConfig.textDetectionBatchSize).toBe(3);
  });

  it("applies textRecognitionBatchSize without user pipelineConfig", async () => {
    const ocr = await PaddleOCR.create({
      lang: "ch",
      ocrVersion: "PP-OCRv5",
      text_recognition_batch_size: 4,
      ...CREATE_WITHOUT_INIT
    });

    expect(ocr.pipelineConfig.textRecognitionBatchSize).toBe(4);
  });

  it("can turn unsupported pipeline warnings into errors", async () => {
    await expect(
      PaddleOCR.create({
        pipelineConfig: pipelineConfigText,
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

    expect(ocr.options.pipelineConfig.assets.det?.url).toBe("https://example.com/custom-det.tar");
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
      ortOptions: {
        backend: "wasm",
        proxy: true
      },
      ...CREATE_WITHOUT_INIT
    });

    const summary = await ocr.initialize();

    expect(summary.backend).toBe("wasm");
    expect(worker.messages[0].type).toBe("init");
    expect(worker.messages[0].payload.options.ortOptions.backend).toBe("wasm");
    expect(worker.messages[0].payload.options.ortOptions.disableWasmProxy).toBe(true);
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
          limitType: "max",
          maxSideLimit: 4000,
          normalize: {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            scale: 1 / 255
          },
          postprocess: {
            thresh: 0.3,
            boxThresh: 0.6,
            unclipRatio: 1.5
          }
        },
        rec: {
          charDict: [],
          imageShape: [3, 48, 320]
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

    expect(params.det.limitType).toBe("min");
    expect(params.det.thresh).toBe(0.4);
    expect(params.det.boxThresh).toBe(0.7);
    expect(params.det.unclipRatio).toBe(2);
    expect(params.pipeline.scoreThresh).toBe(0.2);
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
