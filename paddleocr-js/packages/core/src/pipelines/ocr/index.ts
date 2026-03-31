import { normalizeOcrPipelineConfig, parseOcrPipelineConfigText } from "./config";
import { ensureServedFromHttp, sourceToMat } from "../../platform/browser";
import type { OcrPipelineRunnerOptions } from "./core";
import { OcrPipelineRunner } from "./core";
import { resolvePaddleOCROptions, resolveWorkerOptions } from "./shared";
import { createWorkerBackedPaddleOCR } from "./worker-backed";
import type { WorkerBackedPaddleOCR } from "./worker-backed";
import type { OrtRuntimeOptions } from "../../runtime/ort";
import type { AssetDescriptor } from "../../resources/registry";

export interface PaddleOCRCreateOptions {
  worker?: boolean | { createWorker?: () => Worker };
  fetch?: typeof fetch;
  initialize?: boolean;
  runtime?: OrtRuntimeOptions;

  lang?: string;
  ocrVersion?: string;
  ocr_version?: string;

  pipelineConfig?: unknown;
  pipelineConfigText?: string;
  pipeline?: unknown;
  unsupportedBehavior?: "warn" | "ignore" | "error";

  textDetectionModelName?: string;
  text_detection_model_name?: string;
  textRecognitionModelName?: string;
  text_recognition_model_name?: string;

  textDetectionModelAsset?: AssetDescriptor;
  textDetectionModelDir?: AssetDescriptor;
  text_detection_model_dir?: AssetDescriptor;
  textRecognitionModelAsset?: AssetDescriptor;
  textRecognitionModelDir?: AssetDescriptor;
  text_recognition_model_dir?: AssetDescriptor;

  [key: string]: unknown;
}

export class PaddleOCR extends OcrPipelineRunner {
  constructor(options: OcrPipelineRunnerOptions) {
    super({
      ...options,
      ensureServedFromHttp,
      sourceToMat,
    });
  }

  static async create(options: PaddleOCRCreateOptions = {}): Promise<PaddleOCR | WorkerBackedPaddleOCR> {
    const workerOptions = resolveWorkerOptions(options.worker);
    if (workerOptions.enabled && options.fetch) {
      throw new Error("worker mode does not support a custom fetch implementation.");
    }

    const resolvedOptions = resolvePaddleOCROptions(options);
    const instance = workerOptions.enabled
      ? createWorkerBackedPaddleOCR(resolvedOptions, {
          createWorker: workerOptions.createWorker ?? undefined,
        })
      : new PaddleOCR({
          ...resolvedOptions,
          fetch: options.fetch,
        });

    if (options.initialize !== false) {
      await instance.initialize();
    }
    return instance;
  }

  static async fromPipelineConfig(
    pipelineConfig: unknown,
    options: PaddleOCRCreateOptions = {},
  ): Promise<PaddleOCR | WorkerBackedPaddleOCR> {
    return PaddleOCR.create({ ...options, pipelineConfig });
  }
}

export { normalizeOcrPipelineConfig, parseOcrPipelineConfigText };
