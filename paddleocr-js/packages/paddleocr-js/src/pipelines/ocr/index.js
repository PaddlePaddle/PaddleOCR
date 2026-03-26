import { normalizeOcrPipelineConfig, parseOcrPipelineConfigText } from "./config.js";
import { ensureServedFromHttp, sourceToMat } from "../../platform/browser.js";
import { OcrPipelineRunner } from "./core.js";
import { resolvePaddleOCROptions, resolveWorkerOptions } from "./shared.js";
import { createWorkerBackedPaddleOCR } from "./worker-backed.js";

export class PaddleOCR extends OcrPipelineRunner {
  constructor(options) {
    super({
      ...options,
      ensureServedFromHttp,
      sourceToMat
    });
  }

  static async create(options = {}) {
    const workerOptions = resolveWorkerOptions(options.worker);
    if (workerOptions.enabled && options.fetch) {
      throw new Error("worker mode does not support a custom fetch implementation.");
    }

    const resolvedOptions = resolvePaddleOCROptions(options);
    const instance = workerOptions.enabled
      ? createWorkerBackedPaddleOCR(resolvedOptions, workerOptions)
      : new PaddleOCR({
          ...resolvedOptions,
          fetch: options.fetch
        });

    if (options.initialize !== false) {
      await instance.initialize();
    }
    return instance;
  }

  static async fromPipelineConfig(pipelineConfig, options = {}) {
    return PaddleOCR.create({
      ...options,
      pipelineConfig
    });
  }
}

export { normalizeOcrPipelineConfig, parseOcrPipelineConfigText };
