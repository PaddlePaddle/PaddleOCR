/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { normalizeOcrPipelineConfig, parseOcrPipelineConfigText } from "./config";
import { ensureServedFromHttp, sourceToMat } from "../../platform/browser";
import type { OcrPipelineRunnerOptions } from "./core";
import { OcrPipelineRunner } from "./core";
import { resolvePaddleOCROptions, resolveWorkerOptions } from "./shared";
import { createWorkerBackedPaddleOCR } from "./worker-backed";
import type { WorkerBackedPaddleOCR } from "./worker-backed";
import type { OrtOptions } from "../../runtime/ort";
import type { ModelAsset } from "../../resources/model-asset";
import type { LimitType } from "./runtime-params";

export interface PaddleOCRCreateOptions {
  worker?: boolean | { createWorker?: () => Worker };
  fetch?: typeof fetch;
  initialize?: boolean;
  ortOptions?: OrtOptions;

  pipelineConfig?: unknown;
  unsupportedBehavior?: "warn" | "ignore" | "error";

  lang?: string;
  ocrVersion?: string;
  ocr_version?: string;

  textDetectionModelName?: string;
  text_detection_model_name?: string;
  textRecognitionModelName?: string;
  text_recognition_model_name?: string;

  textDetectionModelAsset?: ModelAsset;
  textDetectionModelDir?: ModelAsset;
  text_detection_model_dir?: ModelAsset;
  textRecognitionModelAsset?: ModelAsset;
  textRecognitionModelDir?: ModelAsset;
  text_recognition_model_dir?: ModelAsset;

  textDetectionBatchSize?: number;
  text_detection_batch_size?: number;
  textRecognitionBatchSize?: number;
  text_recognition_batch_size?: number;
  batch_size?: number;

  textDetLimitSideLen?: number;
  text_det_limit_side_len?: number;
  textDetLimitType?: LimitType;
  text_det_limit_type?: LimitType;
  textDetMaxSideLimit?: number;
  text_det_max_side_limit?: number;
  textDetThresh?: number;
  text_det_thresh?: number;
  textDetBoxThresh?: number;
  text_det_box_thresh?: number;
  textDetUnclipRatio?: number;
  text_det_unclip_ratio?: number;
  textRecScoreThresh?: number;
  text_rec_score_thresh?: number;

  [key: string]: unknown;
}

export class PaddleOCR extends OcrPipelineRunner {
  constructor(options: OcrPipelineRunnerOptions) {
    super({
      ...options,
      ensureServedFromHttp,
      sourceToMat
    });
  }

  static async create(
    options: PaddleOCRCreateOptions = {}
  ): Promise<PaddleOCR | WorkerBackedPaddleOCR> {
    const workerOptions = resolveWorkerOptions(options.worker);
    if (workerOptions.enabled && options.fetch) {
      throw new Error("worker mode does not support a custom fetch implementation.");
    }

    const resolvedOptions = resolvePaddleOCROptions(options);
    const instance = workerOptions.enabled
      ? createWorkerBackedPaddleOCR(resolvedOptions, {
          createWorker: workerOptions.createWorker ?? undefined
        })
      : new PaddleOCR({
          ...resolvedOptions,
          fetch: options.fetch
        });

    if (options.initialize !== false) {
      await instance.initialize();
    }
    return instance;
  }
}

export { normalizeOcrPipelineConfig, parseOcrPipelineConfigText };
