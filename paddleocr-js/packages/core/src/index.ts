/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export {
  PaddleOCR,
  normalizeOcrPipelineConfig,
  parseOcrPipelineConfigText
} from "./pipelines/ocr/index";

export type { Point2D, NormalizeConfig, DetBox } from "./models/common";

export type {
  DetModelConfig,
  DetPostprocessConfig,
  DetModel,
  DetResult,
  DetRuntimeOverrides,
  LimitType
} from "./models/det";

export type { RecModelConfig, RecModel, RecResult, RecRuntimeOverrides } from "./models/rec";

export type {
  OcrRuntimeParamsInput,
  OcrModelConfig,
  ResolvedOcrParams
} from "./pipelines/ocr/runtime-params";

export type {
  OcrResult,
  OcrResultItem,
  OcrResultMetrics,
  OcrResultRuntime,
  InitializationSummary,
  OcrPipelineRunnerOptions
} from "./pipelines/ocr/core";

export type {
  NormalizedPipelineConfig,
  PipelineModelSelection,
  PipelineRuntimeDefaults
} from "./pipelines/ocr/config";

export type {
  ResolvedBackend,
  ResolvedOcrOptions,
  NormalizedOrtOptions,
  WorkerResolvedOptions
} from "./pipelines/ocr/shared";

export type { ModelAsset, ModelAssetsMap } from "./resources/model-asset";

export type { WebGpuState, OrtOptions } from "./runtime/ort";

export type { ImageSource, SourceMatResult } from "./platform/browser";

export type { PaddleOCRCreateOptions } from "./pipelines/ocr/index";
