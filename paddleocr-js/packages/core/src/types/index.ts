/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export type { OpenCv, Mat, MatVector, Size, Rect, Scalar, RotatedRect } from "@techstark/opencv-js";

export type { Point2D, NormalizeConfig, DetBox, MiniBox } from "../models/common";

export type {
  DetModelConfig,
  DetPostprocessConfig,
  DetModel,
  DetResult,
  DetRuntimeOverrides,
  LimitType
} from "../models/det";

export type { RecModelConfig, RecModel, RecResult, RecRuntimeOverrides } from "../models/rec";

export type {
  OcrRuntimeParamsInput,
  OcrModelConfig,
  ResolvedOcrParams
} from "../pipelines/ocr/runtime-params";

export type {
  OcrResult,
  OcrResultItem,
  OcrResultMetrics,
  OcrResultRuntime,
  InitializationSummary,
  OcrPipelineRunnerOptions,
  SourceToMatFn
} from "../pipelines/ocr/core";

export type {
  NormalizedPipelineConfig,
  PipelineModelSelection,
  PipelineRuntimeDefaults
} from "../pipelines/ocr/config";

export type {
  ResolvedBackend,
  ResolvedOcrOptions,
  NormalizedOrtOptions,
  WorkerResolvedOptions
} from "../pipelines/ocr/shared";

export type { PaddleOCRCreateOptions } from "../pipelines/ocr/index";

export type { ModelAsset, ModelAssetsMap } from "../resources/model-asset";

export type {
  OrtModule,
  WebGpuState,
  OrtOptions,
  OrtRuntimeResult,
  SessionState
} from "../runtime/ort";

export type {
  ImageSource,
  SourceMatResult,
  WorkerPayload,
  WorkerPayloadResult
} from "../platform/browser";

export type {
  TransportRequest,
  TransportResponse,
  TransportSuccessResponse,
  TransportErrorResponse,
  SerializedError
} from "../worker/protocol";

export type { WorkerOptions } from "../worker/client";
export type { MessageHandler } from "../worker/entry";
