export type { OpenCv, Mat, MatVector, Size, Rect, Scalar, RotatedRect } from "@techstark/opencv-js";

export type { Point2D, NormalizeConfig, DetBox, MiniBox } from "../models/common";

export type {
  DetModelConfig,
  DetPostprocessConfig,
  DetModel,
  DetResult,
  DetPreprocessResult,
} from "../models/det";

export type {
  RecModelConfig,
  RecModel,
  RecSample,
  RecResult,
} from "../models/rec";

export type {
  LimitType,
  OcrRuntimeParams,
  OcrRuntimeParamsInput,
  OcrModelConfig,
} from "../pipelines/ocr/runtime-params";

export type {
  OcrResult,
  OcrResultItem,
  OcrResultMetrics,
  OcrResultRuntime,
  InitializationSummary,
  OcrPipelineRunnerOptions,
  SourceToMatFn,
} from "../pipelines/ocr/core";

export type {
  NormalizedPipelineConfig,
  PipelineModelSelection,
  PipelineRuntimeDefaults,
} from "../pipelines/ocr/config";

export type {
  ResolvedBackend,
  ResolvedOcrOptions,
  NormalizedRuntimeOptions,
  WorkerResolvedOptions,
} from "../pipelines/ocr/shared";

export type { PaddleOCRCreateOptions } from "../pipelines/ocr/index";

export type {
  AssetDescriptor,
  ModelAssetsMap,
} from "../resources/registry";

export type {
  ResourceAsset,
  AssetFetchResult,
  AssetDownloadSummary,
} from "../resources/cache";

export type {
  OrtModule,
  WebGpuState,
  OrtRuntimeOptions,
  OrtRuntimeResult,
  SessionState,
} from "../runtime/ort";

export type {
  ImageSource,
  SourceMatResult,
  WorkerPayload,
  WorkerPayloadResult,
} from "../platform/browser";

export type {
  TransportRequest,
  TransportResponse,
  TransportSuccessResponse,
  TransportErrorResponse,
  SerializedError,
} from "../worker/protocol";

export type { WorkerOptions } from "../worker/client";
export type { MessageHandler } from "../worker/entry";
