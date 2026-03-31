export {
  PaddleOCR,
  normalizeOcrPipelineConfig,
  parseOcrPipelineConfigText,
} from "./pipelines/ocr/index";

export type {
  Point2D,
  NormalizeConfig,
  DetBox,
} from "./models/common";

export type {
  DetModelConfig,
  DetPostprocessConfig,
  DetModel,
  DetResult,
  DetPreprocessResult,
} from "./models/det";

export type {
  RecModelConfig,
  RecModel,
  RecSample,
  RecResult,
} from "./models/rec";

export type {
  LimitType,
  OcrRuntimeParams,
  OcrRuntimeParamsInput,
  OcrModelConfig,
} from "./pipelines/ocr/runtime-params";

export type {
  OcrResult,
  OcrResultItem,
  OcrResultMetrics,
  OcrResultRuntime,
  InitializationSummary,
  OcrPipelineRunnerOptions,
} from "./pipelines/ocr/core";

export type {
  NormalizedPipelineConfig,
  PipelineModelSelection,
  PipelineRuntimeDefaults,
} from "./pipelines/ocr/config";

export type {
  ResolvedBackend,
  ResolvedOcrOptions,
  NormalizedRuntimeOptions,
  WorkerResolvedOptions,
} from "./pipelines/ocr/shared";

export type {
  AssetDescriptor,
  ModelAssetsMap,
} from "./resources/registry";

export type {
  ResourceAsset,
  AssetFetchResult,
  AssetDownloadSummary,
} from "./resources/cache";

export type {
  WebGpuState,
  OrtRuntimeOptions,
} from "./runtime/ort";

export type {
  ImageSource,
  SourceMatResult,
} from "./platform/browser";

export type { PaddleOCRCreateOptions } from "./pipelines/ocr/index";
