import type { OpenCv } from "@techstark/opencv-js";
import type { AssetDescriptor } from "../../resources/registry";
import type { AssetDownloadSummary } from "../../resources/cache";
import { loadStandardModelAsset } from "../../resources/index";
import { createDetModel, createRecModel, cropByPoly } from "../../models/index";
import type { DetModel } from "../../models/det";
import type { RecModel } from "../../models/rec";
import type { Point2D } from "../../models/common";
import { initOpenCvRuntime } from "../../runtime/opencv";
import { initOrtRuntime } from "../../runtime/ort";
import type { OrtModule, WebGpuState, OrtRuntimeOptions } from "../../runtime/ort";
import { nowMs } from "../../utils/common";
import type { OcrModelConfig, OcrRuntimeParams } from "./runtime-params";
import { getOcrRuntimeParams } from "./runtime-params";
import type { NormalizedPipelineConfig, PipelineRuntimeDefaults } from "./config";
import { cloneDefaultOcrConfig, validateLoadedModelName } from "./shared";
import type { NormalizedRuntimeOptions } from "./shared";
import type { SourceMatResult } from "../../platform/browser";

export interface OcrResultItem {
  originalIndex: number;
  poly: Point2D[];
  text: string;
  score: number;
}

export interface OcrResultMetrics {
  detInferMs: number;
  recPrepMs: number;
  recInferMs: number;
  totalMs: number;
  detectedBoxes: number;
  recognizedCount: number;
}

export interface OcrResultRuntime {
  requestedBackend: string;
  detProvider: string;
  recProvider: string;
  webgpuAvailable: boolean;
}

export interface OcrResult {
  image: { width: number; height: number };
  items: OcrResultItem[];
  metrics: OcrResultMetrics;
  runtime: OcrResultRuntime;
}

export interface InitializationSummary {
  backend: string;
  webgpuAvailable: boolean;
  detProvider: string;
  recProvider: string;
  assets: AssetDownloadSummary[];
  elapsedMs: number;
  cacheHits: number;
  cacheMisses: number;
  pipelineConfigWarnings: string[];
}

export type SourceToMatFn = (cv: OpenCv, source: unknown) => SourceMatResult | Promise<SourceMatResult>;
type EnsureServedFromHttpFn = () => void;

export interface OcrPipelineRunnerOptions {
  assets: Record<string, AssetDescriptor>;
  modelSelection?: Record<string, string | null> | null;
  pipelineConfig?: NormalizedPipelineConfig | null;
  runtimeDefaults?: PipelineRuntimeDefaults;
  runtime?: OrtRuntimeOptions | NormalizedRuntimeOptions;
  fetch?: typeof fetch;
  ensureServedFromHttp?: EnsureServedFromHttpFn;
  sourceToMat?: SourceToMatFn;
}

function noopEnsureServedFromHttp(): void {}

function getResolvedAssets(assets: Record<string, AssetDescriptor> | undefined): { det: AssetDescriptor; rec: AssetDescriptor } {
  if (
    !assets?.det ||
    typeof assets.det !== "object" ||
    typeof assets.rec !== "object"
  ) {
    throw new Error(
      "PaddleOCRCore requires pre-resolved detection and recognition asset descriptors.",
    );
  }
  return assets as { det: AssetDescriptor; rec: AssetDescriptor };
}

export class OcrPipelineRunner {
  protected options: OcrPipelineRunnerOptions;
  protected modelConfig: OcrModelConfig;
  protected runtimeDefaults: Partial<OcrRuntimeParams>;
  protected cv: OpenCv | null;
  protected ort: OrtModule | null;
  protected detModel: DetModel | null;
  protected recModel: RecModel | null;
  protected webgpuState: WebGpuState;
  protected assets: Record<string, AssetDescriptor>;
  protected modelSelection: Record<string, string | null> | null;
  protected pipelineConfig: NormalizedPipelineConfig | null;
  protected lastInitializationSummary: InitializationSummary | null;
  private ensureServedFromHttp: EnsureServedFromHttpFn;
  private sourceToMat: SourceToMatFn | undefined;

  constructor(options: OcrPipelineRunnerOptions) {
    this.options = options;
    this.modelConfig = cloneDefaultOcrConfig();
    this.runtimeDefaults = { ...(options.runtimeDefaults || {}) };
    this.cv = null;
    this.ort = null;
    this.detModel = null;
    this.recModel = null;
    this.webgpuState = { available: false, reason: "" };
    this.assets = options.assets;
    this.modelSelection = options.modelSelection || null;
    this.pipelineConfig = options.pipelineConfig || null;
    this.lastInitializationSummary = null;
    this.ensureServedFromHttp = options.ensureServedFromHttp || noopEnsureServedFromHttp;
    this.sourceToMat = options.sourceToMat;
  }

  async initialize(): Promise<InitializationSummary> {
    this.ensureServedFromHttp();
    const start = nowMs();
    const { cv } = await initOpenCvRuntime();
    this.cv = cv;
    const { ort, webgpuState, backend } = await initOrtRuntime(this.options.runtime || {});
    this.ort = ort;
    this.webgpuState = webgpuState;

    const assets = getResolvedAssets(this.assets);
    const fetchImpl = this.options.fetch || fetch;
    const loadedAssets = await Promise.all([
      loadStandardModelAsset(assets.det, fetchImpl),
      loadStandardModelAsset(assets.rec, fetchImpl),
    ]);
    validateLoadedModelName(
      "TextDetection",
      this.modelSelection?.textDetectionModelName,
      loadedAssets[0].configText,
    );
    validateLoadedModelName(
      "TextRecognition",
      this.modelSelection?.textRecognitionModelName,
      loadedAssets[1].configText,
    );
    await this.disposeModelsOnly();
    const [detModel, recModel] = await Promise.all([
      createDetModel({
        ort: this.ort,
        modelBytes: loadedAssets[0].modelBytes,
        configText: loadedAssets[0].configText,
        backend,
        webgpuState,
      }),
      createRecModel({
        ort: this.ort,
        modelBytes: loadedAssets[1].modelBytes,
        configText: loadedAssets[1].configText,
        backend,
        webgpuState,
      }),
    ]);
    this.detModel = detModel;
    this.recModel = recModel;
    this.modelConfig = {
      det: this.detModel.config,
      rec: this.recModel.config,
    };

    const elapsed = nowMs() - start;
    this.lastInitializationSummary = {
      backend,
      webgpuAvailable: webgpuState.available,
      detProvider: this.detModel.provider,
      recProvider: this.recModel.provider,
      assets: loadedAssets.map((asset) => asset.download),
      elapsedMs: elapsed,
      cacheHits: loadedAssets.filter((asset) => asset.download.cacheHit).length,
      cacheMisses: loadedAssets.filter((asset) => !asset.download.cacheHit).length,
      pipelineConfigWarnings: this.pipelineConfig?.warnings || [],
    };
    return this.lastInitializationSummary;
  }

  getInitializationSummary(): InitializationSummary | null {
    return this.lastInitializationSummary;
  }

  getModelConfig(): OcrModelConfig {
    return this.modelConfig;
  }

  async predict(source: unknown, params: Record<string, unknown> = {}): Promise<OcrResult> {
    if (!this.sourceToMat) {
      throw new Error("PaddleOCR source adapter is not configured.");
    }
    if (!this.detModel || !this.recModel || !this.cv || !this.ort) {
      await this.initialize();
    }

    const cv = this.cv;
    const detModel = this.detModel;
    const recModel = this.recModel;
    if (!cv || !detModel || !recModel) {
      throw new Error("Initialization did not complete. Call initialize() first.");
    }

    const sourceImage = await this.sourceToMat(cv, source);
    const totalStart = nowMs();
    try {
      const runtimeParams = getOcrRuntimeParams(this.modelConfig, this.runtimeDefaults, params);
      const detStart = nowMs();
      const detResult = await detModel.detect({
        cv,
        sourceMat: sourceImage.mat,
        params: runtimeParams,
      });
      const detElapsed = nowMs() - detStart;
      const detBoxes = detResult.boxes;

      const recPrepStart = nowMs();
      const samples = [];
      for (let index = 0; index < detBoxes.length; index += 1) {
        const crop = cropByPoly(cv, sourceImage.mat, detBoxes[index].poly);
        samples.push(
          recModel.prepareSample({
            cv,
            cropMat: crop,
            poly: detBoxes[index].poly,
            originalIndex: index,
          }),
        );
        crop.delete();
      }
      const recPrepElapsed = nowMs() - recPrepStart;

      const recStart = nowMs();
      const recRaw = await recModel.recognize(samples);
      const recElapsed = nowMs() - recStart;

      const items = recRaw
        .filter((item) => item.text && item.score >= runtimeParams.text_rec_score_thresh)
        .sort((a, b) => a.originalIndex - b.originalIndex);

      return {
        image: {
          width: sourceImage.width,
          height: sourceImage.height,
        },
        items,
        metrics: {
          detInferMs: detElapsed,
          recPrepMs: recPrepElapsed,
          recInferMs: recElapsed,
          totalMs: nowMs() - totalStart,
          detectedBoxes: detBoxes.length,
          recognizedCount: items.length,
        },
        runtime: {
          requestedBackend: (this.options.runtime as NormalizedRuntimeOptions | undefined)?.backend || "auto",
          detProvider: detModel.provider,
          recProvider: recModel.provider,
          webgpuAvailable: this.webgpuState.available,
        },
      };
    } finally {
      sourceImage.dispose();
    }
  }

  async disposeModelsOnly(): Promise<void> {
    await Promise.all([this.detModel?.dispose(), this.recModel?.dispose()]);
    this.detModel = null;
    this.recModel = null;
  }

  async dispose(): Promise<void> {
    await this.disposeModelsOnly();
  }
}

export { OcrPipelineRunner as PaddleOCRCore };
