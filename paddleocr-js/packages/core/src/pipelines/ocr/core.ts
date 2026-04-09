import type { OpenCv, Mat } from "@techstark/opencv-js";
import type { ModelAsset, ModelLoadSummary } from "../../resources/model-asset";
import { loadModelAsset } from "../../resources/index";
import { createDetModel, createRecModel } from "../../models/index";
import type { DetModel } from "../../models/det";
import type { RecModel } from "../../models/rec";
import type { Point2D } from "../../models/common";
import { cropByPoly } from "./crop";
import { initOpenCvRuntime } from "../../runtime/opencv";
import { initOrtRuntime } from "../../runtime/ort";
import type { OrtModule, WebGpuState, OrtRuntimeOptions } from "../../runtime/ort";
import { nowMs } from "../../utils/common";
import type { OcrModelConfig, OcrRuntimeParamsInput } from "./runtime-params";
import { getOcrRuntimeParams } from "./runtime-params";
import type { NormalizedPipelineConfig, PipelineRuntimeDefaults } from "./config";
import { cloneDefaultOcrConfig, validateLoadedModelName } from "./shared";
import type { NormalizedRuntimeOptions } from "./shared";
import type { SourceMatResult } from "../../platform/browser";

export interface OcrResultItem {
  poly: Point2D[];
  text: string;
  score: number;
}

export interface OcrResultMetrics {
  detMs: number;
  recMs: number;
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
  assets: ModelLoadSummary[];
  elapsedMs: number;
  pipelineConfigWarnings: string[];
}

export type SourceToMatFn = (
  cv: OpenCv,
  source: unknown
) => SourceMatResult | Promise<SourceMatResult>;
type EnsureServedFromHttpFn = () => void;

export interface OcrPipelineRunnerOptions {
  assets: Record<string, ModelAsset>;
  modelSelection?: Record<string, string | null> | null;
  pipelineConfig?: NormalizedPipelineConfig | null;
  runtimeDefaults?: PipelineRuntimeDefaults;
  runtime?: OrtRuntimeOptions | NormalizedRuntimeOptions;
  fetch?: typeof fetch;
  ensureServedFromHttp?: EnsureServedFromHttpFn;
  sourceToMat?: SourceToMatFn;
}

function noopEnsureServedFromHttp(): void {}

function getResolvedAssets(assets: Record<string, ModelAsset> | undefined): {
  det: ModelAsset;
  rec: ModelAsset;
} {
  if (!assets?.det || typeof assets.det !== "object" || typeof assets.rec !== "object") {
    throw new Error(
      "PaddleOCRCore requires pre-resolved detection and recognition asset descriptors."
    );
  }
  return assets as { det: ModelAsset; rec: ModelAsset };
}

export class OcrPipelineRunner {
  protected options: OcrPipelineRunnerOptions;
  protected modelConfig: OcrModelConfig;
  protected runtimeDefaults: Partial<OcrRuntimeParamsInput>;
  protected cv: OpenCv | null;
  protected ort: OrtModule | null;
  protected detModel: DetModel | null;
  protected recModel: RecModel | null;
  protected webgpuState: WebGpuState;
  protected assets: Record<string, ModelAsset>;
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
      loadModelAsset(assets.det, fetchImpl),
      loadModelAsset(assets.rec, fetchImpl)
    ]);
    validateLoadedModelName(
      "TextDetection",
      this.modelSelection?.textDetectionModelName,
      loadedAssets[0].configText
    );
    validateLoadedModelName(
      "TextRecognition",
      this.modelSelection?.textRecognitionModelName,
      loadedAssets[1].configText
    );
    await this.disposeModelsOnly();
    const detBatchSize = this.pipelineConfig?.textDetectionBatchSize ?? 1;
    const recBatchSize = this.pipelineConfig?.textRecognitionBatchSize ?? 1;

    const [detModel, recModel] = await Promise.all([
      createDetModel({
        ort: this.ort,
        modelBytes: loadedAssets[0].modelBytes,
        configText: loadedAssets[0].configText,
        backend,
        webgpuState,
        batchSize: detBatchSize
      }),
      createRecModel({
        ort: this.ort,
        modelBytes: loadedAssets[1].modelBytes,
        configText: loadedAssets[1].configText,
        backend,
        webgpuState,
        batchSize: recBatchSize
      })
    ]);
    this.detModel = detModel;
    this.recModel = recModel;
    this.modelConfig = {
      det: this.detModel.config,
      rec: this.recModel.config
    };

    const elapsed = nowMs() - start;
    this.lastInitializationSummary = {
      backend,
      webgpuAvailable: webgpuState.available,
      detProvider: this.detModel.provider,
      recProvider: this.recModel.provider,
      assets: loadedAssets.map((asset) => asset.download),
      elapsedMs: elapsed,
      pipelineConfigWarnings: this.pipelineConfig?.warnings || []
    };
    return this.lastInitializationSummary;
  }

  getInitializationSummary(): InitializationSummary | null {
    return this.lastInitializationSummary;
  }

  getModelConfig(): OcrModelConfig {
    return this.modelConfig;
  }

  async predict(input: unknown, params: OcrRuntimeParamsInput = {}): Promise<OcrResult[]> {
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

    const sources = Array.isArray(input) ? input : [input];
    const sourceImages = await Promise.all(sources.map((source) => this.sourceToMat!(cv, source)));

    const totalStart = nowMs();
    try {
      const resolved = getOcrRuntimeParams(this.modelConfig, this.runtimeDefaults, params);

      const detStart = nowMs();
      const detResults = await detModel.predict(
        cv,
        sourceImages.map((s) => s.mat),
        resolved.det
      );
      const detElapsed = nowMs() - detStart;

      const recStart = nowMs();
      const perImageItems: OcrResultItem[][] = [];

      for (let imgIdx = 0; imgIdx < detResults.length; imgIdx += 1) {
        const detBoxes = detResults[imgIdx]?.boxes ?? [];
        const cropMats: Mat[] = [];
        for (let boxIdx = 0; boxIdx < detBoxes.length; boxIdx += 1) {
          cropMats.push(cropByPoly(cv, sourceImages[imgIdx].mat, detBoxes[boxIdx].poly));
        }

        try {
          const recResults = cropMats.length ? await recModel.predict(cv, cropMats) : [];
          const items: OcrResultItem[] = [];
          for (let boxIdx = 0; boxIdx < recResults.length; boxIdx += 1) {
            const rec = recResults[boxIdx];
            if (rec.text && rec.score >= resolved.pipeline.scoreThresh) {
              items.push({
                poly: detBoxes[boxIdx].poly,
                text: rec.text,
                score: rec.score
              });
            }
          }
          perImageItems.push(items);
        } finally {
          for (const mat of cropMats) {
            mat.delete();
          }
        }
      }

      const recElapsed = nowMs() - recStart;
      const totalElapsed = nowMs() - totalStart;

      const results: OcrResult[] = sourceImages.map((sourceImage, imgIdx) => {
        const detBoxes = detResults[imgIdx]?.boxes ?? [];
        const items = perImageItems[imgIdx] ?? [];

        return {
          image: {
            width: sourceImage.width,
            height: sourceImage.height
          },
          items,
          metrics: {
            detMs: detElapsed,
            recMs: recElapsed,
            totalMs: totalElapsed,
            detectedBoxes: detBoxes.length,
            recognizedCount: items.length
          },
          runtime: {
            requestedBackend:
              (this.options.runtime as NormalizedRuntimeOptions | undefined)?.backend ?? "auto",
            detProvider: detModel.provider,
            recProvider: recModel.provider,
            webgpuAvailable: this.webgpuState.available
          }
        };
      });

      return results;
    } finally {
      for (const sourceImage of sourceImages) {
        sourceImage.dispose();
      }
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
