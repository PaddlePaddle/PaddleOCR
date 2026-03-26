import { loadStandardModelAsset } from "../../resources/index.js";
import { createDetModel, createRecModel, cropByPoly } from "../../models/index.js";
import { initOpenCvRuntime } from "../../runtime/opencv.js";
import { initOrtRuntime } from "../../runtime/ort.js";
import { nowMs } from "../../utils/common.js";
import { getOcrRuntimeParams } from "./runtime-params.js";
import { cloneDefaultOcrConfig, validateLoadedModelName } from "./shared.js";

function noopEnsureServedFromHttp() {}

function getResolvedAssets(assets) {
  if (
    !assets?.det ||
    !assets?.rec ||
    typeof assets.det !== "object" ||
    typeof assets.rec !== "object"
  ) {
    throw new Error(
      "PaddleOCRCore requires pre-resolved detection and recognition asset descriptors."
    );
  }
  return assets;
}

export class OcrPipelineRunner {
  constructor(options) {
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

  async initialize() {
    this.ensureServedFromHttp();
    const start = nowMs();
    const { cv } = await initOpenCvRuntime();
    this.cv = cv;
    const { ort, webgpuState, backend } = await initOrtRuntime(this.options.runtime || {});
    this.ort = ort;
    this.webgpuState = webgpuState;

    const assets = getResolvedAssets(this.assets);
    const loadedAssets = await Promise.all([
      loadStandardModelAsset(assets.det, this.options.fetch || fetch),
      loadStandardModelAsset(assets.rec, this.options.fetch || fetch)
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
    const [detModel, recModel] = await Promise.all([
      createDetModel({
        ort: this.ort,
        modelBytes: loadedAssets[0].modelBytes,
        configText: loadedAssets[0].configText,
        backend,
        webgpuState
      }),
      createRecModel({
        ort: this.ort,
        modelBytes: loadedAssets[1].modelBytes,
        configText: loadedAssets[1].configText,
        backend,
        webgpuState
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
      cacheHits: loadedAssets.filter((asset) => asset.download.cacheHit).length,
      cacheMisses: loadedAssets.filter((asset) => !asset.download.cacheHit).length,
      pipelineConfigWarnings: this.pipelineConfig?.warnings || []
    };
    return this.lastInitializationSummary;
  }

  getInitializationSummary() {
    return this.lastInitializationSummary;
  }

  getModelConfig() {
    return this.modelConfig;
  }

  async predict(source, params = {}) {
    if (!this.sourceToMat) {
      throw new Error("PaddleOCR source adapter is not configured.");
    }
    if (!this.detModel || !this.recModel || !this.cv || !this.ort) {
      await this.initialize();
    }

    const sourceImage = await this.sourceToMat(this.cv, source);
    const totalStart = nowMs();
    try {
      const runtimeParams = getOcrRuntimeParams(this.modelConfig, this.runtimeDefaults, params);
      const detStart = nowMs();
      const detResult = await this.detModel.detect({
        cv: this.cv,
        sourceMat: sourceImage.mat,
        params: runtimeParams
      });
      const detElapsed = nowMs() - detStart;
      const detBoxes = detResult.boxes;

      const recPrepStart = nowMs();
      const samples = [];
      for (let index = 0; index < detBoxes.length; index += 1) {
        const crop = cropByPoly(this.cv, sourceImage.mat, detBoxes[index].poly);
        samples.push(
          this.recModel.prepareSample({
            cv: this.cv,
            cropMat: crop,
            poly: detBoxes[index].poly,
            originalIndex: index
          })
        );
        crop.delete();
      }
      const recPrepElapsed = nowMs() - recPrepStart;

      const recStart = nowMs();
      const recRaw = await this.recModel.recognize(samples);
      const recElapsed = nowMs() - recStart;

      const items = recRaw
        .filter((item) => item.text && item.score >= runtimeParams.text_rec_score_thresh)
        .sort((a, b) => a.originalIndex - b.originalIndex);

      return {
        image: {
          width: sourceImage.width,
          height: sourceImage.height
        },
        items,
        metrics: {
          detInferMs: detElapsed,
          recPrepMs: recPrepElapsed,
          recInferMs: recElapsed,
          totalMs: nowMs() - totalStart,
          detectedBoxes: detBoxes.length,
          recognizedCount: items.length
        },
        runtime: {
          requestedBackend: this.options.runtime?.backend || "auto",
          detProvider: this.detModel.provider,
          recProvider: this.recModel.provider,
          webgpuAvailable: this.webgpuState.available
        }
      };
    } finally {
      sourceImage.dispose();
    }
  }

  async disposeModelsOnly() {
    await Promise.all([this.detModel?.dispose(), this.recModel?.dispose()]);
    this.detModel = null;
    this.recModel = null;
  }

  async dispose() {
    await this.disposeModelsOnly();
  }
}

export { OcrPipelineRunner as PaddleOCRCore };
