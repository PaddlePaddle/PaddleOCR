/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ModelAsset } from "../../resources/model-asset";
import { DEFAULT_MODEL_ASSETS } from "../../resources/model-asset";
import { DEFAULT_DET_MODEL_CONFIG } from "../../models/det";
import { DEFAULT_REC_MODEL_CONFIG } from "../../models/rec";
import { extractInferenceModelName } from "../../models/common";
import { deepClone } from "../../utils/common";
import type {
  NormalizedPipelineConfig,
  PipelineModelSelection,
  PipelineRuntimeDefaults
} from "./config";
import { normalizeOcrPipelineConfig, toFiniteNumber } from "./config";
import type { LimitType } from "./runtime-params";
import { DEFAULT_OCR_PIPELINE_CONFIG_TEXT } from "./default-config";
import type { OcrModelConfig } from "./runtime-params";
import type { OrtOptions } from "../../runtime/ort";

export interface ResolvedOcrOptions {
  pipelineConfig: NormalizedPipelineConfig;
  ortOptions: NormalizedOrtOptions;
}

export type ResolvedBackend = "webgpu" | "wasm" | "auto";

export interface NormalizedOrtOptions {
  backend: ResolvedBackend;
  wasmPaths?: string;
  numThreads?: number;
  simd?: boolean;
  proxy?: boolean;
}

export interface WorkerResolvedOptions {
  enabled: boolean;
  createWorker: (() => Worker) | null;
}

export const DEFAULT_OCR_CONFIG: OcrModelConfig = {
  det: DEFAULT_DET_MODEL_CONFIG,
  rec: DEFAULT_REC_MODEL_CONFIG
};

interface ModelRole {
  assetKey: string;
  modelRole: string;
  selectionKey: keyof PipelineModelSelection;
  nameAliases: string[];
  assetAliases: string[];
  nameLabel: string;
  assetLabel: string;
  assetRequirementError: string;
}

const DEFAULT_NORMALIZED_PIPELINE_CONFIG = normalizeOcrPipelineConfig(
  DEFAULT_OCR_PIPELINE_CONFIG_TEXT
);
const DEFAULT_MODEL_SELECTION: Readonly<PipelineModelSelection> = Object.freeze({
  ...DEFAULT_NORMALIZED_PIPELINE_CONFIG.modelSelection
});
const DEFAULT_LANG_VERSION_MODEL_SELECTION: Readonly<PipelineModelSelection> = Object.freeze({
  ...DEFAULT_MODEL_SELECTION
});
const OCR_MODEL_ROLES: Readonly<ModelRole[]> = Object.freeze([
  {
    assetKey: "det",
    modelRole: "TextDetection",
    selectionKey: "textDetectionModelName",
    nameAliases: ["text_detection_model_name", "textDetectionModelName"],
    assetAliases: ["textDetectionModelAsset", "text_detection_model_dir", "textDetectionModelDir"],
    nameLabel: "text detection model name",
    assetLabel: "text detection model asset",
    assetRequirementError: "text_detection_model_dir requires text_detection_model_name."
  },
  {
    assetKey: "rec",
    modelRole: "TextRecognition",
    selectionKey: "textRecognitionModelName",
    nameAliases: ["text_recognition_model_name", "textRecognitionModelName"],
    assetAliases: [
      "textRecognitionModelAsset",
      "text_recognition_model_dir",
      "textRecognitionModelDir"
    ],
    nameLabel: "text recognition model name",
    assetLabel: "text recognition model asset",
    assetRequirementError: "text_recognition_model_dir requires text_recognition_model_name."
  }
]);

const SUPPORTED_LANG_VERSION_MODELS = new Map<string, Readonly<PipelineModelSelection>>([
  ["ch::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
  ["chinese_cht::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
  ["en::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
  ["japan::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION]
]);

function readAliasedOption(
  options: Record<string, unknown>,
  aliases: string[],
  label: string
): unknown {
  let resolved: unknown;
  let hasResolvedValue = false;

  for (const alias of aliases) {
    if (!(alias in options)) continue;
    const value = options[alias];
    if (!hasResolvedValue) {
      resolved = value;
      hasResolvedValue = true;
      continue;
    }
    if (value !== resolved) {
      throw new Error(`Conflicting values provided for ${label}: ${aliases.join(", ")}.`);
    }
  }

  return hasResolvedValue ? resolved : undefined;
}

function isLimitType(value: unknown): value is LimitType {
  return value === "min" || value === "max";
}

function overlayPipelineRuntimeDefaults(
  base: PipelineRuntimeDefaults,
  explicit: Partial<PipelineRuntimeDefaults>
): PipelineRuntimeDefaults {
  const next: Record<string, unknown> = { ...base };
  for (const key of Object.keys(explicit) as Array<keyof PipelineRuntimeDefaults>) {
    const value = explicit[key];
    if (value === undefined) continue;
    next[key as string] = value as unknown;
  }
  return next as PipelineRuntimeDefaults;
}

function readExplicitPipelineRuntimeDefaults(
  options: Record<string, unknown>
): Partial<PipelineRuntimeDefaults> {
  const out: Partial<PipelineRuntimeDefaults> = {};

  const limitSide = readAliasedOption(
    options,
    ["text_det_limit_side_len", "textDetLimitSideLen"],
    "text_det_limit_side_len"
  );
  if (limitSide !== undefined) {
    const n = toFiniteNumber(limitSide);
    if (n !== undefined) out.text_det_limit_side_len = n;
  }

  const limitType = readAliasedOption(
    options,
    ["text_det_limit_type", "textDetLimitType"],
    "text_det_limit_type"
  );
  if (limitType !== undefined && isLimitType(limitType)) {
    out.text_det_limit_type = limitType;
  }

  const maxSide = readAliasedOption(
    options,
    ["text_det_max_side_limit", "textDetMaxSideLimit"],
    "text_det_max_side_limit"
  );
  if (maxSide !== undefined) {
    const n = toFiniteNumber(maxSide);
    if (n !== undefined) out.text_det_max_side_limit = n;
  }

  const detThresh = readAliasedOption(
    options,
    ["text_det_thresh", "textDetThresh"],
    "text_det_thresh"
  );
  if (detThresh !== undefined) {
    const n = toFiniteNumber(detThresh);
    if (n !== undefined) out.text_det_thresh = n;
  }

  const boxThresh = readAliasedOption(
    options,
    ["text_det_box_thresh", "textDetBoxThresh"],
    "text_det_box_thresh"
  );
  if (boxThresh !== undefined) {
    const n = toFiniteNumber(boxThresh);
    if (n !== undefined) out.text_det_box_thresh = n;
  }

  const unclip = readAliasedOption(
    options,
    ["text_det_unclip_ratio", "textDetUnclipRatio"],
    "text_det_unclip_ratio"
  );
  if (unclip !== undefined) {
    const n = toFiniteNumber(unclip);
    if (n !== undefined) out.text_det_unclip_ratio = n;
  }

  const recScore = readAliasedOption(
    options,
    ["text_rec_score_thresh", "textRecScoreThresh"],
    "text_rec_score_thresh"
  );
  if (recScore !== undefined) {
    const n = toFiniteNumber(recScore);
    if (n !== undefined) out.text_rec_score_thresh = n;
  }

  return out;
}

function toBatchSizeOption(value: unknown): number | undefined {
  const n = toFiniteNumber(value);
  return n !== undefined && n >= 1 ? Math.floor(n) : undefined;
}

function readExplicitBatchSizes(options: Record<string, unknown>): {
  det: number | undefined;
  rec: number | undefined;
  pipeline: number | undefined;
} {
  return {
    det: toBatchSizeOption(
      readAliasedOption(
        options,
        ["textDetectionBatchSize", "text_detection_batch_size"],
        "textDetectionBatchSize"
      )
    ),
    rec: toBatchSizeOption(
      readAliasedOption(
        options,
        ["textRecognitionBatchSize", "text_recognition_batch_size"],
        "textRecognitionBatchSize"
      )
    ),
    pipeline: toBatchSizeOption(
      readAliasedOption(
        options,
        ["pipelineBatchSize", "pipeline_batch_size", "batch_size"],
        "pipelineBatchSize"
      )
    )
  };
}

function mergeNormalizedPipelineConfigWithExplicit(
  normalized: NormalizedPipelineConfig,
  options: Record<string, unknown>
): NormalizedPipelineConfig {
  const explicitRuntime = readExplicitPipelineRuntimeDefaults(options);
  const explicitBatch = readExplicitBatchSizes(options);
  const merged = deepClone(normalized);
  merged.runtimeDefaults = overlayPipelineRuntimeDefaults(merged.runtimeDefaults, explicitRuntime);
  if (explicitBatch.det !== undefined) {
    merged.textDetectionBatchSize = explicitBatch.det;
  }
  if (explicitBatch.rec !== undefined) {
    merged.textRecognitionBatchSize = explicitBatch.rec;
  }
  if (explicitBatch.pipeline !== undefined) {
    merged.pipelineBatchSize = explicitBatch.pipeline;
  }
  return merged;
}

function resolveWarningBehavior(value: unknown): "warn" | "ignore" | "error" {
  if (value === "ignore" || value === "error") return value;
  return "warn";
}

function emitPipelineWarnings(warnings: string[], behavior: "warn" | "ignore" | "error"): void {
  if (!warnings.length || behavior === "ignore") return;
  if (behavior === "error") {
    throw new Error(warnings.join(" "));
  }
  for (const warning of warnings) {
    console.warn(`[PaddleOCR.js] ${warning}`);
  }
}

function resolveModelAssetByName(_modelRole: string, modelName: string): ModelAsset {
  const asset = DEFAULT_MODEL_ASSETS[modelName];
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition -- runtime guard for missing Record key
  if (!asset) {
    throw new Error(`Unknown model asset "${modelName}".`);
  }
  return { url: asset.url };
}

function getSelectedModelName(
  baseSelection: PipelineModelSelection | null,
  configSelection: PipelineModelSelection | null,
  explicitSelection: Record<string, string | null> | null,
  selectionKey: keyof PipelineModelSelection
): string | null {
  return (
    explicitSelection?.[selectionKey] ??
    configSelection?.[selectionKey] ??
    baseSelection?.[selectionKey] ??
    null
  );
}

function createResolvedModelSelection(
  baseSelection: PipelineModelSelection | null,
  configSelection: PipelineModelSelection | null,
  explicitSelection: Record<string, string | null> | null
): PipelineModelSelection {
  return Object.fromEntries(
    OCR_MODEL_ROLES.map((role) => [
      role.selectionKey,
      getSelectedModelName(baseSelection, configSelection, explicitSelection, role.selectionKey)
    ])
  ) as unknown as PipelineModelSelection;
}

export function validateLoadedModelName(
  modelRole: string,
  expectedModelName: string | null | undefined,
  configText: string
): void {
  if (!expectedModelName) {
    throw new Error(`${modelRole} model selection must define model_name.`);
  }
  const declaredModelName = extractInferenceModelName(configText);
  if (!declaredModelName) {
    throw new Error(`${modelRole} in inference.yml must define model_name.`);
  }
  if (declaredModelName !== expectedModelName) {
    throw new Error(
      `${modelRole} in inference.yml declares model_name "${declaredModelName}" but requested model_name is "${expectedModelName}".`
    );
  }
}

function resolveSelectedAsset(
  assetRole: string,
  modelRole: string,
  selectionKey: keyof PipelineModelSelection,
  baseSelection: PipelineModelSelection | null,
  configSelection: PipelineModelSelection | null,
  explicitSelection: Record<string, string | null> | null,
  configAssets: Partial<Record<string, ModelAsset>> | null,
  explicitAssets: Record<string, ModelAsset> | null
): ModelAsset | null {
  const explicitAsset = explicitAssets?.[assetRole];
  if (explicitAsset) {
    return explicitAsset;
  }
  const explicitModelName = explicitSelection?.[selectionKey];
  if (explicitModelName) {
    return resolveModelAssetByName(modelRole, explicitModelName);
  }
  const configAsset = configAssets?.[assetRole];
  if (configAsset) {
    return configAsset;
  }
  const configModelName = configSelection?.[selectionKey];
  if (configModelName) {
    return resolveModelAssetByName(modelRole, configModelName);
  }
  const baseModelName = baseSelection?.[selectionKey];
  if (baseModelName) {
    return resolveModelAssetByName(modelRole, baseModelName);
  }
  return null;
}

function createOcrAssets(
  baseSelection: PipelineModelSelection | null,
  configSelection: PipelineModelSelection | null,
  explicitSelection: Record<string, string | null> | null,
  configAssets: Partial<Record<string, ModelAsset>> | null,
  explicitAssets: Record<string, ModelAsset> | null
): Record<string, ModelAsset> {
  const assets = Object.fromEntries(
    OCR_MODEL_ROLES.map((role) => [
      role.assetKey,
      resolveSelectedAsset(
        role.assetKey,
        role.modelRole,
        role.selectionKey,
        baseSelection,
        configSelection,
        explicitSelection,
        configAssets,
        explicitAssets
      )
    ])
  );

  if (Object.values(assets).some((asset) => !asset)) {
    throw new Error("OCR model selection must define both detection and recognition models.");
  }

  return assets as Record<string, ModelAsset>;
}

function getExplicitModelSelection(options: Record<string, unknown>): {
  modelSelection: Record<string, string | null>;
  assets: Record<string, ModelAsset>;
} | null {
  const modelSelection: Record<string, string | null> = {};
  const assets: Record<string, ModelAsset> = {};
  let hasAnyOption = false;

  for (const role of OCR_MODEL_ROLES) {
    const modelName = readAliasedOption(options, role.nameAliases, role.nameLabel) as
      | string
      | undefined;
    const asset = readAliasedOption(options, role.assetAliases, role.assetLabel) as
      | ModelAsset
      | undefined;

    if (modelName !== undefined) {
      modelSelection[role.selectionKey] = modelName;
      hasAnyOption = true;
    }
    if (asset !== undefined) {
      if (modelName === undefined) {
        throw new Error(role.assetRequirementError);
      }
      assets[role.assetKey] = asset;
      hasAnyOption = true;
    }
  }

  if (!hasAnyOption) {
    return null;
  }

  return {
    modelSelection,
    assets
  };
}

function resolveBaseModelSelection(
  options: Record<string, unknown>,
  includeDefaultBase = false
): Readonly<PipelineModelSelection> | null {
  const ocrVersion = readAliasedOption(options, ["ocrVersion", "ocr_version"], "ocrVersion") as
    | string
    | undefined;
  if (!options.lang && !ocrVersion) {
    return includeDefaultBase ? DEFAULT_MODEL_SELECTION : null;
  }

  const lang = (options.lang as string) || "ch";
  const resolvedOcrVersion = ocrVersion || "PP-OCRv5";
  const modelSelection = SUPPORTED_LANG_VERSION_MODELS.get(`${lang}::${resolvedOcrVersion}`);

  if (!modelSelection) {
    throw new Error(
      `Unsupported lang/ocrVersion combination: lang="${lang}", ocrVersion="${resolvedOcrVersion}".`
    );
  }
  return modelSelection;
}

function resolveConstructionOptions(
  options: Record<string, unknown> = {}
): NormalizedPipelineConfig {
  const pipelineInput = options.pipelineConfig;
  const userPipelineConfig =
    pipelineInput != null ? normalizeOcrPipelineConfig(pipelineInput) : null;
  const warningBehavior = resolveWarningBehavior(options.unsupportedBehavior);
  const warnings = userPipelineConfig?.warnings || [];
  const baseSelection = resolveBaseModelSelection(options, !userPipelineConfig);
  const configSelection = userPipelineConfig?.modelSelection || null;
  const configAssets = userPipelineConfig?.assets || null;
  const explicitOptions = getExplicitModelSelection(options);
  const explicitSelection = explicitOptions?.modelSelection || null;
  const explicitAssets = explicitOptions?.assets || null;
  const resolvedModelSelection = createResolvedModelSelection(
    baseSelection,
    configSelection,
    explicitSelection
  );
  const assets = createOcrAssets(
    baseSelection,
    configSelection,
    explicitSelection,
    configAssets,
    explicitAssets
  );

  const baseNormalized = userPipelineConfig ?? DEFAULT_NORMALIZED_PIPELINE_CONFIG;
  if (userPipelineConfig) {
    emitPipelineWarnings(warnings, warningBehavior);
  }
  const merged = mergeNormalizedPipelineConfigWithExplicit(baseNormalized, options);
  merged.modelSelection = resolvedModelSelection;
  merged.assets = { ...assets };
  return merged;
}

function resolveBackend(raw: string | undefined): ResolvedBackend {
  if (raw === "webgpu" || raw === "wasm") return raw;
  return "auto";
}

export function normalizeOrtOptions(ortOptions: OrtOptions = {}): NormalizedOrtOptions {
  const backend = resolveBackend(ortOptions.backend);

  return {
    backend,
    ...(ortOptions.wasmPaths !== undefined ? { wasmPaths: ortOptions.wasmPaths } : {}),
    ...(ortOptions.numThreads !== undefined ? { numThreads: ortOptions.numThreads } : {}),
    ...(ortOptions.simd !== undefined ? { simd: ortOptions.simd } : {}),
    ...(ortOptions.proxy !== undefined ? { proxy: ortOptions.proxy } : {})
  };
}

export function resolveWorkerOptions(workerOption: unknown): WorkerResolvedOptions {
  if (!workerOption) {
    return {
      enabled: false,
      createWorker: null
    };
  }

  if (workerOption === true) {
    return {
      enabled: true,
      createWorker: null
    };
  }

  if (typeof workerOption === "object") {
    const opts = workerOption as Record<string, unknown>;
    return {
      enabled: true,
      createWorker:
        typeof opts.createWorker === "function" ? (opts.createWorker as () => Worker) : null
    };
  }

  throw new Error("worker must be a boolean or an options object.");
}

export function resolvePaddleOCROptions(options: Record<string, unknown> = {}): ResolvedOcrOptions {
  return {
    pipelineConfig: resolveConstructionOptions(options),
    ortOptions: normalizeOrtOptions((options.ortOptions || {}) as OrtOptions)
  };
}

export function cloneDefaultOcrConfig(): OcrModelConfig {
  return deepClone(DEFAULT_OCR_CONFIG);
}
