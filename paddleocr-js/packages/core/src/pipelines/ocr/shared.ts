import type { AssetDescriptor } from "../../resources/registry";
import { DEFAULT_MODEL_ASSETS } from "../../resources/registry";
import { DEFAULT_DET_MODEL_CONFIG } from "../../models/det";
import { DEFAULT_REC_MODEL_CONFIG } from "../../models/rec";
import { extractInferenceModelName } from "../../models/common";
import { deepClone } from "../../utils/common";
import type { NormalizedPipelineConfig, PipelineModelSelection, PipelineRuntimeDefaults } from "./config";
import { normalizeOcrPipelineConfig } from "./config";
import { DEFAULT_OCR_PIPELINE_CONFIG_TEXT } from "./default-config";
import type { OcrModelConfig } from "./runtime-params";
import type { OrtRuntimeOptions } from "../../runtime/ort";

export interface ResolvedOcrOptions {
  assets: Record<string, AssetDescriptor>;
  modelSelection: Record<string, string | null>;
  runtimeDefaults: PipelineRuntimeDefaults;
  pipelineConfig: NormalizedPipelineConfig | null;
  runtime: NormalizedRuntimeOptions;
}

export type ResolvedBackend = "webgpu" | "wasm" | "auto";

export interface NormalizedRuntimeOptions {
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
  rec: DEFAULT_REC_MODEL_CONFIG,
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
  DEFAULT_OCR_PIPELINE_CONFIG_TEXT,
);
const DEFAULT_MODEL_SELECTION: Readonly<PipelineModelSelection> = Object.freeze({
  ...DEFAULT_NORMALIZED_PIPELINE_CONFIG.modelSelection,
});
const DEFAULT_RUNTIME_DEFAULTS: Readonly<PipelineRuntimeDefaults> = Object.freeze({
  ...DEFAULT_NORMALIZED_PIPELINE_CONFIG.runtimeDefaults,
});
const DEFAULT_LANG_VERSION_MODEL_SELECTION: Readonly<PipelineModelSelection> = Object.freeze({
  ...DEFAULT_MODEL_SELECTION,
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
    assetRequirementError: "text_detection_model_dir requires text_detection_model_name.",
  },
  {
    assetKey: "rec",
    modelRole: "TextRecognition",
    selectionKey: "textRecognitionModelName",
    nameAliases: ["text_recognition_model_name", "textRecognitionModelName"],
    assetAliases: [
      "textRecognitionModelAsset",
      "text_recognition_model_dir",
      "textRecognitionModelDir",
    ],
    nameLabel: "text recognition model name",
    assetLabel: "text recognition model asset",
    assetRequirementError: "text_recognition_model_dir requires text_recognition_model_name.",
  },
]);

const SUPPORTED_LANG_VERSION_MODELS = new Map<string, Readonly<PipelineModelSelection>>([
  ["ch::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
  ["chinese_cht::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
  ["en::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
  ["japan::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
]);

function readAliasedOption(options: Record<string, unknown>, aliases: string[], label: string): unknown {
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

function resolveModelAssetByName(_modelRole: string, modelName: string): AssetDescriptor {
  return DEFAULT_MODEL_ASSETS[modelName];
}

function getSelectedModelName(
  baseSelection: PipelineModelSelection | null,
  configSelection: PipelineModelSelection | null,
  explicitSelection: Record<string, string | null> | null,
  selectionKey: keyof PipelineModelSelection,
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
  explicitSelection: Record<string, string | null> | null,
): Record<string, string | null> {
  return Object.fromEntries(
    OCR_MODEL_ROLES.map((role) => [
      role.selectionKey,
      getSelectedModelName(baseSelection, configSelection, explicitSelection, role.selectionKey),
    ]),
  );
}

export function validateLoadedModelName(modelRole: string, expectedModelName: string | null | undefined, configText: string): void {
  if (!expectedModelName) {
    throw new Error(`${modelRole} model selection must define model_name.`);
  }
  const declaredModelName = extractInferenceModelName(configText);
  if (!declaredModelName) {
    throw new Error(`${modelRole} in inference.yml must define model_name.`);
  }
  if (declaredModelName !== expectedModelName) {
    throw new Error(
      `${modelRole} in inference.yml declares model_name "${declaredModelName}" but requested model_name is "${expectedModelName}".`,
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
  configAssets: Partial<Record<string, AssetDescriptor>> | null,
  explicitAssets: Record<string, AssetDescriptor> | null,
): AssetDescriptor | null {
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
  configAssets: Partial<Record<string, AssetDescriptor>> | null,
  explicitAssets: Record<string, AssetDescriptor> | null,
): Record<string, AssetDescriptor> {
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
        explicitAssets,
      ),
    ]),
  );

  if (Object.values(assets).some((asset) => !asset)) {
    throw new Error("OCR model selection must define both detection and recognition models.");
  }

  return assets as Record<string, AssetDescriptor>;
}

function getExplicitModelSelection(options: Record<string, unknown>): {
  modelSelection: Record<string, string | null>;
  assets: Record<string, AssetDescriptor>;
} | null {
  const modelSelection: Record<string, string | null> = {};
  const assets: Record<string, AssetDescriptor> = {};
  let hasAnyOption = false;

  for (const role of OCR_MODEL_ROLES) {
    const modelName = readAliasedOption(options, role.nameAliases, role.nameLabel) as string | undefined;
    const asset = readAliasedOption(options, role.assetAliases, role.assetLabel) as AssetDescriptor | undefined;

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
    assets,
  };
}

function resolveBaseModelSelection(
  options: Record<string, unknown>,
  includeDefaultBase = false,
): Readonly<PipelineModelSelection> | null {
  const ocrVersion = readAliasedOption(options, ["ocrVersion", "ocr_version"], "ocrVersion") as string | undefined;
  if (!options.lang && !ocrVersion) {
    return includeDefaultBase ? DEFAULT_MODEL_SELECTION : null;
  }

  const lang = (options.lang as string) || "ch";
  const resolvedOcrVersion = ocrVersion || "PP-OCRv5";
  const modelSelection = SUPPORTED_LANG_VERSION_MODELS.get(`${lang}::${resolvedOcrVersion}`);

  if (!modelSelection) {
    throw new Error(
      `Unsupported lang/ocrVersion combination: lang="${lang}", ocrVersion="${resolvedOcrVersion}".`,
    );
  }
  return modelSelection;
}

interface ConstructionResult {
  assets: Record<string, AssetDescriptor>;
  modelSelection: Record<string, string | null>;
  runtimeDefaults: PipelineRuntimeDefaults;
  normalizedPipelineConfig: NormalizedPipelineConfig | null;
}

function resolveConstructionOptions(options: Record<string, unknown> = {}): ConstructionResult {
  const pipelineInput = readAliasedOption(
    options,
    ["pipelineConfigText", "pipelineConfig", "pipeline"],
    "pipeline config",
  );
  const normalizedPipelineConfig =
    pipelineInput != null ? normalizeOcrPipelineConfig(pipelineInput) : null;
  const warningBehavior = resolveWarningBehavior(options.unsupportedBehavior);
  const warnings = normalizedPipelineConfig?.warnings || [];
  const baseSelection = resolveBaseModelSelection(options, !normalizedPipelineConfig);
  const configSelection = normalizedPipelineConfig?.modelSelection || null;
  const configAssets = normalizedPipelineConfig?.assets || null;
  const explicitOptions = getExplicitModelSelection(options);
  const explicitSelection = explicitOptions?.modelSelection || null;
  const explicitAssets = explicitOptions?.assets || null;
  const modelSelection = createResolvedModelSelection(
    baseSelection,
    configSelection,
    explicitSelection,
  );
  const assets = createOcrAssets(
    baseSelection,
    configSelection,
    explicitSelection,
    configAssets,
    explicitAssets,
  );

  if (normalizedPipelineConfig) {
    emitPipelineWarnings(warnings, warningBehavior);
    return {
      assets,
      modelSelection,
      runtimeDefaults: normalizedPipelineConfig.runtimeDefaults,
      normalizedPipelineConfig,
    };
  }

  return {
    assets,
    modelSelection,
    runtimeDefaults: DEFAULT_RUNTIME_DEFAULTS,
    normalizedPipelineConfig: null,
  };
}

function resolveBackend(raw: string | undefined): ResolvedBackend {
  if (raw === "webgpu" || raw === "wasm") return raw;
  return "auto";
}

export function normalizeRuntimeOptions(runtimeOptions: OrtRuntimeOptions = {}): NormalizedRuntimeOptions {
  const backend = resolveBackend(runtimeOptions.backend);

  return {
    backend,
    ...(runtimeOptions.wasmPaths !== undefined ? { wasmPaths: runtimeOptions.wasmPaths } : {}),
    ...(runtimeOptions.numThreads !== undefined ? { numThreads: runtimeOptions.numThreads } : {}),
    ...(runtimeOptions.simd !== undefined ? { simd: runtimeOptions.simd } : {}),
    ...(runtimeOptions.proxy !== undefined ? { proxy: runtimeOptions.proxy } : {}),
  };
}

export function resolveWorkerOptions(workerOption: unknown): WorkerResolvedOptions {
  if (!workerOption) {
    return {
      enabled: false,
      createWorker: null,
    };
  }

  if (workerOption === true) {
    return {
      enabled: true,
      createWorker: null,
    };
  }

  if (typeof workerOption === "object") {
    const opts = workerOption as Record<string, unknown>;
    return {
      enabled: true,
      createWorker:
        typeof opts.createWorker === "function" ? (opts.createWorker as () => Worker) : null,
    };
  }

  throw new Error("worker must be a boolean or an options object.");
}

export function resolvePaddleOCROptions(options: Record<string, unknown> = {}): ResolvedOcrOptions {
  const resolved = resolveConstructionOptions(options);
  return {
    assets: resolved.assets,
    modelSelection: resolved.modelSelection,
    runtimeDefaults: resolved.runtimeDefaults,
    pipelineConfig: resolved.normalizedPipelineConfig,
    runtime: normalizeRuntimeOptions((options.runtime || {}) as OrtRuntimeOptions),
  };
}

export function cloneDefaultOcrConfig(): OcrModelConfig {
  return deepClone(DEFAULT_OCR_CONFIG);
}
