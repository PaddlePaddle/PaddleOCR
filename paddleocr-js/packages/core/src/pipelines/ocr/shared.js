import { DEFAULT_MODEL_ASSETS } from "../../resources/registry.js";
import { DEFAULT_DET_MODEL_CONFIG, DEFAULT_REC_MODEL_CONFIG } from "../../models/index.js";
import { extractInferenceModelName } from "../../models/common.js";
import { deepClone } from "../../utils/common.js";
import { normalizeOcrPipelineConfig } from "./config.js";
import { DEFAULT_OCR_PIPELINE_CONFIG_TEXT } from "./default-config.js";

export const DEFAULT_OCR_CONFIG = {
  det: DEFAULT_DET_MODEL_CONFIG,
  rec: DEFAULT_REC_MODEL_CONFIG
};

const DEFAULT_NORMALIZED_PIPELINE_CONFIG = normalizeOcrPipelineConfig(
  DEFAULT_OCR_PIPELINE_CONFIG_TEXT
);
const DEFAULT_MODEL_SELECTION = Object.freeze({
  ...DEFAULT_NORMALIZED_PIPELINE_CONFIG.modelSelection
});
const DEFAULT_RUNTIME_DEFAULTS = Object.freeze({
  ...DEFAULT_NORMALIZED_PIPELINE_CONFIG.runtimeDefaults
});
const DEFAULT_LANG_VERSION_MODEL_SELECTION = Object.freeze({
  ...DEFAULT_MODEL_SELECTION
});
const OCR_MODEL_ROLES = Object.freeze([
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

const SUPPORTED_LANG_VERSION_MODELS = new Map([
  ["ch::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
  ["chinese_cht::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
  ["en::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
  ["japan::PP-OCRv5", DEFAULT_LANG_VERSION_MODEL_SELECTION],
]);

function readAliasedOption(options, aliases, label) {
  let resolved;
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

function resolveWarningBehavior(value) {
  if (value === "ignore" || value === "error") return value;
  return "warn";
}

function emitPipelineWarnings(warnings, behavior) {
  if (!warnings.length || behavior === "ignore") return;
  if (behavior === "error") {
    throw new Error(warnings.join(" "));
  }
  for (const warning of warnings) {
    console.warn(`[PaddleOCR.js] ${warning}`);
  }
}

function resolveModelAssetByName(modelRole, modelName) {
  const asset = DEFAULT_MODEL_ASSETS[modelName];
  if (!asset) {
    throw new Error(`Unsupported ${modelRole} model_name "${modelName}".`);
  }
  return asset;
}

function getSelectedModelName(baseSelection, configSelection, explicitSelection, selectionKey) {
  return (
    explicitSelection?.[selectionKey] ??
    configSelection?.[selectionKey] ??
    baseSelection?.[selectionKey] ??
    null
  );
}

function createResolvedModelSelection(baseSelection, configSelection, explicitSelection) {
  return Object.fromEntries(
    OCR_MODEL_ROLES.map((role) => [
      role.selectionKey,
      getSelectedModelName(baseSelection, configSelection, explicitSelection, role.selectionKey)
    ])
  );
}

export function validateLoadedModelName(modelRole, expectedModelName, configText) {
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
  assetRole,
  modelRole,
  selectionKey,
  baseSelection,
  configSelection,
  explicitSelection,
  configAssets,
  explicitAssets
) {
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
  baseSelection,
  configSelection,
  explicitSelection,
  configAssets,
  explicitAssets
) {
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

  return assets;
}

function getExplicitModelSelection(options) {
  const modelSelection = {};
  const assets = {};
  let hasAnyOption = false;

  for (const role of OCR_MODEL_ROLES) {
    const modelName = readAliasedOption(options, role.nameAliases, role.nameLabel);
    const asset = readAliasedOption(options, role.assetAliases, role.assetLabel);

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

function resolveBaseModelSelection(options, includeDefaultBase = false) {
  const ocrVersion = readAliasedOption(options, ["ocrVersion", "ocr_version"], "ocrVersion");
  if (!options.lang && !ocrVersion) {
    return includeDefaultBase ? DEFAULT_MODEL_SELECTION : null;
  }

  const lang = options.lang || "ch";
  const resolvedOcrVersion = ocrVersion || "PP-OCRv5";
  const modelSelection = SUPPORTED_LANG_VERSION_MODELS.get(`${lang}::${resolvedOcrVersion}`);

  if (!modelSelection) {
    throw new Error(
      `Unsupported lang/ocrVersion combination: lang="${lang}", ocrVersion="${resolvedOcrVersion}".`
    );
  }
  return modelSelection;
}

function resolveConstructionOptions(options = {}) {
  const pipelineInput = readAliasedOption(
    options,
    ["pipelineConfigText", "pipelineConfig", "pipeline"],
    "pipeline config"
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
    explicitSelection
  );
  const assets = createOcrAssets(
    baseSelection,
    configSelection,
    explicitSelection,
    configAssets,
    explicitAssets
  );

  if (normalizedPipelineConfig) {
    emitPipelineWarnings(warnings, warningBehavior);
    return {
      assets,
      modelSelection,
      runtimeDefaults: normalizedPipelineConfig.runtimeDefaults,
      normalizedPipelineConfig
    };
  }

  return {
    assets,
    modelSelection,
    runtimeDefaults: DEFAULT_RUNTIME_DEFAULTS,
    normalizedPipelineConfig: null
  };
}

export function normalizeRuntimeOptions(runtimeOptions = {}) {
  const backend =
    runtimeOptions.backend === "webgpu" || runtimeOptions.backend === "wasm"
      ? runtimeOptions.backend
      : "auto";

  return {
    backend,
    ...(runtimeOptions.wasmPaths !== undefined ? { wasmPaths: runtimeOptions.wasmPaths } : {}),
    ...(runtimeOptions.numThreads !== undefined ? { numThreads: runtimeOptions.numThreads } : {}),
    ...(runtimeOptions.simd !== undefined ? { simd: runtimeOptions.simd } : {}),
    ...(runtimeOptions.proxy !== undefined ? { proxy: runtimeOptions.proxy } : {})
  };
}

export function resolveWorkerOptions(workerOption) {
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
    return {
      enabled: true,
      createWorker:
        typeof workerOption.createWorker === "function" ? workerOption.createWorker : null
    };
  }

  throw new Error("worker must be a boolean or an options object.");
}

export function resolvePaddleOCROptions(options = {}) {
  const resolved = resolveConstructionOptions(options);
  return {
    assets: resolved.assets,
    modelSelection: resolved.modelSelection,
    runtimeDefaults: resolved.runtimeDefaults,
    pipelineConfig: resolved.normalizedPipelineConfig,
    runtime: normalizeRuntimeOptions(options.runtime || {})
  };
}

export function cloneDefaultOcrConfig() {
  return deepClone(DEFAULT_OCR_CONFIG);
}
