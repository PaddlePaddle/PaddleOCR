import yaml from "js-yaml";

import { normalizeAssetDescriptor } from "../../resources/registry.js";

const SUPPORTED_PIPELINE_NAME = "OCR";

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function toFiniteNumber(value) {
  if (value === null || value === undefined || value === "") {
    return undefined;
  }
  const normalized = Number(value);
  return Number.isFinite(normalized) ? normalized : undefined;
}

function parsePipelineConfigInput(input) {
  if (typeof input === "string") {
    const parsed = yaml.load(input);
    if (!isPlainObject(parsed)) {
      throw new Error("OCR pipeline config text must decode to an object.");
    }
    return parsed;
  }
  if (!isPlainObject(input)) {
    throw new Error("OCR pipeline config must be an object or YAML text.");
  }
  return input;
}

function addFeatureWarning(warnings, featureName, reason) {
  warnings.push(
    `${featureName} is not yet supported in PaddleOCR.js${reason ? `: ${reason}` : ""}.`
  );
}

function getModuleModelName(moduleConfig) {
  return typeof moduleConfig?.model_name === "string" ? moduleConfig.model_name : null;
}

function validateModuleAsset(modulePath, modelName) {
  if (!modelName) {
    throw new Error(
      `${modulePath}.model_name must be provided when ${modulePath}.model_dir is set.`
    );
  }
}

function getModuleAsset(assetName, modulePath, moduleConfig) {
  if (moduleConfig?.model_dir == null) {
    return null;
  }
  if (isPlainObject(moduleConfig.model_dir)) {
    const asset = normalizeAssetDescriptor(assetName, moduleConfig.model_dir);
    validateModuleAsset(modulePath, getModuleModelName(moduleConfig));
    return asset;
  }
  throw new Error(
    `${modulePath}.model_dir must be null or an asset descriptor object in browser usage.`
  );
}

export function parseOcrPipelineConfigText(text) {
  return parsePipelineConfigInput(text);
}

export function normalizeOcrPipelineConfig(input) {
  const config = parsePipelineConfigInput(input);
  const pipelineName = config.pipeline_name ?? SUPPORTED_PIPELINE_NAME;

  if (pipelineName !== SUPPORTED_PIPELINE_NAME) {
    throw new Error(
      `Unsupported pipeline_name "${pipelineName}". PaddleOCR.js currently supports only "${SUPPORTED_PIPELINE_NAME}".`
    );
  }

  const warnings = [];
  const subModules = isPlainObject(config.SubModules) ? config.SubModules : {};
  const textDetection = isPlainObject(subModules.TextDetection) ? subModules.TextDetection : null;
  const textRecognition = isPlainObject(subModules.TextRecognition)
    ? subModules.TextRecognition
    : null;

  if (!textDetection || !textRecognition) {
    throw new Error(
      'OCR pipeline config must define both "SubModules.TextDetection" and "SubModules.TextRecognition".'
    );
  }

  const useDocPreprocessor = Boolean(config.use_doc_preprocessor);
  const useTextlineOrientation = Boolean(config.use_textline_orientation);
  const docPreprocessor = isPlainObject(config.SubPipelines?.DocPreprocessor)
    ? config.SubPipelines.DocPreprocessor
    : null;
  const textLineOrientation = isPlainObject(subModules.TextLineOrientation)
    ? subModules.TextLineOrientation
    : null;

  if (useDocPreprocessor || docPreprocessor) {
    addFeatureWarning(warnings, "DocPreprocessor", "config will be ignored for now");
  }
  if (useTextlineOrientation || textLineOrientation) {
    addFeatureWarning(warnings, "TextLineOrientation", "config will be ignored for now");
  }
  if (config.text_type && config.text_type !== "general") {
    warnings.push(`text_type "${config.text_type}" is not used by PaddleOCR.js yet.`);
  }

  const detAsset = getModuleAsset("det", "SubModules.TextDetection", textDetection);
  const recAsset = getModuleAsset("rec", "SubModules.TextRecognition", textRecognition);

  return {
    pipelineName,
    raw: config,
    warnings,
    unsupportedFeatures: [
      ...(useDocPreprocessor || docPreprocessor ? ["DocPreprocessor"] : []),
      ...(useTextlineOrientation || textLineOrientation ? ["TextLineOrientation"] : [])
    ],
    modelSelection: {
      textDetectionModelName: getModuleModelName(textDetection),
      textRecognitionModelName: getModuleModelName(textRecognition)
    },
    assets: {
      ...(detAsset ? { det: detAsset } : {}),
      ...(recAsset ? { rec: recAsset } : {})
    },
    runtimeDefaults: {
      text_det_limit_side_len: toFiniteNumber(textDetection.limit_side_len),
      text_det_limit_type: textDetection.limit_type || undefined,
      text_det_max_side_limit: toFiniteNumber(textDetection.max_side_limit),
      text_det_thresh: toFiniteNumber(textDetection.thresh),
      text_det_box_thresh: toFiniteNumber(textDetection.box_thresh),
      text_det_unclip_ratio: toFiniteNumber(textDetection.unclip_ratio),
      text_rec_score_thresh: toFiniteNumber(textRecognition.score_thresh)
    }
  };
}
