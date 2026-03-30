import yaml from "js-yaml";

import type { AssetDescriptor } from "../../resources/registry";
import { normalizeAssetDescriptor } from "../../resources/registry";

const SUPPORTED_PIPELINE_NAME = "OCR";

export interface NormalizedPipelineConfig {
  pipelineName: string;
  raw: Record<string, unknown>;
  warnings: string[];
  unsupportedFeatures: string[];
  modelSelection: PipelineModelSelection;
  assets: Partial<Record<string, AssetDescriptor>>;
  runtimeDefaults: PipelineRuntimeDefaults;
}

export interface PipelineModelSelection {
  textDetectionModelName: string | null;
  textRecognitionModelName: string | null;
}

export interface PipelineRuntimeDefaults {
  text_det_limit_side_len?: number;
  text_det_limit_type?: string;
  text_det_max_side_limit?: number;
  text_det_thresh?: number;
  text_det_box_thresh?: number;
  text_det_unclip_ratio?: number;
  text_rec_score_thresh?: number;
}

type YamlObject = Record<string, unknown>;

function isPlainObject(value: unknown): value is YamlObject {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function toFiniteNumber(value: unknown): number | undefined {
  if (value === null || value === undefined || value === "") {
    return undefined;
  }
  const normalized = Number(value);
  return Number.isFinite(normalized) ? normalized : undefined;
}

function parsePipelineConfigInput(input: unknown): YamlObject {
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

function addFeatureWarning(warnings: string[], featureName: string, reason?: string): void {
  warnings.push(
    `${featureName} is not yet supported in PaddleOCR.js${reason ? `: ${reason}` : ""}.`,
  );
}

function getModuleModelName(moduleConfig: YamlObject | null): string | null {
  return typeof moduleConfig?.model_name === "string" ? moduleConfig.model_name : null;
}

function validateModuleAsset(modulePath: string, modelName: string | null): void {
  if (!modelName) {
    throw new Error(
      `${modulePath}.model_name must be provided when ${modulePath}.model_dir is set.`,
    );
  }
}

function getModuleAsset(
  assetName: string,
  modulePath: string,
  moduleConfig: YamlObject | null,
): AssetDescriptor | null {
  if (moduleConfig?.model_dir == null) {
    return null;
  }
  if (isPlainObject(moduleConfig.model_dir)) {
    const asset = normalizeAssetDescriptor(assetName, moduleConfig.model_dir);
    validateModuleAsset(modulePath, getModuleModelName(moduleConfig));
    return asset;
  }
  throw new Error(
    `${modulePath}.model_dir must be null or an asset descriptor object in browser usage.`,
  );
}

export function parseOcrPipelineConfigText(text: string): YamlObject {
  return parsePipelineConfigInput(text) as YamlObject;
}

export function normalizeOcrPipelineConfig(input: unknown): NormalizedPipelineConfig {
  const config = parsePipelineConfigInput(input);
  const pipelineName = (config.pipeline_name as string | undefined) ?? SUPPORTED_PIPELINE_NAME;

  if (pipelineName !== SUPPORTED_PIPELINE_NAME) {
    throw new Error(
      `Unsupported pipeline_name "${pipelineName}". PaddleOCR.js currently supports only "${SUPPORTED_PIPELINE_NAME}".`,
    );
  }

  const warnings: string[] = [];
  const subModules = isPlainObject(config.SubModules) ? config.SubModules : {};
  const textDetection = isPlainObject(subModules.TextDetection) ? (subModules.TextDetection as YamlObject) : null;
  const textRecognition = isPlainObject(subModules.TextRecognition) ? (subModules.TextRecognition as YamlObject) : null;

  if (!textDetection || !textRecognition) {
    throw new Error(
      'OCR pipeline config must define both "SubModules.TextDetection" and "SubModules.TextRecognition".',
    );
  }

  const useDocPreprocessor = Boolean(config.use_doc_preprocessor);
  const useTextlineOrientation = Boolean(config.use_textline_orientation);
  const subPipelines = config.SubPipelines as YamlObject | undefined;
  const docPreprocessor = isPlainObject(subPipelines?.DocPreprocessor)
    ? subPipelines!.DocPreprocessor
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
      ...(useTextlineOrientation || textLineOrientation ? ["TextLineOrientation"] : []),
    ],
    modelSelection: {
      textDetectionModelName: getModuleModelName(textDetection),
      textRecognitionModelName: getModuleModelName(textRecognition),
    },
    assets: {
      ...(detAsset ? { det: detAsset } : {}),
      ...(recAsset ? { rec: recAsset } : {}),
    },
    runtimeDefaults: {
      text_det_limit_side_len: toFiniteNumber(textDetection.limit_side_len),
      text_det_limit_type: (textDetection.limit_type as string) || undefined,
      text_det_max_side_limit: toFiniteNumber(textDetection.max_side_limit),
      text_det_thresh: toFiniteNumber(textDetection.thresh),
      text_det_box_thresh: toFiniteNumber(textDetection.box_thresh),
      text_det_unclip_ratio: toFiniteNumber(textDetection.unclip_ratio),
      text_rec_score_thresh: toFiniteNumber(textRecognition.score_thresh),
    },
  };
}
