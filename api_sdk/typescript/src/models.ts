// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

export enum Model {
  PPOCRv5 = "PP-OCRv5",
  PPOCRv5Latin = "PP-OCRv5-latin",
  PPOCRv6 = "PP-OCRv6",
  PPStructureV3 = "PP-StructureV3",
  PaddleOCRVL = "PaddleOCR-VL",
  PaddleOCRVL15 = "PaddleOCR-VL-1.5",
  PaddleOCRVL16 = "PaddleOCR-VL-1.6",
}

const OCR_MODELS = new Set<string>([Model.PPOCRv5, Model.PPOCRv5Latin, Model.PPOCRv6]);
const DOCUMENT_PARSING_MODELS = new Set<string>([
  Model.PPStructureV3,
  Model.PaddleOCRVL,
  Model.PaddleOCRVL15,
  Model.PaddleOCRVL16,
]);
const VL_MODELS = new Set<string>([
  Model.PaddleOCRVL,
  Model.PaddleOCRVL15,
  Model.PaddleOCRVL16,
]);

export function isOCRModel(
  model: string
): model is Model.PPOCRv5 | Model.PPOCRv5Latin | Model.PPOCRv6 {
  return OCR_MODELS.has(model);
}

export function isDocumentParsingModel(model: string): boolean {
  return DOCUMENT_PARSING_MODELS.has(model);
}

export function isVLModel(model: string): boolean {
  return VL_MODELS.has(model);
}

export interface OCROptions {
  useDocOrientationClassify?: boolean;
  useDocUnwarping?: boolean;
  useTextlineOrientation?: boolean;
  textDetLimitSideLen?: number;
  textDetLimitType?: string;
  textDetThresh?: number;
  textDetBoxThresh?: number;
  textDetUnclipRatio?: number;
  textRecScoreThresh?: number;
  visualize?: boolean;
  [key: string]: unknown;
}

export interface PPStructureV3Options {
  useDocOrientationClassify?: boolean;
  useDocUnwarping?: boolean;
  useTextlineOrientation?: boolean;
  useSealRecognition?: boolean;
  useTableRecognition?: boolean;
  useFormulaRecognition?: boolean;
  useChartRecognition?: boolean;
  useRegionDetection?: boolean;
  layoutThreshold?: number | Record<string, number>;
  layoutNms?: boolean;
  layoutUnclipRatio?: number | number[] | Record<string, number>;
  layoutMergeBboxesMode?: string | Record<string, string>;
  formatBlockContent?: boolean;
  textDetLimitSideLen?: number;
  textDetLimitType?: string;
  textDetThresh?: number;
  textDetBoxThresh?: number;
  textDetUnclipRatio?: number;
  textRecScoreThresh?: number;
  useWiredTableCellsTransToHtml?: boolean;
  useWirelessTableCellsTransToHtml?: boolean;
  useTableOrientationClassify?: boolean;
  useOcrResultsWithTableCells?: boolean;
  useE2eWiredTableRecModel?: boolean;
  useE2eWirelessTableRecModel?: boolean;
  markdownIgnoreLabels?: string[];
  prettifyMarkdown?: boolean;
  showFormulaNumber?: boolean;
  returnMarkdownImages?: boolean;
  outputFormats?: string[];
  visualize?: boolean;
  [key: string]: unknown;
}

export interface PaddleOCRVLOptions {
  useDocOrientationClassify?: boolean;
  useDocUnwarping?: boolean;
  useLayoutDetection?: boolean;
  useChartRecognition?: boolean;
  useSealRecognition?: boolean;
  useOcrForImageBlock?: boolean;
  layoutThreshold?: number | Record<string, number>;
  layoutNms?: boolean;
  layoutUnclipRatio?: number | number[] | Record<string, number>;
  layoutMergeBboxesMode?: string | Record<string, string>;
  layoutShapeMode?: "rect" | "quad" | "poly" | "auto";
  promptLabel?: "ocr" | "formula" | "table" | "chart" | "seal" | "spotting";
  formatBlockContent?: boolean;
  repetitionPenalty?: number;
  temperature?: number;
  topP?: number;
  minPixels?: number;
  maxPixels?: number;
  maxNewTokens?: number;
  vlmExtraArgs?: Record<string, unknown>;
  mergeLayoutBlocks?: boolean;
  markdownIgnoreLabels?: string[];
  prettifyMarkdown?: boolean;
  showFormulaNumber?: boolean;
  restructurePages?: boolean;
  mergeTables?: boolean;
  relevelTitles?: boolean;
  returnMarkdownImages?: boolean;
  outputFormats?: string[];
  visualize?: boolean;
  [key: string]: unknown;
}

export type DocParsingOptions = PPStructureV3Options | PaddleOCRVLOptions;

export interface OCRRequest {
  model?: Model | string;
  fileUrl?: string;
  filePath?: string;
  pageRanges?: string;
  batchId?: string;
  options?: OCROptions;
}

export interface DocParsingRequest {
  model?: Model | string;
  fileUrl?: string;
  filePath?: string;
  pageRanges?: string;
  batchId?: string;
  options?: DocParsingOptions;
}

export interface ClientOptions {
  token?: string;
  baseUrl?: string;
  timeout?: number;
  requestTimeout?: number;
  pollTimeout?: number;
  clientPlatform?: string;
  fetch?: typeof fetch;
}

export interface SaveResourceOptions {
  overwrite?: boolean;
  filename?: string;
}
