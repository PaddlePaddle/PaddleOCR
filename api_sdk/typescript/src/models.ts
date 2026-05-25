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
  PPStructureV3 = "PP-StructureV3",
  PaddleOCRVL = "PaddleOCR-VL",
  PaddleOCRVL15 = "PaddleOCR-VL-1.5",
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
}

export interface DocParsingOptions {
  useDocOrientationClassify?: boolean;
  useDocUnwarping?: boolean;
  useTextlineOrientation?: boolean;
  useSealRecognition?: boolean;
  useTableRecognition?: boolean;
  useFormulaRecognition?: boolean;
  useChartRecognition?: boolean;
  useRegionDetection?: boolean;
  useLayoutDetection?: boolean;
  layoutThreshold?: number | Record<string, number>;
  layoutNms?: boolean;
  layoutUnclipRatio?: number | number[] | Record<string, number>;
  layoutMergeBboxesMode?: string;
  textDetLimitSideLen?: number;
  textDetLimitType?: string;
  textDetThresh?: number;
  textDetBoxThresh?: number;
  textDetUnclipRatio?: number;
  textRecScoreThresh?: number;
  visualize?: boolean;
}

export interface OCRRequest {
  fileUrl?: string;
  filePath?: string;
  pageRanges?: string;
  batchId?: string;
  options?: OCROptions;
}

export interface DocParsingRequest {
  model: Model;
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
}
