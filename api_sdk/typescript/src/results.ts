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

export interface OCRPage {
  prunedResult: unknown;
  ocrImageUrl?: string;
  docPreprocessingImageUrl?: string;
  inputImageUrl?: string;
  raw?: unknown;
}

export interface DocParsingPage {
  markdownText: string;
  markdownImages: Record<string, string>;
  outputImages: Record<string, string>;
  prunedResult?: unknown;
  inputImageUrl?: string;
  exports?: Record<string, unknown>;
  markdown?: Record<string, unknown>;
  raw?: unknown;
}

export interface OCRResult {
  jobId: string;
  pages: OCRPage[];
  dataInfo?: Record<string, unknown>;
}

export interface DocParsingResult {
  jobId: string;
  pages: DocParsingPage[];
  dataInfo?: Record<string, unknown>;
}

export interface Progress {
  totalPages: number;
  extractedPages: number;
  startTime?: string;
  endTime?: string;
}

export interface Job {
  jobId: string;
  model: string;
  task: "ocr" | "document_parsing";
  pageRanges?: string;
  batchId?: string;
}

export interface JobStatus {
  jobId: string;
  state: "pending" | "running" | "done" | "failed";
  progress?: Progress;
  resultUrl?: Record<string, string>;
  errorMsg?: string;
}

export interface BatchStatus {
  batchId: string;
  jobs: JobStatus[];
}
