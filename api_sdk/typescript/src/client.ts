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

import { AuthError, InvalidRequestError } from "./errors.js";
import { HttpClient } from "./internal/http.js";
import { Poller } from "./internal/poller.js";
import type { ClientOptions, DocParsingRequest, OCRRequest } from "./models.js";
import { Model } from "./models.js";
import type { DocParsingResult, Job, JobStatus, OCRResult } from "./results.js";

const DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs";

export class PaddleOCRClient {
  private http: HttpClient;
  private poller: Poller;

  constructor(options: ClientOptions = {}) {
    const token = options.token || process.env.PADDLE_OCR_TOKEN || "";
    if (!token) {
      throw new AuthError("Token is required. Set PADDLE_OCR_TOKEN or pass token option.");
    }
    const baseUrl = options.baseUrl || DEFAULT_BASE_URL;
    const timeout = options.timeout || 300000;

    this.http = new HttpClient(token, baseUrl, timeout);
    this.poller = new Poller(this.http, timeout);
  }

  async ocr(req: OCRRequest, options?: { signal?: AbortSignal }): Promise<OCRResult> {
    const jobId = await this.submit(Model.PPOCRv5, req);
    const jsonlData = await this.poller.pollUntilDone(jobId, options?.signal);
    return this.parseOCRResult(jobId, jsonlData);
  }

  async docParsing(req: DocParsingRequest, options?: { signal?: AbortSignal }): Promise<DocParsingResult> {
    const jobId = await this.submit(req.model, req);
    const jsonlData = await this.poller.pollUntilDone(jobId, options?.signal);
    return this.parseDocParsingResult(jobId, jsonlData);
  }

  async submitOcr(req: OCRRequest): Promise<Job> {
    const jobId = await this.submit(Model.PPOCRv5, req);
    return { jobId };
  }

  async submitDocParsing(req: DocParsingRequest): Promise<Job> {
    const jobId = await this.submit(req.model, req);
    return { jobId };
  }

  async waitForResult(jobId: string, options?: { signal?: AbortSignal }): Promise<OCRResult | DocParsingResult> {
    const jsonlData = await this.poller.pollUntilDone(jobId, options?.signal);
    const first = jsonlData[0]?.result || {};
    if ("ocrResults" in first) {
      return this.parseOCRResult(jobId, jsonlData);
    }
    return this.parseDocParsingResult(jobId, jsonlData);
  }

  async getResult(jobId: string): Promise<JobStatus> {
    return this.poller.getStatus(jobId);
  }

  private async submit(model: string, req: { fileUrl?: string; filePath?: string; options?: object }): Promise<string> {
    if (!req.fileUrl && !req.filePath) {
      throw new InvalidRequestError("Either fileUrl or filePath is required.");
    }
    if (req.fileUrl && req.filePath) {
      throw new InvalidRequestError("fileUrl and filePath are mutually exclusive.");
    }

    const payload = req.options || this.defaultPayload(model);

    if (req.fileUrl) {
      return this.http.submitUrl(model, req.fileUrl, payload);
    }
    return this.http.submitFile(model, req.filePath!, payload);
  }

  private defaultPayload(_model: string): object {
    return {};
  }

  private parseOCRResult(jobId: string, jsonlData: any[]): OCRResult {
    const pages = jsonlData.flatMap((lineObj) => {
      const ocrResults = lineObj.result?.ocrResults || [];
      return ocrResults.map((item: any) => ({
        prunedResult: item.prunedResult,
        ocrImageUrl: item.ocrImage,
      }));
    });
    return { jobId, pages };
  }

  private parseDocParsingResult(jobId: string, jsonlData: any[]): DocParsingResult {
    const pages = jsonlData.flatMap((lineObj) => {
      const lpResults = lineObj.result?.layoutParsingResults || [];
      return lpResults.map((item: any) => ({
        markdownText: item.markdown?.text || "",
        markdownImages: item.markdown?.images || {},
        outputImages: item.outputImages || {},
      }));
    });
    return { jobId, pages };
  }
}
