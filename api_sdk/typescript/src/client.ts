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

import { stat, writeFile } from "node:fs/promises";
import { basename, dirname, extname, join } from "node:path";
import { AuthError, FileNotFoundError, InvalidRequestError, ResultParseError } from "./errors.js";
import { HttpClient } from "./internal/http.js";
import { Poller } from "./internal/poller.js";
import type { ClientOptions, DocParsingRequest, OCRRequest, SaveResourceOptions } from "./models.js";
import { isDocumentParsingModel, isOCRModel, Model } from "./models.js";
import type { BatchStatus, DocParsingResult, Job, JobStatus, OCRResult } from "./results.js";

const DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com";

interface ResourceSavePlan {
  resourceUrl: string;
  filename: string;
}

export class PaddleOCRClient {
  private http: HttpClient;
  private poller: Poller;

  constructor(options: ClientOptions = {}) {
    const token = options.token || process.env.PADDLEOCR_ACCESS_TOKEN || "";
    if (!token) {
      throw new AuthError("Token is required. Set PADDLEOCR_ACCESS_TOKEN or pass token option.");
    }
    const baseUrl = options.baseUrl || process.env.PADDLEOCR_BASE_URL || DEFAULT_BASE_URL;
    const requestTimeout = options.requestTimeout || options.timeout || 300000;
    const pollTimeout = options.pollTimeout || options.timeout || 600000;

    this.http = new HttpClient(
      token,
      baseUrl,
      requestTimeout,
      options.fetch,
      options.clientPlatform,
    );
    this.poller = new Poller(this.http, pollTimeout);
  }

  async ocr(req: OCRRequest, options?: { signal?: AbortSignal }): Promise<OCRResult> {
    const job = await this.submitOcr(req, options);
    return this.waitOcrResult(job, options);
  }

  async parseDocument(req: DocParsingRequest, options?: { signal?: AbortSignal }): Promise<DocParsingResult> {
    const job = await this.submitDocumentParsing(req, options);
    return this.waitDocumentParsingResult(job, options);
  }

  async submitOcr(req: OCRRequest, options?: { signal?: AbortSignal }): Promise<Job> {
    const model = req.model ?? Model.PPOCRv5;
    const jobId = await this.submit(model, "ocr", req, options?.signal);
    return { jobId, model, task: "ocr", pageRanges: req.pageRanges, batchId: req.batchId };
  }

  async submitDocumentParsing(req: DocParsingRequest, options?: { signal?: AbortSignal }): Promise<Job> {
    const model = req.model ?? Model.PaddleOCRVL16;
    if (!isDocumentParsingModel(model)) {
      throw new InvalidRequestError(`Model ${model} is not a document parsing model.`);
    }
    const jobId = await this.submit(model, "document_parsing", req, options?.signal);
    return { jobId, model, task: "document_parsing", pageRanges: req.pageRanges, batchId: req.batchId };
  }

  async waitOcrResult(job: Job | string, options?: { signal?: AbortSignal }): Promise<OCRResult> {
    const resolved = this.resolveJob(job, "ocr");
    const jsonlData = await this.poller.pollUntilDone(resolved.jobId, options?.signal);
    return this.parseOCRResult(resolved.jobId, jsonlData);
  }

  async waitDocumentParsingResult(job: Job | string, options?: { signal?: AbortSignal }): Promise<DocParsingResult> {
    const resolved = this.resolveJob(job, "document_parsing");
    const jsonlData = await this.poller.pollUntilDone(resolved.jobId, options?.signal);
    return this.parseDocParsingResult(resolved.jobId, jsonlData);
  }

  async getStatus(jobId: string, options?: { signal?: AbortSignal }): Promise<JobStatus> {
    return this.poller.getStatus(jobId, options?.signal);
  }

  async getBatchStatus(batchId: string, options?: { signal?: AbortSignal }): Promise<BatchStatus> {
    return this.poller.getBatchStatus(batchId, options?.signal);
  }

  async saveResource(
    resourceUrl: string,
    destination: string,
    options: SaveResourceOptions = {},
  ): Promise<string> {
    return this.saveResourceUrl(resourceUrl, destination, options);
  }

  async saveOcrResultResources(
    result: OCRResult,
    destination: string,
    options: SaveResourceOptions = {},
  ): Promise<string[]> {
    return this.saveResultResources(result, destination, options);
  }

  async saveDocumentParsingResultResources(
    result: DocParsingResult,
    destination: string,
    options: SaveResourceOptions = {},
  ): Promise<string[]> {
    return this.saveResultResources(result, destination, options);
  }

  private async saveResourceUrl(resourceUrl: string, destination: string, options: SaveResourceOptions): Promise<string> {
    if (!resourceUrl) {
      throw new InvalidRequestError("resourceUrl is required.");
    }
    let url: URL;
    try {
      url = new URL(resourceUrl);
    } catch (error) {
      throw new InvalidRequestError(`Invalid resource URL: ${resourceUrl}`, { cause: error });
    }
    const target = await this.resolveDestination(url, destination, options);
    const content = await this.http.fetchResource(resourceUrl);
    await writeFile(target, Buffer.from(content), { flag: options.overwrite ? "w" : "wx" });
    return target;
  }

  private async saveResultResources(
    result: OCRResult | DocParsingResult,
    destination: string,
    options: SaveResourceOptions,
  ): Promise<string[]> {
    await this.requireExistingDirectory(destination);
    const plans = this.collectResultResourcePlans(result);
    const targets = plans.map((plan) => join(destination, plan.filename));
    for (const target of targets) {
      await this.requireWritableTarget(target, options);
    }
    this.requireUniqueTargets(targets, options);
    const savedPaths: string[] = [];
    for (const [index, plan] of plans.entries()) {
      await this.saveResourceUrl(plan.resourceUrl, targets[index], options);
      savedPaths.push(targets[index]);
    }
    return savedPaths;
  }

  private collectResultResourcePlans(result: OCRResult | DocParsingResult): ResourceSavePlan[] {
    if (isDocParsingResult(result)) {
      return result.pages.flatMap((page) => [
        ...this.collectMappedResourcePlans(page.markdownImages),
        ...this.collectMappedResourcePlans(page.outputImages),
      ]);
    }
    return result.pages.flatMap((page, index) => {
      if (!page.ocrImageUrl) {
        return [];
      }
      return [{
        resourceUrl: page.ocrImageUrl,
        filename: `ocr-page-${index + 1}${resourceExtension(page.ocrImageUrl)}`,
      }];
    });
  }

  private collectMappedResourcePlans(resources: Record<string, string>): ResourceSavePlan[] {
    return Object.keys(resources)
      .sort()
      .map((key) => ({
        resourceUrl: resources[key],
        filename: safeMapKeyFilename(key),
      }));
  }

  private async requireExistingDirectory(destination: string): Promise<void> {
    let destinationStat;
    try {
      destinationStat = await stat(destination);
    } catch {
      throw new FileNotFoundError(destination);
    }
    if (!destinationStat.isDirectory()) {
      throw new InvalidRequestError(`Destination must be an existing directory: ${destination}`);
    }
  }

  private async requireWritableTarget(target: string, options: SaveResourceOptions): Promise<void> {
    try {
      await stat(target);
    } catch {
      return;
    }
    if (!options.overwrite) {
      throw new InvalidRequestError(`Destination already exists: ${target}`);
    }
  }

  private requireUniqueTargets(targets: string[], options: SaveResourceOptions): void {
    if (options.overwrite) {
      return;
    }
    const seen = new Set<string>();
    for (const target of targets) {
      if (seen.has(target)) {
        throw new InvalidRequestError(`Destination already exists: ${target}`);
      }
      seen.add(target);
    }
  }

  private async resolveDestination(url: URL, destination: string, options: SaveResourceOptions): Promise<string> {
    let destinationStat;
    try {
      destinationStat = await stat(destination);
    } catch {
      destinationStat = undefined;
    }
    const target = destinationStat?.isDirectory() ? join(destination, safeUrlBasename(url)) : destination;
    await this.requireWritableTarget(target, options);
    const parent = dirname(target);
    try {
      const parentStat = await stat(parent);
      if (!parentStat.isDirectory()) {
        throw new InvalidRequestError(`Destination parent must be a directory: ${parent}`);
      }
    } catch (error) {
      if (error instanceof InvalidRequestError) {
        throw error;
      }
      throw new FileNotFoundError(parent, { cause: error });
    }
    return target;
  }

  private async submit(
    model: string,
    task: Job["task"],
    req: { fileUrl?: string; filePath?: string; pageRanges?: string; batchId?: string; options?: object },
    signal?: AbortSignal,
  ): Promise<string> {
    if (!req.fileUrl && !req.filePath) {
      throw new InvalidRequestError("Either fileUrl or filePath is required.");
    }
    if (req.fileUrl && req.filePath) {
      throw new InvalidRequestError("fileUrl and filePath are mutually exclusive.");
    }

    this.validateModelForTask(model, task);
    const payload = req.options || {};

    if (req.fileUrl) {
      return this.http.submitUrl(model, req.fileUrl, payload, {
        pageRanges: req.pageRanges,
        batchId: req.batchId,
        signal,
      });
    }
    return this.http.submitFile(model, req.filePath!, payload, {
      pageRanges: req.pageRanges,
      batchId: req.batchId,
      signal,
    });
  }

  private parseOCRResult(jobId: string, jsonlData: unknown[]): OCRResult {
    const pages = jsonlData.flatMap((lineObj) => {
      if (!isRecord(lineObj) || !isRecord(lineObj.result) || !Array.isArray(lineObj.result.ocrResults)) {
        throw new ResultParseError("OCR result item is missing result.ocrResults.");
      }
      return lineObj.result.ocrResults.map((item) => {
        if (!isRecord(item) || !("prunedResult" in item)) {
          throw new ResultParseError("OCR result page is missing prunedResult.");
        }
        return {
          prunedResult: item.prunedResult,
          ocrImageUrl: typeof item.ocrImage === "string" ? item.ocrImage : undefined,
          raw: item,
        };
      });
    });
    return { jobId, pages };
  }

  private parseDocParsingResult(jobId: string, jsonlData: unknown[]): DocParsingResult {
    const pages = jsonlData.flatMap((lineObj) => {
      if (!isRecord(lineObj) || !isRecord(lineObj.result) || !Array.isArray(lineObj.result.layoutParsingResults)) {
        throw new ResultParseError("Document parsing result item is missing result.layoutParsingResults.");
      }
      return lineObj.result.layoutParsingResults.map((item) => {
        if (!isRecord(item) || !isRecord(item.markdown) || typeof item.markdown.text !== "string") {
          throw new ResultParseError("Document parsing result page is missing markdown.text.");
        }
        return {
          markdownText: item.markdown.text,
          markdownImages: isRecord(item.markdown.images) ? stringMap(item.markdown.images) : {},
          outputImages: isRecord(item.outputImages) ? stringMap(item.outputImages) : {},
        };
      });
    });
    return { jobId, pages };
  }

  private resolveJob(job: Job | string, expectedTask: Job["task"]): Job {
    if (typeof job === "string") {
      return {
        jobId: job,
        model: expectedTask === "ocr" ? Model.PPOCRv5 : Model.PaddleOCRVL16,
        task: expectedTask,
      };
    }
    if (job.task !== expectedTask) {
      throw new InvalidRequestError(`Job ${job.jobId} is a ${job.task} job, not a ${expectedTask} job.`);
    }
    this.validateModelForTask(job.model, expectedTask);
    return job;
  }

  private validateModelForTask(model: string, task: Job["task"]): void {
    if (task === "ocr" && !isOCRModel(model)) {
      throw new InvalidRequestError(`Model ${model} is not an OCR model.`);
    }
    if (task === "document_parsing" && !isDocumentParsingModel(model)) {
      throw new InvalidRequestError(`Model ${model} is not a document parsing model.`);
    }
  }
}

function stringMap(value: Record<string, unknown>): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, val] of Object.entries(value)) {
    if (typeof val === "string") {
      result[key] = val;
    }
  }
  return result;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isDocParsingResult(result: OCRResult | DocParsingResult): result is DocParsingResult {
  return result.pages.some((page) => "markdownText" in page);
}

function safeMapKeyFilename(key: string): string {
  if (!key || key === "." || key === ".." || key.includes("/") || key.includes("\\") || key.startsWith(".")) {
    throw new InvalidRequestError(`Unsafe resource filename: ${key}`);
  }
  return key;
}

function safeUrlBasename(url: URL): string {
  const name = basename(url.pathname) || "resource";
  if (name === "." || name === "..") {
    return "resource";
  }
  return name;
}

function resourceExtension(resourceUrl: string): string {
  try {
    const url = new URL(resourceUrl);
    return extname(url.pathname);
  } catch {
    return "";
  }
}
