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
import { basename, dirname, extname, isAbsolute, join } from "node:path";
import { AuthError, FileNotFoundError, InvalidRequestError, ResultParseError } from "./errors.js";
import { HttpClient } from "./internal/http.js";
import { Poller } from "./internal/poller.js";
import type { ClientOptions, DocParsingRequest, OCRRequest, SaveResourceOptions } from "./models.js";
import { Model, isDocumentParsingModel, isOCRModel } from "./models.js";
import type { DocParsingResult, Job, JobStatus, OCRResult } from "./results.js";

const DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs";
const DEFAULT_REQUEST_TIMEOUT = 300000;
const DEFAULT_POLL_TIMEOUT = 600000;

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
    const baseUrl = options.baseUrl || DEFAULT_BASE_URL;
    const requestTimeout = options.requestTimeout ?? DEFAULT_REQUEST_TIMEOUT;
    const pollTimeout = options.pollTimeout ?? DEFAULT_POLL_TIMEOUT;

    this.http = new HttpClient(token, baseUrl, requestTimeout, options.fetch);
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
    const model = this.resolveOCRModel(req.model);
    const jobId = await this.submit(model, req, options?.signal);
    return this.toJob(jobId, model, "ocr", req);
  }

  async submitDocumentParsing(req: DocParsingRequest, options?: { signal?: AbortSignal }): Promise<Job> {
    const model = this.resolveDocumentModel(req.model);
    const jobId = await this.submit(model, req, options?.signal);
    return this.toJob(jobId, model, "document_parsing", req);
  }

  async getStatus(jobId: string, options?: { signal?: AbortSignal }): Promise<JobStatus> {
    return this.poller.getStatus(jobId, options?.signal);
  }

  async waitOcrResult(job: Job | string, options?: { signal?: AbortSignal }): Promise<OCRResult> {
    const resolved = this.resolveWaitJob(job, "ocr");
    const jsonlData = await this.poller.pollUntilDone(resolved.jobId, options?.signal);
    return this.parseOCRResult(resolved.jobId, jsonlData);
  }

  async waitDocumentParsingResult(job: Job | string, options?: { signal?: AbortSignal }): Promise<DocParsingResult> {
    const resolved = this.resolveWaitJob(job, "document_parsing");
    const jsonlData = await this.poller.pollUntilDone(resolved.jobId, options?.signal);
    return this.parseDocParsingResult(resolved.jobId, jsonlData);
  }

  async saveResource(resourceUrl: string, destination: string, options?: SaveResourceOptions): Promise<{ savedPaths: string[] }>;
  async saveResource(result: OCRResult | DocParsingResult, destination: string, options?: SaveResourceOptions): Promise<{ savedPaths: string[] }>;
  async saveResource(
    resource: string | OCRResult | DocParsingResult,
    destination: string,
    options: SaveResourceOptions = {},
  ): Promise<{ savedPaths: string[] }> {
    if (typeof resource === "string") {
      const savedPath = await this.saveResourceUrl(resource, destination, options);
      return { savedPaths: [savedPath] };
    }

    const savedPaths = await this.saveResultResources(resource, destination, options);
    return { savedPaths };
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
    const targets = await Promise.all(
      plans.map(async (plan) => {
        const target = join(destination, plan.filename);
        await this.requireWritableTarget(target, options);
        return target;
      }),
    );
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
      return [
        {
          resourceUrl: page.ocrImageUrl,
          filename: `ocr-page-${index + 1}${resourceExtension(page.ocrImageUrl)}`,
        },
      ];
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

  private resolveWaitJob(job: Job | string, task: Job["task"]): Job {
    if (typeof job === "string") {
      return {
        jobId: job,
        model: task === "ocr" ? Model.PPOCRv5 : Model.PaddleOCRVL15,
        task,
      };
    }
    if (job.task !== task) {
      throw new InvalidRequestError(`Cannot wait for ${task} result from ${job.task} job.`);
    }
    if (task === "ocr" && !isOCRModel(job.model)) {
      throw new InvalidRequestError(`Cannot wait for OCR result from model ${job.model}.`);
    }
    if (task === "document_parsing") {
      this.validateDocumentModel(job.model);
    }
    return job;
  }

  private async submit(
    model: string,
    req: { fileUrl?: string; filePath?: string; pageRanges?: string; batchId?: string; options?: object },
    signal?: AbortSignal,
  ): Promise<string> {
    if (!req.fileUrl && !req.filePath) {
      throw new InvalidRequestError("Either fileUrl or filePath is required.");
    }
    if (req.fileUrl && req.filePath) {
      throw new InvalidRequestError("fileUrl and filePath are mutually exclusive.");
    }
    this.validatePageRanges(req.pageRanges);

    const payload = req.options || this.defaultPayload(model);

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

  private defaultPayload(_model: string): object {
    return {};
  }

  private parseOCRResult(jobId: string, jsonlData: unknown[]): OCRResult {
    const pages = jsonlData.flatMap((lineObj) => {
      if (!isRecord(lineObj) || !isRecord(lineObj.result)) {
        throw new ResultParseError("OCR result line is missing result.");
      }
      if (!Array.isArray(lineObj.result.ocrResults)) {
        throw new ResultParseError("OCR result line is missing ocrResults.");
      }
      const ocrResults = lineObj.result.ocrResults;
      return ocrResults.map((item) => {
        if (!isRecord(item)) {
          throw new ResultParseError("OCR result item must be an object.");
        }
        if (!("prunedResult" in item)) {
          throw new ResultParseError("OCR result item is missing prunedResult.");
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
      if (!isRecord(lineObj) || !isRecord(lineObj.result)) {
        throw new ResultParseError("Document parsing result line is missing result.");
      }
      if (!Array.isArray(lineObj.result.layoutParsingResults)) {
        throw new ResultParseError("Document parsing result line is missing layoutParsingResults.");
      }
      const lpResults = lineObj.result.layoutParsingResults;
      return lpResults.map((item) => {
        if (!isRecord(item)) {
          throw new ResultParseError("Document parsing result item must be an object.");
        }
        if (!isRecord(item.markdown)) {
          throw new ResultParseError("Document parsing result item is missing markdown.");
        }
        if (typeof item.markdown.text !== "string") {
          throw new ResultParseError("Document parsing result item is missing markdown.text.");
        }
        return {
          markdownText: item.markdown.text,
          markdownImages: optionalStringMap(item.markdown.images, "markdown.images"),
          outputImages: optionalStringMap(item.outputImages, "outputImages"),
        };
      });
    });
    return { jobId, pages };
  }

  private toJob(
    jobId: string,
    model: string,
    task: Job["task"],
    req: { pageRanges?: string; batchId?: string },
  ): Job {
    return {
      jobId,
      model,
      task,
      pageRanges: req.pageRanges,
      batchId: req.batchId,
    };
  }

  private resolveOCRModel(model: Model = Model.PPOCRv5): Model {
    if (!isOCRModel(model)) {
      throw new InvalidRequestError(`Unsupported OCR model: ${model}`);
    }
    return model;
  }

  private resolveDocumentModel(model: Model = Model.PaddleOCRVL15): Model {
    this.validateDocumentModel(model);
    return model;
  }

  private validateDocumentModel(model: string): void {
    if (isOCRModel(model)) {
      throw new InvalidRequestError(`${model} is an OCR model and cannot be used for document parsing.`);
    }
    if (!isDocumentParsingModel(model)) {
      throw new InvalidRequestError(`Unsupported document parsing model: ${model}`);
    }
  }

  private validatePageRanges(pageRanges?: string): void {
    if (pageRanges === undefined) return;
    if (!/^\d+(-\d+)?(,\d+(-\d+)?)*$/.test(pageRanges)) {
      throw new InvalidRequestError(`Invalid pageRanges format: ${pageRanges}`);
    }
  }

  private async resolveDestination(url: URL, destination: string, options: SaveResourceOptions): Promise<string> {
    let destinationStat;
    try {
      destinationStat = await stat(destination);
    } catch {
      const parent = dirname(destination);
      try {
        await stat(parent);
      } catch {
        throw new FileNotFoundError(parent);
      }
      return destination;
    }

    if (!destinationStat.isDirectory()) {
      if (!options.overwrite) {
        throw new InvalidRequestError(`Destination already exists: ${destination}`);
      }
      return destination;
    }

    const filename = safeFilename(options.filename || basename(url.pathname) || "resource");
    const target = join(destination, filename);
    try {
      await stat(target);
      if (!options.overwrite) {
        throw new InvalidRequestError(`Destination already exists: ${target}`);
      }
    } catch (error) {
      if (!(error instanceof InvalidRequestError)) {
        return target;
      }
      throw error;
    }
    return target;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function optionalStringMap(value: unknown, fieldName: string): Record<string, string> {
  if (value === undefined || value === null) {
    return {};
  }
  if (isRecord(value) && Object.values(value).every((item) => typeof item === "string")) {
    return value as Record<string, string>;
  }
  throw new ResultParseError(`Document parsing result item ${fieldName} must be a string map.`);
}

function isDocParsingResult(result: OCRResult | DocParsingResult): result is DocParsingResult {
  return result.pages.some((page) => "markdownImages" in page || "outputImages" in page || "markdownText" in page);
}

function resourceExtension(resourceUrl: string): string {
  try {
    return extname(basename(new URL(resourceUrl).pathname));
  } catch {
    return "";
  }
}

function safeFilename(filename: string): string {
  const decoded = decodeURIComponent(filename);
  const safe = decoded.replace(/[^a-zA-Z0-9._-]/g, "_").replace(/^\.+/, "");
  return safe || "resource";
}

function safeMapKeyFilename(key: string): string {
  let decoded: string;
  try {
    decoded = decodeURIComponent(key);
  } catch (error) {
    throw new InvalidRequestError(`Invalid resource filename: ${key}`, { cause: error });
  }

  if (
    key.trim() === "" ||
    decoded.trim() === "" ||
    isAbsolute(key) ||
    isAbsolute(decoded) ||
    hasPathTraversalSegment(key) ||
    hasPathTraversalSegment(decoded) ||
    hasPathSeparator(key) ||
    hasPathSeparator(decoded)
  ) {
    throw new InvalidRequestError(`Unsafe resource filename: ${key}`);
  }

  return key;
}

function hasPathTraversalSegment(filename: string): boolean {
  return filename.split(/[\\/]/u).includes("..");
}

function hasPathSeparator(filename: string): boolean {
  return filename.includes("/") || filename.includes("\\");
}
