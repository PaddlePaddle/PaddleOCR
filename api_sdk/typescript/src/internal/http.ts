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

import {
  APIError,
  AuthError,
  InvalidRequestError,
  NetworkError,
  RateLimitError,
  RequestTimeoutError,
  ResponseFormatError,
  ResultParseError,
  ServiceUnavailableError,
} from "../errors.js";
import { userAbortReason } from "./abort.js";

const DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com";
const API_PATH = "/api/v2/ocr/jobs";

interface SubmitOptions {
  pageRanges?: string;
  batchId?: string;
  signal?: AbortSignal;
}

interface APIResponse<T> {
  code?: number;
  msg?: string;
  data: T;
}

interface SubmitResponse {
  jobId: string;
}

export class HttpClient {
  private baseUrl: string;
  private jobsUrl: string;
  private token: string;
  private requestTimeout: number;
  private fetchImpl: typeof fetch;
  private clientPlatform?: string;

  constructor(
    token: string,
    baseUrl: string = DEFAULT_BASE_URL,
    requestTimeout: number = 300000,
    fetchImpl: typeof fetch = fetch,
    clientPlatform?: string,
  ) {
    this.token = token;
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.jobsUrl = `${this.baseUrl}${API_PATH}`;
    this.requestTimeout = requestTimeout;
    this.fetchImpl = fetchImpl;
    this.clientPlatform = clientPlatform;
  }

  async submitUrl(model: string, fileUrl: string, optionalPayload: object, options: SubmitOptions = {}): Promise<string> {
    const body: Record<string, unknown> = { fileUrl, model, optionalPayload };
    if (options.pageRanges !== undefined) {
      body.pageRanges = options.pageRanges;
    }
    if (options.batchId !== undefined) {
      body.batchId = options.batchId;
    }
    const data = await this.fetchJson<SubmitResponse>(this.jobsUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }, options.signal);
    return requireJobId(data);
  }

  async submitFile(model: string, filePath: string, optionalPayload: object, options: SubmitOptions = {}): Promise<string> {
    const fs = await import("fs");
    const path = await import("path");

    if (!fs.existsSync(filePath)) {
      const { FileNotFoundError } = await import("../errors.js");
      throw new FileNotFoundError(filePath);
    }

    const form = new FormData();
    form.append("model", model);
    form.append("optionalPayload", JSON.stringify(optionalPayload));
    if (options.pageRanges !== undefined) {
      form.append("pageRanges", options.pageRanges);
    }
    if (options.batchId !== undefined) {
      form.append("batchId", options.batchId);
    }

    const fileBuffer = fs.readFileSync(filePath);
    const blob = new Blob([fileBuffer]);
    form.append("file", blob, path.basename(filePath));

    const data = await this.fetchJson<SubmitResponse>(this.jobsUrl, {
      method: "POST",
      body: form,
    }, options.signal);
    return requireJobId(data);
  }

  async getJobStatus(jobId: string, signal?: AbortSignal, timeoutMs?: number): Promise<unknown> {
    return this.fetchJson<unknown>(
      `${this.jobsUrl}/${encodeURIComponent(jobId)}`,
      { method: "GET" },
      signal,
      true,
      timeoutMs,
    );
  }

  async getBatchStatus(batchId: string, signal?: AbortSignal, timeoutMs?: number): Promise<unknown> {
    return this.fetchJson<unknown>(
      `${this.jobsUrl}/batch/${encodeURIComponent(batchId)}`,
      { method: "GET" },
      signal,
      true,
      timeoutMs,
    );
  }

  async fetchJsonl(url: string, signal?: AbortSignal, timeoutMs?: number): Promise<unknown[]> {
    const resp = await this.fetch(url, { method: "GET" }, signal, false, timeoutMs);
    const text = await resp.text();
    try {
      return text
        .trim()
        .split("\n")
        .filter((line) => line.trim())
        .map((line) => JSON.parse(line) as unknown);
    } catch (error) {
      throw new ResultParseError("Failed to parse JSONL result payload.", { cause: error });
    }
  }

  async fetchResource(url: string, signal?: AbortSignal, timeoutMs?: number): Promise<ArrayBuffer> {
    const resp = await this.fetch(url, { method: "GET" }, signal, false, timeoutMs);
    return resp.arrayBuffer();
  }

  private async fetchJson<T>(
    url: string,
    init: RequestInit,
    signal?: AbortSignal,
    withAuth: boolean = true,
    timeoutMs?: number,
  ): Promise<T> {
    const resp = await this.fetch(url, init, signal, withAuth, timeoutMs);
    let payload: APIResponse<T>;
    try {
      payload = await resp.json() as APIResponse<T>;
    } catch (error) {
      throw new ResponseFormatError("Expected a JSON response body.", { cause: error });
    }
    if (payload.code !== undefined && payload.code !== 0) {
      throw new APIError(resp.status, payload.msg || "PaddleOCR official API request failed.");
    }
    if (!payload || typeof payload !== "object" || !("data" in payload)) {
      throw new ResponseFormatError("Response body is missing data.");
    }
    return payload.data;
  }

  private async fetch(
    url: string,
    init: RequestInit,
    signal?: AbortSignal,
    withAuth: boolean = true,
    timeoutMs?: number,
  ): Promise<Response> {
    const headers: Record<string, string> = {
      ...(init.headers as Record<string, string> || {}),
    };
    if (withAuth) {
      headers.Authorization = `Bearer ${this.token}`;
      if (this.clientPlatform) {
        headers["Client-Platform"] = this.clientPlatform;
      }
    }

    let resp: Response;
    const timeoutController = new AbortController();
    const effectiveTimeout = Math.max(0, Math.min(this.requestTimeout, timeoutMs ?? this.requestTimeout));
    const timeoutID = setTimeout(() => timeoutController.abort(), effectiveTimeout);
    const abortController = new AbortController();
    const abort = () => abortController.abort();
    timeoutController.signal.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) {
      abort();
    } else {
      signal?.addEventListener("abort", abort, { once: true });
    }
    try {
      resp = await this.fetchImpl(url, {
        ...init,
        headers,
        signal: abortController.signal,
      });
    } catch (e: unknown) {
      if (signal?.aborted) {
        throw userAbortReason(signal);
      }
      if (timeoutController.signal.aborted) {
        throw new RequestTimeoutError(effectiveTimeout, { cause: e });
      }
      const message = e instanceof Error ? e.message : String(e);
      throw new NetworkError(`Connection failed: ${message}`);
    } finally {
      clearTimeout(timeoutID);
      signal?.removeEventListener("abort", abort);
    }

    if (resp.ok) return resp;

    let text = await resp.text();
    try {
      const payload = JSON.parse(text) as { msg?: string; message?: string; errorMsg?: string };
      text = payload.msg || payload.message || payload.errorMsg || text;
    } catch {
      // Keep raw response text.
    }
    if (resp.status === 401 || resp.status === 403) {
      throw new AuthError(`Authentication failed: ${text}`);
    } else if (resp.status === 400) {
      throw new InvalidRequestError(`Bad request: ${text}`);
    } else if (resp.status === 429) {
      throw new RateLimitError(`Rate limit exceeded: ${text}`);
    } else if (resp.status === 503 || resp.status === 504) {
      throw new ServiceUnavailableError(resp.status, `Service unavailable: ${text}`);
    } else {
      throw new APIError(resp.status, text);
    }
  }
}

function requireJobId(data: SubmitResponse): string {
  if (!data || typeof data.jobId !== "string" || data.jobId.length === 0) {
    throw new ResponseFormatError("Submit response is missing jobId.");
  }
  return data.jobId;
}
