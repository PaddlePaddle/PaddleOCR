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

import { readFile, stat } from "node:fs/promises";
import { basename } from "node:path";
import {
  APIError,
  AuthError,
  FileNotFoundError,
  InvalidRequestError,
  NetworkError,
  RequestTimeoutError,
  ResponseFormatError,
  ResultParseError,
} from "../errors.js";
import { userAbortReason } from "./abort.js";

const DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs";

interface SubmitOptions {
  pageRanges?: string;
  batchId?: string;
  signal?: AbortSignal;
  timeoutMs?: number;
}

interface SubmitResponse {
  jobId: string;
}

export class HttpClient {
  private baseUrl: string;
  private token: string;
  private requestTimeout: number;
  private fetchImpl: typeof fetch;

  constructor(
    token: string,
    baseUrl: string = DEFAULT_BASE_URL,
    requestTimeout: number = 300000,
    fetchImpl: typeof fetch = fetch,
  ) {
    this.token = token;
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.requestTimeout = requestTimeout;
    this.fetchImpl = fetchImpl;
  }

  async submitUrl(model: string, fileUrl: string, optionalPayload: object, options: SubmitOptions = {}): Promise<string> {
    const body: Record<string, unknown> = { fileUrl, model, optionalPayload };
    if (options.pageRanges !== undefined) {
      body.pageRanges = options.pageRanges;
    }
    if (options.batchId !== undefined) {
      body.batchId = options.batchId;
    }
    const data = await this.fetchJson<SubmitResponse>(
      this.baseUrl,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      { signal: options.signal, timeoutMs: options.timeoutMs },
    );
    return this.requireJobId(data);
  }

  async submitFile(model: string, filePath: string, optionalPayload: object, options: SubmitOptions = {}): Promise<string> {
    try {
      await stat(filePath);
    } catch (error) {
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

    const fileBuffer = await readFile(filePath);
    const blob = new Blob([fileBuffer]);
    form.append("file", blob, basename(filePath));

    const data = await this.fetchJson<SubmitResponse>(
      this.baseUrl,
      {
        method: "POST",
        body: form,
      },
      { signal: options.signal, timeoutMs: options.timeoutMs },
    );
    return this.requireJobId(data);
  }

  async getJobStatus(jobId: string, signal?: AbortSignal, timeoutMs?: number): Promise<unknown> {
    return this.fetchJson<unknown>(`${this.baseUrl}/${encodeURIComponent(jobId)}`, { method: "GET" }, { signal, timeoutMs });
  }

  async fetchJsonl(url: string, signal?: AbortSignal, timeoutMs?: number): Promise<unknown[]> {
    const resp = await this.fetch(url, { method: "GET" }, { signal, timeoutMs, withAuth: false });
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
    const resp = await this.fetch(url, { method: "GET" }, { signal, timeoutMs, withAuth: false });
    return resp.arrayBuffer();
  }

  private async fetchJson<T>(url: string, init: RequestInit, options: FetchOptions = {}): Promise<T> {
    const resp = await this.fetch(url, init, options);
    let parsed: unknown;
    try {
      parsed = await resp.json();
    } catch (error) {
      throw new ResponseFormatError("Expected a JSON response body.", { cause: error });
    }
    if (!isRecord(parsed) || !("data" in parsed)) {
      throw new ResponseFormatError("Response body is missing data.");
    }
    return parsed.data as T;
  }

  private async fetch(
    url: string,
    init: RequestInit,
    options: FetchOptions = {},
  ): Promise<Response> {
    const withAuth = options.withAuth ?? true;
    const timeoutMs = Math.max(0, Math.min(this.requestTimeout, options.timeoutMs ?? this.requestTimeout));
    const headers: Record<string, string> = {
      ...(init.headers as Record<string, string> || {}),
    };
    if (withAuth) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    let resp: Response;
    const timeoutController = new AbortController();
    const timeoutID = setTimeout(() => timeoutController.abort(), timeoutMs);
    const abortController = new AbortController();
    const abort = () => abortController.abort();
    timeoutController.signal.addEventListener("abort", abort, { once: true });
    if (options.signal?.aborted) {
      abort();
    } else {
      options.signal?.addEventListener("abort", abort, { once: true });
    }
    try {
      resp = await this.fetchImpl(url, {
        ...init,
        headers,
        signal: abortController.signal,
      });
    } catch (e: unknown) {
      if (options.signal?.aborted) {
        throw userAbortReason(options.signal);
      }
      if (timeoutController.signal.aborted) {
        throw new RequestTimeoutError(timeoutMs, { cause: e });
      }
      const message = e instanceof Error ? e.message : String(e);
      throw new NetworkError(`Connection failed: ${message}`, { cause: e });
    } finally {
      clearTimeout(timeoutID);
      options.signal?.removeEventListener("abort", abort);
    }

    if (resp.ok) return resp;

    const text = await resp.text();
    if (resp.status === 401 || resp.status === 403) {
      throw new AuthError(`Authentication failed: ${text}`);
    } else if (resp.status === 400) {
      throw new InvalidRequestError(`Bad request: ${text}`);
    } else {
      throw new APIError(resp.status, text);
    }
  }

  private requireJobId(data: SubmitResponse): string {
    if (!isRecord(data) || typeof data.jobId !== "string" || data.jobId.length === 0) {
      throw new ResponseFormatError("Submit response is missing jobId.");
    }
    return data.jobId;
  }
}

interface FetchOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  withAuth?: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
