// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import { APIError, AuthError, InvalidRequestError, NetworkError } from "../errors.js";

const DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs";

export class HttpClient {
  private baseUrl: string;
  private token: string;
  private timeout: number;

  constructor(token: string, baseUrl: string = DEFAULT_BASE_URL, timeout: number = 300000) {
    this.token = token;
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = timeout;
  }

  async submitUrl(model: string, fileUrl: string, optionalPayload: object): Promise<string> {
    const body = { fileUrl, model, optionalPayload };
    const resp = await this.fetch(this.baseUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    return data.data.jobId;
  }

  async submitFile(model: string, filePath: string, optionalPayload: object): Promise<string> {
    const fs = await import("fs");
    const path = await import("path");

    if (!fs.existsSync(filePath)) {
      const { FileNotFoundError } = await import("../errors.js");
      throw new FileNotFoundError(filePath);
    }

    const form = new FormData();
    form.append("model", model);
    form.append("optionalPayload", JSON.stringify(optionalPayload));

    const fileBuffer = fs.readFileSync(filePath);
    const blob = new Blob([fileBuffer]);
    form.append("file", blob, path.basename(filePath));

    const resp = await this.fetch(this.baseUrl, {
      method: "POST",
      body: form,
    });
    const data = await resp.json();
    return data.data.jobId;
  }

  async getJobStatus(jobId: string): Promise<any> {
    const resp = await this.fetch(`${this.baseUrl}/${jobId}`, { method: "GET" });
    const data = await resp.json();
    return data.data;
  }

  async fetchJsonl(url: string): Promise<any[]> {
    const resp = await fetch(url, {
      signal: AbortSignal.timeout(this.timeout),
    });
    const text = await resp.text();
    return text
      .trim()
      .split("\n")
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line));
  }

  private async fetch(url: string, init: RequestInit): Promise<Response> {
    const headers: Record<string, string> = {
      Authorization: `bearer ${this.token}`,
      ...(init.headers as Record<string, string> || {}),
    };

    let resp: Response;
    try {
      resp = await fetch(url, {
        ...init,
        headers,
        signal: AbortSignal.timeout(this.timeout),
      });
    } catch (e: any) {
      throw new NetworkError(`Connection failed: ${e.message}`);
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
}
