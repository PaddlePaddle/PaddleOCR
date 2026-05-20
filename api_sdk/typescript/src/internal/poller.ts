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

import { JobFailedError, TimeoutError } from "../errors.js";
import type { JobStatus, Progress } from "../results.js";
import { HttpClient } from "./http.js";

const INITIAL_INTERVAL = 3000;
const MULTIPLIER = 1.5;
const MAX_INTERVAL = 15000;
const MAX_WAIT_TIME = 600000;

export class Poller {
  private http: HttpClient;
  private maxWaitTime: number;

  constructor(http: HttpClient, maxWaitTime: number = MAX_WAIT_TIME) {
    this.http = http;
    this.maxWaitTime = maxWaitTime;
  }

  async pollUntilDone(jobId: string, signal?: AbortSignal): Promise<any[]> {
    let interval = INITIAL_INTERVAL;
    let elapsed = 0;

    while (elapsed < this.maxWaitTime) {
      if (signal?.aborted) {
        throw new Error("Aborted");
      }

      await this.sleep(interval, signal);
      elapsed += interval;

      const data = await this.http.getJobStatus(jobId);

      if (data.state === "done") {
        const jsonUrl = data.resultUrl.jsonUrl;
        return await this.http.fetchJsonl(jsonUrl);
      }

      if (data.state === "failed") {
        throw new JobFailedError(jobId, data.errorMsg || "Unknown error");
      }

      interval = Math.min(interval * MULTIPLIER, MAX_INTERVAL);
    }

    throw new TimeoutError(jobId, elapsed / 1000);
  }

  async getStatus(jobId: string): Promise<JobStatus> {
    const data = await this.http.getJobStatus(jobId);
    let progress: Progress | undefined;
    if (data.extractProgress) {
      progress = {
        totalPages: data.extractProgress.totalPages || 0,
        extractedPages: data.extractProgress.extractedPages || 0,
        startTime: data.extractProgress.startTime,
        endTime: data.extractProgress.endTime,
      };
    }
    return {
      jobId,
      state: data.state,
      progress,
      errorMsg: data.errorMsg,
    };
  }

  private sleep(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, ms);
      signal?.addEventListener("abort", () => {
        clearTimeout(timer);
        reject(new Error("Aborted"));
      });
    });
  }
}
