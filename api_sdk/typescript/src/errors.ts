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

export class PaddleOCRAPIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PaddleOCRAPIError";
  }
}

export class AuthError extends PaddleOCRAPIError {
  constructor(message: string) {
    super(message);
    this.name = "AuthError";
  }
}

export class InvalidRequestError extends PaddleOCRAPIError {
  constructor(message: string) {
    super(message);
    this.name = "InvalidRequestError";
  }
}

export class APIError extends PaddleOCRAPIError {
  statusCode: number;
  constructor(statusCode: number, message: string) {
    super(`HTTP ${statusCode}: ${message}`);
    this.name = "APIError";
    this.statusCode = statusCode;
  }
}

export class JobFailedError extends PaddleOCRAPIError {
  jobId: string;
  errorMsg: string;
  constructor(jobId: string, errorMsg: string) {
    super(`Job ${jobId} failed: ${errorMsg}`);
    this.name = "JobFailedError";
    this.jobId = jobId;
    this.errorMsg = errorMsg;
  }
}

export class TimeoutError extends PaddleOCRAPIError {
  jobId: string;
  elapsed: number;
  constructor(jobId: string, elapsed: number) {
    super(`Timed out after ${elapsed.toFixed(1)}s waiting for job ${jobId}`);
    this.name = "TimeoutError";
    this.jobId = jobId;
    this.elapsed = elapsed;
  }
}

export class NetworkError extends PaddleOCRAPIError {
  constructor(message: string) {
    super(message);
    this.name = "NetworkError";
  }
}

export class FileNotFoundError extends PaddleOCRAPIError {
  path: string;
  constructor(path: string) {
    super(`File not found: ${path}`);
    this.name = "FileNotFoundError";
    this.path = path;
  }
}
