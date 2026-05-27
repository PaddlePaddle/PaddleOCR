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
  constructor(message: string, options?: ErrorOptions) {
    super(message);
    this.name = "PaddleOCRAPIError";
    this.cause = options?.cause;
  }
}

export class AuthError extends PaddleOCRAPIError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "AuthError";
  }
}

export class InvalidRequestError extends PaddleOCRAPIError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "InvalidRequestError";
  }
}

export class APIError extends PaddleOCRAPIError {
  statusCode: number;
  constructor(statusCode: number, message: string, options?: ErrorOptions) {
    super(`HTTP ${statusCode}: ${message}`, options);
    this.name = "APIError";
    this.statusCode = statusCode;
  }
}

export class RateLimitError extends APIError {
  constructor(message: string, options?: ErrorOptions) {
    super(429, message, options);
    this.name = "RateLimitError";
  }
}

export class ServiceUnavailableError extends APIError {
  constructor(statusCode: number, message: string, options?: ErrorOptions) {
    super(statusCode, message, options);
    this.name = "ServiceUnavailableError";
  }
}

export class JobFailedError extends PaddleOCRAPIError {
  jobId: string;
  errorMsg: string;
  constructor(jobId: string, errorMsg: string, options?: ErrorOptions) {
    super(`Job ${jobId} failed: ${errorMsg}`, options);
    this.name = "JobFailedError";
    this.jobId = jobId;
    this.errorMsg = errorMsg;
  }
}

export class RequestTimeoutError extends PaddleOCRAPIError {
  timeoutMs: number;
  constructor(timeoutMs: number, options?: ErrorOptions) {
    super(`Request timed out after ${timeoutMs}ms`, options);
    this.name = "RequestTimeoutError";
    this.timeoutMs = timeoutMs;
  }
}

export class PollTimeoutError extends PaddleOCRAPIError {
  jobId: string;
  timeoutMs: number;
  constructor(jobId: string, timeoutMs: number, options?: ErrorOptions) {
    super(`Timed out after ${timeoutMs}ms waiting for job ${jobId}`, options);
    this.name = "PollTimeoutError";
    this.jobId = jobId;
    this.timeoutMs = timeoutMs;
  }
}

export class NetworkError extends PaddleOCRAPIError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "NetworkError";
  }
}

export class FileNotFoundError extends PaddleOCRAPIError {
  path: string;
  constructor(path: string, options?: ErrorOptions) {
    super(`File not found: ${path}`, options);
    this.name = "FileNotFoundError";
    this.path = path;
  }
}

export class ResponseFormatError extends PaddleOCRAPIError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "ResponseFormatError";
  }
}

export class ResultParseError extends PaddleOCRAPIError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "ResultParseError";
  }
}
