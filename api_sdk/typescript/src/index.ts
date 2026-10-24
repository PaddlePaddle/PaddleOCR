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

export { PaddleOCRClient } from "./client.js";
export { Model, isDocumentParsingModel, isOCRModel, isVLModel } from "./models.js";
export type {
  ClientOptions,
  DocParsingOptions,
  DocParsingRequest,
  OCROptions,
  OCRRequest,
  PaddleOCRVLOptions,
  PPStructureV3Options,
  SaveResourceOptions,
} from "./models.js";
export type {
  BatchStatus,
  DocParsingPage,
  DocParsingResult,
  Job,
  JobStatus,
  OCRPage,
  OCRResult,
  Progress,
} from "./results.js";
export {
  APIError,
  AuthError,
  FileNotFoundError,
  InvalidRequestError,
  JobFailedError,
  NetworkError,
  PaddleOCRAPIError,
  PollTimeoutError,
  RateLimitError,
  RequestTimeoutError,
  ResponseFormatError,
  ResultParseError,
  ServiceUnavailableError,
} from "./errors.js";
