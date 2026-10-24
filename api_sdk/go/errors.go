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

package paddleocr

import "fmt"

type PaddleOCRAPIError struct {
	Message string
	Cause   error
}

func (e *PaddleOCRAPIError) Error() string {
	if e.Message == "" && e.Cause != nil {
		return e.Cause.Error()
	}
	return e.Message
}

func (e *PaddleOCRAPIError) Unwrap() error {
	return e.Cause
}

type AuthError struct {
	PaddleOCRAPIError
}

type InvalidRequestError struct {
	PaddleOCRAPIError
}

type APIError struct {
	StatusCode int
	PaddleOCRAPIError
}

func (e *APIError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message)
}

type RateLimitError struct {
	APIError
}

type ServiceUnavailableError struct {
	APIError
}

type JobFailedError struct {
	JobID    string
	ErrorMsg string
	PaddleOCRAPIError
}

func (e *JobFailedError) Error() string {
	return fmt.Sprintf("Job %s failed: %s", e.JobID, e.ErrorMsg)
}

type RequestTimeoutError struct {
	PaddleOCRAPIError
}

type PollTimeoutError struct {
	JobID   string
	Elapsed float64
	PaddleOCRAPIError
}

func (e *PollTimeoutError) Error() string {
	return fmt.Sprintf("Timed out after %.1fs waiting for job %s", e.Elapsed, e.JobID)
}

type NetworkError struct {
	PaddleOCRAPIError
}

type FileNotFoundError struct {
	Path string
	PaddleOCRAPIError
}

func (e *FileNotFoundError) Error() string {
	return fmt.Sprintf("File not found: %s", e.Path)
}

type ResponseFormatError struct {
	PaddleOCRAPIError
}

type ResultParseError struct {
	PaddleOCRAPIError
}
