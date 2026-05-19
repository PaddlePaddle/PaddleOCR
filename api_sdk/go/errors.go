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

package paddleocr

import "fmt"

type PaddleOCRError struct {
	Message string
}

func (e *PaddleOCRError) Error() string {
	return e.Message
}

type AuthError struct {
	PaddleOCRError
}

type InvalidRequestError struct {
	PaddleOCRError
}

type APIError struct {
	StatusCode int
	PaddleOCRError
}

func (e *APIError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message)
}

type JobFailedError struct {
	JobID    string
	ErrorMsg string
	PaddleOCRError
}

func (e *JobFailedError) Error() string {
	return fmt.Sprintf("Job %s failed: %s", e.JobID, e.ErrorMsg)
}

type TimeoutError struct {
	JobID   string
	Elapsed float64
	PaddleOCRError
}

func (e *TimeoutError) Error() string {
	return fmt.Sprintf("Timed out after %.1fs waiting for job %s", e.Elapsed, e.JobID)
}

type NetworkError struct {
	PaddleOCRError
}

type FileNotFoundError struct {
	Path string
	PaddleOCRError
}

func (e *FileNotFoundError) Error() string {
	return fmt.Sprintf("File not found: %s", e.Path)
}
