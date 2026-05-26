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

import "context"

// Operation represents an accepted asynchronous job bound to the client that
// submitted it. Its embedded Job carries the task/model metadata needed for
// typed waits without inspecting result payloads.
type Operation struct {
	Job
	client *Client
}

// WaitOCR waits for an OCR operation and parses an OCR result.
func (op *Operation) WaitOCR(ctx context.Context) (*OCRResult, error) {
	if op == nil || op.client == nil {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "operation is nil"}}
	}
	return op.client.WaitOCRResult(ctx, &op.Job)
}

// WaitDocumentParsing waits for a document parsing operation and parses a
// document parsing result.
func (op *Operation) WaitDocumentParsing(ctx context.Context) (*DocParsingResult, error) {
	if op == nil || op.client == nil {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "operation is nil"}}
	}
	return op.client.WaitDocumentParsingResult(ctx, &op.Job)
}

// Poll checks the current job status without waiting for completion.
func (op *Operation) Poll(ctx context.Context) (*JobStatus, bool, error) {
	if op == nil || op.client == nil {
		return nil, false, &InvalidRequestError{PaddleOCRAPIError{Message: "operation is nil"}}
	}
	status, err := op.client.getJobStatus(ctx, op.JobID)
	if err != nil {
		return nil, false, err
	}
	js, err := convertStatus(op.JobID, op.Model, op.Task, status)
	if err != nil {
		return nil, false, err
	}
	return js, js.State == JobStateDone, nil
}
