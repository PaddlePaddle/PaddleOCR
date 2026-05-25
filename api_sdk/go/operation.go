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

import (
	"context"
	"encoding/json"
)

// Operation represents an in-progress job. Use Wait() to block until done,
// or Poll() to check status without blocking.
type Operation struct {
	client *Client
	JobID  string
	model  string
}

// Wait blocks until the job completes and returns the parsed result.
func (op *Operation) Wait(ctx context.Context) (interface{}, error) {
	jsonlData, err := op.client.pollUntilDone(ctx, op.JobID)
	if err != nil {
		return nil, err
	}
	if op.model == PPOCRv5 {
		return parseOCRResult(op.JobID, jsonlData)
	}
	return parseDocParsingResult(op.JobID, jsonlData)
}

// Poll checks the current job status without waiting.
// Returns the status, whether the job is done, and any error.
func (op *Operation) Poll(ctx context.Context) (*JobStatus, bool, error) {
	status, err := op.client.getJobStatus(ctx, op.JobID)
	if err != nil {
		return nil, false, err
	}

	js := &JobStatus{
		JobID:    op.JobID,
		State:    status.State,
		ErrorMsg: status.ErrorMsg,
	}

	if status.ExtractProgress != nil {
		var ep extractProgress
		if err := json.Unmarshal(status.ExtractProgress, &ep); err == nil {
			js.Progress = &Progress{
				TotalPages:     ep.TotalPages,
				ExtractedPages: ep.ExtractedPages,
				StartTime:      ep.StartTime,
				EndTime:        ep.EndTime,
			}
		}
	}

	return js, status.State == "done", nil
}
