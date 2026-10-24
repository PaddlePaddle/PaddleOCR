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
	"fmt"
	"time"
)

const (
	initialInterval = 3 * time.Second
	multiplier      = 1.5
	maxInterval     = 15 * time.Second
)

func (c *Client) pollUntilDone(ctx context.Context, jobID string) ([]map[string]interface{}, error) {
	interval := initialInterval
	deadline := time.Now().Add(c.pollTimeout)
	pollCtx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()

	for time.Now().Before(deadline) {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			break
		}
		if interval > remaining {
			interval = remaining
		}
		timer := time.NewTimer(interval)
		select {
		case <-pollCtx.Done():
			timer.Stop()
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, &PollTimeoutError{
				JobID:             jobID,
				Elapsed:           c.pollTimeout.Seconds(),
				PaddleOCRAPIError: PaddleOCRAPIError{Message: fmt.Sprintf("Timed out after %.1fs", c.pollTimeout.Seconds())},
			}
		case <-timer.C:
		}

		status, err := c.getJobStatus(pollCtx, jobID)
		if err != nil {
			return nil, err
		}

		switch status.State {
		case "done":
			jsonURL := status.ResultURL["jsonUrl"]
			if jsonURL == "" {
				return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "done job response is missing resultUrl.jsonUrl"}}
			}
			return c.fetchJSONL(pollCtx, jsonURL)
		case "failed":
			return nil, &JobFailedError{
				JobID:             jobID,
				ErrorMsg:          status.ErrorMsg,
				PaddleOCRAPIError: PaddleOCRAPIError{Message: status.ErrorMsg},
			}
		}

		next := time.Duration(float64(interval) * multiplier)
		if next > maxInterval {
			next = maxInterval
		}
		interval = next
	}

	return nil, &PollTimeoutError{
		JobID:             jobID,
		Elapsed:           c.pollTimeout.Seconds(),
		PaddleOCRAPIError: PaddleOCRAPIError{Message: fmt.Sprintf("Timed out after %.1fs", c.pollTimeout.Seconds())},
	}
}
