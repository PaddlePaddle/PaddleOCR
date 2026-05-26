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
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	initialInterval = 3 * time.Second
	multiplier      = 1.5
	maxInterval     = 15 * time.Second
)

type Job struct {
	JobID string
	Model Model
	Task  Task
}

type apiResponse struct {
	Data json.RawMessage `json:"data"`
}

type submitResponse struct {
	JobID string `json:"jobId"`
}

type jobStatusResponse struct {
	State           JobState        `json:"state"`
	ExtractProgress json.RawMessage `json:"extractProgress"`
	ResultURL       *resultURL      `json:"resultUrl"`
	ErrorMsg        string          `json:"errorMsg"`
}

type resultURL struct {
	JSONURL string `json:"jsonUrl"`
}

type extractProgress struct {
	TotalPages     int    `json:"totalPages"`
	ExtractedPages int    `json:"extractedPages"`
	StartTime      string `json:"startTime"`
	EndTime        string `json:"endTime"`
}

func (c *Client) submitURL(ctx context.Context, model Model, fileURL string, payload interface{}, pageRanges, batchID string) (string, error) {
	body := map[string]interface{}{
		"fileUrl":         fileURL,
		"model":           string(model),
		"optionalPayload": payload,
	}
	if pageRanges != "" {
		body["pageRanges"] = pageRanges
	}
	if batchID != "" {
		body["batchId"] = batchID
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", &InvalidRequestError{PaddleOCRAPIError{Message: "encode request: " + err.Error(), Cause: err}}
	}

	reqCtx, cancel := c.contextWithRequestTimeout(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.baseURL, strings.NewReader(string(jsonBody)))
	if err != nil {
		return "", &NetworkError{PaddleOCRAPIError{Message: err.Error(), Cause: err}}
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if err := raiseForResponse(resp); err != nil {
		return "", err
	}
	return decodeSubmitResponse(resp.Body)
}

func (c *Client) submitFile(ctx context.Context, model Model, filePath string, payload interface{}, pageRanges, batchID string) (string, error) {
	if _, err := os.Stat(filePath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", &FileNotFoundError{Path: filePath, PaddleOCRAPIError: PaddleOCRAPIError{Message: "File not found: " + filePath, Cause: err}}
		}
		return "", err
	}

	pr, pw := io.Pipe()
	w := multipart.NewWriter(pw)
	reqCtx, cancel := c.contextWithRequestTimeout(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.baseURL, pr)
	if err != nil {
		_ = pr.Close()
		_ = pw.Close()
		return "", &NetworkError{PaddleOCRAPIError{Message: err.Error(), Cause: err}}
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", w.FormDataContentType())

	errc := make(chan error, 1)
	go func() {
		defer close(errc)
		defer pw.Close()

		if err := w.WriteField("model", string(model)); err != nil {
			errc <- err
			_ = pw.CloseWithError(err)
			return
		}
		payloadJSON, err := json.Marshal(payload)
		if err != nil {
			invalidErr := &InvalidRequestError{PaddleOCRAPIError{Message: "encode request payload: " + err.Error(), Cause: err}}
			errc <- invalidErr
			_ = pw.CloseWithError(invalidErr)
			return
		}
		if err := w.WriteField("optionalPayload", string(payloadJSON)); err != nil {
			errc <- err
			_ = pw.CloseWithError(err)
			return
		}
		if pageRanges != "" {
			if err := w.WriteField("pageRanges", pageRanges); err != nil {
				errc <- err
				_ = pw.CloseWithError(err)
				return
			}
		}
		if batchID != "" {
			if err := w.WriteField("batchId", batchID); err != nil {
				errc <- err
				_ = pw.CloseWithError(err)
				return
			}
		}

		file, err := os.Open(filePath)
		if err != nil {
			errc <- err
			_ = pw.CloseWithError(err)
			return
		}
		defer file.Close()

		fw, err := w.CreateFormFile("file", filepath.Base(filePath))
		if err != nil {
			errc <- err
			_ = pw.CloseWithError(err)
			return
		}
		if _, err := io.Copy(fw, file); err != nil {
			errc <- err
			_ = pw.CloseWithError(err)
			return
		}
		if err := w.Close(); err != nil {
			errc <- err
			_ = pw.CloseWithError(err)
			return
		}
		errc <- nil
	}()

	resp, err := c.do(req)
	_ = pr.CloseWithError(err)
	writeErr := <-errc
	if writeErr != nil {
		var invalidErr *InvalidRequestError
		if errors.As(writeErr, &invalidErr) {
			return "", writeErr
		}
		if err == nil {
			err = writeErr
		}
	}
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if err := raiseForResponse(resp); err != nil {
		return "", err
	}
	return decodeSubmitResponse(resp.Body)
}

func (c *Client) GetStatus(ctx context.Context, jobID string) (*JobStatus, error) {
	if jobID == "" {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "jobID is required"}}
	}
	status, err := c.getJobStatus(ctx, jobID)
	if err != nil {
		return nil, err
	}
	return convertStatus(jobID, "", "", status)
}

func (c *Client) getJobStatus(ctx context.Context, jobID string) (*jobStatusResponse, error) {
	reqCtx, cancel := c.contextWithRequestTimeout(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, c.baseURL+"/"+jobID, nil)
	if err != nil {
		return nil, &NetworkError{PaddleOCRAPIError{Message: err.Error(), Cause: err}}
	}
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if err := raiseForResponse(resp); err != nil {
		return nil, err
	}

	var apiResp apiResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "decode status response: " + err.Error(), Cause: err}}
	}
	if len(apiResp.Data) == 0 || string(apiResp.Data) == "null" {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "status response missing data"}}
	}
	var status jobStatusResponse
	if err := json.Unmarshal(apiResp.Data, &status); err != nil {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "decode status data: " + err.Error(), Cause: err}}
	}
	if err := validateState(status.State); err != nil {
		return nil, err
	}
	return &status, nil
}

func (c *Client) fetchJSONL(ctx context.Context, url string) ([]map[string]interface{}, error) {
	reqCtx, cancel := c.contextWithRequestTimeout(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return nil, &NetworkError{PaddleOCRAPIError{Message: err.Error(), Cause: err}}
	}
	resp, err := c.do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if err := raiseForResponse(resp); err != nil {
		return nil, err
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, mapBodyReadError("read result payload", err)
	}
	lines := strings.Split(strings.TrimSpace(string(body)), "\n")
	var results []map[string]interface{}
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "parse JSONL result: " + err.Error(), Cause: err}}
		}
		results = append(results, obj)
	}
	if results == nil {
		return nil, &ResultParseError{PaddleOCRAPIError{Message: "result payload is empty"}}
	}
	return results, nil
}

func (c *Client) pollUntilDone(ctx context.Context, job *Job) ([]map[string]interface{}, error) {
	start := time.Now()
	deadline := start.Add(c.pollTimeout)
	interval := c.pollInterval

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if time.Now().After(deadline) {
			return nil, &PollTimeoutError{
				JobID:             job.JobID,
				Elapsed:           time.Since(start).Seconds(),
				PaddleOCRAPIError: PaddleOCRAPIError{Message: fmt.Sprintf("Timed out after %.1fs", time.Since(start).Seconds())},
			}
		}

		statusCtx, cancel := context.WithDeadline(ctx, minTime(deadline, time.Now().Add(c.requestTimeout)))
		status, err := c.getJobStatus(statusCtx, job.JobID)
		cancel()
		if err != nil {
			if errors.Is(err, context.DeadlineExceeded) && time.Now().After(deadline) {
				return nil, &PollTimeoutError{JobID: job.JobID, Elapsed: time.Since(start).Seconds(), PaddleOCRAPIError: PaddleOCRAPIError{Message: "poll timeout exceeded", Cause: err}}
			}
			return nil, err
		}

		switch status.State {
		case JobStateDone:
			if status.ResultURL == nil || status.ResultURL.JSONURL == "" {
				return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "done job missing result URL"}}
			}
			fetchCtx, cancel := context.WithDeadline(ctx, minTime(deadline, time.Now().Add(c.requestTimeout)))
			result, err := c.fetchJSONL(fetchCtx, status.ResultURL.JSONURL)
			cancel()
			if err != nil {
				if errors.Is(err, context.DeadlineExceeded) && time.Now().After(deadline) {
					return nil, &PollTimeoutError{JobID: job.JobID, Elapsed: time.Since(start).Seconds(), PaddleOCRAPIError: PaddleOCRAPIError{Message: "poll timeout exceeded", Cause: err}}
				}
				return nil, err
			}
			return result, nil
		case JobStateFailed:
			return nil, &JobFailedError{
				JobID:             job.JobID,
				ErrorMsg:          status.ErrorMsg,
				PaddleOCRAPIError: PaddleOCRAPIError{Message: status.ErrorMsg},
			}
		case JobStatePending, JobStateRunning:
		default:
			return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "unknown job state: " + string(status.State)}}
		}

		wait := interval
		remaining := time.Until(deadline)
		if wait > remaining {
			wait = remaining
		}
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return nil, ctx.Err()
		case <-timer.C:
		}
		next := time.Duration(float64(interval) * multiplier)
		if next > maxInterval {
			next = maxInterval
		}
		interval = next
	}
}

func decodeSubmitResponse(r io.Reader) (string, error) {
	var apiResp apiResponse
	if err := json.NewDecoder(r).Decode(&apiResp); err != nil {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "decode submit response: " + err.Error(), Cause: err}}
	}
	if len(apiResp.Data) == 0 || string(apiResp.Data) == "null" {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "submit response missing data"}}
	}
	var sr submitResponse
	if err := json.Unmarshal(apiResp.Data, &sr); err != nil {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "decode submit data: " + err.Error(), Cause: err}}
	}
	if sr.JobID == "" {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "submit response missing jobId"}}
	}
	return sr.JobID, nil
}

func convertStatus(jobID string, model Model, task Task, status *jobStatusResponse) (*JobStatus, error) {
	if status.State == "" {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "status response missing state"}}
	}
	if err := validateState(status.State); err != nil {
		return nil, err
	}
	js := &JobStatus{
		JobID:    jobID,
		State:    status.State,
		Model:    model,
		Task:     task,
		ErrorMsg: status.ErrorMsg,
	}
	if status.ExtractProgress != nil {
		var ep extractProgress
		if err := json.Unmarshal(status.ExtractProgress, &ep); err != nil {
			return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "decode progress: " + err.Error(), Cause: err}}
		}
		js.Progress = &Progress{
			TotalPages:     ep.TotalPages,
			ExtractedPages: ep.ExtractedPages,
			StartTime:      ep.StartTime,
			EndTime:        ep.EndTime,
		}
	}
	return js, nil
}

func validateState(state JobState) error {
	switch state {
	case JobStatePending, JobStateRunning, JobStateDone, JobStateFailed:
		return nil
	default:
		return &ResponseFormatError{PaddleOCRAPIError{Message: "unknown job state: " + string(state)}}
	}
}

func raiseForResponse(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	body, _ := io.ReadAll(resp.Body)
	msg := strings.TrimSpace(string(body))

	switch {
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return &AuthError{PaddleOCRAPIError{Message: "Authentication failed: " + msg}}
	case resp.StatusCode == http.StatusBadRequest:
		return &InvalidRequestError{PaddleOCRAPIError{Message: "Bad request: " + msg}}
	default:
		return &APIError{StatusCode: resp.StatusCode, PaddleOCRAPIError: PaddleOCRAPIError{Message: msg}}
	}
}

func (c *Client) do(req *http.Request) (*http.Response, error) {
	resp, err := c.httpClient.Do(req)
	if err == nil {
		return resp, nil
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return nil, &RequestTimeoutError{PaddleOCRAPIError{Message: "request timeout exceeded", Cause: err}}
	}
	if errors.Is(req.Context().Err(), context.DeadlineExceeded) {
		return nil, &RequestTimeoutError{PaddleOCRAPIError{Message: "request timeout exceeded", Cause: err}}
	}
	if errors.Is(req.Context().Err(), context.Canceled) {
		return nil, req.Context().Err()
	}
	return nil, &NetworkError{PaddleOCRAPIError{Message: err.Error(), Cause: err}}
}

func mapBodyReadError(operation string, err error) error {
	if isTimeoutError(err) {
		return &RequestTimeoutError{PaddleOCRAPIError{Message: operation + ": request timeout exceeded", Cause: err}}
	}
	return &NetworkError{PaddleOCRAPIError{Message: operation + ": " + err.Error(), Cause: err}}
}

func isTimeoutError(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var timeout interface{ Timeout() bool }
	return errors.As(err, &timeout) && timeout.Timeout()
}

func (c *Client) contextWithRequestTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if c.requestTimeout <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, c.requestTimeout)
}

func minTime(a, b time.Time) time.Time {
	if a.Before(b) {
		return a
	}
	return b
}
