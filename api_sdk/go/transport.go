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
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type apiResponse struct {
	Code int             `json:"code"`
	Msg  string          `json:"msg"`
	Data json.RawMessage `json:"data"`
}

type submitResponse struct {
	JobID string `json:"jobId"`
}

type jobStatusResponse struct {
	JobID           string            `json:"jobId"`
	State           string            `json:"state"`
	ExtractProgress json.RawMessage   `json:"extractProgress"`
	ResultURL       map[string]string `json:"resultUrl"`
	ErrorMsg        string            `json:"errorMsg"`
}

type extractProgress struct {
	TotalPages     int    `json:"totalPages"`
	ExtractedPages int    `json:"extractedPages"`
	StartTime      string `json:"startTime"`
	EndTime        string `json:"endTime"`
}

func (c *Client) submitURL(ctx context.Context, model, fileURL string, payload interface{}, pageRanges, batchID string) (string, error) {
	body := map[string]interface{}{
		"fileUrl":         fileURL,
		"model":           model,
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
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "failed to encode submit request", Cause: err}}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.jobsURL, bytes.NewReader(jsonBody))
	if err != nil {
		return "", &NetworkError{PaddleOCRAPIError{Message: err.Error()}}
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", "application/json")
	c.setClientPlatformHeader(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", classifyHTTPError(err)
	}
	defer resp.Body.Close()

	if err := raiseForResponse(resp); err != nil {
		return "", err
	}

	apiResp, err := decodeAPIResponse(resp)
	if err != nil {
		return "", err
	}
	var sr submitResponse
	if err := json.Unmarshal(apiResp.Data, &sr); err != nil {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "submit response is malformed", Cause: err}}
	}
	if sr.JobID == "" {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "submit response is missing jobId"}}
	}
	return sr.JobID, nil
}

func (c *Client) submitFile(ctx context.Context, model, filePath string, payload interface{}, pageRanges, batchID string) (string, error) {
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return "", &FileNotFoundError{Path: filePath, PaddleOCRAPIError: PaddleOCRAPIError{Message: "File not found: " + filePath}}
	}

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	_ = w.WriteField("model", model)
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "failed to encode optionalPayload", Cause: err}}
	}
	if err := w.WriteField("optionalPayload", string(payloadJSON)); err != nil {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "failed to prepare multipart body", Cause: err}}
	}
	if pageRanges != "" {
		_ = w.WriteField("pageRanges", pageRanges)
	}
	if batchID != "" {
		_ = w.WriteField("batchId", batchID)
	}

	file, err := os.Open(filePath)
	if err != nil {
		return "", &FileNotFoundError{Path: filePath, PaddleOCRAPIError: PaddleOCRAPIError{Message: "Failed to open file: " + filePath, Cause: err}}
	}
	defer file.Close()

	fw, err := w.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "failed to prepare multipart body", Cause: err}}
	}
	if _, err := io.Copy(fw, file); err != nil {
		return "", &NetworkError{PaddleOCRAPIError{Message: "failed to read file content", Cause: err}}
	}
	if err := w.Close(); err != nil {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "failed to finalize multipart body", Cause: err}}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.jobsURL, &buf)
	if err != nil {
		return "", &NetworkError{PaddleOCRAPIError{Message: err.Error()}}
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", w.FormDataContentType())
	c.setClientPlatformHeader(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", classifyHTTPError(err)
	}
	defer resp.Body.Close()

	if err := raiseForResponse(resp); err != nil {
		return "", err
	}

	apiResp, err := decodeAPIResponse(resp)
	if err != nil {
		return "", err
	}
	var sr submitResponse
	if err := json.Unmarshal(apiResp.Data, &sr); err != nil {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "submit response is malformed", Cause: err}}
	}
	if sr.JobID == "" {
		return "", &ResponseFormatError{PaddleOCRAPIError{Message: "submit response is missing jobId"}}
	}
	return sr.JobID, nil
}

func (c *Client) getJobStatus(ctx context.Context, jobID string) (*jobStatusResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.jobsURL+"/"+jobID, nil)
	if err != nil {
		return nil, &NetworkError{PaddleOCRAPIError{Message: err.Error()}}
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	c.setClientPlatformHeader(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, classifyHTTPError(err)
	}
	defer resp.Body.Close()

	if err := raiseForResponse(resp); err != nil {
		return nil, err
	}

	apiResp, err := decodeAPIResponse(resp)
	if err != nil {
		return nil, err
	}
	var status jobStatusResponse
	if err := json.Unmarshal(apiResp.Data, &status); err != nil {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "status response is malformed", Cause: err}}
	}
	if status.State == "" {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "status response is missing state"}}
	}
	return &status, nil
}

func (c *Client) getBatchStatus(ctx context.Context, batchID string) (*BatchStatus, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.jobsURL+"/batch/"+batchID, nil)
	if err != nil {
		return nil, &NetworkError{PaddleOCRAPIError{Message: err.Error(), Cause: err}}
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	c.setClientPlatformHeader(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, classifyHTTPError(err)
	}
	defer resp.Body.Close()

	if err := raiseForResponse(resp); err != nil {
		return nil, err
	}
	apiResp, err := decodeAPIResponse(resp)
	if err != nil {
		return nil, err
	}
	var payload struct {
		ExtractResult []jobStatusResponse `json:"extractResult"`
	}
	if err := json.Unmarshal(apiResp.Data, &payload); err != nil {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "batch response is malformed", Cause: err}}
	}
	result := &BatchStatus{BatchID: batchID}
	for _, item := range payload.ExtractResult {
		jobID := item.JobID
		if jobID == "" {
			return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "batch response item is missing jobId"}}
		}
		status, err := normalizeStatus(jobID, &item)
		if err != nil {
			return nil, err
		}
		result.Jobs = append(result.Jobs, status)
	}
	return result, nil
}

func (c *Client) fetchJSONL(ctx context.Context, url string) ([]map[string]interface{}, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, &NetworkError{PaddleOCRAPIError{Message: err.Error()}}
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, classifyHTTPError(err)
	}
	defer resp.Body.Close()
	if err := raiseForResponse(resp); err != nil {
		return nil, err
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
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
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "failed to parse JSONL result payload", Cause: err}}
		}
		results = append(results, obj)
	}
	return results, nil
}

func raiseForResponse(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	body, _ := io.ReadAll(resp.Body)
	msg := string(body)

	switch {
	case resp.StatusCode == 401 || resp.StatusCode == 403:
		return &AuthError{PaddleOCRAPIError{Message: "Authentication failed: " + msg}}
	case resp.StatusCode == 400:
		return &InvalidRequestError{PaddleOCRAPIError{Message: "Bad request: " + msg}}
	case resp.StatusCode == 429:
		return &RateLimitError{APIError{StatusCode: 429, PaddleOCRAPIError: PaddleOCRAPIError{Message: "Rate limit exceeded: " + msg}}}
	case resp.StatusCode == 503 || resp.StatusCode == 504:
		return &ServiceUnavailableError{APIError{StatusCode: resp.StatusCode, PaddleOCRAPIError: PaddleOCRAPIError{Message: "Service unavailable: " + msg}}}
	default:
		return &APIError{StatusCode: resp.StatusCode, PaddleOCRAPIError: PaddleOCRAPIError{Message: msg}}
	}
}

func decodeAPIResponse(resp *http.Response) (*apiResponse, error) {
	var apiResp apiResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "expected a JSON response body", Cause: err}}
	}
	if apiResp.Code != 0 {
		msg := apiResp.Msg
		if msg == "" {
			msg = "PaddleOCR official API request failed"
		}
		return nil, &APIError{StatusCode: resp.StatusCode, PaddleOCRAPIError: PaddleOCRAPIError{Message: msg}}
	}
	if len(apiResp.Data) == 0 || string(apiResp.Data) == "null" {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "response body is missing data"}}
	}
	return &apiResp, nil
}

func classifyHTTPError(err error) error {
	if ne, ok := err.(net.Error); ok && ne.Timeout() {
		return &RequestTimeoutError{PaddleOCRAPIError: PaddleOCRAPIError{Message: err.Error(), Cause: err}}
	}
	return &NetworkError{PaddleOCRAPIError{Message: err.Error(), Cause: err}}
}
