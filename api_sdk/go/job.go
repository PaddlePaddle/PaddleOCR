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

import (
	"bytes"
	"context"
	"encoding/json"
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
	maxWaitTime     = 600 * time.Second
)

type apiResponse struct {
	Data json.RawMessage `json:"data"`
}

type submitResponse struct {
	JobID string `json:"jobId"`
}

type jobStatusResponse struct {
	State           string          `json:"state"`
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

func (c *Client) submitURL(model, fileURL string, payload interface{}) (string, error) {
	body := map[string]interface{}{
		"fileUrl":         fileURL,
		"model":           model,
		"optionalPayload": payload,
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", c.baseURL, bytes.NewReader(jsonBody))
	if err != nil {
		return "", &NetworkError{PaddleOCRError{Message: err.Error()}}
	}
	req.Header.Set("Authorization", "bearer "+c.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", &NetworkError{PaddleOCRError{Message: err.Error()}}
	}
	defer resp.Body.Close()

	if err := raiseForResponse(resp); err != nil {
		return "", err
	}

	var apiResp apiResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return "", err
	}
	var sr submitResponse
	if err := json.Unmarshal(apiResp.Data, &sr); err != nil {
		return "", err
	}
	return sr.JobID, nil
}

func (c *Client) submitFile(model, filePath string, payload interface{}) (string, error) {
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return "", &FileNotFoundError{Path: filePath, PaddleOCRError: PaddleOCRError{Message: "File not found: " + filePath}}
	}

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	_ = w.WriteField("model", model)
	payloadJSON, _ := json.Marshal(payload)
	_ = w.WriteField("optionalPayload", string(payloadJSON))

	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	fw, err := w.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(fw, file); err != nil {
		return "", err
	}
	w.Close()

	req, err := http.NewRequest("POST", c.baseURL, &buf)
	if err != nil {
		return "", &NetworkError{PaddleOCRError{Message: err.Error()}}
	}
	req.Header.Set("Authorization", "bearer "+c.token)
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", &NetworkError{PaddleOCRError{Message: err.Error()}}
	}
	defer resp.Body.Close()

	if err := raiseForResponse(resp); err != nil {
		return "", err
	}

	var apiResp apiResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return "", err
	}
	var sr submitResponse
	if err := json.Unmarshal(apiResp.Data, &sr); err != nil {
		return "", err
	}
	return sr.JobID, nil
}

func (c *Client) getJobStatus(jobID string) (*jobStatusResponse, error) {
	req, err := http.NewRequest("GET", c.baseURL+"/"+jobID, nil)
	if err != nil {
		return nil, &NetworkError{PaddleOCRError{Message: err.Error()}}
	}
	req.Header.Set("Authorization", "bearer "+c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, &NetworkError{PaddleOCRError{Message: err.Error()}}
	}
	defer resp.Body.Close()

	if err := raiseForResponse(resp); err != nil {
		return nil, err
	}

	var apiResp apiResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, err
	}
	var status jobStatusResponse
	if err := json.Unmarshal(apiResp.Data, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

func (c *Client) fetchJSONL(url string) ([]map[string]interface{}, error) {
	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, &NetworkError{PaddleOCRError{Message: err.Error()}}
	}
	defer resp.Body.Close()

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
			return nil, err
		}
		results = append(results, obj)
	}
	return results, nil
}

func (c *Client) pollUntilDone(ctx context.Context, jobID string) ([]map[string]interface{}, error) {
	interval := initialInterval
	var elapsed time.Duration

	for elapsed < maxWaitTime {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(interval):
		}
		elapsed += interval

		status, err := c.getJobStatus(jobID)
		if err != nil {
			return nil, err
		}

		switch status.State {
		case "done":
			if status.ResultURL == nil {
				return nil, fmt.Errorf("job done but no result URL")
			}
			return c.fetchJSONL(status.ResultURL.JSONURL)
		case "failed":
			return nil, &JobFailedError{
				JobID:          jobID,
				ErrorMsg:       status.ErrorMsg,
				PaddleOCRError: PaddleOCRError{Message: status.ErrorMsg},
			}
		}

		next := time.Duration(float64(interval) * multiplier)
		if next > maxInterval {
			next = maxInterval
		}
		interval = next
	}

	return nil, &TimeoutError{
		JobID:          jobID,
		Elapsed:        elapsed.Seconds(),
		PaddleOCRError: PaddleOCRError{Message: fmt.Sprintf("Timed out after %.1fs", elapsed.Seconds())},
	}
}

func raiseForResponse(resp *http.Response) error {
	if resp.StatusCode == 200 {
		return nil
	}
	body, _ := io.ReadAll(resp.Body)
	msg := string(body)

	switch {
	case resp.StatusCode == 401 || resp.StatusCode == 403:
		return &AuthError{PaddleOCRError{Message: "Authentication failed: " + msg}}
	case resp.StatusCode == 400:
		return &InvalidRequestError{PaddleOCRError{Message: "Bad request: " + msg}}
	default:
		return &APIError{StatusCode: resp.StatusCode, PaddleOCRError: PaddleOCRError{Message: msg}}
	}
}
