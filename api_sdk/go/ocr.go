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
	"os"
)

// OCR submits an OCR job, waits for completion, and returns the parsed OCR result.
func (c *Client) OCR(ctx context.Context, req *OCRRequest) (*OCRResult, error) {
	op, err := c.SubmitOCR(ctx, req)
	if err != nil {
		return nil, err
	}
	return op.WaitOCR(ctx)
}

// ParseDocument submits a document parsing job, waits for completion, and
// returns the parsed document parsing result.
func (c *Client) ParseDocument(ctx context.Context, req *DocParsingRequest) (*DocParsingResult, error) {
	op, err := c.SubmitDocumentParsing(ctx, req)
	if err != nil {
		return nil, err
	}
	return op.WaitDocumentParsing(ctx)
}

// SubmitOCR submits an OCR job and returns an Operation for status checks or a
// later typed wait.
func (c *Client) SubmitOCR(ctx context.Context, req *OCRRequest) (*Operation, error) {
	if req == nil {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "OCR request is nil"}}
	}
	model := req.Model
	if model == "" {
		model = PPOCRv5
	}
	if err := validateOCRModel(model); err != nil {
		return nil, err
	}
	jobID, err := c.submit(ctx, model, req.FileURL, req.FilePath, req.Options, req.PageRanges, req.BatchID)
	if err != nil {
		return nil, err
	}
	return &Operation{client: c, Job: Job{JobID: jobID, Model: model, Task: TaskOCR}}, nil
}

// SubmitDocumentParsing submits a document parsing job and returns an Operation
// for status checks or a later typed wait.
func (c *Client) SubmitDocumentParsing(ctx context.Context, req *DocParsingRequest) (*Operation, error) {
	if req == nil {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "document parsing request is nil"}}
	}
	model := req.Model
	if model == "" {
		model = PaddleOCRVL15
	}
	if err := validateDocumentParsingModel(model); err != nil {
		return nil, err
	}
	jobID, err := c.submit(ctx, model, req.FileURL, req.FilePath, req.Options, req.PageRanges, req.BatchID)
	if err != nil {
		return nil, err
	}
	return &Operation{client: c, Job: Job{JobID: jobID, Model: model, Task: TaskDocumentParsing}}, nil
}

// WaitOCRResult waits for an OCR job and parses OCR result JSONL.
func (c *Client) WaitOCRResult(ctx context.Context, job *Job) (*OCRResult, error) {
	if err := validateWaitJob(job, TaskOCR); err != nil {
		return nil, err
	}
	jsonlData, err := c.pollUntilDone(ctx, job)
	if err != nil {
		return nil, err
	}
	return parseOCRResult(job.JobID, jsonlData)
}

// WaitDocumentParsingResult waits for a document parsing job and parses
// document parsing result JSONL.
func (c *Client) WaitDocumentParsingResult(ctx context.Context, job *Job) (*DocParsingResult, error) {
	if err := validateWaitJob(job, TaskDocumentParsing); err != nil {
		return nil, err
	}
	jsonlData, err := c.pollUntilDone(ctx, job)
	if err != nil {
		return nil, err
	}
	return parseDocParsingResult(job.JobID, jsonlData)
}

func (c *Client) submit(ctx context.Context, model Model, fileURL, filePath string, options interface{}, pageRanges, batchID string) (string, error) {
	if fileURL == "" && filePath == "" {
		return "", &InvalidRequestError{PaddleOCRAPIError{Message: "Either FileURL or FilePath is required."}}
	}
	if filePath != "" {
		if _, err := os.Stat(filePath); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return "", &FileNotFoundError{Path: filePath, PaddleOCRAPIError: PaddleOCRAPIError{Message: "File not found: " + filePath, Cause: err}}
			}
			return "", err
		}
	}
	if fileURL != "" && filePath != "" {
		return "", &InvalidRequestError{PaddleOCRAPIError{Message: "FileURL and FilePath are mutually exclusive."}}
	}

	payload := defaultPayload(model, options)
	if fileURL != "" {
		return c.submitURL(ctx, model, fileURL, payload, pageRanges, batchID)
	}
	return c.submitFile(ctx, model, filePath, payload, pageRanges, batchID)
}

func defaultPayload(model Model, options interface{}) interface{} {
	if options != nil {
		return options
	}
	if IsOCRModel(model) {
		return &OCROptions{}
	}
	return &DocParsingOptions{}
}

func validateOCRModel(model Model) error {
	if IsOCRModel(model) {
		return nil
	}
	return &InvalidRequestError{PaddleOCRAPIError{Message: "unsupported OCR model: " + string(model)}}
}

func validateDocumentParsingModel(model Model) error {
	if IsDocumentParsingModel(model) {
		return nil
	}
	if IsOCRModel(model) {
		return &InvalidRequestError{PaddleOCRAPIError{Message: fmt.Sprintf("%s is an OCR model and cannot be used for document parsing", model)}}
	}
	return &InvalidRequestError{PaddleOCRAPIError{Message: "unsupported document parsing model: " + string(model)}}
}

func validateWaitJob(job *Job, expected Task) error {
	if job == nil {
		return &InvalidRequestError{PaddleOCRAPIError{Message: "job is nil"}}
	}
	if job.JobID == "" {
		return &InvalidRequestError{PaddleOCRAPIError{Message: "jobID is required"}}
	}
	if job.Task != expected {
		return &InvalidRequestError{PaddleOCRAPIError{Message: fmt.Sprintf("job task %q cannot be used with %q wait", job.Task, expected)}}
	}
	if expected == TaskOCR && !IsOCRModel(job.Model) {
		return &InvalidRequestError{PaddleOCRAPIError{Message: "OCR wait requires an OCR model"}}
	}
	if expected == TaskDocumentParsing {
		return validateDocumentParsingModel(job.Model)
	}
	return nil
}

func parseOCRResult(jobID string, jsonlData []map[string]interface{}) (*OCRResult, error) {
	result := &OCRResult{JobID: jobID}
	for _, lineObj := range jsonlData {
		resultData, ok := lineObj["result"].(map[string]interface{})
		if !ok {
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "OCR result missing result object"}}
		}
		ocrResults, ok := resultData["ocrResults"].([]interface{})
		if !ok {
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "OCR result missing ocrResults"}}
		}
		for _, item := range ocrResults {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				return nil, &ResultParseError{PaddleOCRAPIError{Message: "OCR result page is not an object"}}
			}
			prunedResult, ok := itemMap["prunedResult"]
			if !ok {
				return nil, &ResultParseError{PaddleOCRAPIError{Message: "OCR result page missing prunedResult"}}
			}
			page := OCRPage{
				PrunedResult: prunedResult,
				OCRImageURL:  getString(itemMap, "ocrImage"),
			}
			result.Pages = append(result.Pages, page)
		}
	}
	if len(result.Pages) == 0 {
		return nil, &ResultParseError{PaddleOCRAPIError{Message: "OCR result contains no pages"}}
	}
	return result, nil
}

func parseDocParsingResult(jobID string, jsonlData []map[string]interface{}) (*DocParsingResult, error) {
	result := &DocParsingResult{JobID: jobID}
	for _, lineObj := range jsonlData {
		resultData, ok := lineObj["result"].(map[string]interface{})
		if !ok {
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing result missing result object"}}
		}
		lpResults, ok := resultData["layoutParsingResults"].([]interface{})
		if !ok {
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing result missing layoutParsingResults"}}
		}
		for _, item := range lpResults {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing page is not an object"}}
			}
			markdown, ok := itemMap["markdown"].(map[string]interface{})
			if !ok {
				return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing page missing markdown object"}}
			}
			markdownText, ok := markdown["text"].(string)
			if !ok {
				return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing markdown missing text"}}
			}
			page := DocParsingPage{
				MarkdownText:   markdownText,
				MarkdownImages: getStringMap(markdown, "images"),
				OutputImages:   getStringMap(itemMap, "outputImages"),
			}
			result.Pages = append(result.Pages, page)
		}
	}
	if len(result.Pages) == 0 {
		return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing result contains no pages"}}
	}
	return result, nil
}

func getString(m map[string]interface{}, key string) string {
	if m == nil {
		return ""
	}
	v, _ := m[key].(string)
	return v
}

func getStringMap(m map[string]interface{}, key string) map[string]string {
	if m == nil {
		return nil
	}
	raw, ok := m[key]
	if !ok {
		return nil
	}
	switch v := raw.(type) {
	case map[string]interface{}:
		result := make(map[string]string, len(v))
		for k, val := range v {
			if s, ok := val.(string); ok {
				result[k] = s
			}
		}
		return result
	case map[string]string:
		return v
	default:
		b, err := json.Marshal(raw)
		if err != nil {
			return nil
		}
		var result map[string]string
		if err := json.Unmarshal(b, &result); err != nil {
			return nil
		}
		return result
	}
}
