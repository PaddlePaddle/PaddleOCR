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

func (c *Client) OCR(ctx context.Context, req *OCRRequest) (*OCRResult, error) {
	job, err := c.SubmitOCR(ctx, req)
	if err != nil {
		return nil, err
	}
	return c.WaitOCRResult(ctx, job.JobID)
}

// ParseDocument performs document parsing. Blocks until result is ready.
func (c *Client) ParseDocument(ctx context.Context, req *DocParsingRequest) (*DocParsingResult, error) {
	job, err := c.SubmitDocumentParsing(ctx, req)
	if err != nil {
		return nil, err
	}
	return c.WaitDocumentParsingResult(ctx, job.JobID)
}

// SubmitOCR submits an OCR job and returns job metadata for tracking.
func (c *Client) SubmitOCR(ctx context.Context, req *OCRRequest) (*Job, error) {
	if req == nil {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "OCR request is nil"}}
	}
	model := req.Model
	if model == "" {
		model = PPOCRv6
	}
	if !IsOCRModel(model) {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "model is not an OCR model: " + model}}
	}
	jobID, err := c.submit(ctx, model, req.FileURL, req.FilePath, req.Options, req.PageRanges, req.BatchID)
	if err != nil {
		return nil, err
	}
	return &Job{JobID: jobID, Model: model, Task: "ocr", PageRanges: req.PageRanges, BatchID: req.BatchID}, nil
}

// SubmitDocumentParsing submits a document parsing job and returns job metadata for tracking.
func (c *Client) SubmitDocumentParsing(ctx context.Context, req *DocParsingRequest) (*Job, error) {
	if req == nil {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "document parsing request is nil"}}
	}
	model := req.Model
	if model == "" {
		model = PaddleOCRVL16
	}
	if !IsDocumentParsingModel(model) {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "model is not a document parsing model: " + model}}
	}
	jobID, err := c.submit(ctx, model, req.FileURL, req.FilePath, req.Options, req.PageRanges, req.BatchID)
	if err != nil {
		return nil, err
	}
	return &Job{JobID: jobID, Model: model, Task: "document_parsing", PageRanges: req.PageRanges, BatchID: req.BatchID}, nil
}

func (c *Client) WaitOCRResult(ctx context.Context, jobID string) (*OCRResult, error) {
	jsonlData, err := c.pollUntilDone(ctx, jobID)
	if err != nil {
		return nil, err
	}
	return parseOCRResult(jobID, jsonlData)
}

func (c *Client) WaitDocumentParsingResult(ctx context.Context, jobID string) (*DocParsingResult, error) {
	jsonlData, err := c.pollUntilDone(ctx, jobID)
	if err != nil {
		return nil, err
	}
	return parseDocParsingResult(jobID, jsonlData)
}

func (c *Client) GetStatus(ctx context.Context, jobID string) (*JobStatus, error) {
	status, err := c.getJobStatus(ctx, jobID)
	if err != nil {
		return nil, err
	}
	return normalizeStatus(jobID, status)
}

func (c *Client) GetBatchStatus(ctx context.Context, batchID string) (*BatchStatus, error) {
	if batchID == "" {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "batchID is required"}}
	}
	return c.getBatchStatus(ctx, batchID)
}

func (c *Client) submit(ctx context.Context, model, fileURL, filePath string, options interface{}, pageRanges, batchID string) (string, error) {
	if fileURL == "" && filePath == "" {
		return "", &InvalidRequestError{PaddleOCRAPIError{Message: "Either FileURL or FilePath is required."}}
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

func defaultPayload(model string, options interface{}) interface{} {
	switch typed := options.(type) {
	case *OCROptions:
		if typed != nil {
			return payloadWithExtraOptions(typed)
		}
	case *PPStructureV3Options:
		if typed != nil {
			return payloadWithExtraOptions(typed)
		}
	case *PaddleOCRVLOptions:
		if typed != nil {
			return payloadWithExtraOptions(typed)
		}
	default:
		if options != nil {
			return payloadWithExtraOptions(options)
		}
	}
	if IsOCRModel(model) {
		return payloadWithExtraOptions(&OCROptions{})
	}
	if IsVLModel(model) {
		return payloadWithExtraOptions(&PaddleOCRVLOptions{})
	}
	return payloadWithExtraOptions(&PPStructureV3Options{})
}

func payloadWithExtraOptions(options interface{}) interface{} {
	payloadBytes, _ := json.Marshal(options)
	payload := map[string]interface{}{}
	_ = json.Unmarshal(payloadBytes, &payload)

	var extraOptions map[string]interface{}
	switch typed := options.(type) {
	case *OCROptions:
		extraOptions = typed.ExtraOptions
	case *PPStructureV3Options:
		extraOptions = typed.ExtraOptions
	case *PaddleOCRVLOptions:
		extraOptions = typed.ExtraOptions
	}
	for key, value := range extraOptions {
		payload[key] = value
	}
	return payload
}

func parseOCRResult(jobID string, jsonlData []map[string]interface{}) (*OCRResult, error) {
	result := &OCRResult{JobID: jobID, DataInfo: map[string]interface{}{}}
	for _, lineObj := range jsonlData {
		resultData, ok := lineObj["result"].(map[string]interface{})
		if !ok {
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "OCR result item is missing result"}}
		}
		if dataInfo, ok := resultData["dataInfo"].(map[string]interface{}); ok {
			for key, value := range dataInfo {
				result.DataInfo[key] = value
			}
		}
		ocrResults, ok := resultData["ocrResults"].([]interface{})
		if !ok {
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "OCR result item is missing ocrResults"}}
		}
		for _, item := range ocrResults {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				return nil, &ResultParseError{PaddleOCRAPIError{Message: "OCR result page is malformed"}}
			}
			if _, ok := itemMap["prunedResult"]; !ok {
				return nil, &ResultParseError{PaddleOCRAPIError{Message: "OCR result page is missing prunedResult"}}
			}
			page := OCRPage{
				PrunedResult:             itemMap["prunedResult"],
				OCRImageURL:              getString(itemMap, "ocrImage"),
				DocPreprocessingImageURL: getString(itemMap, "docPreprocessingImage"),
				InputImageURL:            getString(itemMap, "inputImage"),
				Raw:                      itemMap,
			}
			result.Pages = append(result.Pages, page)
		}
	}
	return result, nil
}

func parseDocParsingResult(jobID string, jsonlData []map[string]interface{}) (*DocParsingResult, error) {
	result := &DocParsingResult{JobID: jobID, DataInfo: map[string]interface{}{}}
	for _, lineObj := range jsonlData {
		resultData, ok := lineObj["result"].(map[string]interface{})
		if !ok {
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing result item is missing result"}}
		}
		if dataInfo, ok := resultData["dataInfo"].(map[string]interface{}); ok {
			for key, value := range dataInfo {
				result.DataInfo[key] = value
			}
		}
		lpResults, ok := resultData["layoutParsingResults"].([]interface{})
		if !ok {
			return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing result item is missing layoutParsingResults"}}
		}
		for _, item := range lpResults {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing result page is malformed"}}
			}
			markdown, ok := itemMap["markdown"].(map[string]interface{})
			if !ok || getString(markdown, "text") == "" {
				return nil, &ResultParseError{PaddleOCRAPIError{Message: "document parsing result page is missing markdown.text"}}
			}
			page := DocParsingPage{
				MarkdownText:   getString(markdown, "text"),
				MarkdownImages: getStringMap(markdown, "images"),
				OutputImages:   getStringMap(itemMap, "outputImages"),
				PrunedResult:   itemMap["prunedResult"],
				InputImageURL:  getString(itemMap, "inputImage"),
				Exports:        getMap(itemMap, "exports"),
				Markdown:       markdown,
				Raw:            itemMap,
			}
			result.Pages = append(result.Pages, page)
		}
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

func getMap(m map[string]interface{}, key string) map[string]interface{} {
	value, ok := m[key].(map[string]interface{})
	if !ok {
		return map[string]interface{}{}
	}
	return value
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
	default:
		b, _ := json.Marshal(raw)
		var result map[string]string
		json.Unmarshal(b, &result)
		return result
	}
}

func normalizeStatus(jobID string, status *jobStatusResponse) (*JobStatus, error) {
	if status == nil || status.State == "" {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "status response is missing state"}}
	}
	if status.State != "pending" && status.State != "running" && status.State != "done" && status.State != "failed" {
		return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "unknown job state: " + status.State}}
	}
	result := &JobStatus{
		JobID:     jobID,
		State:     status.State,
		ResultURL: status.ResultURL,
		ErrorMsg:  status.ErrorMsg,
	}
	if len(status.ExtractProgress) != 0 && string(status.ExtractProgress) != "null" {
		var progress extractProgress
		if err := json.Unmarshal(status.ExtractProgress, &progress); err != nil {
			return nil, &ResponseFormatError{PaddleOCRAPIError{Message: "status progress is malformed", Cause: err}}
		}
		result.Progress = &Progress{
			TotalPages:     progress.TotalPages,
			ExtractedPages: progress.ExtractedPages,
			StartTime:      progress.StartTime,
			EndTime:        progress.EndTime,
		}
	}
	return result, nil
}
