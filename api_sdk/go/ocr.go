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

// OCR performs PP-OCRv5 text recognition. Blocks until result is ready.
func (c *Client) OCR(ctx context.Context, req *OCRRequest) (*OCRResult, error) {
	jobID, err := c.submit(ctx, PPOCRv5, req.FileURL, req.FilePath, req.Options)
	if err != nil {
		return nil, err
	}
	jsonlData, err := c.pollUntilDone(ctx, jobID)
	if err != nil {
		return nil, err
	}
	return parseOCRResult(jobID, jsonlData)
}

// DocParsing performs document layout parsing. Blocks until result is ready.
func (c *Client) DocParsing(ctx context.Context, req *DocParsingRequest) (*DocParsingResult, error) {
	jobID, err := c.submit(ctx, req.Model, req.FileURL, req.FilePath, req.Options)
	if err != nil {
		return nil, err
	}
	jsonlData, err := c.pollUntilDone(ctx, jobID)
	if err != nil {
		return nil, err
	}
	return parseDocParsingResult(jobID, jsonlData)
}

// SubmitOCR submits an OCR job and returns an Operation for tracking.
func (c *Client) SubmitOCR(ctx context.Context, req *OCRRequest) (*Operation, error) {
	jobID, err := c.submit(ctx, PPOCRv5, req.FileURL, req.FilePath, req.Options)
	if err != nil {
		return nil, err
	}
	return &Operation{client: c, JobID: jobID, model: PPOCRv5}, nil
}

// SubmitDocParsing submits a doc parsing job and returns an Operation for tracking.
func (c *Client) SubmitDocParsing(ctx context.Context, req *DocParsingRequest) (*Operation, error) {
	jobID, err := c.submit(ctx, req.Model, req.FileURL, req.FilePath, req.Options)
	if err != nil {
		return nil, err
	}
	return &Operation{client: c, JobID: jobID, model: req.Model}, nil
}

func (c *Client) submit(ctx context.Context, model, fileURL, filePath string, options interface{}) (string, error) {
	if fileURL == "" && filePath == "" {
		return "", &InvalidRequestError{PaddleOCRAPIError{Message: "Either FileURL or FilePath is required."}}
	}
	if fileURL != "" && filePath != "" {
		return "", &InvalidRequestError{PaddleOCRAPIError{Message: "FileURL and FilePath are mutually exclusive."}}
	}

	payload := defaultPayload(model, options)

	if fileURL != "" {
		return c.submitURL(model, fileURL, payload)
	}
	return c.submitFile(model, filePath, payload)
}

func defaultPayload(model string, options interface{}) interface{} {
	if options != nil {
		return options
	}
	if model == PPOCRv5 {
		return &OCROptions{}
	}
	return &DocParsingOptions{}
}

func parseOCRResult(jobID string, jsonlData []map[string]interface{}) (*OCRResult, error) {
	result := &OCRResult{JobID: jobID}
	for _, lineObj := range jsonlData {
		resultData, ok := lineObj["result"].(map[string]interface{})
		if !ok {
			continue
		}
		ocrResults, ok := resultData["ocrResults"].([]interface{})
		if !ok {
			continue
		}
		for _, item := range ocrResults {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			page := OCRPage{
				PrunedResult: itemMap["prunedResult"],
				OCRImageURL:  getString(itemMap, "ocrImage"),
			}
			result.Pages = append(result.Pages, page)
		}
	}
	return result, nil
}

func parseDocParsingResult(jobID string, jsonlData []map[string]interface{}) (*DocParsingResult, error) {
	result := &DocParsingResult{JobID: jobID}
	for _, lineObj := range jsonlData {
		resultData, ok := lineObj["result"].(map[string]interface{})
		if !ok {
			continue
		}
		lpResults, ok := resultData["layoutParsingResults"].([]interface{})
		if !ok {
			continue
		}
		for _, item := range lpResults {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			markdown, _ := itemMap["markdown"].(map[string]interface{})
			page := DocParsingPage{
				MarkdownText:   getString(markdown, "text"),
				MarkdownImages: getStringMap(markdown, "images"),
				OutputImages:   getStringMap(itemMap, "outputImages"),
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
