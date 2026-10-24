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

type OCRPage struct {
	PrunedResult             interface{}            `json:"prunedResult"`
	OCRImageURL              string                 `json:"ocrImageUrl,omitempty"`
	DocPreprocessingImageURL string                 `json:"docPreprocessingImageUrl,omitempty"`
	InputImageURL            string                 `json:"inputImageUrl,omitempty"`
	Raw                      map[string]interface{} `json:"raw,omitempty"`
}

type DocParsingPage struct {
	MarkdownText   string                 `json:"markdownText"`
	MarkdownImages map[string]string      `json:"markdownImages"`
	OutputImages   map[string]string      `json:"outputImages"`
	PrunedResult   interface{}            `json:"prunedResult,omitempty"`
	InputImageURL  string                 `json:"inputImageUrl,omitempty"`
	Exports        map[string]interface{} `json:"exports,omitempty"`
	Markdown       map[string]interface{} `json:"markdown,omitempty"`
	Raw            map[string]interface{} `json:"raw,omitempty"`
}

type OCRResult struct {
	JobID    string                 `json:"jobId"`
	Pages    []OCRPage              `json:"pages"`
	DataInfo map[string]interface{} `json:"dataInfo,omitempty"`
}

type DocParsingResult struct {
	JobID    string                 `json:"jobId"`
	Pages    []DocParsingPage       `json:"pages"`
	DataInfo map[string]interface{} `json:"dataInfo,omitempty"`
}

type Progress struct {
	TotalPages     int    `json:"totalPages"`
	ExtractedPages int    `json:"extractedPages"`
	StartTime      string `json:"startTime,omitempty"`
	EndTime        string `json:"endTime,omitempty"`
}

type Job struct {
	JobID      string `json:"jobId"`
	Model      string `json:"model"`
	Task       string `json:"task"`
	PageRanges string `json:"pageRanges,omitempty"`
	BatchID    string `json:"batchId,omitempty"`
}

type JobStatus struct {
	JobID     string            `json:"jobId"`
	State     string            `json:"state"`
	Progress  *Progress         `json:"progress,omitempty"`
	ResultURL map[string]string `json:"resultUrl,omitempty"`
	ErrorMsg  string            `json:"errorMsg,omitempty"`
}

type BatchStatus struct {
	BatchID string       `json:"batchId"`
	Jobs    []*JobStatus `json:"jobs"`
}
