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

const (
	PPOCRv5        = "PP-OCRv5"
	PPStructureV3  = "PP-StructureV3"
	PaddleOCRVL    = "PaddleOCR-VL"
	PaddleOCRVL15  = "PaddleOCR-VL-1.5"
)

type OCROptions struct {
	UseDocOrientationClassify bool `json:"useDocOrientationClassify"`
	UseDocUnwarping           bool `json:"useDocUnwarping"`
	UseTextlineOrientation    bool `json:"useTextlineOrientation"`
}

type DocParsingOptions struct {
	UseDocOrientationClassify bool `json:"useDocOrientationClassify"`
	UseDocUnwarping           bool `json:"useDocUnwarping"`
	UseChartRecognition       bool `json:"useChartRecognition"`
}

type OCRRequest struct {
	FileURL  string
	FilePath string
	Options  *OCROptions
}

type DocParsingRequest struct {
	Model    string
	FileURL  string
	FilePath string
	Options  *DocParsingOptions
}
