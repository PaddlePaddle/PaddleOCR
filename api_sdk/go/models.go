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

const (
	PPOCRv5       = "PP-OCRv5"
	PPStructureV3 = "PP-StructureV3"
	PaddleOCRVL   = "PaddleOCR-VL"
	PaddleOCRVL15 = "PaddleOCR-VL-1.5"
)

type OCROptions struct {
	UseDocOrientationClassify *bool    `json:"useDocOrientationClassify,omitempty"`
	UseDocUnwarping           *bool    `json:"useDocUnwarping,omitempty"`
	UseTextlineOrientation    *bool    `json:"useTextlineOrientation,omitempty"`
	TextDetLimitSideLen       *int     `json:"textDetLimitSideLen,omitempty"`
	TextDetLimitType          *string  `json:"textDetLimitType,omitempty"`
	TextDetThresh             *float64 `json:"textDetThresh,omitempty"`
	TextDetBoxThresh          *float64 `json:"textDetBoxThresh,omitempty"`
	TextDetUnclipRatio        *float64 `json:"textDetUnclipRatio,omitempty"`
	TextRecScoreThresh        *float64 `json:"textRecScoreThresh,omitempty"`
	Visualize                 *bool    `json:"visualize,omitempty"`
}

type DocParsingOptions struct {
	UseDocOrientationClassify *bool       `json:"useDocOrientationClassify,omitempty"`
	UseDocUnwarping           *bool       `json:"useDocUnwarping,omitempty"`
	UseTextlineOrientation    *bool       `json:"useTextlineOrientation,omitempty"`
	UseSealRecognition        *bool       `json:"useSealRecognition,omitempty"`
	UseTableRecognition       *bool       `json:"useTableRecognition,omitempty"`
	UseFormulaRecognition     *bool       `json:"useFormulaRecognition,omitempty"`
	UseChartRecognition       *bool       `json:"useChartRecognition,omitempty"`
	UseRegionDetection        *bool       `json:"useRegionDetection,omitempty"`
	UseLayoutDetection        *bool       `json:"useLayoutDetection,omitempty"`
	LayoutThreshold           interface{} `json:"layoutThreshold,omitempty"`
	LayoutNms                 *bool       `json:"layoutNms,omitempty"`
	LayoutUnclipRatio         interface{} `json:"layoutUnclipRatio,omitempty"`
	LayoutMergeBboxesMode     *string     `json:"layoutMergeBboxesMode,omitempty"`
	TextDetLimitSideLen       *int        `json:"textDetLimitSideLen,omitempty"`
	TextDetLimitType          *string     `json:"textDetLimitType,omitempty"`
	TextDetThresh             *float64    `json:"textDetThresh,omitempty"`
	TextDetBoxThresh          *float64    `json:"textDetBoxThresh,omitempty"`
	TextDetUnclipRatio        *float64    `json:"textDetUnclipRatio,omitempty"`
	TextRecScoreThresh        *float64    `json:"textRecScoreThresh,omitempty"`
	Visualize                 *bool       `json:"visualize,omitempty"`
}

type OCRRequest struct {
	FileURL    string
	FilePath   string
	PageRanges string
	BatchID    string
	Options    *OCROptions
}

type DocParsingRequest struct {
	Model      string
	FileURL    string
	FilePath   string
	PageRanges string
	BatchID    string
	Options    *DocParsingOptions
}
