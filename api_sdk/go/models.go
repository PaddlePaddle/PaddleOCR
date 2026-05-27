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
	PaddleOCRVL16 = "PaddleOCR-VL-1.6"
)

// IsOCRModel reports whether model is supported by OCR APIs.
func IsOCRModel(model string) bool {
	return model == PPOCRv5
}

// IsDocumentParsingModel reports whether model is supported by document parsing APIs.
func IsDocumentParsingModel(model string) bool {
	switch model {
	case PPStructureV3, PaddleOCRVL, PaddleOCRVL15, PaddleOCRVL16:
		return true
	default:
		return false
	}
}

// IsVLModel reports whether model is a PaddleOCR-VL family model.
func IsVLModel(model string) bool {
	switch model {
	case PaddleOCRVL, PaddleOCRVL15, PaddleOCRVL16:
		return true
	default:
		return false
	}
}

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

type PPStructureV3Options struct {
	UseDocOrientationClassify        *bool       `json:"useDocOrientationClassify,omitempty"`
	UseDocUnwarping                  *bool       `json:"useDocUnwarping,omitempty"`
	UseTextlineOrientation           *bool       `json:"useTextlineOrientation,omitempty"`
	UseSealRecognition               *bool       `json:"useSealRecognition,omitempty"`
	UseTableRecognition              *bool       `json:"useTableRecognition,omitempty"`
	UseFormulaRecognition            *bool       `json:"useFormulaRecognition,omitempty"`
	UseChartRecognition              *bool       `json:"useChartRecognition,omitempty"`
	UseRegionDetection               *bool       `json:"useRegionDetection,omitempty"`
	LayoutThreshold                  interface{} `json:"layoutThreshold,omitempty"`
	LayoutNms                        *bool       `json:"layoutNms,omitempty"`
	LayoutUnclipRatio                interface{} `json:"layoutUnclipRatio,omitempty"`
	LayoutMergeBboxesMode            *string     `json:"layoutMergeBboxesMode,omitempty"`
	TextDetLimitSideLen              *int        `json:"textDetLimitSideLen,omitempty"`
	TextDetLimitType                 *string     `json:"textDetLimitType,omitempty"`
	TextDetThresh                    *float64    `json:"textDetThresh,omitempty"`
	TextDetBoxThresh                 *float64    `json:"textDetBoxThresh,omitempty"`
	TextDetUnclipRatio               *float64    `json:"textDetUnclipRatio,omitempty"`
	TextRecScoreThresh               *float64    `json:"textRecScoreThresh,omitempty"`
	UseWiredTableCellsTransToHtml    *bool       `json:"useWiredTableCellsTransToHtml,omitempty"`
	UseWirelessTableCellsTransToHtml *bool       `json:"useWirelessTableCellsTransToHtml,omitempty"`
	UseTableOrientationClassify      *bool       `json:"useTableOrientationClassify,omitempty"`
	UseOcrResultsWithTableCells      *bool       `json:"useOcrResultsWithTableCells,omitempty"`
	UseE2eWiredTableRecModel         *bool       `json:"useE2eWiredTableRecModel,omitempty"`
	UseE2eWirelessTableRecModel      *bool       `json:"useE2eWirelessTableRecModel,omitempty"`
	PrettifyMarkdown                 *bool       `json:"prettifyMarkdown,omitempty"`
	ShowFormulaNumber                *bool       `json:"showFormulaNumber,omitempty"`
	Visualize                        *bool       `json:"visualize,omitempty"`
}

type PaddleOCRVLOptions struct {
	UseDocOrientationClassify *bool       `json:"useDocOrientationClassify,omitempty"`
	UseDocUnwarping           *bool       `json:"useDocUnwarping,omitempty"`
	UseLayoutDetection        *bool       `json:"useLayoutDetection,omitempty"`
	UseChartRecognition       *bool       `json:"useChartRecognition,omitempty"`
	UseSealRecognition        *bool       `json:"useSealRecognition,omitempty"`
	LayoutThreshold           interface{} `json:"layoutThreshold,omitempty"`
	LayoutNms                 *bool       `json:"layoutNms,omitempty"`
	LayoutUnclipRatio         interface{} `json:"layoutUnclipRatio,omitempty"`
	LayoutMergeBboxesMode     interface{} `json:"layoutMergeBboxesMode,omitempty"`
	LayoutShapeMode           *string     `json:"layoutShapeMode,omitempty"`
	PromptLabel               *string     `json:"promptLabel,omitempty"`
	RepetitionPenalty         *float64    `json:"repetitionPenalty,omitempty"`
	Temperature               *float64    `json:"temperature,omitempty"`
	TopP                      *float64    `json:"topP,omitempty"`
	MinPixels                 *int        `json:"minPixels,omitempty"`
	MaxPixels                 *int        `json:"maxPixels,omitempty"`
	MaxNewTokens              *int        `json:"maxNewTokens,omitempty"`
	MergeLayoutBlocks         *bool       `json:"mergeLayoutBlocks,omitempty"`
	PrettifyMarkdown          *bool       `json:"prettifyMarkdown,omitempty"`
	ShowFormulaNumber         *bool       `json:"showFormulaNumber,omitempty"`
	RestructurePages          *bool       `json:"restructurePages,omitempty"`
	MergeTables               *bool       `json:"mergeTables,omitempty"`
	RelevelTitles             *bool       `json:"relevelTitles,omitempty"`
	Visualize                 *bool       `json:"visualize,omitempty"`
}

// DocParsingOptionsProvider marks document parsing option structs.
type DocParsingOptionsProvider interface {
	isDocParsingOptions()
}

func (*PPStructureV3Options) isDocParsingOptions() {}
func (*PaddleOCRVLOptions) isDocParsingOptions()   {}

type OCRRequest struct {
	Model      string
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
	Options    DocParsingOptionsProvider
}
