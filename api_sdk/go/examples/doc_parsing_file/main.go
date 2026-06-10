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

package main

import (
	"context"
	"fmt"
	"log"

	paddleocr "github.com/PaddlePaddle/PaddleOCR/api_sdk/go"
)

func main() {
	client, err := paddleocr.NewClient()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	// Convenience method (blocks until done)
	result, err := client.ParseDocument(ctx, &paddleocr.DocParsingRequest{
		Model:    paddleocr.PPStructureV3,
		FilePath: "./sample.pdf",
		Options:  &paddleocr.PPStructureV3Options{UseChartRecognition: paddleocr.Bool(true)},
	})
	if err != nil {
		log.Fatal(err)
	}
	for i, page := range result.Pages {
		fmt.Printf("Page %d:\n%s\n", i+1, page.MarkdownText)
	}

	// Manual control with typed job metadata and typed wait methods.
	ocrJob, _ := client.SubmitOCR(ctx, &paddleocr.OCRRequest{FileURL: "https://example.com/f1.pdf"})
	docJob, _ := client.SubmitDocumentParsing(ctx, &paddleocr.DocParsingRequest{
		Model: paddleocr.PPStructureV3, FilePath: "./sample.pdf",
	})

	ocrResult, err := client.WaitOCRResult(ctx, ocrJob.JobID)
	if err != nil {
		log.Printf("OCR job error: %v", err)
	}
	docResult, err := client.WaitDocumentParsingResult(ctx, docJob.JobID)
	if err != nil {
		log.Printf("document parsing job error: %v", err)
	}
	fmt.Printf("OCR done: %v\n", ocrResult)
	fmt.Printf("Document parsing done: %v\n", docResult)
}
