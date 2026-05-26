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
	"errors"
	"fmt"
	"log"
	"sync"

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
		Model:    paddleocr.PaddleOCRVL15,
		FilePath: "./sample.pdf",
		Options:  &paddleocr.DocParsingOptions{UseChartRecognition: paddleocr.Bool(true)},
	})
	if err != nil {
		var fileErr *paddleocr.FileNotFoundError
		if errors.As(err, &fileErr) {
			log.Fatalf("input file missing: %v", fileErr)
		}
		log.Fatal(err)
	}
	for i, page := range result.Pages {
		fmt.Printf("Page %d:\n%s\n", i+1, page.MarkdownText)
	}

	// Manual control with Operation objects
	op1, err := client.SubmitOCR(ctx, &paddleocr.OCRRequest{FileURL: "https://example.com/f1.pdf"})
	if err != nil {
		log.Fatal(err)
	}
	op2, err := client.SubmitDocumentParsing(ctx, &paddleocr.DocParsingRequest{
		Model: paddleocr.PaddleOCRVL15, FilePath: "./sample.pdf",
	})
	if err != nil {
		log.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		r, err := op1.WaitOCR(ctx)
		if err != nil {
			log.Printf("op1 error: %v", err)
			return
		}
		fmt.Printf("OCR done: %v\n", r)
	}()
	go func() {
		defer wg.Done()
		r, err := op2.WaitDocumentParsing(ctx)
		if err != nil {
			log.Printf("op2 error: %v", err)
			return
		}
		fmt.Printf("DocParsing done: %v\n", r)
	}()
	wg.Wait()
}
