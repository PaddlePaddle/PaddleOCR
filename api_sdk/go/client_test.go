package paddleocr

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClientPlatformHeader(t *testing.T) {
	var got string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Client-Platform")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"code": 0,
			"data": map[string]string{"jobId": "job-1"},
		})
	}))
	defer server.Close()

	client, err := NewClient(
		WithToken("token"),
		WithBaseURL(server.URL),
		WithClientPlatform("my-app"),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	job, err := client.SubmitOCR(context.Background(), &OCRRequest{
		FileURL: "https://example.test/input.pdf",
	})
	if err != nil {
		t.Fatalf("SubmitOCR() error = %v", err)
	}
	if job.JobID != "job-1" {
		t.Fatalf("JobID = %q, want job-1", job.JobID)
	}
	if got != "my-app" {
		t.Fatalf("Client-Platform = %q, want my-app", got)
	}
}

func TestDocumentParsingOptionsIncludeCurrentAndFutureServiceParameters(t *testing.T) {
	trueValue := true
	falseValue := false
	var got map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			OptionalPayload map[string]interface{} `json:"optionalPayload"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("Decode request body error = %v", err)
		}
		got = body.OptionalPayload
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"code": 0,
			"data": map[string]string{"jobId": "job-doc"},
		})
	}))
	defer server.Close()

	client, err := NewClient(
		WithToken("token"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	_, err = client.SubmitDocumentParsing(context.Background(), &DocParsingRequest{
		Model:   PaddleOCRVL16,
		FileURL: "https://example.test/doc.pdf",
		Options: &PaddleOCRVLOptions{
			UseOcrForImageBlock:  &trueValue,
			FormatBlockContent:   &trueValue,
			MarkdownIgnoreLabels: []string{"image"},
			VlmExtraArgs:         map[string]interface{}{"temperature": 0.1},
			ReturnMarkdownImages: &falseValue,
			OutputFormats:        []string{"docx"},
			ExtraOptions:         map[string]interface{}{"futureOption": "enabled"},
		},
	})
	if err != nil {
		t.Fatalf("SubmitDocumentParsing() error = %v", err)
	}

	if got["useOcrForImageBlock"] != true || got["formatBlockContent"] != true {
		t.Fatalf("got boolean options %#v", got)
	}
	if got["returnMarkdownImages"] != false || got["futureOption"] != "enabled" {
		t.Fatalf("got passthrough options %#v", got)
	}
}

func TestResultParsersPreserveRawFieldsAndDataInfo(t *testing.T) {
	ocrLine := map[string]interface{}{
		"result": map[string]interface{}{
			"dataInfo": map[string]interface{}{"numPages": float64(1)},
			"ocrResults": []interface{}{
				map[string]interface{}{
					"prunedResult":          map[string]interface{}{"text": "hello"},
					"ocrImage":              "ocr.png",
					"docPreprocessingImage": "pre.png",
					"inputImage":            "input.png",
				},
			},
		},
	}
	ocrResult, err := parseOCRResult("job-ocr", []map[string]interface{}{ocrLine})
	if err != nil {
		t.Fatalf("parseOCRResult() error = %v", err)
	}
	if ocrResult.DataInfo["numPages"] != float64(1) {
		t.Fatalf("OCR metadata not preserved: %#v", ocrResult)
	}
	if ocrResult.Pages[0].DocPreprocessingImageURL != "pre.png" || ocrResult.Pages[0].InputImageURL != "input.png" {
		t.Fatalf("OCR page image URLs not preserved: %#v", ocrResult.Pages[0])
	}
	if ocrResult.Pages[0].Raw["ocrImage"] != "ocr.png" {
		t.Fatalf("OCR raw page not preserved: %#v", ocrResult.Pages[0].Raw)
	}

	docPage := map[string]interface{}{
		"prunedResult": map[string]interface{}{"blocks": []interface{}{map[string]interface{}{"label": "text"}}},
		"markdown":     map[string]interface{}{"text": "hello", "images": map[string]interface{}{"figure.png": "figure-url"}, "isStart": true},
		"outputImages": map[string]interface{}{"page.png": "page-url"},
		"inputImage":   "input.png",
		"exports":      map[string]interface{}{"docx": "docx-url"},
	}
	docLine := map[string]interface{}{
		"result": map[string]interface{}{
			"dataInfo":             map[string]interface{}{"numPages": float64(1)},
			"layoutParsingResults": []interface{}{docPage},
		},
	}
	docResult, err := parseDocParsingResult("job-doc", []map[string]interface{}{docLine})
	if err != nil {
		t.Fatalf("parseDocParsingResult() error = %v", err)
	}
	if docResult.DataInfo["numPages"] != float64(1) {
		t.Fatalf("document metadata not preserved: %#v", docResult)
	}
	if docResult.Pages[0].PrunedResult == nil || docResult.Pages[0].Raw["inputImage"] != "input.png" {
		t.Fatalf("document page raw fields not preserved: %#v", docResult.Pages[0])
	}
	if docResult.Pages[0].Exports["docx"] != "docx-url" || docResult.Pages[0].Markdown["isStart"] != true {
		t.Fatalf("document structured fields not preserved: %#v", docResult.Pages[0])
	}
}

func TestSubmitOCRAcceptsOfficialModelNameString(t *testing.T) {
	var got string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("Decode request body error = %v", err)
		}
		got = body.Model
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"code": 0,
			"data": map[string]string{"jobId": "job-1"},
		})
	}))
	defer server.Close()

	client, err := NewClient(
		WithToken("token"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	job, err := client.SubmitOCR(context.Background(), &OCRRequest{
		Model:   "PP-OCRv5",
		FileURL: "https://example.test/input.pdf",
	})
	if err != nil {
		t.Fatalf("SubmitOCR() error = %v", err)
	}
	if job.Model != "PP-OCRv5" {
		t.Fatalf("Job model = %q, want PP-OCRv5", job.Model)
	}
	if got != "PP-OCRv5" {
		t.Fatalf("Request model = %q, want PP-OCRv5", got)
	}
}

func TestSubmitOCRAcceptsPPOCRv5LatinModelNameString(t *testing.T) {
	var got string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("Decode request body error = %v", err)
		}
		got = body.Model
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"code": 0,
			"data": map[string]string{"jobId": "job-latin"},
		})
	}))
	defer server.Close()

	client, err := NewClient(
		WithToken("token"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	job, err := client.SubmitOCR(context.Background(), &OCRRequest{
		Model:   PPOCRv5Latin,
		FileURL: "https://example.test/latin.pdf",
	})
	if err != nil {
		t.Fatalf("SubmitOCR() error = %v", err)
	}
	if job.Model != PPOCRv5Latin {
		t.Fatalf("Job model = %q, want %s", job.Model, PPOCRv5Latin)
	}
	if got != PPOCRv5Latin {
		t.Fatalf("Request model = %q, want %s", got, PPOCRv5Latin)
	}
}
