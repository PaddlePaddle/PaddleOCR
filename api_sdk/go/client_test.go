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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewClientAuthEnvAndBaseURL(t *testing.T) {
	t.Setenv("PADDLEOCR_ACCESS_TOKEN", "")
	if _, err := NewClient(); err == nil {
		t.Fatal("expected missing token to fail")
	} else {
		var authErr *AuthError
		if !errors.As(err, &authErr) {
			t.Fatalf("expected AuthError, got %T", err)
		}
	}

	t.Setenv("PADDLEOCR_ACCESS_TOKEN", "env-token")
	client, err := NewClient(WithBaseURL("https://example.test/jobs///"))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if client.token != "env-token" {
		t.Fatalf("token = %q", client.token)
	}
	if client.baseURL != "https://example.test/jobs" {
		t.Fatalf("baseURL = %q", client.baseURL)
	}
}

func TestSubmitValidationAndPayloads(t *testing.T) {
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer token" {
			t.Fatalf("Authorization = %q", got)
		}
		if !strings.Contains(r.Header.Get("Content-Type"), "application/json") {
			t.Fatalf("Content-Type = %q", r.Header.Get("Content-Type"))
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body["fileUrl"] != "https://example.test/invoice.pdf" {
			t.Fatalf("fileUrl = %#v", body["fileUrl"])
		}
		if body["model"] != string(PPOCRv5) {
			t.Fatalf("model = %#v", body["model"])
		}
		if body["pageRanges"] != "1-2" {
			t.Fatalf("pageRanges = %#v", body["pageRanges"])
		}
		writeJSON(w, http.StatusAccepted, map[string]any{"data": map[string]any{"jobId": "job-1"}})
	}))

	if _, err := client.SubmitOCR(context.Background(), nil); err == nil {
		t.Fatal("expected nil OCR request to fail")
	} else {
		var invalid *InvalidRequestError
		if !errors.As(err, &invalid) {
			t.Fatalf("expected InvalidRequestError, got %T", err)
		}
	}
	if _, err := client.SubmitOCR(context.Background(), &OCRRequest{}); err == nil {
		t.Fatal("expected missing file to fail")
	}
	if _, err := client.SubmitOCR(context.Background(), &OCRRequest{FileURL: "u", FilePath: "p"}); err == nil {
		t.Fatal("expected mutually exclusive file inputs to fail")
	}

	op, err := client.SubmitOCR(context.Background(), &OCRRequest{
		FileURL:    "https://example.test/invoice.pdf",
		PageRanges: "1-2",
	})
	if err != nil {
		t.Fatalf("SubmitOCR() error = %v", err)
	}
	if op.JobID != "job-1" || op.Model != PPOCRv5 || op.Task != TaskOCR {
		t.Fatalf("operation metadata = %#v", op)
	}
}

func TestOCRModelDefaultPropagationAndClassification(t *testing.T) {
	var models []string
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		models = append(models, fmt.Sprint(body["model"]))
		writeJSON(w, http.StatusAccepted, map[string]any{"data": map[string]any{"jobId": fmt.Sprintf("job-%d", len(models))}})
	}))

	defaultOp, err := client.SubmitOCR(context.Background(), &OCRRequest{FileURL: "https://example.test/default.pdf"})
	if err != nil {
		t.Fatalf("SubmitOCR(default) error = %v", err)
	}
	explicitOp, err := client.SubmitOCR(context.Background(), &OCRRequest{Model: PPOCRv5, FileURL: "https://example.test/explicit.pdf"})
	if err != nil {
		t.Fatalf("SubmitOCR(explicit) error = %v", err)
	}

	if defaultOp.Model != PPOCRv5 || explicitOp.Model != PPOCRv5 {
		t.Fatalf("OCR operation models = %#v %#v", defaultOp.Model, explicitOp.Model)
	}
	if len(models) != 2 || models[0] != string(PPOCRv5) || models[1] != string(PPOCRv5) {
		t.Fatalf("submitted models = %#v", models)
	}
	if !IsOCRModel(PPOCRv5) || IsOCRModel(PPStructureV3) || IsOCRModel(Model("future-unknown-model")) {
		t.Fatal("unexpected OCR model classification")
	}
	if !IsDocumentParsingModel(PaddleOCRVL) || IsDocumentParsingModel(PPOCRv5) {
		t.Fatal("unexpected document parsing model classification")
	}
}

func TestSubmitOCRRejectsNonOCRModel(t *testing.T) {
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unexpected network call")
	}))

	_, err := client.SubmitOCR(context.Background(), &OCRRequest{
		Model:   PPStructureV3,
		FileURL: "https://example.test/doc.pdf",
	})
	if err == nil {
		t.Fatal("expected document model to fail for OCR")
	}
	var invalid *InvalidRequestError
	if !errors.As(err, &invalid) {
		t.Fatalf("expected InvalidRequestError, got %T", err)
	}
}

func TestDocumentParsingModelValidation(t *testing.T) {
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unexpected network call")
	}))

	_, err := client.SubmitDocumentParsing(context.Background(), &DocParsingRequest{
		Model:   PPOCRv5,
		FileURL: "https://example.test/doc.pdf",
	})
	if err == nil {
		t.Fatal("expected PPOCRv5 document parsing model to fail")
	}
	var invalid *InvalidRequestError
	if !errors.As(err, &invalid) {
		t.Fatalf("expected InvalidRequestError, got %T", err)
	}
	if got := invalid.Message; !strings.Contains(got, string(PPOCRv5)) || !strings.Contains(got, "OCR model") {
		t.Fatalf("unexpected validation message %q", got)
	}
}

func TestFileUploadAndFileNotFoundPrecedence(t *testing.T) {
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.Header.Get("Content-Type"), "multipart/form-data") {
			t.Fatalf("Content-Type = %q", r.Header.Get("Content-Type"))
		}
		reader, err := r.MultipartReader()
		if err != nil {
			t.Fatalf("MultipartReader: %v", err)
		}
		fields := map[string]string{}
		for {
			part, err := reader.NextPart()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatalf("NextPart: %v", err)
			}
			b, err := io.ReadAll(part)
			if err != nil {
				t.Fatalf("ReadAll: %v", err)
			}
			fields[part.FormName()] = string(b)
		}
		if fields["model"] != string(PPStructureV3) {
			t.Fatalf("model field = %q", fields["model"])
		}
		if fields["file"] != "hello" {
			t.Fatalf("file field = %q", fields["file"])
		}
		writeJSON(w, http.StatusCreated, map[string]any{"data": map[string]any{"jobId": "job-file"}})
	}))

	missing := "/definitely/missing.pdf"
	_, err := client.SubmitOCR(context.Background(), &OCRRequest{FilePath: missing})
	if err == nil {
		t.Fatal("expected missing file to fail")
	}
	var notFound *FileNotFoundError
	if !errors.As(err, &notFound) {
		t.Fatalf("expected FileNotFoundError, got %T", err)
	}

	f, err := os.CreateTemp(t.TempDir(), "doc-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("hello"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	op, err := client.SubmitDocumentParsing(context.Background(), &DocParsingRequest{
		Model:    PPStructureV3,
		FilePath: f.Name(),
	})
	if err != nil {
		t.Fatalf("SubmitDocumentParsing() error = %v", err)
	}
	if op.JobID != "job-file" {
		t.Fatalf("JobID = %q", op.JobID)
	}
}

func TestFileUploadTransportErrorDoesNotDeadlock(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "upload-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(strings.Repeat("x", 1024*1024)); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	client := clientWithTransport(t, roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return nil, errors.New("transport failed before reading body")
	}))

	done := make(chan error, 1)
	go func() {
		_, err := client.SubmitOCR(context.Background(), &OCRRequest{FilePath: f.Name()})
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected transport error")
		}
		var network *NetworkError
		if !errors.As(err, &network) {
			t.Fatalf("expected NetworkError, got %T", err)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("SubmitOCR deadlocked after transport returned without reading body")
	}
}

func TestFileUploadRequestConstructionErrorDoesNotLeak(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "upload-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(strings.Repeat("x", 1024*1024)); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	client, err := NewClient(WithToken("token"), WithBaseURL("://bad-url"))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	baseline := runtime.NumGoroutine()
	for i := 0; i < 20; i++ {
		done := make(chan error, 1)
		go func() {
			_, err := client.SubmitOCR(context.Background(), &OCRRequest{FilePath: f.Name()})
			done <- err
		}()
		select {
		case err := <-done:
			if err == nil {
				t.Fatal("expected request construction error")
			}
			var network *NetworkError
			if !errors.As(err, &network) {
				t.Fatalf("expected NetworkError, got %T", err)
			}
		case <-time.After(200 * time.Millisecond):
			t.Fatal("SubmitOCR hung after request construction failed")
		}
	}
	deadline := time.Now().Add(200 * time.Millisecond)
	for time.Now().Before(deadline) && runtime.NumGoroutine() > baseline+5 {
		time.Sleep(10 * time.Millisecond)
	}
	if got := runtime.NumGoroutine(); got > baseline+5 {
		t.Fatalf("possible leaked upload goroutines: before=%d after=%d", baseline, got)
	}
}

func TestFileUploadRequestConstructionErrorReturnsNetworkError(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "upload-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("x"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	client, err := NewClient(WithToken("token"), WithBaseURL("://bad-url"))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	_, err = client.SubmitOCR(context.Background(), &OCRRequest{FilePath: f.Name()})
	if err == nil {
		t.Fatal("expected request construction error")
	}
	var network *NetworkError
	if !errors.As(err, &network) {
		t.Fatalf("expected NetworkError, got %T", err)
	}
}

func TestFileUploadPayloadMarshalErrorIsInvalidRequest(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "upload-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("hello"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	client := clientWithTransport(t, roundTripFunc(func(req *http.Request) (*http.Response, error) {
		_, err := io.Copy(io.Discard, req.Body)
		if err != nil {
			return nil, err
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"data":{"jobId":"unexpected"}}`)),
			Header:     make(http.Header),
			Request:    req,
		}, nil
	}))

	_, err = client.SubmitDocumentParsing(context.Background(), &DocParsingRequest{
		Model:    PPStructureV3,
		FilePath: f.Name(),
		Options:  &DocParsingOptions{LayoutThreshold: make(chan int)},
	})
	if err == nil {
		t.Fatal("expected payload marshal error")
	}
	var invalid *InvalidRequestError
	if !errors.As(err, &invalid) {
		t.Fatalf("expected InvalidRequestError, got %T: %v", err, err)
	}
}

func TestGetStatusAndPollingTerminalSemantics(t *testing.T) {
	var statusCalls atomic.Int32
	resultServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Fatalf("result Authorization = %q", got)
		}
		fmt.Fprintln(w, `{"result":{"ocrResults":[{"prunedResult":{"text":"hello"},"ocrImage":"https://img.test/1.png"}]}}`)
	}))
	defer resultServer.Close()

	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/jobs/job-1":
			call := statusCalls.Add(1)
			if call == 1 {
				writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{
					"state":           string(JobStateRunning),
					"extractProgress": map[string]any{"totalPages": 2, "extractedPages": 1},
				}})
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{
				"state":     string(JobStateDone),
				"resultUrl": map[string]any{"jsonUrl": resultServer.URL},
			}})
		case "/jobs/job-failed":
			writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{
				"state":    string(JobStateFailed),
				"errorMsg": "boom",
			}})
		case "/jobs/job-weird":
			writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{"state": "paused"}})
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))

	status, err := client.GetStatus(context.Background(), "job-1")
	if err != nil {
		t.Fatalf("GetStatus() error = %v", err)
	}
	if status.JobID != "job-1" || status.State != JobStateRunning || status.Progress == nil || status.Progress.ExtractedPages != 1 {
		t.Fatalf("status = %#v", status)
	}

	result, err := client.WaitOCRResult(context.Background(), &Job{JobID: "job-1", Model: PPOCRv5, Task: TaskOCR})
	if err != nil {
		t.Fatalf("WaitOCRResult() error = %v", err)
	}
	if len(result.Pages) != 1 {
		t.Fatalf("pages = %#v", result.Pages)
	}
	if statusCalls.Load() != 2 {
		t.Fatalf("status calls = %d, want first check before sleep and one retry", statusCalls.Load())
	}

	missingResultCases := []struct {
		name string
		data map[string]any
	}{
		{
			name: "missingResultUrl",
			data: map[string]any{"state": string(JobStateDone)},
		},
		{
			name: "missingJSONUrl",
			data: map[string]any{"state": string(JobStateDone), "resultUrl": map[string]any{}},
		},
	}
	for _, tc := range missingResultCases {
		t.Run(tc.name, func(t *testing.T) {
			client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				writeJSON(w, http.StatusOK, map[string]any{"data": tc.data})
			}))
			_, err := client.WaitOCRResult(context.Background(), &Job{JobID: "job-no-result", Model: PPOCRv5, Task: TaskOCR})
			if err == nil {
				t.Fatal("expected done job without result URL to fail")
			}
			var format *ResponseFormatError
			if !errors.As(err, &format) {
				t.Fatalf("expected ResponseFormatError, got %T", err)
			}
		})
	}

	_, err = client.WaitOCRResult(context.Background(), &Job{JobID: "job-failed", Model: PPOCRv5, Task: TaskOCR})
	if err == nil {
		t.Fatal("expected failed job error")
	}
	var failed *JobFailedError
	if !errors.As(err, &failed) {
		t.Fatalf("expected JobFailedError, got %T", err)
	}

	_, err = client.GetStatus(context.Background(), "job-weird")
	if err == nil {
		t.Fatal("expected unknown state to fail")
	}
	var format *ResponseFormatError
	if !errors.As(err, &format) {
		t.Fatalf("expected ResponseFormatError, got %T", err)
	}
}

func TestPollingTimeoutAndContextCancel(t *testing.T) {
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{"state": string(JobStateRunning)}})
	}), WithPollTimeout(20*time.Millisecond), withPollInterval(5*time.Millisecond))

	_, err := client.WaitOCRResult(context.Background(), &Job{JobID: "job-timeout", Model: PPOCRv5, Task: TaskOCR})
	if err == nil {
		t.Fatal("expected poll timeout")
	}
	var pollTimeout *PollTimeoutError
	if !errors.As(err, &pollTimeout) {
		t.Fatalf("expected PollTimeoutError, got %T", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = client.WaitOCRResult(ctx, &Job{JobID: "job-cancel", Model: PPOCRv5, Task: TaskOCR})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context canceled, got %v", err)
	}
}

func TestHTTPErrorMappingAndMalformedResponses(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       string
		check      func(error) bool
	}{
		{"auth401", http.StatusUnauthorized, "bad token", func(err error) bool { var target *AuthError; return errors.As(err, &target) }},
		{"auth403", http.StatusForbidden, "forbidden", func(err error) bool { var target *AuthError; return errors.As(err, &target) }},
		{"badRequest", http.StatusBadRequest, "bad input", func(err error) bool { var target *InvalidRequestError; return errors.As(err, &target) }},
		{"server", http.StatusInternalServerError, "server down", func(err error) bool { var target *APIError; return errors.As(err, &target) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, tt.body, tt.statusCode)
			}))
			_, err := client.SubmitOCR(context.Background(), &OCRRequest{FileURL: "https://example.test/doc.pdf"})
			if err == nil {
				t.Fatal("expected error")
			}
			if !tt.check(err) {
				t.Fatalf("unexpected error type %T", err)
			}
		})
	}

	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	_, err := client.SubmitOCR(context.Background(), &OCRRequest{FileURL: "https://example.test/doc.pdf"})
	if err == nil {
		t.Fatal("expected malformed 2xx response")
	}
	var format *ResponseFormatError
	if !errors.As(err, &format) {
		t.Fatalf("expected ResponseFormatError, got %T", err)
	}
}

func TestRequestTimeout(t *testing.T) {
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
		writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{"jobId": "slow-job"}})
	}), WithRequestTimeout(5*time.Millisecond))

	_, err := client.SubmitOCR(context.Background(), &OCRRequest{FileURL: "https://example.test/doc.pdf"})
	if err == nil {
		t.Fatal("expected request timeout")
	}
	var timeout *RequestTimeoutError
	if !errors.As(err, &timeout) {
		t.Fatalf("expected RequestTimeoutError, got %T", err)
	}
}

func TestSaveResource(t *testing.T) {
	var calls int
	resourceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if got := r.Header.Get("Authorization"); got != "" {
			t.Fatalf("resource Authorization = %q", got)
		}
		fmt.Fprint(w, "resource-body")
	}))
	defer resourceServer.Close()
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unexpected API server call")
	}))

	dir := t.TempDir()
	saved, err := client.SaveResource(context.Background(), resourceServer.URL+"/assets/page.png", dir)
	if err != nil {
		t.Fatalf("SaveResource() error = %v", err)
	}
	if saved != filepath.Join(dir, "page.png") {
		t.Fatalf("saved path = %q", saved)
	}
	body, err := os.ReadFile(saved)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != "resource-body" {
		t.Fatalf("saved body = %q", body)
	}
	if calls != 1 {
		t.Fatalf("resource calls = %d", calls)
	}

	if _, err := client.SaveResource(context.Background(), resourceServer.URL+"/assets/page.png", saved); err == nil {
		t.Fatal("expected existing destination to fail without overwrite")
	} else {
		var invalid *InvalidRequestError
		if !errors.As(err, &invalid) {
			t.Fatalf("expected InvalidRequestError, got %T", err)
		}
	}

	if _, err := client.SaveResource(context.Background(), resourceServer.URL+"/assets/page.png", saved, WithOverwrite(true)); err != nil {
		t.Fatalf("SaveResource(overwrite) error = %v", err)
	}

	_, err = client.SaveResource(context.Background(), resourceServer.URL+"/assets/page.png", filepath.Join(dir, "missing", "page.png"))
	if err == nil {
		t.Fatal("expected missing parent to fail")
	}
	var notFound *FileNotFoundError
	if !errors.As(err, &notFound) {
		t.Fatalf("expected FileNotFoundError, got %T", err)
	}
}

func TestSaveOCRResultResources(t *testing.T) {
	resourceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Fatalf("resource Authorization = %q", got)
		}
		fmt.Fprint(w, strings.TrimPrefix(r.URL.Path, "/"))
	}))
	defer resourceServer.Close()
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unexpected API server call")
	}))

	dir := t.TempDir()
	saved, err := client.SaveOCRResultResources(context.Background(), &OCRResult{
		Pages: []OCRPage{
			{OCRImageURL: resourceServer.URL + "/assets/page-one.png"},
			{OCRImageURL: ""},
			{OCRImageURL: resourceServer.URL + "/assets/page-three.jpg?signature=opaque"},
		},
	}, dir)
	if err != nil {
		t.Fatalf("SaveOCRResultResources() error = %v", err)
	}
	expected := []string{
		filepath.Join(dir, "ocr-page-1.png"),
		filepath.Join(dir, "ocr-page-3.jpg"),
	}
	if fmt.Sprint(saved) != fmt.Sprint(expected) {
		t.Fatalf("saved paths = %#v, want %#v", saved, expected)
	}
	assertFileBody(t, saved[0], "assets/page-one.png")
	assertFileBody(t, saved[1], "assets/page-three.jpg")
}

func TestSaveDocumentParsingResultResources(t *testing.T) {
	var requestPaths []string
	resourceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Fatalf("resource Authorization = %q", got)
		}
		requestPaths = append(requestPaths, r.URL.Path)
		fmt.Fprint(w, r.URL.Path)
	}))
	defer resourceServer.Close()
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unexpected API server call")
	}))

	dir := t.TempDir()
	saved, err := client.SaveDocumentParsingResultResources(context.Background(), &DocParsingResult{
		Pages: []DocParsingPage{
			{
				MarkdownImages: map[string]string{
					"z-markdown.png": resourceServer.URL + "/opaque-z",
					"a-markdown.jpg": resourceServer.URL + "/opaque-a",
					"figure 1.png":   resourceServer.URL + "/opaque-figure",
				},
				OutputImages: map[string]string{
					"m-output.webp": resourceServer.URL + "/opaque-output",
				},
			},
		},
	}, dir)
	if err != nil {
		t.Fatalf("SaveDocumentParsingResultResources() error = %v", err)
	}
	expected := []string{
		filepath.Join(dir, "a-markdown.jpg"),
		filepath.Join(dir, "figure 1.png"),
		filepath.Join(dir, "z-markdown.png"),
		filepath.Join(dir, "m-output.webp"),
	}
	if fmt.Sprint(saved) != fmt.Sprint(expected) {
		t.Fatalf("saved paths = %#v, want %#v", saved, expected)
	}
	if fmt.Sprint(requestPaths) != fmt.Sprint([]string{"/opaque-a", "/opaque-figure", "/opaque-z", "/opaque-output"}) {
		t.Fatalf("request order = %#v", requestPaths)
	}
	assertFileBody(t, filepath.Join(dir, "a-markdown.jpg"), "/opaque-a")
	assertFileBody(t, filepath.Join(dir, "figure 1.png"), "/opaque-figure")
	assertFileBody(t, filepath.Join(dir, "z-markdown.png"), "/opaque-z")
	assertFileBody(t, filepath.Join(dir, "m-output.webp"), "/opaque-output")
}

func TestSaveResultResourcesOverwriteAndValidation(t *testing.T) {
	resourceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "new")
	}))
	defer resourceServer.Close()
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unexpected API server call")
	}))

	dir := t.TempDir()
	existing := filepath.Join(dir, "ocr-page-1.png")
	if err := os.WriteFile(existing, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	result := &OCRResult{Pages: []OCRPage{{OCRImageURL: resourceServer.URL + "/page.png"}}}
	_, err := client.SaveOCRResultResources(context.Background(), result, dir)
	if err == nil {
		t.Fatal("expected existing destination to fail without overwrite")
	}
	var invalid *InvalidRequestError
	if !errors.As(err, &invalid) {
		t.Fatalf("expected InvalidRequestError, got %T", err)
	}
	assertFileBody(t, existing, "old")

	if _, err := client.SaveOCRResultResources(context.Background(), result, dir, WithOverwrite(true)); err != nil {
		t.Fatalf("SaveOCRResultResources(overwrite) error = %v", err)
	}
	assertFileBody(t, existing, "new")

	unsafeKeys := []string{
		"",
		"..",
		"../escape.png",
		"/tmp/escape.png",
		"nested/escape.png",
		`nested\escape.png`,
	}
	for _, key := range unsafeKeys {
		_, err = client.SaveDocumentParsingResultResources(context.Background(), &DocParsingResult{
			Pages: []DocParsingPage{{MarkdownImages: map[string]string{key: resourceServer.URL + "/escape.png"}}},
		}, dir)
		if err == nil {
			t.Fatalf("expected unsafe map key %q to fail", key)
		}
		if !errors.As(err, &invalid) {
			t.Fatalf("expected InvalidRequestError for unsafe key %q, got %T", key, err)
		}
	}

	missingDir := filepath.Join(dir, "missing")
	_, err = client.SaveOCRResultResources(context.Background(), result, missingDir)
	if err == nil {
		t.Fatal("expected missing destination directory to fail")
	}
	var notFound *FileNotFoundError
	if !errors.As(err, &notFound) {
		t.Fatalf("expected FileNotFoundError, got %T", err)
	}
}

func TestSaveResourceHTTPAndTimeoutErrors(t *testing.T) {
	errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusNotFound)
	}))
	defer errorServer.Close()
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unexpected API server call")
	}))

	_, err := client.SaveResource(context.Background(), errorServer.URL+"/missing.png", filepath.Join(t.TempDir(), "missing.png"))
	if err == nil {
		t.Fatal("expected non-2xx download to fail")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}

	slowServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
		fmt.Fprint(w, "late")
	}))
	defer slowServer.Close()
	timeoutClient := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unexpected API server call")
	}), WithRequestTimeout(5*time.Millisecond))

	_, err = timeoutClient.SaveResource(context.Background(), slowServer.URL+"/slow.png", filepath.Join(t.TempDir(), "slow.png"))
	if err == nil {
		t.Fatal("expected resource request timeout")
	}
	var timeout *RequestTimeoutError
	if !errors.As(err, &timeout) {
		t.Fatalf("expected RequestTimeoutError, got %T", err)
	}
}

func TestSaveResourceBodyReadErrors(t *testing.T) {
	tests := []struct {
		name  string
		err   error
		check func(error) bool
	}{
		{
			name: "timeout",
			err:  timeoutErr{message: "body timeout"},
			check: func(err error) bool {
				var timeout *RequestTimeoutError
				return errors.As(err, &timeout)
			},
		},
		{
			name: "network",
			err:  errors.New("connection reset while reading"),
			check: func(err error) bool {
				var network *NetworkError
				return errors.As(err, &network)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := clientWithTransport(t, roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       errReadCloser{err: tt.err},
					Header:     make(http.Header),
					Request:    req,
				}, nil
			}))

			_, err := client.SaveResource(context.Background(), "https://resources.test/file.bin", filepath.Join(t.TempDir(), "file.bin"))
			if err == nil {
				t.Fatal("expected body read error")
			}
			if !tt.check(err) {
				t.Fatalf("unexpected error type %T: %v", err, err)
			}
		})
	}
}

func TestSaveResourcePartialFailureIsAtomic(t *testing.T) {
	client := clientWithTransport(t, roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       partialErrReadCloser{reader: bytes.NewReader([]byte("partial")), err: errors.New("mid-stream reset")},
			Header:     make(http.Header),
			Request:    req,
		}, nil
	}))

	dir := t.TempDir()
	existing := filepath.Join(dir, "existing.bin")
	if err := os.WriteFile(existing, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := client.SaveResource(context.Background(), "https://resources.test/existing.bin", existing, WithOverwrite(true))
	if err == nil {
		t.Fatal("expected partial download failure")
	}
	var network *NetworkError
	if !errors.As(err, &network) {
		t.Fatalf("expected NetworkError, got %T: %v", err, err)
	}
	body, err := os.ReadFile(existing)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != "original" {
		t.Fatalf("existing file was modified: %q", body)
	}

	newPath := filepath.Join(dir, "new.bin")
	_, err = client.SaveResource(context.Background(), "https://resources.test/new.bin", newPath)
	if err == nil {
		t.Fatal("expected partial download failure")
	}
	if !errors.As(err, &network) {
		t.Fatalf("expected NetworkError, got %T: %v", err, err)
	}
	if _, err := os.Stat(newPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("final partial file exists or stat failed unexpectedly: %v", err)
	}
}

func TestFetchJSONLBodyReadErrors(t *testing.T) {
	tests := []struct {
		name  string
		err   error
		check func(error) bool
	}{
		{
			name: "timeout",
			err:  timeoutErr{message: "body timeout"},
			check: func(err error) bool {
				var timeout *RequestTimeoutError
				return errors.As(err, &timeout)
			},
		},
		{
			name: "network",
			err:  errors.New("connection reset while reading"),
			check: func(err error) bool {
				var network *NetworkError
				return errors.As(err, &network)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := clientWithTransport(t, roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       errReadCloser{err: tt.err},
					Header:     make(http.Header),
					Request:    req,
				}, nil
			}))

			_, err := client.fetchJSONL(context.Background(), "https://resources.test/result.jsonl")
			if err == nil {
				t.Fatal("expected body read error")
			}
			if !tt.check(err) {
				t.Fatalf("unexpected error type %T: %v", err, err)
			}
		})
	}
}

func TestJSONLAndParserErrors(t *testing.T) {
	resultServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"result":`)
	}))
	defer resultServer.Close()
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{
			"state":     string(JobStateDone),
			"resultUrl": map[string]any{"jsonUrl": resultServer.URL},
		}})
	}))

	_, err := client.WaitOCRResult(context.Background(), &Job{JobID: "job-jsonl", Model: PPOCRv5, Task: TaskOCR})
	if err == nil {
		t.Fatal("expected malformed JSONL error")
	}
	var parse *ResultParseError
	if !errors.As(err, &parse) {
		t.Fatalf("expected ResultParseError, got %T", err)
	}

	if _, err := parseOCRResult("job", []map[string]any{{"result": map[string]any{"layoutParsingResults": []any{}}}}); err == nil {
		t.Fatal("expected malformed OCR payload")
	} else if !errors.As(err, &parse) {
		t.Fatalf("expected ResultParseError, got %T", err)
	}
	if _, err := parseOCRResult("job", []map[string]any{{"result": map[string]any{
		"ocrResults": []any{map[string]any{"ocrImage": "https://img.test/1.png"}},
	}}}); err == nil {
		t.Fatal("expected OCR payload missing prunedResult to fail")
	} else if !errors.As(err, &parse) {
		t.Fatalf("expected ResultParseError, got %T", err)
	}

	docMalformedCases := []struct {
		name string
		page map[string]any
	}{
		{
			name: "missingMarkdown",
			page: map[string]any{},
		},
		{
			name: "nonObjectMarkdown",
			page: map[string]any{"markdown": "not-object"},
		},
		{
			name: "missingMarkdownText",
			page: map[string]any{"markdown": map[string]any{"images": map[string]any{"a": "https://a"}}},
		},
		{
			name: "nonStringMarkdownText",
			page: map[string]any{"markdown": map[string]any{"text": 123}},
		},
	}
	for _, tc := range docMalformedCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseDocParsingResult("job-doc", []map[string]any{{"result": map[string]any{
				"layoutParsingResults": []any{tc.page},
			}}})
			if err == nil {
				t.Fatal("expected malformed document parsing payload to fail")
			}
			if !errors.As(err, &parse) {
				t.Fatalf("expected ResultParseError, got %T", err)
			}
		})
	}

	doc, err := parseDocParsingResult("job-doc", []map[string]any{{"result": map[string]any{
		"layoutParsingResults": []any{map[string]any{
			"markdown":     map[string]any{"text": "md", "images": map[string]any{"a": "https://a"}},
			"outputImages": map[string]any{"b": "https://b"},
		}},
	}}})
	if err != nil {
		t.Fatalf("parseDocParsingResult() error = %v", err)
	}
	if len(doc.Pages) != 1 || doc.Pages[0].MarkdownText != "md" || doc.Pages[0].MarkdownImages["a"] == "" {
		t.Fatalf("doc result = %#v", doc)
	}
}

func TestTypedWaitsRejectMismatchesAndOperationMethods(t *testing.T) {
	client := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{"jobId": "job-op"}})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{
			"state":     string(JobStateDone),
			"resultUrl": map[string]any{"jsonUrl": "unused"},
		}})
	}))

	op, err := client.SubmitOCR(context.Background(), &OCRRequest{FileURL: "https://example.test/doc.pdf"})
	if err != nil {
		t.Fatalf("SubmitOCR() error = %v", err)
	}
	status, done, err := op.Poll(context.Background())
	if err != nil {
		t.Fatalf("Poll() error = %v", err)
	}
	if !done || status.State != JobStateDone || status.Model != PPOCRv5 || status.Task != TaskOCR {
		t.Fatalf("status=%#v done=%v", status, done)
	}

	_, err = op.WaitDocumentParsing(context.Background())
	if err == nil {
		t.Fatal("expected mismatched operation wait to fail")
	}
	var invalid *InvalidRequestError
	if !errors.As(err, &invalid) {
		t.Fatalf("expected InvalidRequestError, got %T", err)
	}

	_, err = client.WaitOCRResult(context.Background(), &Job{JobID: "job-doc", Model: PPStructureV3, Task: TaskDocumentParsing})
	if err == nil {
		t.Fatal("expected mismatched client wait to fail")
	}
	if !errors.As(err, &invalid) {
		t.Fatalf("expected InvalidRequestError, got %T", err)
	}
}

func TestOperationZeroValueReturnsInvalidRequest(t *testing.T) {
	var op Operation
	var invalid *InvalidRequestError

	if _, err := op.WaitOCR(context.Background()); err == nil {
		t.Fatal("expected zero-value WaitOCR to fail")
	} else if !errors.As(err, &invalid) {
		t.Fatalf("expected InvalidRequestError, got %T", err)
	}

	if _, err := op.WaitDocumentParsing(context.Background()); err == nil {
		t.Fatal("expected zero-value WaitDocumentParsing to fail")
	} else if !errors.As(err, &invalid) {
		t.Fatalf("expected InvalidRequestError, got %T", err)
	}

	if _, _, err := op.Poll(context.Background()); err == nil {
		t.Fatal("expected zero-value Poll to fail")
	} else if !errors.As(err, &invalid) {
		t.Fatalf("expected InvalidRequestError, got %T", err)
	}
}

func testClient(t *testing.T, handler http.Handler, opts ...ClientOption) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	options := append([]ClientOption{
		WithToken("token"),
		WithBaseURL(server.URL + "/jobs/"),
		WithRequestTimeout(2 * time.Second),
		WithPollTimeout(2 * time.Second),
		WithHTTPClient(server.Client()),
	}, opts...)
	client, err := NewClient(options...)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	return client
}

func clientWithTransport(t *testing.T, transport http.RoundTripper) *Client {
	t.Helper()
	client, err := NewClient(
		WithToken("token"),
		WithBaseURL("https://api.test/jobs"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	return client
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type errReadCloser struct {
	err error
}

func (r errReadCloser) Read([]byte) (int, error) {
	return 0, r.err
}

func (r errReadCloser) Close() error {
	return nil
}

type partialErrReadCloser struct {
	reader *bytes.Reader
	err    error
}

func (r partialErrReadCloser) Read(p []byte) (int, error) {
	if r.reader.Len() > 0 {
		return r.reader.Read(p)
	}
	return 0, r.err
}

func (r partialErrReadCloser) Close() error {
	return nil
}

type timeoutErr struct {
	message string
}

func (e timeoutErr) Error() string {
	return e.message
}

func (e timeoutErr) Timeout() bool {
	return true
}

func (e timeoutErr) Temporary() bool {
	return true
}

var _ net.Error = timeoutErr{}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func assertFileBody(t *testing.T, path, expected string) {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != expected {
		t.Fatalf("%s body = %q, want %q", path, body, expected)
	}
}

var _ = multipart.ErrMessageTooLarge
