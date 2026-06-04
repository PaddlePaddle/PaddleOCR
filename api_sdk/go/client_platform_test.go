package paddleocr

import (
	"context"
	"encoding/json"
	"errors"
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

func TestTypedHTTPStatusErrors(t *testing.T) {
	tests := []struct {
		name    string
		status  int
		wantErr any
	}{
		{name: "rate limit", status: http.StatusTooManyRequests, wantErr: &RateLimitError{}},
		{name: "service unavailable", status: http.StatusServiceUnavailable, wantErr: &ServiceUnavailableError{}},
		{name: "gateway timeout", status: http.StatusGatewayTimeout, wantErr: &ServiceUnavailableError{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(`{"msg":"temporary failure"}`))
			}))
			defer server.Close()

			client, err := NewClient(
				WithToken("token"),
				WithBaseURL(server.URL),
			)
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}

			_, err = client.SubmitOCR(context.Background(), &OCRRequest{
				FileURL: "https://example.test/input.pdf",
			})
			if err == nil {
				t.Fatal("SubmitOCR() error = nil, want typed error")
			}

			switch tt.wantErr.(type) {
			case *RateLimitError:
				var got *RateLimitError
				if !errors.As(err, &got) {
					t.Fatalf("SubmitOCR() error = %T, want RateLimitError", err)
				}
			case *ServiceUnavailableError:
				var got *ServiceUnavailableError
				if !errors.As(err, &got) {
					t.Fatalf("SubmitOCR() error = %T, want ServiceUnavailableError", err)
				}
			default:
				t.Fatalf("unsupported test error type %T", tt.wantErr)
			}
		})
	}
}
