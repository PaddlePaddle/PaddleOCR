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
