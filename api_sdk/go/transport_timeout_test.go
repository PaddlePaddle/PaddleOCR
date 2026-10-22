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
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestAPIResponseBodyErrors(t *testing.T) {
	requests := map[string]func(*Client) error{
		"submit": func(c *Client) error {
			_, err := c.SubmitOCR(context.Background(), &OCRRequest{FileURL: "https://example.test/input.pdf"})
			return err
		},
		"status": func(c *Client) error {
			_, err := c.GetStatus(context.Background(), "job-1")
			return err
		},
	}
	for name, request := range requests {
		for _, delayed := range []bool{false, true} {
			kind := "malformed"
			if delayed {
				kind = "timeout"
			}
			t.Run(name+"/"+kind, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					if delayed {
						w.WriteHeader(http.StatusOK)
						w.(http.Flusher).Flush()
						<-r.Context().Done()
					} else {
						_, _ = io.WriteString(w, "not JSON")
					}
				}))
				defer server.Close()
				client, err := NewClient(WithToken("test-token"), WithBaseURL(server.URL), WithRequestTimeout(200*time.Millisecond))
				if err != nil {
					t.Fatal(err)
				}
				err = request(client)
				var timeout *RequestTimeoutError
				var format *ResponseFormatError
				if delayed {
					if !errors.As(err, &timeout) || !errors.Is(err, context.DeadlineExceeded) {
						t.Fatalf("expected RequestTimeoutError wrapping deadline, got %T: %v", err, err)
					}
					if errors.As(err, &format) {
						t.Fatal("timeout reported as malformed JSON")
					}
				} else if !errors.As(err, &format) {
					t.Fatalf("expected ResponseFormatError, got %T: %v", err, err)
				}
			})
		}
	}
}
