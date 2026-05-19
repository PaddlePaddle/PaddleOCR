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

import (
	"net/http"
	"os"
	"time"
)

type Client struct {
	token      string
	baseURL    string
	timeout    time.Duration
	httpClient *http.Client
}

func NewClient(opts ...ClientOption) (*Client, error) {
	c := &Client{
		baseURL: DefaultBaseURL,
		timeout: 5 * time.Minute,
	}
	for _, opt := range opts {
		opt(c)
	}
	if c.token == "" {
		c.token = os.Getenv("PADDLE_OCR_TOKEN")
	}
	if c.token == "" {
		return nil, &AuthError{PaddleOCRError{Message: "Token is required. Set PADDLE_OCR_TOKEN or use WithToken()."}}
	}
	if c.httpClient == nil {
		c.httpClient = &http.Client{Timeout: c.timeout}
	}
	return c, nil
}
