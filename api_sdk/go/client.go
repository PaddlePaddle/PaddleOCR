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
	"net/http"
	"os"
	"strings"
	"time"
)

type Client struct {
	token          string
	baseURL        string
	jobsURL        string
	requestTimeout time.Duration
	pollTimeout    time.Duration
	clientPlatform string
	httpClient     *http.Client
}

func NewClient(opts ...ClientOption) (*Client, error) {
	c := &Client{
		requestTimeout: 5 * time.Minute,
		pollTimeout:    10 * time.Minute,
	}
	for _, opt := range opts {
		opt(c)
	}
	if c.token == "" {
		c.token = os.Getenv("PADDLEOCR_ACCESS_TOKEN")
	}
	if c.token == "" {
		return nil, &AuthError{PaddleOCRAPIError{Message: "Token is required. Set PADDLEOCR_ACCESS_TOKEN or use WithToken()."}}
	}
	if c.baseURL == "" {
		c.baseURL = os.Getenv("PADDLEOCR_BASE_URL")
	}
	if c.baseURL == "" {
		c.baseURL = DefaultBaseURL
	}
	c.baseURL = strings.TrimRight(c.baseURL, "/")
	c.jobsURL = c.baseURL + apiPath
	if c.httpClient == nil {
		c.httpClient = &http.Client{Timeout: c.requestTimeout}
	}
	return c, nil
}

func (c *Client) setClientPlatformHeader(req *http.Request) {
	if c.clientPlatform != "" {
		req.Header.Set("Client-Platform", c.clientPlatform)
	}
}
