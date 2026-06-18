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
	"time"
)

const DefaultBaseURL = "https://paddleocr.aistudio-app.com"

const apiPath = "/api/v2/ocr/jobs"

type ClientOption func(*Client)

func WithToken(token string) ClientOption {
	return func(c *Client) {
		c.token = token
	}
}

func WithBaseURL(url string) ClientOption {
	return func(c *Client) {
		c.baseURL = url
	}
}

func WithTimeout(d time.Duration) ClientOption {
	return func(c *Client) {
		c.requestTimeout = d
		c.pollTimeout = d
	}
}

func WithRequestTimeout(d time.Duration) ClientOption {
	return func(c *Client) {
		c.requestTimeout = d
	}
}

func WithPollTimeout(d time.Duration) ClientOption {
	return func(c *Client) {
		c.pollTimeout = d
	}
}

func WithClientPlatform(clientPlatform string) ClientOption {
	return func(c *Client) {
		c.clientPlatform = clientPlatform
	}
}

func WithHTTPClient(hc *http.Client) ClientOption {
	return func(c *Client) {
		c.httpClient = hc
	}
}

func Bool(v bool) *bool {
	return &v
}
