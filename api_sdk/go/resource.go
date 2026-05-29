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
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

type SaveResourceOption func(*saveResourceOptions)

type saveResourceOptions struct {
	overwrite bool
}

func WithOverwrite(overwrite bool) SaveResourceOption {
	return func(o *saveResourceOptions) {
		o.overwrite = overwrite
	}
}

// SaveResource downloads one result resource URL.
func (c *Client) SaveResource(ctx context.Context, resourceURL, dest string, opts ...SaveResourceOption) (string, error) {
	if resourceURL == "" {
		return "", &InvalidRequestError{PaddleOCRAPIError{Message: "resource URL is required"}}
	}
	if dest == "" {
		return "", &InvalidRequestError{PaddleOCRAPIError{Message: "destination path is required"}}
	}
	options := saveResourceOptions{}
	for _, opt := range opts {
		opt(&options)
	}
	savedPath, err := resolveSavePath(resourceURL, dest)
	if err != nil {
		return "", err
	}
	parent := filepath.Dir(savedPath)
	if st, err := os.Stat(parent); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", &FileNotFoundError{Path: parent, PaddleOCRAPIError: PaddleOCRAPIError{Message: "destination parent not found: " + parent, Cause: err}}
		}
		return "", err
	} else if !st.IsDir() {
		return "", &FileNotFoundError{Path: parent, PaddleOCRAPIError: PaddleOCRAPIError{Message: "destination parent is not a directory: " + parent}}
	}
	if !options.overwrite {
		if _, err := os.Stat(savedPath); err == nil {
			return "", &InvalidRequestError{PaddleOCRAPIError{Message: "destination already exists: " + savedPath}}
		} else if err != nil && !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, resourceURL, nil)
	if err != nil {
		return "", &NetworkError{PaddleOCRAPIError{Message: err.Error(), Cause: err}}
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", classifyHTTPError(err)
	}
	defer resp.Body.Close()
	if err := raiseForResourceResponse(resp); err != nil {
		return "", err
	}

	tmp, err := os.CreateTemp(parent, "."+filepath.Base(savedPath)+".tmp-*")
	if err != nil {
		return "", err
	}
	tmpPath := tmp.Name()
	cleanupTemp := true
	defer func() {
		if cleanupTemp {
			_ = os.Remove(tmpPath)
		}
	}()
	if _, err := io.Copy(tmp, resp.Body); err != nil {
		_ = tmp.Close()
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if options.overwrite {
		if err := os.Rename(tmpPath, savedPath); err != nil {
			return "", err
		}
	} else {
		if err := os.Link(tmpPath, savedPath); err != nil {
			if errors.Is(err, os.ErrExist) {
				return "", &InvalidRequestError{PaddleOCRAPIError{Message: "destination already exists: " + savedPath, Cause: err}}
			}
			return "", err
		}
		if err := os.Remove(tmpPath); err != nil {
			return "", err
		}
	}
	cleanupTemp = false
	return savedPath, nil
}

func (c *Client) SaveOCRResultResources(ctx context.Context, result *OCRResult, destDir string, opts ...SaveResourceOption) ([]string, error) {
	if result == nil {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "OCR result is required"}}
	}
	if err := requireExistingDirectory(destDir); err != nil {
		return nil, err
	}
	saved := make([]string, 0, len(result.Pages))
	for i, page := range result.Pages {
		if page.OCRImageURL == "" {
			continue
		}
		name := fmt.Sprintf("ocr-page-%d%s", i+1, safeResourceExtension(page.OCRImageURL))
		if err := validateResultResourceFilename(name); err != nil {
			return nil, err
		}
		path, err := c.SaveResource(ctx, page.OCRImageURL, filepath.Join(destDir, name), opts...)
		if err != nil {
			return nil, err
		}
		saved = append(saved, path)
	}
	return saved, nil
}

func (c *Client) SaveDocumentParsingResultResources(ctx context.Context, result *DocParsingResult, destDir string, opts ...SaveResourceOption) ([]string, error) {
	if result == nil {
		return nil, &InvalidRequestError{PaddleOCRAPIError{Message: "document parsing result is required"}}
	}
	if err := requireExistingDirectory(destDir); err != nil {
		return nil, err
	}
	var saved []string
	for _, page := range result.Pages {
		paths, err := c.saveNamedResourceMap(ctx, page.MarkdownImages, destDir, opts...)
		if err != nil {
			return nil, err
		}
		saved = append(saved, paths...)
		paths, err = c.saveNamedResourceMap(ctx, page.OutputImages, destDir, opts...)
		if err != nil {
			return nil, err
		}
		saved = append(saved, paths...)
	}
	return saved, nil
}

func (c *Client) saveNamedResourceMap(ctx context.Context, resources map[string]string, destDir string, opts ...SaveResourceOption) ([]string, error) {
	keys := make([]string, 0, len(resources))
	for key := range resources {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	saved := make([]string, 0, len(keys))
	for _, key := range keys {
		resourceURL := resources[key]
		if resourceURL == "" {
			continue
		}
		name, err := resultResourceFilename(key)
		if err != nil {
			return nil, err
		}
		path, err := c.SaveResource(ctx, resourceURL, filepath.Join(destDir, name), opts...)
		if err != nil {
			return nil, err
		}
		saved = append(saved, path)
	}
	return saved, nil
}

func resolveSavePath(resourceURL, dest string) (string, error) {
	if st, err := os.Stat(dest); err == nil && st.IsDir() {
		name := "resource"
		u, parseErr := url.Parse(resourceURL)
		if parseErr == nil {
			if base := path.Base(u.Path); base != "." && base != "/" && base != "" {
				name = base
			}
		}
		return filepath.Join(dest, name), nil
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	return dest, nil
}

func raiseForResourceResponse(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	body, _ := io.ReadAll(resp.Body)
	return &APIError{StatusCode: resp.StatusCode, PaddleOCRAPIError: PaddleOCRAPIError{Message: string(body)}}
}

func requireExistingDirectory(destDir string) error {
	if destDir == "" {
		return &InvalidRequestError{PaddleOCRAPIError{Message: "destination directory is required"}}
	}
	st, err := os.Stat(destDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return &FileNotFoundError{Path: destDir, PaddleOCRAPIError: PaddleOCRAPIError{Message: "destination directory not found: " + destDir, Cause: err}}
		}
		return err
	}
	if !st.IsDir() {
		return &FileNotFoundError{Path: destDir, PaddleOCRAPIError: PaddleOCRAPIError{Message: "destination is not a directory: " + destDir}}
	}
	return nil
}

func resultResourceFilename(key string) (string, error) {
	if err := validateResultResourceFilename(key); err != nil {
		return "", err
	}
	return key, nil
}

func validateResultResourceFilename(name string) error {
	if name == "" || name == "." || name == ".." || filepath.IsAbs(name) || path.IsAbs(name) || strings.ContainsAny(name, `/\`) {
		return &InvalidRequestError{PaddleOCRAPIError{Message: "unsafe resource filename: " + name}}
	}
	return nil
}

func safeResourceExtension(resourceURL string) string {
	u, err := url.Parse(resourceURL)
	if err != nil {
		return ""
	}
	ext := path.Ext(path.Base(u.Path))
	if len(ext) <= 1 {
		return ""
	}
	if err := validateResultResourceFilename("resource" + ext); err != nil {
		return ""
	}
	return ext
}
