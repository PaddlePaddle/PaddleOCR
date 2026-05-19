# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class PaddleOCRError(Exception):
    """Base exception for PaddleOCR API SDK."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class AuthError(PaddleOCRError):
    """Token missing, invalid, or expired (HTTP 401/403)."""


class InvalidRequestError(PaddleOCRError):
    """Invalid parameters (HTTP 400)."""


class APIError(PaddleOCRError):
    """Non-2xx response from the API server."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class JobFailedError(PaddleOCRError):
    """Job execution failed on the server side."""

    def __init__(self, job_id: str, error_msg: str):
        self.job_id = job_id
        self.error_msg = error_msg
        super().__init__(f"Job {job_id} failed: {error_msg}")


class TimeoutError(PaddleOCRError):
    """Polling timed out waiting for job completion."""

    def __init__(self, job_id: str, elapsed: float):
        self.job_id = job_id
        self.elapsed = elapsed
        super().__init__(
            f"Timed out after {elapsed:.1f}s waiting for job {job_id}"
        )


class NetworkError(PaddleOCRError):
    """Network connection failure."""


class FileNotFoundError(PaddleOCRError):
    """Local file does not exist."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"File not found: {path}")
