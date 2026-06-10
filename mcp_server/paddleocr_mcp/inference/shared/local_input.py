# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import io
import re
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse


class LocalInputProcessor:
    @staticmethod
    def is_file_path(value: str) -> bool:
        return Path(value).expanduser().exists()

    @staticmethod
    def is_url(value: str) -> bool:
        if not (value.startswith("http://") or value.startswith("https://")):
            return False

        result = urlparse(value)
        return all([result.scheme, result.netloc]) and result.scheme in (
            "http",
            "https",
        )

    @staticmethod
    def is_base64(value: str) -> bool:
        if value.startswith("data:"):
            return "," in value and LocalInputProcessor.is_base64(
                value.split(",", 1)[1]
            )

        pattern = r"^[A-Za-z0-9+/]+={0,2}$"
        return bool(re.fullmatch(pattern, value))

    @staticmethod
    def infer_file_type_from_bytes(data: bytes) -> Optional[str]:
        import puremagic

        mime = puremagic.from_string(data, mime=True)
        if mime.startswith("image/"):
            return "image"
        if mime == "application/pdf":
            return "pdf"
        return None

    @classmethod
    def process_for_local(cls, input_data: str) -> Union[str, Any]:
        if cls.is_url(input_data) or cls.is_file_path(input_data):
            return input_data

        if cls.is_base64(input_data):
            import numpy as np
            from PIL import Image as PILImage

            base64_data = (
                input_data.split(",", 1)[1]
                if input_data.startswith("data:")
                else input_data
            )
            try:
                image_bytes = base64.b64decode(base64_data)
                file_type = cls.infer_file_type_from_bytes(image_bytes)
                if file_type != "image":
                    raise ValueError("Currently, only images can be passed via Base64.")
                image_pil = PILImage.open(io.BytesIO(image_bytes))
                image_arr = np.array(image_pil.convert("RGB"))
                return np.ascontiguousarray(image_arr[..., ::-1])
            except Exception as e:
                raise ValueError(f"Failed to decode Base64 image: {str(e)}") from e

        raise ValueError("Invalid input data format")
