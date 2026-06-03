# mcp_server/paddleocr_mcp/executors/base.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, Dict, Optional


class ExecutorError(RuntimeError):
    """Executor 执行错误基类"""


class AuthenticationError(ExecutorError):
    """认证失败"""


class ResourceUnavailableError(ExecutorError):
    """服务不可用"""


class ExecutionTimeoutError(ExecutorError):
    """超时"""


class Executor(abc.ABC):
    """执行器抽象基类，负责底层推理执行"""

    @abc.abstractmethod
    async def execute(
        self, input_data: str, file_type: Optional[str] = None, **options
    ) -> Dict[str, Any]:
        """执行推理，返回统一格式的结果

        Args:
            input_data: 输入数据（文件路径、URL 或 base64）
            file_type: 文件类型（"image" 或 "pdf"）
            **options: 其他选项

        Returns:
            统一格式的结果字典
        """

    @abc.abstractmethod
    async def start(self) -> None:
        """初始化资源"""

    @abc.abstractmethod
    async def stop(self) -> None:
        """清理资源"""
