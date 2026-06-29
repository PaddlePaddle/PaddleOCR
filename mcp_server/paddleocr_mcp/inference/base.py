# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import abc
from typing import Any, Dict

from .shared.input_adapters import InputAdapter
from .types import InferenceRequest, InferenceResult


class Inference(abc.ABC):
    @property
    @abc.abstractmethod
    def input_adapter(self) -> InputAdapter:
        """Input adapter that validates and prepares user input for this source."""
        pass

    @abc.abstractmethod
    async def start(self) -> None:
        """Initialize inference resources."""
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        """Clean up inference resources."""
        pass

    @abc.abstractmethod
    async def predict(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Run inference for the given request."""
        pass

    @abc.abstractmethod
    def get_valid_params(self) -> set[str]:
        """Get valid runtime parameter names for this inference.

        Returns:
            Set of valid parameter names.
        """
        pass

    @abc.abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default runtime parameters for this inference.

        Returns:
            Dictionary of default parameters.
        """
        pass

    def validate_params(self, params: Dict[str, Any]) -> None:
        """Validate runtime parameters against allowed set.

        Args:
            params: Runtime parameters dictionary.

        Raises:
            ValueError: If any parameter is not in the valid set.
        """
        valid_params = self.get_valid_params()
        invalid_params = set(params.keys()) - valid_params
        if invalid_params:
            raise ValueError(
                f"Invalid runtime parameters: {invalid_params}. "
                f"Valid parameters are: {sorted(valid_params)}."
            )

    def get_final_params(self, runtime_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get final parameters with defaults merged.

        Args:
            runtime_params: User-provided runtime parameters.

        Returns:
            Dictionary with defaults overridden by user params.
        """
        final_params = self.get_default_params()
        final_params.update(runtime_params)
        return final_params
