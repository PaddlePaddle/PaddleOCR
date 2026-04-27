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

"""Shared helpers (calibration I/O) for ``quantize_onnx_model`` / ``debug_onnx_qdq`` / ``build_onnx_calib_npy``.

Entry scripts prepend this file's directory to ``sys.path`` before ``from utils import ...`` so
``import utils`` does not depend on the process working directory. The top-level name ``utils`` is
generic; the scripts resolve ``utils.py`` in this same ``scripts/`` directory first.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

__all__ = [
    "die",
    "user_input_names",
    "build_npy_dir_reader",
]


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def user_input_names(model_path: Path) -> list[str]:
    """Return non-initializer graph input names (user-visible inputs)."""
    import onnx

    m = onnx.load(str(model_path), load_external_data=True)
    init = {t.name for t in m.graph.initializer}
    return [i.name for i in m.graph.input if i.name not in init]


def build_npy_dir_reader(
    model_path: Path, data_dir: Path, *, max_samples: int | None = None
) -> Any:
    """ONNX Runtime ``CalibrationDataReader``: one float32 ``(NCHW)`` ``.npy`` per sample; sorted by filename.

    *model_path* is used to discover the single graph input name (``user_input_names``); it may be
    the original or an augmented model as long as the first user input is unchanged.
    """
    from onnxruntime.quantization import CalibrationDataReader

    class NpyDirectoryDataReader(CalibrationDataReader):
        def __init__(self) -> None:
            names = user_input_names(model_path)
            if len(names) != 1:
                die(
                    "expected exactly one non-initializer graph input; "
                    f"found {len(names)}: {names}"
                )
            self._input_name = names[0]
            files = sorted(data_dir.glob("*.npy"))
            if not files:
                die(f"no .npy files under {data_dir}")
            if max_samples is not None:
                files = files[: max(0, max_samples)]
            self._it = iter(files)

        def get_next(self) -> Any:
            import numpy as np

            try:
                path = next(self._it)
            except StopIteration:
                return None
            arr = np.load(str(path))
            if not isinstance(arr, np.ndarray):
                die(f"expected ndarray in {path}, got {type(arr)}")
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            return {self._input_name: arr}

    return NpyDirectoryDataReader()
