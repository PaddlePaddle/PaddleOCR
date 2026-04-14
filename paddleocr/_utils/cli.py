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

import argparse
import time
from typing import Any, Dict, Optional, Set

from .logging import logger


def str2bool(v: str, /) -> bool:
    return v.lower() in ("true", "yes", "t", "y", "1")


def get_subcommand_args(args: argparse.Namespace) -> Dict[str, Any]:
    args_dict = vars(args).copy()
    args_dict.pop("subcommand")
    args_dict.pop("executor")
    return args_dict


def add_simple_inference_args(
    subparser: argparse.ArgumentParser, *, input_help: Optional[str] = None
) -> None:
    if input_help is None:
        input_help = "Input path or URL."
    subparser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help=input_help,
    )
    subparser.add_argument(
        "--save_path",
        type=str,
        help="Path to the output directory.",
    )


def perform_simple_inference(
    wrapper_cls: type,
    params: Dict[str, Any],
    predict_param_names: Optional[Set[str]] = None,
) -> None:
    params = params.copy()

    input_ = params.pop("input")
    save_path = params.pop("save_path")

    if predict_param_names is not None:
        predict_params: Dict[str, Any] = {}
        for name in predict_param_names:
            predict_params[name] = params.pop(name)
    else:
        predict_params = {}
    init_params = params

    wrapper = wrapper_cls(**init_params)

    try:
        result = wrapper.predict_iter(input_, **predict_params)

        t1 = time.time()
        for i, res in enumerate(result):
            logger.info(f"Processed item {i} in {(time.time()-t1) * 1000} ms")
            t1 = time.time()
            res.print()
            if save_path:
                res.save_all(save_path)
    finally:
        wrapper.close()
