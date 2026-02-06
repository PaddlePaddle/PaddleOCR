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

import time

from .logging import logger


def str2bool(v, /):
    return v.lower() in ("true", "yes", "t", "y", "1")


def get_subcommand_args(args):
    args = vars(args).copy()
    args.pop("subcommand")
    args.pop("executor")
    return args


def add_simple_inference_args(subparser, *, input_help=None):
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


def perform_simple_inference(wrapper_cls, params, predict_param_names=None):
    params = params.copy()

    input_ = params.pop("input")
    save_path = params.pop("save_path")

    if predict_param_names is not None:
        predict_params = {}
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
