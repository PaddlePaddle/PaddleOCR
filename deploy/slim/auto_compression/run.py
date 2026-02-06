# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import logging
from tqdm import tqdm
import numpy as np
import argparse
import paddle
from paddleslim.common import load_config as load_slim_config
from paddleslim.common import get_logger
from paddleslim.auto_compression import AutoCompression
from paddleslim.common.dataloader import get_feed_vars

import sys

sys.path.append("../../../")
from ppocr.data import build_dataloader
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric

logger = get_logger(__name__, level=logging.INFO)


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output",
        help="directory to save compressed model.",
    )
    parser.add_argument(
        "--devices", type=str, default="gpu", help="which device used to compress."
    )
    return parser


def reader_wrapper(reader, input_name):
    if isinstance(input_name, list) and len(input_name) == 1:
        input_name = input_name[0]

    def gen():  # 形成一个字典输入
        for i, batch in enumerate(reader()):
            yield {input_name: batch[0]}

    return gen


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    post_process_class = build_post_process(all_config["PostProcess"], global_config)
    eval_class = build_metric(all_config["Metric"])
    model_type = global_config["model_type"]

    with tqdm(
        total=len(val_loader),
        bar_format="Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}",
        ncols=80,
    ) as t:
        for batch_id, batch in enumerate(val_loader):
            images = batch[0]

            try:
                (preds,) = exe.run(
                    compiled_test_program,
                    feed={test_feed_names[0]: images},
                    fetch_list=test_fetch_list,
                )
            except:
                preds, _ = exe.run(
                    compiled_test_program,
                    feed={test_feed_names[0]: images},
                    fetch_list=test_fetch_list,
                )

            batch_numpy = []
            for item in batch:
                batch_numpy.append(np.array(item))

            if model_type == "det":
                preds_map = {"maps": preds}
                post_result = post_process_class(preds_map, batch_numpy[1])
                eval_class(post_result, batch_numpy)
            elif model_type == "rec":
                post_result = post_process_class(preds, batch_numpy[1])
                eval_class(post_result, batch_numpy)
            t.update()
        metric = eval_class.get_metric()
    logger.info("metric eval ***************")
    for k, v in metric.items():
        logger.info("{}:{}".format(k, v))

    if model_type == "det":
        return metric["hmean"]
    elif model_type == "rec":
        return metric["acc"]
    return metric


def main():
    rank_id = paddle.distributed.get_rank()
    if args.devices == "gpu":
        place = paddle.CUDAPlace(rank_id)
        paddle.set_device("gpu")
    else:
        place = paddle.CPUPlace()
        paddle.set_device("cpu")

    global all_config, global_config
    all_config = load_slim_config(args.config_path)

    if "Global" not in all_config:
        raise KeyError(f"Key 'Global' not found in config file. \n{all_config}")
    global_config = all_config["Global"]

    gpu_num = paddle.distributed.get_world_size()

    train_dataloader = build_dataloader(all_config, "Train", args.devices, logger)

    global val_loader
    val_loader = build_dataloader(all_config, "Eval", args.devices, logger)

    if (
        isinstance(all_config["TrainConfig"]["learning_rate"], dict)
        and all_config["TrainConfig"]["learning_rate"]["type"] == "CosineAnnealingDecay"
    ):
        steps = len(train_dataloader) * all_config["TrainConfig"]["epochs"]
        all_config["TrainConfig"]["learning_rate"]["T_max"] = steps
        print("total training steps:", steps)

    global_config["input_name"] = get_feed_vars(
        global_config["model_dir"],
        global_config["model_filename"],
        global_config["params_filename"],
    )

    ac = AutoCompression(
        model_dir=global_config["model_dir"],
        model_filename=global_config["model_filename"],
        params_filename=global_config["params_filename"],
        save_dir=args.save_dir,
        config=all_config,
        train_dataloader=reader_wrapper(train_dataloader, global_config["input_name"]),
        eval_callback=eval_function if rank_id == 0 else None,
        eval_dataloader=reader_wrapper(val_loader, global_config["input_name"]),
    )
    ac.compress()


if __name__ == "__main__":
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main()
