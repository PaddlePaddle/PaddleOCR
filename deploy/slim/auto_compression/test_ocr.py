# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys
import cv2
import numpy as np
import paddle
import logging
import numpy as np
import argparse
from tqdm import tqdm
import paddle
from paddleslim.common import load_config as load_slim_config
from paddleslim.common import get_logger

import sys
sys.path.append('../../../')
from ppocr.data import build_dataloader
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

logger = get_logger(__name__, level=logging.INFO)


def load_predictor(args):
    """
    load predictor func
    """
    rerun_flag = False
    model_file = os.path.join(args.model_path, args.model_filename)
    params_file = os.path.join(args.model_path, args.params_filename)
    pred_cfg = PredictConfig(model_file, params_file)
    pred_cfg.enable_memory_optim()
    pred_cfg.switch_ir_optim(True)
    if args.device == "GPU":
        pred_cfg.enable_use_gpu(100, 0)
    else:
        pred_cfg.disable_gpu()
        pred_cfg.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.use_mkldnn:
            pred_cfg.enable_mkldnn()
            if args.precision == "int8":
                pred_cfg.enable_mkldnn_int8({
                    "conv2d", "depthwise_conv2d", "pool2d", "elementwise_mul"
                })

    if args.use_trt:
        # To collect the dynamic shapes of inputs for TensorRT engine
        dynamic_shape_file = os.path.join(args.model_path, "dynamic_shape.txt")
        if os.path.exists(dynamic_shape_file):
            pred_cfg.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                         True)
            # pred_cfg.exp_disable_tensorrt_ops(["reshape2"])
            print("trt set dynamic shape done!")
            precision_map = {
                "fp16": PrecisionType.Half,
                "fp32": PrecisionType.Float32,
                "int8": PrecisionType.Int8
            }
            pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=4,
                precision_mode=precision_map[args.precision],
                use_static=True,
                use_calib_mode=False, )
        else:
            pred_cfg.disable_gpu()
            pred_cfg.set_cpu_math_library_num_threads(10)
            pred_cfg.collect_shape_range_info(dynamic_shape_file)
            print("Start collect dynamic shape...")
            rerun_flag = True

    pred_cfg.exp_disable_tensorrt_ops(["reshape2"])
    # pred_cfg.delete_pass("gpu_cpu_map_matmul_v2_to_mul_pass")
    # pred_cfg.delete_pass("delete_quant_dequant_linear_op_pass")
    # pred_cfg.delete_pass("delete_weight_dequant_linear_op_pass")
    predictor = create_predictor(pred_cfg)
    return predictor, rerun_flag


def eval(args):
    """
    eval mIoU func
    """
    # DataLoader need run on cpu
    paddle.set_device("cpu")
    devices = paddle.device.get_device().split(':')[0]

    val_loader = build_dataloader(all_config, 'Eval', devices, logger)
    post_process_class = build_post_process(all_config['PostProcess'],
                                            global_config)
    eval_class = build_metric(all_config['Metric'])
    model_type = global_config['model_type']

    predictor, rerun_flag = load_predictor(args)

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    sample_nums = len(val_loader)
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    print("Start evaluating ( total_iters: {}).".format(sample_nums))

    # preprocess_op = create_operators(pre_process_list)
    # data = transform(data, self.preprocess_op)
    for batch_id, batch in enumerate(val_loader):
        
        images = np.array(batch[0])

        batch_numpy = []
        for item in batch:
            batch_numpy.append(np.array(item))

        # ori_shape = np.array(batch_numpy).shape[-2:]
        input_handle.reshape(images.shape)
        input_handle.copy_from_cpu(images)
        start_time = time.time()

        predictor.run()
        preds = output_handle.copy_to_cpu()

        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed

        if model_type == 'det':
            preds_map = {'maps': preds}
            post_result = post_process_class(preds_map, batch_numpy[1])
            eval_class(post_result, batch_numpy)
        elif model_type == 'rec':
            post_result = post_process_class(preds, batch_numpy[1])
            eval_class(post_result, batch_numpy)

        if rerun_flag:
            print(
                "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
            )
            return
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()


        metric = eval_class.get_metric()
        
    time_avg = predict_time / sample_nums
    print(
        "[Benchmark] Inference time(ms): min={}, max={}, avg={}".
        format(
               round(time_min * 1000, 2),
               round(time_max * 1000, 1), round(time_avg * 1000, 1)))
    for k, v in metric.items():
        print('{}:{}'.format(k, v))
    sys.stdout.flush()



def main():
    global all_config, global_config
    all_config = load_slim_config(args.config_path)
    global_config = all_config["Global"]
    eval(args)


if __name__ == "__main__":
    paddle.enable_static()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, help="inference model filepath")
    parser.add_argument(
        "--config_path", 
        type=str,
        default='./configs/ppocrv3_det_qat_dist.yaml',
        help="path of compression strategy config.")
    parser.add_argument(
        "--model_filename",
        type=str,
        default="inference.pdmodel",
        help="model file name")
    parser.add_argument(
        "--params_filename",
        type=str,
        default="inference.pdiparams",
        help="params file name")
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["CPU", "GPU"],
        help="Choose the device you want to run, it can be: CPU/GPU, default is GPU",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="The precision of inference. It can be 'fp32', 'fp16' or 'int8'. Default is 'fp16'.",
    )
    parser.add_argument(
        "--use_trt",
        type=bool,
        default=False,
        help="Whether to use tensorrt engine or not.")
    parser.add_argument(
        "--use_mkldnn",
        type=bool,
        default=False,
        help="Whether use mkldnn or not.")
    parser.add_argument(
        "--cpu_threads", type=int, default=10, help="Num of cpu threads.")
    args = parser.parse_args()
    main()