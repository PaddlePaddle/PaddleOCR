# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import time
import logging

import paddle
import paddle.inference as paddle_infer

from pathlib import Path

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH_ROOT = f"{CUR_DIR}/../../tools/output"


class PaddleInferBenchmark(object):
    def __init__(self,
                 config,
                 model_info: dict,
                 data_info: dict,
                 perf_info: dict,
                 mem_info: dict,
                 **kwargs):
        """
        Construct PaddleInferBenchmark Class to format logs.
        args:
            config(paddle.inference.Config): paddle inference config
            model_info(dict): basic model info
            data_info(dict): input data info
            perf_info(dict): speed performance
            mem_info(dict): cpu and gpu resources
        """
        # PaddleInferBenchmark Log Version
        self.log_version = 1.0

        # Paddle Version
        self.paddle_version = paddle.__version__
        self.paddle_commit = paddle.__git_commit__
        paddle_infer_info = paddle_infer.get_version()
        self.paddle_branch = paddle_infer_info.strip().split(': ')[-1]

        # model info
        self.model_info = model_info
        self.model_name = model_info['model_name']
        self.precision = model_info['precision']

        # data info
        self.data_info = data_info
        self.batch_size = data_info['batch_size']
        self.shape = data_info['shape']
        self.data_num = data_info['data_num']

        # conf info
        self.config_status = self.parse_config(config)

        # perf info
        self.perf_info = perf_info
        self.preprocess_time = round(perf_info['preprocess_time'], 4)
        self.inference_time = round(perf_info['inference_time'], 4)
        self.postprocess_time = round(perf_info['postprocess_time'], 4)
        self.total_time = round(perf_info['total_time'], 4)

        # mem info
        self.mem_info = mem_info
        if mem_info:
            self.cpu_rss = int(mem_info['cpu_rss'])
            self.gpu_rss = int(mem_info['gpu_rss'])
            self.gpu_util = round(mem_info['gpu_util'], 2)
        else:
            self.cpu_rss = None
            self.gpu_rss = None
            self.gpu_util = None

        # init benchmark logger
        self.benchmark_logger()

    def benchmark_logger(self):
        """
        benchmark logger
        """
        # Init logger
        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_output = f"{LOG_PATH_ROOT}/{self.model_name}.log"
        Path(f"{LOG_PATH_ROOT}").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=FORMAT,
            handlers=[
                logging.FileHandler(
                    filename=log_output, mode='w'),
                logging.StreamHandler(),
            ])
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Paddle Inference benchmark log will be saved to {log_output}")

    def parse_config(self, config) -> dict:
        """
        parse paddle predictor config
        args:
            config(paddle.inference.Config): paddle inference config
        return:
            config_status(dict): dict style config info
        """
        config_status = {}
        config_status['runtime_device'] = "gpu" if config.use_gpu() else "cpu"
        config_status['ir_optim'] = config.ir_optim()
        config_status['enable_tensorrt'] = config.tensorrt_engine_enabled()
        config_status['precision'] = self.precision
        config_status['enable_mkldnn'] = config.mkldnn_enabled()
        config_status[
            'cpu_math_library_num_threads'] = config.cpu_math_library_num_threads(
            )
        return config_status

    def report(self, identifier=None):
        """
        print log report
        args:
            identifier(string): identify log
        """
        if identifier:
            identifier = f"[{identifier}]"
        else:
            identifier = ""

        self.logger.info("\n")
        self.logger.info(
            "---------------------- Paddle info ----------------------")
        self.logger.info(f"{identifier} paddle_version: {self.paddle_version}")
        self.logger.info(f"{identifier} paddle_commit: {self.paddle_commit}")
        self.logger.info(f"{identifier} paddle_branch: {self.paddle_branch}")
        self.logger.info(f"{identifier} log_style_version: {self.log_version}")
        self.logger.info(
            "----------------------- Conf info -----------------------")
        self.logger.info(
            f"{identifier} runtime_device: {self.config_status['runtime_device']}"
        )
        self.logger.info(
            f"{identifier} ir_optim: {self.config_status['ir_optim']}")
        self.logger.info(f"{identifier} enable_memory_optim: {True}")
        self.logger.info(
            f"{identifier} enable_tensorrt: {self.config_status['enable_tensorrt']}"
        )
        self.logger.info(f"{identifier} precision: {self.precision}")
        self.logger.info(
            f"{identifier} enable_mkldnn: {self.config_status['enable_mkldnn']}")
        self.logger.info(
            f"{identifier} cpu_math_library_num_threads: {self.config_status['cpu_math_library_num_threads']}"
        )
        self.logger.info(
            "----------------------- Model info ----------------------")
        self.logger.info(f"{identifier} model_name: {self.model_name}")
        self.logger.info(
            "----------------------- Data info -----------------------")
        self.logger.info(f"{identifier} batch_size: {self.batch_size}")
        self.logger.info(f"{identifier} input_shape: {self.shape}")
        self.logger.info(f"{identifier} data_num: {self.data_num}")
        self.logger.info(
            "----------------------- Perf info -----------------------")
        self.logger.info(
            f"{identifier} cpu_rss(MB): {self.cpu_rss}, gpu_rss(MB): {self.gpu_rss}, gpu_util: {self.gpu_util}%"
        )
        self.logger.info(f"{identifier} total time spent(s): {self.total_time}")
        self.logger.info(
            f"{identifier} preproce_time(ms): {round(self.preprocess_time*1000, 1)}, inference_time(ms): {round(self.inference_time*1000, 1)}, postprocess_time(ms): {round(self.postprocess_time*1000, 1)}"
        )

    def print_help(self):
        """
        print function help
        """
        self.logger.info("""Usage: 
            Print inference benchmark logs.
            log = PaddleInferBenchmark(config, model_name, batch_size, shape, precision, times, mem_info)
            log()
            """)

    def __call__(self, identifier=None):
        """
        __call__
        args:
            identifier(string): identify log
        """
        self.report(identifier)
