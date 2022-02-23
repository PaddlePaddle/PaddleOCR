# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import os
import abc
from abc import abstractmethod
import paddle
from ..utils.env import PPNLP_HOME
from ..utils.log import logger
from .utils import download_check, static_mode_guard, dygraph_mode_guard, download_file


class Task(metaclass=abc.ABCMeta):
    """
    The meta classs of task in Taskflow. The meta class has the five abstract function,
        the subclass need to inherit from the meta class.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, model, task, **kwargs):
        self.model = model
        self.task = task
        self.kwargs = kwargs
        self._usage = ""
        # The dygraph model instantce 
        self._model = None
        # The static model instantce
        self._input_spec = None
        self._config = None
        # The root directory for storing Taskflow related files, default to ~/.paddlenlp.
        self._home_path = self.kwargs[
            'home_path'] if 'home_path' in self.kwargs else PPNLP_HOME
        self._task_flag = self.kwargs[
            'task_flag'] if 'task_flag' in self.kwargs else self.model
        if 'task_path' in self.kwargs:
            self._task_path = self.kwargs['task_path']
        else:
            self._task_path = os.path.join(self._home_path, "taskflow", 
                self.task, self.model)
        download_check(self._task_flag)

    @abstractmethod
    def _construct_model(self, model):
        """
       Construct the inference model for the predictor.
       """

    @abstractmethod
    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """

    @abstractmethod
    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """

    @abstractmethod
    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """

    @abstractmethod
    def _postprocess(self, inputs):
        """
        The model output is the logits and pros, this function will convert the model output to raw text.
        """

    @abstractmethod
    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """

    def _check_task_files(self):
        """
        Check files required by the task.
        """
        for file_id, file_name in self.resource_files_names.items():
            path = os.path.join(self._task_path, file_name)
            if not os.path.exists(path):
                url = self.resource_files_urls[self.model][file_id]
                download_file(self._task_path, file_name, url[0], url[1])

    def _prepare_static_mode(self):
        """
        Construct the input data and predictor in the PaddlePaddele static mode. 
        """
        place = paddle.get_device()
        if place == 'cpu':
            self._config.disable_gpu()
        else:
            self._config.enable_use_gpu(100, self.kwargs['device_id'])
            # TODO(linjieccc): enable embedding_eltwise_layernorm_fuse_pass after fixed
            self._config.delete_pass(
                "embedding_eltwise_layernorm_fuse_pass")
        self._config.switch_use_feed_fetch_ops(False)
        self._config.disable_glog_info()
        self._config.enable_memory_optim()
        self.predictor = paddle.inference.create_predictor(self._config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = [
            self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        ]

    def _get_inference_model(self):
        """
        Return the inference program, inputs and outputs in static mode. 
        """
        inference_model_path = os.path.join(self._task_path, "static",
                                            "inference")
        if not os.path.exists(inference_model_path + ".pdiparams"):
            with dygraph_mode_guard():
                self._construct_model(self.model)
                self._construct_input_spec()
                self._convert_dygraph_to_static()

        model_file = inference_model_path + ".pdmodel"
        params_file = inference_model_path + ".pdiparams"
        self._config = paddle.inference.Config(model_file, params_file)
        self._prepare_static_mode()

    def _convert_dygraph_to_static(self):
        """
        Convert the dygraph model to static model.
        """
        assert self._model is not None, 'The dygraph model must be created before converting the dygraph model to static model.'
        assert self._input_spec is not None, 'The input spec must be created before converting the dygraph model to static model.'
        logger.info("Converting to the inference model cost a little time.")
        static_model = paddle.jit.to_static(
            self._model, input_spec=self._input_spec)
        save_path = os.path.join(self._task_path, "static", "inference")
        paddle.jit.save(static_model, save_path)
        logger.info("The inference model save in the path:{}".format(save_path))

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError(
                    "Invalid inputs, input text should not be empty text, please check your input.".
                    format(type(inputs)))
            inputs = [inputs]
        elif isinstance(inputs, list):
            if not (isinstance(inputs[0], str) and len(inputs[0].strip()) > 0):
                raise TypeError(
                    "Invalid inputs, input text should be list of str, and first element of list should not be empty text.".
                    format(type(inputs[0])))
        else:
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, but type of {} found!".
                format(type(inputs)))
        return inputs

    def help(self):
        """
        Return the usage message of the current task.
        """
        print("Examples:\n{}".format(self._usage))

    def __call__(self, *args):
        inputs = self._preprocess(*args)
        outputs = self._run_model(inputs)
        results = self._postprocess(outputs)
        return results
