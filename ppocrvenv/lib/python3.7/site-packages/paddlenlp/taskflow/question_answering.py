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

import glob
import json
import math
import os
import copy
import itertools

import numpy as np
from .utils import download_file
from .text_generation import TextGenerationTask

usage = r"""
           from paddlenlp import Taskflow 

           qa = Taskflow("question_answering")
           qa("中国的国土面积有多大？")
           '''
           [{'text': '中国的国土面积有多大？', 'answer': '960万平方公里。'}]
           '''

           qa(["中国国土面积有多大？", "中国的首都在哪里？"])
           '''
           [{'text': '中国国土面积有多大？', 'answer': '960万平方公里。'}, {'text': '中国的首都在哪里？', 'answer': '北京。'}]
           '''
           
         """

URLS = {
    "gpt-cpm-large-cn": [
        "https://bj.bcebos.com/paddlenlp/taskflow/text_generation/gpt-cpm/gpt-cpm-large-cn_params.tar",
        "5aad6f81053cfdbba4797f044fcf66d1"
    ],
}


class QuestionAnsweringTask(TextGenerationTask):
    """
    The text generation model to predict the question or chinese  poetry. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
