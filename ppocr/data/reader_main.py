#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import random
import numpy as np

import paddle
from ppocr.utils.utility import create_module
from copy import deepcopy

from .rec.img_tools import process_image
import cv2

import sys
import signal


# handle terminate reader process, do not print stack frame
def _reader_quit(signum, frame):
    print("Reader process exit.")
    sys.exit()


def _term_group(sig_num, frame):
    print('pid {} terminated, terminate group '
          '{}...'.format(os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


signal.signal(signal.SIGTERM, _reader_quit)
signal.signal(signal.SIGINT, _term_group)


def reader_main(config=None, mode=None):
    """Create a reader for trainning

    Args:
        settings: arguments

    Returns:
        train reader
    """
    assert mode in ["train", "eval", "test"],\
        "Nonsupport mode:{}".format(mode)
    global_params = config['Global']
    if mode == "train":
        params = deepcopy(config['TrainReader'])
    elif mode == "eval":
        params = deepcopy(config['EvalReader'])
    else:
        params = deepcopy(config['TestReader'])
    params['mode'] = mode
    params.update(global_params)
    reader_function = params['reader_function']
    function = create_module(reader_function)(params)
    if mode == "train":
        if sys.platform == "win32":
            return function(0)
        readers = []
        num_workers = params['num_workers']
        for process_id in range(num_workers):
            readers.append(function(process_id))
        return paddle.reader.multiprocess_reader(readers, False)
    else:
        return function(mode)
