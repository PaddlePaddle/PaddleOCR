#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

from paddle.fluid.dygraph.parallel import ParallelEnv

__all__ = []


def setup_logger(output=None, name="hapi", log_level=logging.INFO):
    """
    Initialize logger of hapi and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger. Default: 'hapi'.
        log_level (enum): log level. eg.'INFO', 'DEBUG', 'ERROR'. Default: logging.INFO.
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(log_level)

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # stdout logging: only local rank==0
    local_rank = ParallelEnv().local_rank
    if local_rank == 0 and len(logger.handlers) == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(log_level)

        ch.setFormatter(logging.Formatter(format_str))
        logger.addHandler(ch)

    # file logging if output is not None: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        if local_rank > 0:
            filename = filename + ".rank{}".format(local_rank)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        fh = logging.StreamHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter(format_str))
        logger.addHandler(fh)

    return logger
