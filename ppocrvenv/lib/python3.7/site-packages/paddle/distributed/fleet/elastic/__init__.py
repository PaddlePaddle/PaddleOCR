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

import signal
import os, sys

from .manager import ElasticManager
from .manager import ElasticStatus
from .manager import ELASTIC_EXIT_CODE
from .collective import CollectiveLauncher

from paddle.distributed.fleet.launch_utils import DistributeMode


def enable_elastic(args, distribute_mode):
    if distribute_mode != DistributeMode.COLLECTIVE:
        return False

    if not args.elastic_server and not os.getenv('PADDLE_ELASTIC_SERVER'):
        return False

    if not args.job_id and not os.getenv('PADDLE_ELASTIC_JOB_ID'):
        return False

    if not args.np and not int(os.getenv('PADDLE_ELASTIC_NP', 0)):
        return False

    return True


def launch_elastic(args, distribute_mode):

    elastic = ElasticManager(args)

    signal.signal(signal.SIGTERM, elastic.signal_handler)
    signal.signal(signal.SIGABRT, elastic.signal_handler)
    signal.signal(signal.SIGINT, elastic.signal_handler)

    while True:

        # wait for all nodes ready to run
        elastic.wait()

        # run self with specified launcher
        elastic.run(CollectiveLauncher)

        # keep wathing the health status of self and being notified for other's failure
        ret = elastic.watch()
        if ret == ElasticStatus.COMPLETED:
            break
        if ret == ElasticStatus.HOLD:
            continue
        if ret == ElasticStatus.EXIT:
            break
        if ret == ElasticStatus.ERROR:
            sys.exit(3)
        if ret == ElasticStatus.RESTART:
            sys.exit(ELASTIC_EXIT_CODE)

    if int(elastic.sigint) > 0:
        sys.exit(128 + int(elastic.sigint))
    else:
        sys.exit(0)
