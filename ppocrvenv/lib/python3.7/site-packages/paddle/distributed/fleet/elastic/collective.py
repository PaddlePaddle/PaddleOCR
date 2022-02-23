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

from paddle.distributed.fleet import launch_utils
from paddle.distributed.fleet import cloud_utils
from paddle.distributed.fleet import ascend_utils

from paddle.distributed.fleet.launch_utils import *

from paddle.distributed.fleet.elastic.manager import LauncherInterface


class CollectiveLauncher(LauncherInterface):
    def __init__(self, args):
        self.args = args
        self.procs = []

    def launch(self):
        logger.info("collective lauchner launch ...")
        args = self.args
        # parse arguments, used for cloud-single-machine and local
        (device_mode,
         devices_per_proc) = launch_utils.get_device_proc_info(args)
        trainers_num = cloud_utils.get_trainers_num()
        logger.debug("parsed from args trainerss_num:{} mode:{} devices:{}".
                     format(trainers_num, device_mode, devices_per_proc))

        cluster = None
        pod = None

        start_port = 6170
        if os.environ.get('FLAGS_START_PORT') is not None:
            start_port = os.environ.get('FLAGS_START_PORT')
        if cloud_utils.use_paddlecloud() and trainers_num != 1:
            cluster, pod = cloud_utils.get_cloud_cluster(
                args.ips, device_mode, devices_per_proc, start_port)
            logger.debug("get cluster from cloud:{}".format(cluster))
        elif device_mode == DeviceMode.ASCEND_NPU:
            # for ascend
            cluster, pod = ascend_utils.get_cloud_cluster(
                rank_table_file=os.getenv("RANK_TABLE_FILE", None),
                device_mode=device_mode,
                start_port=start_port)
        else:
            # trainers_num = 1 or not use paddlecloud ips="a,b"
            cluster, pod = paddle.distributed.fleet.launch.get_cluster_from_args(
                args, device_mode, devices_per_proc)
            logger.debug("get cluster from args:{}".format(cluster))

        global_envs = copy.copy(os.environ.copy())
        self.gloo_rendezvous_dir = tempfile.mkdtemp()
        # add gloo env
        global_envs["PADDLE_WITH_GLOO"] = str(
            os.getenv("PADDLE_WITH_GLOO", "0"))
        global_envs["PADDLE_GLOO_RENDEZVOUS"] = "3"
        global_envs["PADDLE_GLOO_FS_PATH"] = self.gloo_rendezvous_dir

        self.procs = start_local_trainers(
            cluster,
            pod,
            training_script=args.training_script,
            training_script_args=args.training_script_args,
            log_dir=args.log_dir,
            envs=global_envs)

        for idx, proc in enumerate(self.procs):
            logger.info("launch proc_id:{} idx:{}".format(proc.proc.pid, idx))

    def stop(self):
        logger.info("collective lauchner stop ...")
        if not self._terminate_procs():
            logger.error("kill process failed")
        if os.path.exists(self.gloo_rendezvous_dir):
            shutil.rmtree(self.gloo_rendezvous_dir)

    def watch(self):
        logger.debug("collective lauchner watch ...")
        for p in self.procs:
            if p.log_fn and p.local_rank == 0:
                pull_worker_log(p)
        ret = self._check_procs()
        return ret
