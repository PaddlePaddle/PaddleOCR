# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time
import multiprocessing
import numpy as np

# from paddle.fluid.contrib.model_stat import summary


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid
from ppocr.utils.utility import create_module
from ppocr.utils.utility import load_config, merge_config
import ppocr.data.det.reader_main as reader
from ppocr.utils.utility import ArgsParser
from ppocr.utils.character import CharacterOps, cal_predicts_accuracy
from ppocr.utils.check import check_gpu
from ppocr.utils.stats import TrainingStats
from ppocr.utils.checkpoint import load_pretrain, load_checkpoint, save, save_model
from ppocr.utils.eval_utils import eval_run
from ppocr.utils.eval_utils import eval_det_run

from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.utils.utility import create_multi_devices_program


def main():
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    print(config)

    alg = config['Global']['algorithm']
    assert alg in ['EAST', 'DB']

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    det_model = create_module(config['Architecture']['function'])(params=config)

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            train_loader, train_outputs = det_model(mode="train")
            train_fetch_list = [v.name for v in train_outputs]
            train_loss = train_outputs[0]
            opt_params = config['Optimizer']
            optimizer = create_module(opt_params['function'])(opt_params)
            optimizer.minimize(train_loss)
            global_lr = optimizer._global_learning_rate()
            global_lr.persistable = True
            train_fetch_list.append(global_lr.name)

    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            eval_loader, eval_outputs = det_model(mode="eval")
            eval_fetch_list = [v.name for v in eval_outputs]
    eval_prog = eval_prog.clone(for_test=True)

    train_reader = reader.train_reader(config=config)
    train_loader.set_sample_list_generator(train_reader, places=place)

    exe.run(startup_prog)

    # compile program for multi-devices
    train_compile_program = create_multi_devices_program(train_prog,
                                                         train_loss.name)

    pretrain_weights = config['Global']['pretrain_weights']
    if pretrain_weights is not None:
        load_pretrain(exe, train_prog, pretrain_weights)
        print("pretrain weights loaded!")

    train_batch_id = 0
    if alg == 'EAST':
        train_log_keys = ['loss_total', 'loss_cls', 'loss_offset']
    elif alg == 'DB':
        train_log_keys = [
            'loss_total', 'loss_shrink', 'loss_threshold', 'loss_binary'
        ]
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_step = config['Global']['print_step']
    eval_step = config['Global']['eval_step']
    save_epoch_step = config['Global']['save_epoch_step']
    save_dir = config['Global']['save_dir']
    train_stats = TrainingStats(log_smooth_window, train_log_keys)
    best_eval_hmean = -1
    best_batch_id = 0
    best_epoch = 0
    for epoch in range(epoch_num):
        train_loader.start()
        try:
            while True:
                t1 = time.time()
                train_outs = exe.run(program=train_compile_program,
                                     fetch_list=train_fetch_list,
                                     return_numpy=False)
                loss_total = np.mean(np.array(train_outs[0]))
                if alg == 'EAST':
                    loss_cls = np.mean(np.array(train_outs[1]))
                    loss_offset = np.mean(np.array(train_outs[2]))
                    stats = {'loss_total':loss_total, 'loss_cls':loss_cls,\
                        'loss_offset':loss_offset}
                elif alg == 'DB':
                    loss_shrink_maps = np.mean(np.array(train_outs[1]))
                    loss_threshold_maps = np.mean(np.array(train_outs[2]))
                    loss_binary_maps = np.mean(np.array(train_outs[3]))
                    stats = {'loss_total':loss_total, 'loss_shrink':loss_shrink_maps, \
                        'loss_threshold':loss_threshold_maps, 'loss_binary':loss_binary_maps}
                lr = np.mean(np.array(train_outs[-1]))
                t2 = time.time()
                train_batch_elapse = t2 - t1

                # stats = {'loss_total':loss_total, 'loss_cls':loss_cls,\
                #     'loss_offset':loss_offset}
                train_stats.update(stats)
                if train_batch_id > 0 and train_batch_id % print_step == 0:
                    logs = train_stats.log()
                    strs = 'epoch: {}, iter: {}, lr: {:.6f}, {}, time: {:.3f}'.format(
                        epoch, train_batch_id, lr, logs, train_batch_elapse)
                    logger.info(strs)

                if train_batch_id > 0 and\
                    train_batch_id % eval_step == 0:
                    metrics = eval_det_run(exe, eval_prog, eval_fetch_list,
                                           config, "eval")
                    hmean = metrics['hmean']
                    if hmean >= best_eval_hmean:
                        best_eval_hmean = hmean
                        best_batch_id = train_batch_id
                        best_epoch = epoch
                        save_path = save_dir + "/best_accuracy"
                        save_model(train_prog, save_path)
                    strs = 'Test iter: {}, metrics:{}, best_hmean:{:.6f}, best_epoch:{}, best_batch_id:{}'.format(
                        train_batch_id, metrics, best_eval_hmean, best_epoch,
                        best_batch_id)
                    logger.info(strs)
                train_batch_id += 1

        except fluid.core.EOFException:
            train_loader.reset()

        if epoch > 0 and epoch % save_epoch_step == 0:
            save_path = save_dir + "/iter_epoch_%d" % (epoch)
            save_model(train_prog, save_path)


def test_reader():
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    print(config)
    tmp_reader = reader.train_reader(config=config)
    count = 0
    print_count = 0
    import time
    while True:
        starttime = time.time()
        count = 0
        for data in tmp_reader():
            count += 1
            if print_count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                print("reader:", count, len(data), batch_time)
    print("finish reader:", count)
    print("success")


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        default=None,
        type=str,
        help="Checkpoint path for resuming training.")
    FLAGS = parser.parse_args()
    main()
    # test_reader()
