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

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import sys
import yaml
import os
from ppocr.utils.utility import create_module
from ppocr.utils.utility import initial_logger
logger = initial_logger()

import paddle.fluid as fluid
import time
from ppocr.utils.stats import TrainingStats
from eval_utils.eval_det_utils import eval_det_run
from eval_utils.eval_rec_utils import eval_rec_run
from ppocr.utils.save_load import save_model
import numpy as np
from ppocr.utils.character import cal_predicts_accuracy


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


global_config = AttrDict()


def load_config(file_path):
    """
    Load config from yml/yaml file.

    Args:
        file_path (str): Path of the config file to be loaded.

    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    merge_config(yaml.load(open(file_path), Loader=yaml.Loader))
    assert "reader_yml" in global_config['Global'],\
        "absence reader_yml in global"
    reader_file_path = global_config['Global']['reader_yml']
    _, ext = os.path.splitext(reader_file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for reader"
    merge_config(yaml.load(open(reader_file_path), Loader=yaml.Loader))
    return global_config


def merge_config(config):
    """
    Merge config into global config.

    Args:
        config (dict): Config to be merged.

    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                assert (sub_key in cur)
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            logger.error(err)
            sys.exit(1)
    except Exception as e:
        pass


def build(config, main_prog, startup_prog, mode):
    """
    Build a program using a model and an optimizer
        1. create feeds
        2. create a dataloader
        3. create a model
        4. create fetchs
        5. create an optimizer

    Args:
        config(dict): config
        main_prog(): main program
        startup_prog(): startup program
        is_train(bool): train or valid

    Returns:
        dataloader(): a bridge between the model and the data
        fetchs(dict): dict of model outputs(included loss and measures)
    """
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            func_infor = config['Architecture']['function']
            model = create_module(func_infor)(params=config)
            dataloader, outputs = model(mode=mode)
            fetch_name_list = list(outputs.keys())
            fetch_varname_list = [outputs[v].name for v in fetch_name_list]
            opt_loss_name = None
            if mode == "train":
                opt_loss = outputs['total_loss']
                opt_params = config['Optimizer']
                optimizer = create_module(opt_params['function'])(opt_params)
                optimizer.minimize(opt_loss)
                opt_loss_name = opt_loss.name
                global_lr = optimizer._global_learning_rate()
                fetch_name_list.insert(0, "lr")
                fetch_varname_list.insert(0, global_lr.name)
    return (dataloader, fetch_name_list, fetch_varname_list, opt_loss_name)


def build_export(config, main_prog, startup_prog):
    """
    """
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            func_infor = config['Architecture']['function']
            model = create_module(func_infor)(params=config)
            image, outputs = model(mode='export')
            fetches_var_name = sorted([name for name in outputs.keys()])
            fetches_var = [outputs[name] for name in fetches_var_name]
    feeded_var_names = [image.name]
    target_vars = fetches_var
    return feeded_var_names, target_vars, fetches_var_name


def create_multi_devices_program(program, loss_var_name):
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = True
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_iteration_per_drop_scope = 1
    compile_program = fluid.CompiledProgram(program).with_data_parallel(
        loss_name=loss_var_name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    return compile_program


def train_eval_det_run(config, exe, train_info_dict, eval_info_dict):
    train_batch_id = 0
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        logger.info(
            "During the training process, after the {}th iteration, an evaluation is run every {} iterations".
            format(start_eval_step, eval_batch_step))
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    train_stats = TrainingStats(log_smooth_window,
                                train_info_dict['fetch_name_list'])
    best_eval_hmean = -1
    best_batch_id = 0
    best_epoch = 0
    train_loader = train_info_dict['reader']
    for epoch in range(epoch_num):
        train_loader.start()
        try:
            while True:
                t1 = time.time()
                train_outs = exe.run(
                    program=train_info_dict['compile_program'],
                    fetch_list=train_info_dict['fetch_varname_list'],
                    return_numpy=False)
                stats = {}
                for tno in range(len(train_outs)):
                    fetch_name = train_info_dict['fetch_name_list'][tno]
                    fetch_value = np.mean(np.array(train_outs[tno]))
                    stats[fetch_name] = fetch_value
                t2 = time.time()
                train_batch_elapse = t2 - t1
                train_stats.update(stats)
                if train_batch_id > start_eval_step and (train_batch_id -start_eval_step)  \
                    % print_batch_step == 0:
                    logs = train_stats.log()
                    strs = 'epoch: {}, iter: {}, {}, time: {:.3f}'.format(
                        epoch, train_batch_id, logs, train_batch_elapse)
                    logger.info(strs)

                if train_batch_id > 0 and\
                    train_batch_id % eval_batch_step == 0:
                    metrics = eval_det_run(exe, config, eval_info_dict, "eval")
                    hmean = metrics['hmean']
                    if hmean >= best_eval_hmean:
                        best_eval_hmean = hmean
                        best_batch_id = train_batch_id
                        best_epoch = epoch
                        save_path = save_model_dir + "/best_accuracy"
                        save_model(train_info_dict['train_program'], save_path)
                    strs = 'Test iter: {}, metrics:{}, best_hmean:{:.6f}, best_epoch:{}, best_batch_id:{}'.format(
                        train_batch_id, metrics, best_eval_hmean, best_epoch,
                        best_batch_id)
                    logger.info(strs)
                train_batch_id += 1

        except fluid.core.EOFException:
            train_loader.reset()
        if epoch == 0 and save_epoch_step == 1:
            save_path = save_model_dir + "/iter_epoch_0"
            save_model(train_info_dict['train_program'], save_path)
        if epoch > 0 and epoch % save_epoch_step == 0:
            save_path = save_model_dir + "/iter_epoch_%d" % (epoch)
            save_model(train_info_dict['train_program'], save_path)
    return


def train_eval_rec_run(config, exe, train_info_dict, eval_info_dict):
    train_batch_id = 0
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        logger.info(
            "During the training process, after the {}th iteration, an evaluation is run every {} iterations".
            format(start_eval_step, eval_batch_step))
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    train_stats = TrainingStats(log_smooth_window, ['loss', 'acc'])
    best_eval_acc = -1
    best_batch_id = 0
    best_epoch = 0
    train_loader = train_info_dict['reader']
    for epoch in range(epoch_num):
        train_loader.start()
        try:
            while True:
                t1 = time.time()
                train_outs = exe.run(
                    program=train_info_dict['compile_program'],
                    fetch_list=train_info_dict['fetch_varname_list'],
                    return_numpy=False)
                fetch_map = dict(
                    zip(train_info_dict['fetch_name_list'],
                        range(len(train_outs))))

                loss = np.mean(np.array(train_outs[fetch_map['total_loss']]))
                lr = np.mean(np.array(train_outs[fetch_map['lr']]))
                preds_idx = fetch_map['decoded_out']
                preds = np.array(train_outs[preds_idx])
                preds_lod = train_outs[preds_idx].lod()[0]
                labels_idx = fetch_map['label']
                labels = np.array(train_outs[labels_idx])
                labels_lod = train_outs[labels_idx].lod()[0]

                acc, acc_num, img_num = cal_predicts_accuracy(
                    config['Global']['char_ops'], preds, preds_lod, labels,
                    labels_lod)
                t2 = time.time()
                train_batch_elapse = t2 - t1
                stats = {'loss': loss, 'acc': acc}
                train_stats.update(stats)
                if train_batch_id > start_eval_step and (train_batch_id - start_eval_step) \
                    % print_batch_step == 0:
                    logs = train_stats.log()
                    strs = 'epoch: {}, iter: {}, lr: {:.6f}, {}, time: {:.3f}'.format(
                        epoch, train_batch_id, lr, logs, train_batch_elapse)
                    logger.info(strs)

                if train_batch_id > 0 and\
                    train_batch_id % eval_batch_step == 0:
                    metrics = eval_rec_run(exe, config, eval_info_dict, "eval")
                    eval_acc = metrics['avg_acc']
                    eval_sample_num = metrics['total_sample_num']
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        best_batch_id = train_batch_id
                        best_epoch = epoch
                        save_path = save_model_dir + "/best_accuracy"
                        save_model(train_info_dict['train_program'], save_path)
                    strs = 'Test iter: {}, acc:{:.6f}, best_acc:{:.6f}, best_epoch:{}, best_batch_id:{}, eval_sample_num:{}'.format(
                        train_batch_id, eval_acc, best_eval_acc, best_epoch,
                        best_batch_id, eval_sample_num)
                    logger.info(strs)
                train_batch_id += 1

        except fluid.core.EOFException:
            train_loader.reset()
        if epoch == 0 and save_epoch_step == 1:
            save_path = save_model_dir + "/iter_epoch_0"
            save_model(train_info_dict['train_program'], save_path)
        if epoch > 0 and epoch % save_epoch_step == 0:
            save_path = save_model_dir + "/iter_epoch_%d" % (epoch)
            save_model(train_info_dict['train_program'], save_path)
    return
