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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import platform
import yaml
import time
import datetime
import paddle
import paddle.distributed as dist
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from ppocr.utils.stats import TrainingStats
from ppocr.utils.save_load import save_model
from ppocr.utils.utility import print_dict, AverageMeter
from ppocr.utils.logging import get_logger
from ppocr.utils.loggers import VDLLogger, WandbLogger, Loggers
from ppocr.utils import profiler
from ppocr.data import build_dataloader


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )

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


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def check_device(use_gpu, use_xpu=False, use_npu=False, use_mlu=False):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config {} cannot be set as true while your paddle " \
          "is not compiled with {} ! \nPlease try: \n" \
          "\t1. Install paddlepaddle to run model on {} \n" \
          "\t2. Set {} as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and use_xpu:
            print("use_xpu and use_gpu can not both be ture.")
        if use_gpu and not paddle.is_compiled_with_cuda():
            print(err.format("use_gpu", "cuda", "gpu", "use_gpu"))
            sys.exit(1)
        if use_xpu and not paddle.device.is_compiled_with_xpu():
            print(err.format("use_xpu", "xpu", "xpu", "use_xpu"))
            sys.exit(1)
        if use_npu and not paddle.device.is_compiled_with_npu():
            print(err.format("use_npu", "npu", "npu", "use_npu"))
            sys.exit(1)
        if use_mlu and not paddle.device.is_compiled_with_mlu():
            print(err.format("use_mlu", "mlu", "mlu", "use_mlu"))
            sys.exit(1)
    except Exception as e:
        pass


def to_float32(preds):
    if isinstance(preds, dict):
        for k in preds:
            if isinstance(preds[k], dict) or isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, list):
        for k in range(len(preds)):
            if isinstance(preds[k], dict):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, paddle.Tensor):
        preds = preds.astype(paddle.float32)
    return preds


def train(config,
          train_dataloader,
          valid_dataloader,
          device,
          model,
          loss_class,
          optimizer,
          lr_scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          log_writer=None,
          scaler=None,
          amp_level='O2',
          amp_custom_black_list=[]):
    cal_metric_during_train = config['Global'].get('cal_metric_during_train',
                                                   False)
    calc_epoch_interval = config['Global'].get('calc_epoch_interval', 1)
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']
    profiler_options = config['profiler_options']

    global_step = 0
    if 'global_step' in pre_best_model_dict:
        global_step = pre_best_model_dict['global_step']
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        if len(valid_dataloader) == 0:
            logger.info(
                'No Images in eval dataset, evaluation during training ' \
                'will be disabled'
            )
            start_eval_step = 1e111
        logger.info(
            "During the training process, after the {}th iteration, " \
            "an evaluation is run every {} iterations".
            format(start_eval_step, eval_batch_step))
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    train_stats = TrainingStats(log_smooth_window, ['lr'])
    model_average = False
    model.train()

    use_srn = config['Architecture']['algorithm'] == "SRN"
    extra_input_models = [
        "SRN", "NRTR", "SAR", "SEED", "SVTR", "SPIN", "VisionLAN",
        "RobustScanner", "RFL", 'DRRG'
    ]
    extra_input = False
    if config['Architecture']['algorithm'] == 'Distillation':
        for key in config['Architecture']["Models"]:
            extra_input = extra_input or config['Architecture']['Models'][key][
                'algorithm'] in extra_input_models
    else:
        extra_input = config['Architecture']['algorithm'] in extra_input_models
    try:
        model_type = config['Architecture']['model_type']
    except:
        model_type = None

    algorithm = config['Architecture']['algorithm']

    start_epoch = best_model_dict[
        'start_epoch'] if 'start_epoch' in best_model_dict else 1

    total_samples = 0
    train_reader_cost = 0.0
    train_batch_cost = 0.0
    reader_start = time.time()
    eta_meter = AverageMeter()

    max_iter = len(train_dataloader) - 1 if platform.system(
    ) == "Windows" else len(train_dataloader)

    for epoch in range(start_epoch, epoch_num + 1):
        if train_dataloader.dataset.need_reset:
            train_dataloader = build_dataloader(
                config, 'Train', device, logger, seed=epoch)
            max_iter = len(train_dataloader) - 1 if platform.system(
            ) == "Windows" else len(train_dataloader)

        for idx, batch in enumerate(train_dataloader):
            profiler.add_profiler_step(profiler_options)
            train_reader_cost += time.time() - reader_start
            if idx >= max_iter:
                break
            lr = optimizer.get_lr()
            images = batch[0]
            if use_srn:
                model_average = True
            # use amp
            if scaler:
                with paddle.amp.auto_cast(
                        level=amp_level,
                        custom_black_list=amp_custom_black_list):
                    if model_type == 'table' or extra_input:
                        preds = model(images, data=batch[1:])
                    elif model_type in ["kie"]:
                        preds = model(batch)
                    elif algorithm in ['CAN']:
                        preds = model(batch[:3])
                    else:
                        preds = model(images)
                preds = to_float32(preds)
                loss = loss_class(preds, batch)
                avg_loss = loss['loss']
                scaled_avg_loss = scaler.scale(avg_loss)
                scaled_avg_loss.backward()
                scaler.minimize(optimizer, scaled_avg_loss)
            else:
                if model_type == 'table' or extra_input:
                    preds = model(images, data=batch[1:])
                elif model_type in ["kie", 'sr']:
                    preds = model(batch)
                elif algorithm in ['CAN']:
                    preds = model(batch[:3])
                else:
                    preds = model(images)
                loss = loss_class(preds, batch)
                avg_loss = loss['loss']
                avg_loss.backward()
                optimizer.step()

            optimizer.clear_grad()

            if cal_metric_during_train and epoch % calc_epoch_interval == 0:  # only rec and cls need
                batch = [item.numpy() for item in batch]
                if model_type in ['kie', 'sr']:
                    eval_class(preds, batch)
                elif model_type in ['table']:
                    post_result = post_process_class(preds, batch)
                    eval_class(post_result, batch)
                elif algorithm in ['CAN']:
                    model_type = 'can'
                    eval_class(preds[0], batch[2:], epoch_reset=(idx == 0))
                else:
                    if config['Loss']['name'] in ['MultiLoss', 'MultiLoss_v2'
                                                  ]:  # for multi head loss
                        post_result = post_process_class(
                            preds['ctc'], batch[1])  # for CTC head out
                    elif config['Loss']['name'] in ['VLLoss']:
                        post_result = post_process_class(preds, batch[1],
                                                         batch[-1])
                    else:
                        post_result = post_process_class(preds, batch[1])
                    eval_class(post_result, batch)
                metric = eval_class.get_metric()
                train_stats.update(metric)

            train_batch_time = time.time() - reader_start
            train_batch_cost += train_batch_time
            eta_meter.update(train_batch_time)
            global_step += 1
            total_samples += len(images)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            # logger and visualdl
            stats = {k: v.numpy().mean() for k, v in loss.items()}
            stats['lr'] = lr
            train_stats.update(stats)

            if log_writer is not None and dist.get_rank() == 0:
                log_writer.log_metrics(
                    metrics=train_stats.get(), prefix="TRAIN", step=global_step)

            if dist.get_rank() == 0 and (
                (global_step > 0 and global_step % print_batch_step == 0) or
                (idx >= len(train_dataloader) - 1)):
                logs = train_stats.log()

                eta_sec = ((epoch_num + 1 - epoch) * \
                    len(train_dataloader) - idx - 1) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                strs = 'epoch: [{}/{}], global_step: {}, {}, avg_reader_cost: ' \
                    '{:.5f} s, avg_batch_cost: {:.5f} s, avg_samples: {}, ' \
                    'ips: {:.5f} samples/s, eta: {}'.format(
                    epoch, epoch_num, global_step, logs,
                    train_reader_cost / print_batch_step,
                    train_batch_cost / print_batch_step,
                    total_samples / print_batch_step,
                    total_samples / train_batch_cost, eta_sec_format)
                logger.info(strs)

                total_samples = 0
                train_reader_cost = 0.0
                train_batch_cost = 0.0
            # eval
            if global_step > start_eval_step and \
                    (global_step - start_eval_step) % eval_batch_step == 0 \
                    and dist.get_rank() == 0:
                if model_average:
                    Model_Average = paddle.incubate.optimizer.ModelAverage(
                        0.15,
                        parameters=model.parameters(),
                        min_average_window=10000,
                        max_average_window=15625)
                    Model_Average.apply()
                cur_metric = eval(
                    model,
                    valid_dataloader,
                    post_process_class,
                    eval_class,
                    model_type,
                    extra_input=extra_input,
                    scaler=scaler,
                    amp_level=amp_level,
                    amp_custom_black_list=amp_custom_black_list)
                cur_metric_str = 'cur metric, {}'.format(', '.join(
                    ['{}: {}'.format(k, v) for k, v in cur_metric.items()]))
                logger.info(cur_metric_str)

                # logger metric
                if log_writer is not None:
                    log_writer.log_metrics(
                        metrics=cur_metric, prefix="EVAL", step=global_step)

                if cur_metric[main_indicator] >= best_model_dict[
                        main_indicator]:
                    best_model_dict.update(cur_metric)
                    best_model_dict['best_epoch'] = epoch
                    save_model(
                        model,
                        optimizer,
                        save_model_dir,
                        logger,
                        config,
                        is_best=True,
                        prefix='best_accuracy',
                        best_model_dict=best_model_dict,
                        epoch=epoch,
                        global_step=global_step)
                best_str = 'best metric, {}'.format(', '.join([
                    '{}: {}'.format(k, v) for k, v in best_model_dict.items()
                ]))
                logger.info(best_str)
                # logger best metric
                if log_writer is not None:
                    log_writer.log_metrics(
                        metrics={
                            "best_{}".format(main_indicator):
                            best_model_dict[main_indicator]
                        },
                        prefix="EVAL",
                        step=global_step)

                    log_writer.log_model(
                        is_best=True,
                        prefix="best_accuracy",
                        metadata=best_model_dict)

            reader_start = time.time()
        if dist.get_rank() == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                config,
                is_best=False,
                prefix='latest',
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step)

            if log_writer is not None:
                log_writer.log_model(is_best=False, prefix="latest")

        if dist.get_rank() == 0 and epoch > 0 and epoch % save_epoch_step == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                config,
                is_best=False,
                prefix='iter_epoch_{}'.format(epoch),
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step)
            if log_writer is not None:
                log_writer.log_model(
                    is_best=False, prefix='iter_epoch_{}'.format(epoch))

    best_str = 'best metric, {}'.format(', '.join(
        ['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    logger.info(best_str)
    if dist.get_rank() == 0 and log_writer is not None:
        log_writer.close()
    return


def eval(model,
         valid_dataloader,
         post_process_class,
         eval_class,
         model_type=None,
         extra_input=False,
         scaler=None,
         amp_level='O2',
         amp_custom_black_list=[]):
    model.eval()
    with paddle.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader),
            desc='eval model:',
            position=0,
            leave=True)
        max_iter = len(valid_dataloader) - 1 if platform.system(
        ) == "Windows" else len(valid_dataloader)
        sum_images = 0
        for idx, batch in enumerate(valid_dataloader):
            if idx >= max_iter:
                break
            images = batch[0]
            start = time.time()

            # use amp
            if scaler:
                with paddle.amp.auto_cast(
                        level=amp_level,
                        custom_black_list=amp_custom_black_list):
                    if model_type == 'table' or extra_input:
                        preds = model(images, data=batch[1:])
                    elif model_type in ["kie"]:
                        preds = model(batch)
                    elif model_type in ['can']:
                        preds = model(batch[:3])
                    elif model_type in ['sr']:
                        preds = model(batch)
                        sr_img = preds["sr_img"]
                        lr_img = preds["lr_img"]
                    else:
                        preds = model(images)
                preds = to_float32(preds)
            else:
                if model_type == 'table' or extra_input:
                    preds = model(images, data=batch[1:])
                elif model_type in ["kie"]:
                    preds = model(batch)
                elif model_type in ['can']:
                    preds = model(batch[:3])
                elif model_type in ['sr']:
                    preds = model(batch)
                    sr_img = preds["sr_img"]
                    lr_img = preds["lr_img"]
                else:
                    preds = model(images)

            batch_numpy = []
            for item in batch:
                if isinstance(item, paddle.Tensor):
                    batch_numpy.append(item.numpy())
                else:
                    batch_numpy.append(item)
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch
            if model_type in ['table', 'kie']:
                if post_process_class is None:
                    eval_class(preds, batch_numpy)
                else:
                    post_result = post_process_class(preds, batch_numpy)
                    eval_class(post_result, batch_numpy)
            elif model_type in ['sr']:
                eval_class(preds, batch_numpy)
            elif model_type in ['can']:
                eval_class(preds[0], batch_numpy[2:], epoch_reset=(idx == 0))
            else:
                post_result = post_process_class(preds, batch_numpy[1])
                eval_class(post_result, batch_numpy)

            pbar.update(1)
            total_frame += len(images)
            sum_images += 1
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric['fps'] = total_frame / total_time
    return metric


def update_center(char_center, post_result, preds):
    result, label = post_result
    feats, logits = preds
    logits = paddle.argmax(logits, axis=-1)
    feats = feats.numpy()
    logits = logits.numpy()

    for idx_sample in range(len(label)):
        if result[idx_sample][0] == label[idx_sample][0]:
            feat = feats[idx_sample]
            logit = logits[idx_sample]
            for idx_time in range(len(logit)):
                index = logit[idx_time]
                if index in char_center.keys():
                    char_center[index][0] = (
                        char_center[index][0] * char_center[index][1] +
                        feat[idx_time]) / (char_center[index][1] + 1)
                    char_center[index][1] += 1
                else:
                    char_center[index] = [feat[idx_time], 1]
    return char_center


def get_center(model, eval_dataloader, post_process_class):
    pbar = tqdm(total=len(eval_dataloader), desc='get center:')
    max_iter = len(eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(eval_dataloader)
    char_center = dict()
    for idx, batch in enumerate(eval_dataloader):
        if idx >= max_iter:
            break
        images = batch[0]
        start = time.time()
        preds = model(images)

        batch = [item.numpy() for item in batch]
        # Obtain usable results from post-processing methods
        post_result = post_process_class(preds, batch[1])

        #update char_center
        char_center = update_center(char_center, post_result, preds)
        pbar.update(1)

    pbar.close()
    for key in char_center.keys():
        char_center[key] = char_center[key][0]
    return char_center


def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profile_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profile_dic)

    if is_train:
        # save_config
        save_model_dir = config['Global']['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(
                dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
    else:
        log_file = None
    logger = get_logger(log_file=log_file)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global'].get('use_gpu', False)
    use_xpu = config['Global'].get('use_xpu', False)
    use_npu = config['Global'].get('use_npu', False)
    use_mlu = config['Global'].get('use_mlu', False)

    alg = config['Architecture']['algorithm']
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN',
        'CLS', 'PGNet', 'Distillation', 'NRTR', 'TableAttn', 'SAR', 'PSE',
        'SEED', 'SDMGR', 'LayoutXLM', 'LayoutLM', 'LayoutLMv2', 'PREN', 'FCE',
        'SVTR', 'ViTSTR', 'ABINet', 'DB++', 'TableMaster', 'SPIN', 'VisionLAN',
        'Gestalt', 'SLANet', 'RobustScanner', 'CT', 'RFL', 'DRRG', 'CAN',
        'Telescope'
    ]

    if use_xpu:
        device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
    elif use_npu:
        device = 'npu:{0}'.format(os.getenv('FLAGS_selected_npus', 0))
    elif use_mlu:
        device = 'mlu:{0}'.format(os.getenv('FLAGS_selected_mlus', 0))
    else:
        device = 'gpu:{}'.format(dist.ParallelEnv()
                                 .dev_id) if use_gpu else 'cpu'
    check_device(use_gpu, use_xpu, use_npu, use_mlu)

    device = paddle.set_device(device)

    config['Global']['distributed'] = dist.get_world_size() != 1

    loggers = []

    if 'use_visualdl' in config['Global'] and config['Global']['use_visualdl']:
        save_model_dir = config['Global']['save_model_dir']
        vdl_writer_path = '{}/vdl/'.format(save_model_dir)
        log_writer = VDLLogger(vdl_writer_path)
        loggers.append(log_writer)
    if ('use_wandb' in config['Global'] and
            config['Global']['use_wandb']) or 'wandb' in config:
        save_dir = config['Global']['save_model_dir']
        wandb_writer_path = "{}/wandb".format(save_dir)
        if "wandb" in config:
            wandb_params = config['wandb']
        else:
            wandb_params = dict()
        wandb_params.update({'save_dir': save_model_dir})
        log_writer = WandbLogger(**wandb_params, config=config)
        loggers.append(log_writer)
    else:
        log_writer = None
    print_dict(config, logger)

    if loggers:
        log_writer = Loggers(loggers)
    else:
        log_writer = None

    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    return config, device, logger, log_writer
