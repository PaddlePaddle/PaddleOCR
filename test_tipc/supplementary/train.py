import paddle
import numpy as np
import os
import paddle.nn as nn
import paddle.distributed as dist
dist.get_world_size()
dist.init_parallel_env()

from loss import build_loss, LossDistill, DMLLoss, KLJSLoss
from optimizer import create_optimizer
from data_loader import build_dataloader
from metric import create_metric
from mv3 import MobileNetV3_large_x0_5, distillmv3_large_x0_5, build_model
from config import preprocess
import time

from paddleslim.dygraph.quant import QAT
from slim.slim_quant import PACT, quant_config
from slim.slim_fpgm import prune_model
from utils import load_model


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def save_model(model,
               optimizer,
               model_path,
               logger,
               is_best=False,
               prefix='ppocr',
               **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    paddle.save(model.state_dict(), model_prefix + '.pdparams')
    if type(optimizer) is list:
        paddle.save(optimizer[0].state_dict(), model_prefix + '.pdopt')
        paddle.save(optimizer[1].state_dict(), model_prefix + "_1" + '.pdopt')

    else:
        paddle.save(optimizer.state_dict(), model_prefix + '.pdopt')

    # # save metric and config
    # with open(model_prefix + '.states', 'wb') as f:
    #     pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))


def amp_scaler(config):
    if 'AMP' in config and config['AMP']['use_amp'] is True:
        AMP_RELATED_FLAGS_SETTING = {
            'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
            'FLAGS_max_inplace_grad_add': 8,
        }
        paddle.fluid.set_flags(AMP_RELATED_FLAGS_SETTING)
        scale_loss = config["AMP"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["AMP"].get("use_dynamic_loss_scaling",
                                                     False)
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling)
        return scaler
    else:
        return None


def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)


def train(config, scaler=None):
    EPOCH = config['epoch']
    topk = config['topk']

    batch_size = config['TRAIN']['batch_size']
    num_workers = config['TRAIN']['num_workers']
    train_loader = build_dataloader(
        'train', batch_size=batch_size, num_workers=num_workers)

    # build metric
    metric_func = create_metric

    # build model
    # model = MobileNetV3_large_x0_5(class_dim=100)
    model = build_model(config)

    # build_optimizer 
    optimizer, lr_scheduler = create_optimizer(
        config, parameter_list=model.parameters())

    # load model
    pre_best_model_dict = load_model(config, model, optimizer)
    if len(pre_best_model_dict) > 0:
        pre_str = 'The metric of loaded metric as follows {}'.format(', '.join(
            ['{}: {}'.format(k, v) for k, v in pre_best_model_dict.items()]))
        logger.info(pre_str)

    # about slim prune and quant
    if "quant_train" in config and config['quant_train'] is True:
        quanter = QAT(config=quant_config, act_preprocess=PACT)
        quanter.quantize(model)
    elif "prune_train" in config and config['prune_train'] is True:
        model = prune_model(model, [1, 3, 32, 32], 0.1)
    else:
        pass

    # distribution
    model.train()
    model = paddle.DataParallel(model)
    # build loss function
    loss_func = build_loss(config)

    data_num = len(train_loader)

    best_acc = {}
    for epoch in range(EPOCH):
        st = time.time()
        for idx, data in enumerate(train_loader):
            img_batch, label = data
            img_batch = paddle.transpose(img_batch, [0, 3, 1, 2])
            label = paddle.unsqueeze(label, -1)

            if scaler is not None:
                with paddle.amp.auto_cast():
                    outs = model(img_batch)
            else:
                outs = model(img_batch)

            # cal metric 
            acc = metric_func(outs, label)

            # cal loss
            avg_loss = loss_func(outs, label)

            if scaler is None:
                # backward
                avg_loss.backward()
                optimizer.step()
                optimizer.clear_grad()
            else:
                scaled_avg_loss = scaler.scale(avg_loss)
                scaled_avg_loss.backward()
                scaler.minimize(optimizer, scaled_avg_loss)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            if idx % 10 == 0:
                et = time.time()
                strs = f"epoch: [{epoch}/{EPOCH}], iter: [{idx}/{data_num}], "
                strs += f"loss: {avg_loss.numpy()[0]}"
                strs += f", acc_topk1: {acc['top1'].numpy()[0]}, acc_top5: {acc['top5'].numpy()[0]}"
                strs += f", batch_time: {round(et-st, 4)} s"
                logger.info(strs)
                st = time.time()

        if epoch % 10 == 0:
            acc = eval(config, model)
            if len(best_acc) < 1 or acc['top5'].numpy()[0] > best_acc['top5']:
                best_acc = acc
                best_acc['epoch'] = epoch
                is_best = True
            else:
                is_best = False
            logger.info(
                f"The best acc: acc_topk1: {best_acc['top1'].numpy()[0]}, acc_top5: {best_acc['top5'].numpy()[0]}, best_epoch: {best_acc['epoch']}"
            )
            save_model(
                model,
                optimizer,
                config['save_model_dir'],
                logger,
                is_best,
                prefix="cls")


def train_distill(config, scaler=None):
    EPOCH = config['epoch']
    topk = config['topk']

    batch_size = config['TRAIN']['batch_size']
    num_workers = config['TRAIN']['num_workers']
    train_loader = build_dataloader(
        'train', batch_size=batch_size, num_workers=num_workers)

    # build metric
    metric_func = create_metric

    # model = distillmv3_large_x0_5(class_dim=100)
    model = build_model(config)

    # pact quant train
    if "quant_train" in config and config['quant_train'] is True:
        quanter = QAT(config=quant_config, act_preprocess=PACT)
        quanter.quantize(model)
    elif "prune_train" in config and config['prune_train'] is True:
        model = prune_model(model, [1, 3, 32, 32], 0.1)
    else:
        pass

    # build_optimizer 
    optimizer, lr_scheduler = create_optimizer(
        config, parameter_list=model.parameters())

    # load model
    pre_best_model_dict = load_model(config, model, optimizer)
    if len(pre_best_model_dict) > 0:
        pre_str = 'The metric of loaded metric as follows {}'.format(', '.join(
            ['{}: {}'.format(k, v) for k, v in pre_best_model_dict.items()]))
        logger.info(pre_str)

    model.train()
    model = paddle.DataParallel(model)

    # build loss function
    loss_func_distill = LossDistill(model_name_list=['student', 'student1'])
    loss_func_dml = DMLLoss(model_name_pairs=['student', 'student1'])
    loss_func_js = KLJSLoss(mode='js')

    data_num = len(train_loader)

    best_acc = {}
    for epoch in range(EPOCH):
        st = time.time()
        for idx, data in enumerate(train_loader):
            img_batch, label = data
            img_batch = paddle.transpose(img_batch, [0, 3, 1, 2])
            label = paddle.unsqueeze(label, -1)
            if scaler is not None:
                with paddle.amp.auto_cast():
                    outs = model(img_batch)
            else:
                outs = model(img_batch)

            # cal metric 
            acc = metric_func(outs['student'], label)

            # cal loss
            avg_loss = loss_func_distill(outs, label)['student'] + \
                       loss_func_distill(outs, label)['student1'] + \
                       loss_func_dml(outs, label)['student_student1']

            # backward
            if scaler is None:
                avg_loss.backward()
                optimizer.step()
                optimizer.clear_grad()
            else:
                scaled_avg_loss = scaler.scale(avg_loss)
                scaled_avg_loss.backward()
                scaler.minimize(optimizer, scaled_avg_loss)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            if idx % 10 == 0:
                et = time.time()
                strs = f"epoch: [{epoch}/{EPOCH}], iter: [{idx}/{data_num}], "
                strs += f"loss: {avg_loss.numpy()[0]}"
                strs += f", acc_topk1: {acc['top1'].numpy()[0]}, acc_top5: {acc['top5'].numpy()[0]}"
                strs += f", batch_time: {round(et-st, 4)} s"
                logger.info(strs)
                st = time.time()

        if epoch % 10 == 0:
            acc = eval(config, model._layers.student)
            if len(best_acc) < 1 or acc['top5'].numpy()[0] > best_acc['top5']:
                best_acc = acc
                best_acc['epoch'] = epoch
                is_best = True
            else:
                is_best = False
            logger.info(
                f"The best acc: acc_topk1: {best_acc['top1'].numpy()[0]}, acc_top5: {best_acc['top5'].numpy()[0]}, best_epoch: {best_acc['epoch']}"
            )

            save_model(
                model,
                optimizer,
                config['save_model_dir'],
                logger,
                is_best,
                prefix="cls_distill")


def train_distill_multiopt(config, scaler=None):
    EPOCH = config['epoch']
    topk = config['topk']

    batch_size = config['TRAIN']['batch_size']
    num_workers = config['TRAIN']['num_workers']
    train_loader = build_dataloader(
        'train', batch_size=batch_size, num_workers=num_workers)

    # build metric
    metric_func = create_metric

    # model = distillmv3_large_x0_5(class_dim=100)
    model = build_model(config)

    # build_optimizer 
    optimizer, lr_scheduler = create_optimizer(
        config, parameter_list=model.student.parameters())
    optimizer1, lr_scheduler1 = create_optimizer(
        config, parameter_list=model.student1.parameters())

    # load model
    pre_best_model_dict = load_model(config, model, optimizer)
    if len(pre_best_model_dict) > 0:
        pre_str = 'The metric of loaded metric as follows {}'.format(', '.join(
            ['{}: {}'.format(k, v) for k, v in pre_best_model_dict.items()]))
        logger.info(pre_str)

    # quant train
    if "quant_train" in config and config['quant_train'] is True:
        quanter = QAT(config=quant_config, act_preprocess=PACT)
        quanter.quantize(model)
    elif "prune_train" in config and config['prune_train'] is True:
        model = prune_model(model, [1, 3, 32, 32], 0.1)
    else:
        pass

    model.train()

    model = paddle.DataParallel(model)

    # build loss function
    loss_func_distill = LossDistill(model_name_list=['student', 'student1'])
    loss_func_dml = DMLLoss(model_name_pairs=['student', 'student1'])
    loss_func_js = KLJSLoss(mode='js')

    data_num = len(train_loader)
    best_acc = {}
    for epoch in range(EPOCH):
        st = time.time()
        for idx, data in enumerate(train_loader):
            img_batch, label = data
            img_batch = paddle.transpose(img_batch, [0, 3, 1, 2])
            label = paddle.unsqueeze(label, -1)

            if scaler is not None:
                with paddle.amp.auto_cast():
                    outs = model(img_batch)
            else:
                outs = model(img_batch)

            # cal metric 
            acc = metric_func(outs['student'], label)

            # cal loss
            avg_loss = loss_func_distill(outs,
                                         label)['student'] + loss_func_dml(
                                             outs, label)['student_student1']
            avg_loss1 = loss_func_distill(outs,
                                          label)['student1'] + loss_func_dml(
                                              outs, label)['student_student1']

            if scaler is None:
                # backward
                avg_loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.clear_grad()

                avg_loss1.backward()
                optimizer1.step()
                optimizer1.clear_grad()
            else:
                scaled_avg_loss = scaler.scale(avg_loss)
                scaled_avg_loss.backward()
                scaler.minimize(optimizer, scaled_avg_loss)

                scaled_avg_loss = scaler.scale(avg_loss1)
                scaled_avg_loss.backward()
                scaler.minimize(optimizer1, scaled_avg_loss)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()
            if not isinstance(lr_scheduler1, float):
                lr_scheduler1.step()

            if idx % 10 == 0:
                et = time.time()
                strs = f"epoch: [{epoch}/{EPOCH}], iter: [{idx}/{data_num}], "
                strs += f"loss: {avg_loss.numpy()[0]}, loss1: {avg_loss1.numpy()[0]}"
                strs += f", acc_topk1: {acc['top1'].numpy()[0]}, acc_top5: {acc['top5'].numpy()[0]}"
                strs += f", batch_time: {round(et-st, 4)} s"
                logger.info(strs)
                st = time.time()

        if epoch % 10 == 0:
            acc = eval(config, model._layers.student)
            if len(best_acc) < 1 or acc['top5'].numpy()[0] > best_acc['top5']:
                best_acc = acc
                best_acc['epoch'] = epoch
                is_best = True
            else:
                is_best = False
            logger.info(
                f"The best acc: acc_topk1: {best_acc['top1'].numpy()[0]}, acc_top5: {best_acc['top5'].numpy()[0]}, best_epoch: {best_acc['epoch']}"
            )
            save_model(
                model, [optimizer, optimizer1],
                config['save_model_dir'],
                logger,
                is_best,
                prefix="cls_distill_multiopt")


def eval(config, model):
    batch_size = config['VALID']['batch_size']
    num_workers = config['VALID']['num_workers']
    valid_loader = build_dataloader(
        'test', batch_size=batch_size, num_workers=num_workers)

    # build metric
    metric_func = create_metric

    outs = []
    labels = []
    for idx, data in enumerate(valid_loader):
        img_batch, label = data
        img_batch = paddle.transpose(img_batch, [0, 3, 1, 2])
        label = paddle.unsqueeze(label, -1)
        out = model(img_batch)

        outs.append(out)
        labels.append(label)

    outs = paddle.concat(outs, axis=0)
    labels = paddle.concat(labels, axis=0)
    acc = metric_func(outs, labels)

    strs = f"The metric are as follows: acc_topk1: {acc['top1'].numpy()[0]}, acc_top5: {acc['top5'].numpy()[0]}"
    logger.info(strs)
    return acc


if __name__ == "__main__":

    config, logger = preprocess(is_train=False)

    # AMP scaler
    scaler = amp_scaler(config)

    model_type = config['model_type']

    if model_type == "cls":
        train(config)
    elif model_type == "cls_distill":
        train_distill(config)
    elif model_type == "cls_distill_multiopt":
        train_distill_multiopt(config)
    else:
        raise ValueError("model_type should be one of ['']")
