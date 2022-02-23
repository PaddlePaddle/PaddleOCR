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
"""Fleet Metrics"""

import math
import numpy as np
from paddle.static import Variable
import paddle

__all__ = []


def sum(input, scope=None, util=None):
    """
    distributed sum in fleet

    Args:
        input(numpy.array|Variable|string): output of a layer
        scope(Scope): specific scope

    Returns:
        global_metric(numpy.array): sum array

    Example:
        .. code-block:: python

          # in model.py
          input = fluid.layers.cast(some_input, dtype='float32')
          cnt = fluid.layers.reduce_sum(input)
          global_cnt = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp = fluid.layers.elementwise_add(cnt, global_cnt)
          fluid.layers.assign(tmp, global_cnt)
          
          # in train.py, after train or infer
          res = np.array(scope.find_var(global_cnt.name).get_tensor())
          print("sum array: ", paddle.distributed.fleet.sum(res))
    """
    if scope is None:
        scope = paddle.static.global_scope()
    if util is None:
        util = paddle.distributed.fleet.util
    if isinstance(input, Variable):
        input = np.array(scope.find_var(input.name).get_tensor())
    elif isinstance(input, str):
        input = np.array(scope.find_var(input).get_tensor())
    old_shape = np.array(input.shape)
    output = np.copy(input) * 0
    output = util.all_reduce(input, "sum")
    output = output.reshape(old_shape)
    return output


def max(input, scope=None, util=None):
    """
    distributed max in fleet

    Args:
        input(numpy.array|Variable|string): output of a layer
        scope(Scope): specific scope

    Returns:
        global_metric(numpy.array): max array

    Example:
        .. code-block:: python

          # in model.py
          input = fluid.layers.cast(some_input, dtype='float32')
          cnt = fluid.layers.reduce_sum(input)
          global_cnt = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp = fluid.layers.elementwise_max(cnt, global_cnt)
          fluid.layers.assign(tmp, global_cnt)

          # in train.py, after train or infer
          res = np.array(scope.find_var(global_cnt.name).get_tensor())
          print("max array: ", paddle.distributed.fleet.max(res))
    """
    if scope is None:
        scope = paddle.static.global_scope()
    if util is None:
        util = paddle.distributed.fleet.util
    if isinstance(input, Variable):
        input = np.array(scope.find_var(input.name).get_tensor())
    elif isinstance(input, str):
        input = np.array(scope.find_var(input).get_tensor())
    old_shape = np.array(input.shape)
    output = np.copy(input) * 0
    output = util.all_reduce(input, "max")
    output = output.reshape(old_shape)
    return output


def min(input, scope=None, util=None):
    """
    distributed min in fleet

    Args:
        input(numpy.array|Variable|string): output of a layer
        scope(Scope): specific scope

    Returns:
        global_metric(numpy.array): min array

    Example:
        .. code-block:: python

          # in model.py
          input = fluid.layers.cast(some_input, dtype='float32')
          cnt = fluid.layers.reduce_sum(input)
          global_cnt = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp = fluid.layers.elementwise_min(cnt, global_cnt)
          fluid.layers.assign(tmp, global_cnt)

          # in train.py, after train or infer
          res = np.array(scope.find_var(global_cnt.name).get_tensor())
          print("min array: ", paddle.distributed.fleet.min(res))
    """
    if scope is None:
        scope = paddle.static.global_scope()
    if util is None:
        util = paddle.distributed.fleet.util
    if isinstance(input, Variable):
        input = np.array(scope.find_var(input.name).get_tensor())
    elif isinstance(input, str):
        input = np.array(scope.find_var(input).get_tensor())
    old_shape = np.array(input.shape)
    output = np.copy(input) * 0
    output = util.all_reduce(input, "min")
    output = output.reshape(old_shape)
    return output


def auc(stat_pos, stat_neg, scope=None, util=None):
    """
    distributed auc in fleet

    Args:
        stat_pos(numpy.array|Variable|string): stat_pos in output of fluid.layers.auc
        stat_neg(numpy.array|Variable|string): stat_neg in output of fluid.layers.auc
        scope(Scope): specific scope

    Returns:
        auc_value(float): auc value

    Example:
        .. code-block:: python

          # in model.py
          similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(output, min=-15.0, max=15.0))
          binary_predict = fluid.layers.concat(
              input=[fluid.layers.elementwise_sub(fluid.layers.ceil(similarity_norm), similarity_norm), similarity_norm], axis=1)
          self.auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg] =
              fluid.layers.auc(input=binary_predict, label=label, curve='ROC', num_thresholds=4096)

          # in train.py, after train or infer
          pos = np.array(scope.find_var(stat_pos.name).get_tensor())
          neg = np.array(scope.find_var(stat_neg.name).get_tensor())
          print("auc: ", paddle.distributed.fleet.auc(pos, neg))
    """
    if scope is None:
        scope = paddle.static.global_scope()
    if util is None:
        util = paddle.distributed.fleet.util

    if isinstance(stat_pos, Variable):
        stat_pos = np.array(scope.find_var(stat_pos.name).get_tensor())
    elif isinstance(stat_pos, str):
        stat_pos = np.array(scope.find_var(stat_pos).get_tensor())
    if isinstance(stat_neg, Variable):
        stat_neg = np.array(scope.find_var(stat_neg.name).get_tensor())
    elif isinstance(stat_neg, str):
        stat_neg = np.array(scope.find_var(stat_neg).get_tensor())
    # auc pos bucket shape
    old_pos_shape = np.array(stat_pos.shape)
    # reshape to one dim
    stat_pos = stat_pos.reshape(-1)
    global_pos = np.copy(stat_pos) * 0
    # mpi allreduce
    global_pos = util.all_reduce(stat_pos, "sum")
    global_pos = global_pos.reshape(old_pos_shape)

    # auc neg bucket
    old_neg_shape = np.array(stat_neg.shape)
    stat_neg = stat_neg.reshape(-1)
    global_neg = np.copy(stat_neg) * 0
    global_neg = util.all_reduce(stat_neg, "sum")
    global_neg = global_neg.reshape(old_neg_shape)

    # calculate auc
    num_bucket = len(global_pos[0])
    area = 0.0
    pos = 0.0
    neg = 0.0
    new_pos = 0.0
    new_neg = 0.0
    total_ins_num = 0
    for i in range(num_bucket):
        index = num_bucket - 1 - i
        new_pos = pos + global_pos[0][index]
        total_ins_num += global_pos[0][index]
        new_neg = neg + global_neg[0][index]
        total_ins_num += global_neg[0][index]
        area += (new_neg - neg) * (pos + new_pos) / 2
        pos = new_pos
        neg = new_neg

    auc_value = None
    if pos * neg == 0 or total_ins_num == 0:
        auc_value = 0.5
    else:
        auc_value = area / (pos * neg)

    return auc_value


def mae(abserr, total_ins_num, scope=None, util=None):
    """
    distributed mae in fleet

    Args:
        abserr(numpy.array|Variable|string): abserr in output of fluid.contrib.layers.ctr_metric_bundle
        total_ins_num(numpy.array|Variable|string): total variable
        scope(Scope): specific scope

    Returns:
        mae(float): mae value

    Example:
        .. code-block:: python

          # in model.py
          sqrerr, abserr, prob, q, pos, total = fluid.contrib.layers.ctr_metric_bundle(similarity_norm, fluid.layers.cast(x=label, dtype='float32'))

          # in train.py, after train or infer
          res = np.array(scope.find_var(abserr.name).get_tensor())
          print("mae: ", paddle.distributed.fleet.mae(res, total_ins_num))
    """
    if scope is None:
        scope = paddle.static.global_scope()
    if util is None:
        util = paddle.distributed.fleet.util

    if isinstance(abserr, Variable):
        abserr = np.array(scope.find_var(abserr.name).get_tensor())
    elif isinstance(abserr, str):
        abserr = np.array(scope.find_var(abserr).get_tensor())
    if isinstance(total_ins_num, Variable):
        total_ins_num = np.array(
            scope.find_var(total_ins_num.name).get_tensor())
    elif isinstance(total_ins_num, str):
        total_ins_num = np.array(scope.find_var(total_ins_num).get_tensor())

    old_metric_shape = np.array(abserr.shape)
    abserr = abserr.reshape(-1)
    global_metric = np.copy(abserr) * 0

    global_metric = util.all_reduce(abserr, "sum")
    global_metric = global_metric.reshape(old_metric_shape)
    global_total_num = util.all_reduce(total_ins_num, "sum")

    mae_value = float(global_metric[0]) / float(global_total_num[0])
    return mae_value


def rmse(sqrerr, total_ins_num, scope=None, util=None):
    """
    distributed rmse in fleet

    Args:
        sqrerr(numpy.array|Variable|string): sqrerr in output of fluid.contrib.layers.ctr_metric_bundle
        total_ins_num(numpy.array|Variable|string): total variable
        scope(Scope): specific scope

    Returns:
        rmse(float): rmse value

    Example:
        .. code-block:: python

          # in model.py
          sqrerr, abserr, prob, q, pos, total = fluid.contrib.layers.ctr_metric_bundle(similarity_norm, fluid.layers.cast(x=label, dtype='float32'))

          # in train.py, after train or infer
          res = np.array(scope.find_var(sqrerr.name).get_tensor())
          print("rmse: ", paddle.distributed.fleet.rmse(res, total_ins_num))
    """
    if scope is None:
        scope = paddle.static.global_scope()
    if util is None:
        util = paddle.distributed.fleet.util

    if isinstance(sqrerr, Variable):
        sqrerr = np.array(scope.find_var(sqrerr.name).get_tensor())
    elif isinstance(sqrerr, str):
        sqrerr = np.array(scope.find_var(sqrerr).get_tensor())
    if isinstance(total_ins_num, Variable):
        total_ins_num = np.array(
            scope.find_var(total_ins_num.name).get_tensor())
    elif isinstance(total_ins_num, str):
        total_ins_num = np.array(scope.find_var(total_ins_num).get_tensor())
    old_metric_shape = np.array(sqrerr.shape)
    sqrerr = sqrerr.reshape(-1)
    global_metric = np.copy(sqrerr) * 0

    global_metric = util.all_reduce(sqrerr, "sum")
    global_metric = global_metric.reshape(old_metric_shape)
    global_total_num = util.all_reduce(total_ins_num, "sum")

    rmse_value = math.sqrt(float(global_metric[0]) / float(global_total_num[0]))

    return rmse_value


def mse(sqrerr, total_ins_num, scope=None, util=None):
    """
    distributed mse in fleet

    Args:
        sqrerr(numpy.array|Variable|string): sqrerr in output of fluid.contrib.layers.ctr_metric_bundle
        total_ins_num(numpy.array|Variable|string): total variable
        scope(Scope): specific scope

    Returns:
        mse(float): mse value

    Example:
        .. code-block:: python

          # in model.py
          sqrerr, abserr, prob, q, pos, total = fluid.contrib.layers.ctr_metric_bundle(similarity_norm, fluid.layers.cast(x=label, dtype='float32'))

          # in train.py, after train or infer
          metric = np.array(scope.find_var(sqrerr.name).get_tensor())
          print("mse: ", paddle.distributed.fleet.mse(metric, total_ins_num))
    """
    if scope is None:
        scope = paddle.static.global_scope()
    if util is None:
        util = paddle.distributed.fleet.util

    if isinstance(sqrerr, Variable):
        sqrerr = np.array(scope.find_var(sqrerr.name).get_tensor())
    elif isinstance(sqrerr, str):
        sqrerr = np.array(scope.find_var(sqrerr).get_tensor())
    if isinstance(total_ins_num, Variable):
        total_ins_num = np.array(
            scope.find_var(total_ins_num.name).get_tensor())
    elif isinstance(total_ins_num, str):
        total_ins_num = np.array(scope.find_var(total_ins_num).get_tensor())
    old_metric_shape = np.array(sqrerr.shape)
    sqrerr = sqrerr.reshape(-1)
    global_metric = np.copy(sqrerr) * 0

    global_metric = util.all_reduce(sqrerr, "sum")
    global_metric = global_metric.reshape(old_metric_shape)
    global_total_num = util.all_reduce(total_ins_num, "sum")

    mse_value = float(global_metric[0]) / float(global_total_num[0])
    return mse_value


def acc(correct, total, scope=None, util=None):
    """
    distributed accuracy in fleet

    Args:
        correct(numpy.array|Variable|string): correct Variable
        total(numpy.array|Variable): total Variable
        scope(Scope): specific scope

    Returns:
        acc(float): accuracy value

    Example:
        .. code-block:: python

          # in model.py
          correct = fluid.layers.create_global_var(dtype='float32', shape=[1], value=0)
          total = fluid.layers.create_global_var(dtype='float32', shape=[1], value=0)
          acc = fluid.layers.acc(predict, label, k=1, correct=correct, total=total)

          global_correct = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp1 = fluid.layers.elementwise_min(correct, global_correct)
          fluid.layers.assign(tmp1, global_correct)

          global_total = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp2 = fluid.layers.elementwise_min(total, global_total)
          fluid.layers.assign(tmp2, global_total)

          # in train.py, after train or infer
          correct_num = np.array(scope.find_var(correct.name).get_tensor())
          total_num = np.array(scope.find_var(total.name).get_tensor())
          print("accuracy: ", paddle.distributed.fleet.acc(correct_num, total_num))
    """
    if scope is None:
        scope = paddle.static.global_scope()
    if util is None:
        util = paddle.distributed.fleet.util

    if isinstance(correct, Variable):
        correct = np.array(scope.find_var(correct.name).get_tensor())
    elif isinstance(correct, str):
        correct = np.array(scope.find_var(correct).get_tensor())
    if isinstance(total, Variable):
        total = np.array(scope.find_var(total.name).get_tensor())
    elif isinstance(total, str):
        total = np.array(scope.find_var(total).get_tensor())

    global_correct_num = np.copy(correct) * 0
    global_total_num = np.copy(total) * 0

    global_correct_num = util.all_reduce(correct, "sum")
    global_total_num = util.all_reduce(total, "sum")

    return float(global_correct_num[0]) / float(global_total_num[0])
