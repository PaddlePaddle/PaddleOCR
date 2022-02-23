#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
from ..layer_helper import LayerHelper, unique_name
from ..framework import Variable


def _allreduce(x, out=None, reduce_type="sum", sync_mode=False):
    helper = LayerHelper("allreduce", **locals())
    # Convert string reduce type to op int type
    red_typ_int = 0
    if reduce_type == "sum":
        red_typ_int = 0
    elif reduce_type == "prod":
        red_typ_int = 1
    elif reduce_type == "max":
        red_typ_int = 2
    elif reduce_type == "min":
        red_typ_int = 3
    else:
        raise TypeError("reduce type can only be [sum|prod|max|min]")

    if out is None:
        out = helper.create_variable(
            name=unique_name.generate_with_ignorable_key(".".join(
                [x.name, 'tmp'])),
            shape=x.shape,
            dtype=x.dtype,
            type=x.type,
            persistable=x.persistable,
            stop_gradient=True)
    helper.append_op(
        type='allreduce',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={"reduce_type": red_typ_int,
               "sync_mode": sync_mode})
    return out


def _broadcast(x, root, sync_mode=False):
    helper = LayerHelper("broadcast", **locals())
    helper.append_op(
        type='broadcast',
        inputs={'X': [x]},
        outputs={'Out': [x]},
        attrs={"sync_mode": sync_mode,
               "root": root})
    return x


def _c_allreduce(x,
                 out=None,
                 reduce_type='sum',
                 ring_id=0,
                 use_calc_stream=False):
    helper = LayerHelper('c_allreduce', **locals())

    if reduce_type not in ['sum', 'prob', 'max', 'min']:
        raise TypeError('reduce type can only be "sum|prod|max|min]"')

    op_type = 'c_allreduce_' + reduce_type
    if out is None:
        out = helper.create_variable(
            name=unique_name.generate_with_ignorable_key('.'.join(
                [x.name, op_type])),
            shape=x.shape,
            dtype=x.dtype,
            type=x.type,
            persistable=x.persistable)

    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'ring_id': ring_id,
               'use_calc_stream': use_calc_stream})
    return out


def _c_broadcast(x, root=0, ring_id=0, use_calc_stream=False):
    op_type = 'c_broadcast'
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [x]},
        attrs={
            'root': root,
            'ring_id': ring_id,
            'use_calc_stream': use_calc_stream
        })
    return x


def _c_allgather(x, nranks, ring_id=0, use_calc_stream=False):
    op_type = 'c_allgather'
    helper = LayerHelper(op_type, **locals())
    out_shape = list(x.shape[:])
    if out_shape[0] > 0:
        out_shape[0] *= nranks
    out = helper.create_variable(
        name=unique_name.generate_with_ignorable_key('.'.join(
            [x.name, op_type])),
        shape=out_shape,
        dtype=x.dtype,
        type=x.type,
        persistable=x.persistable)
    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={
            'nranks': nranks,
            'ring_id': ring_id,
            'use_calc_stream': use_calc_stream
        })
    return out


def _c_reducescatter(x, nranks, ring_id=0, use_calc_stream=False):
    if not isinstance(x, Variable):
        raise TypeError('x must be a Variable')

    if x.shape[0] > 0 and x.shape[0] % nranks != 0:
        raise ValueError('x.shape[0](%d) cannot be evenly divided by nranks(%d)'
                         % (x.shape[0], nranks))

    op_type = 'c_reducescatter'
    helper = LayerHelper(op_type, **locals())
    out_shape = list(x.shape[:])
    if out_shape[0] > 0:
        out_shape[0] //= nranks
    out = helper.create_variable(
        name=unique_name.generate_with_ignorable_key('.'.join(
            [x.name, op_type])),
        shape=out_shape,
        dtype=x.dtype,
        type=x.type,
        persistable=x.persistable)
    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={
            'nranks': nranks,
            'ring_id': ring_id,
            'use_calc_stream': use_calc_stream
        })
    return out


def _c_sync_calc_stream(x):
    op_type = 'c_sync_calc_stream'
    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type, inputs={'X': [x]}, outputs={'Out': [x]})
    return x


def _c_sync_comm_stream(x, ring_id):
    op_type = 'c_sync_comm_stream'
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [x]},
        attrs={'ring_id': ring_id})
    return x
