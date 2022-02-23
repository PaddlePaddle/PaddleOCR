# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import collections
import functools
from ..framework import Variable, default_main_program, in_dygraph_mode, dygraph_only, Parameter, ParamBase, _varbase_creator, _dygraph_tracer
import pickle
from . import learning_rate_scheduler
import warnings
from .. import core
from .base import guard
from paddle.fluid.dygraph.jit import _SaveLoadConfig
from paddle.fluid.dygraph.io import _construct_program_holders, _construct_params_and_buffers

__all__ = [
    'save_dygraph',
    'load_dygraph',
]


def _parse_load_config(configs):
    supported_configs = ['model_filename', 'params_filename', 'keep_name_table']

    # input check
    for key in configs:
        if key not in supported_configs:
            raise ValueError(
                "The additional config (%s) of `paddle.fluid.load_dygraph` is not supported."
                % (key))

    # construct inner config
    inner_config = _SaveLoadConfig()
    inner_config.model_filename = configs.get('model_filename', None)
    inner_config.params_filename = configs.get('params_filename', None)
    inner_config.keep_name_table = configs.get('keep_name_table', None)

    return inner_config


@dygraph_only
def save_dygraph(state_dict, model_path):
    '''
    :api_attr: imperative

    Save Layer's state_dict to disk. This will generate a file with suffix ".pdparams"
    
    The state_dict is get from Layers.state_dict function
    
    Args:
        state_dict(dict) : The state dict to be saved.
        model_path(str) : the file prefix to save the state_dict. The format is "dirname/file_prefix". If file_prefix is empty str. A exception will be raised

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            with fluid.dygraph.guard():
                emb = fluid.dygraph.Embedding([10, 10])

                state_dict = emb.state_dict()
                fluid.save_dygraph( state_dict, "paddle_dy")

                adam = fluid.optimizer.Adam( learning_rate = fluid.layers.noam_decay( 100, 10000),
                                             parameter_list = emb.parameters() )

                state_dict = adam.state_dict()
                fluid.save_dygraph( state_dict, "paddle_dy")

    '''

    base_name = os.path.basename(model_path)
    assert base_name != "", "The input model_path MUST be format of dirname/filename [dirname\\filename in Windows system], but received filename is empty string."

    suffix = ".pdparams"
    assert len(state_dict) > 0, "state_dict is empty, no need to save"

    param_num = 0
    for k, v in state_dict.items():
        if isinstance(v, ParamBase):
            param_num += 1

    if param_num == 0:
        suffix = ".pdopt"

    model_dict = {}
    name_table = {}
    for k, v in state_dict.items():
        if isinstance(v, (Variable, core.VarBase)):
            model_dict[k] = v.numpy()
            name_table[k] = v.name
        else:
            model_dict[k] = v
    model_dict["StructuredToParameterName@@"] = name_table

    file_name = model_path + suffix
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(file_name, 'wb') as f:
        pickle.dump(model_dict, f, protocol=2)


# NOTE(chenweihang): load_dygraph will deprecated in future, we don't 
# support new loading features for it
# TODO(qingqing01): remove dygraph_only to support loading static model.
# maybe need to unify the loading interface after 2.0 API is ready.
# @dygraph_only
def load_dygraph(model_path, **configs):
    '''
    :api_attr: imperative
    
    Load parameter state dict from disk.

    .. note::
        Due to some historical reasons, if you load ``state_dict`` from the saved 
        result of `paddle.static.save_inference_model`, the structured variable name 
        will cannot be restored. You need to set the argument `use_structured_name=False` 
        when using `Layer.set_state_dict` later.

    Args:
        model_path(str) : The file prefix store the state_dict. 
            (The path should Not contain suffix '.pdparams') 
        **configs (dict, optional): Other load configuration options for compatibility. We do not 
            recommend using these configurations, if not necessary, DO NOT use them. Default None.
            The following options are currently supported:
            (1) model_filename (str): The inference model file name of the paddle 1.x ``save_inference_model`` 
            save format. Default file name is :code:`__model__` . 
            (2) params_filename (str): The persistable variables file name of the paddle 1.x ``save_inference_model`` 
            save format. No default file name, save variables separately by default.

    Returns:
        state_dict(dict) : the dict store the state_dict

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            paddle.disable_static()

            emb = paddle.nn.Embedding(10, 10)

            state_dict = emb.state_dict()
            fluid.save_dygraph(state_dict, "paddle_dy")

            scheduler = paddle.optimizer.lr.NoamDecay(	
                d_model=0.01, warmup_steps=100, verbose=True)
            adam = paddle.optimizer.Adam(
                learning_rate=scheduler,
                parameters=emb.parameters())
            state_dict = adam.state_dict()
            fluid.save_dygraph(state_dict, "paddle_dy")

            para_state_dict, opti_state_dict = fluid.load_dygraph("paddle_dy")
    '''
    # deal with argument `model_path`
    model_prefix = model_path
    if model_prefix.endswith(".pdparams"):
        model_prefix = model_prefix[:-9]
    elif model_prefix.endswith(".pdopt"):
        model_prefix = model_prefix[:-6]

    para_dict = None
    opti_dict = None
    params_file_path = model_prefix + ".pdparams"
    opti_file_path = model_prefix + ".pdopt"

    # deal with argument `config`
    config = _parse_load_config(configs)

    if os.path.exists(params_file_path) or os.path.exists(opti_file_path):
        # Load state dict by `save_dygraph` save format
        para_dict = {}
        if os.path.exists(params_file_path):
            with open(params_file_path, 'rb') as f:
                para_dict = pickle.load(f, encoding='latin1')

        if not config.keep_name_table and "StructuredToParameterName@@" in para_dict:
            del para_dict["StructuredToParameterName@@"]

        if os.path.exists(opti_file_path):
            with open(opti_file_path, 'rb') as f:
                opti_dict = pickle.load(f, encoding='latin1')
    else:
        # check model path
        if not os.path.isdir(model_prefix):
            raise ValueError("Model saved directory '%s' is not exists." %
                             model_prefix)

        # check whether model file exists
        if config.model_filename is None:
            model_filename = '__model__'
        else:
            model_filename = config.model_filename
        model_file_path = os.path.join(model_path, model_filename)

        if os.path.exists(model_file_path):
            # Load state dict by `jit.save/io.save_inference_model` save format
            # NOTE(chenweihang): [ Compatibility of save_inference_model save format ]
            # The model saved by `save_inference_model` does not completely correspond to 
            # the information required by the `state_dict` under the dygraph. 
            # `save_inference_model` not save structured name, we need to remind 
            # the user to configure the `use_structured_name` argument when `set_state_dict`
            # NOTE(chenweihang): `jit.save` doesn't save optimizer state 

            # 1. load program desc & construct _ProgramHolder
            programs = _construct_program_holders(model_path,
                                                  config.model_filename)

            # 2. load layer parameters & buffers
            # NOTE: using fluid.dygraph.guard() here will cause import error in py2
            with guard():
                persistable_var_dict = _construct_params_and_buffers(
                    model_prefix,
                    programs,
                    config.params_filename,
                    append_suffix=False)

                # 3. construct state_dict
                para_dict = dict()
                for var_name in persistable_var_dict:
                    para_dict[var_name] = persistable_var_dict[var_name].numpy()

                # if *.info exists, we can recover structured_name
                var_info_filename = str(config.params_filename) + ".info"
                var_info_path = os.path.join(model_prefix, var_info_filename)
                if os.path.exists(var_info_path):
                    with open(var_info_path, 'rb') as f:
                        extra_var_info = pickle.load(f)
                    structured_para_dict = dict()
                    for var_name in para_dict:
                        structured_name = extra_var_info[var_name].get(
                            'structured_name', None)
                        assert structured_name is not None, "Cannot find saved variable (%s)'s structured name in saved model." % var_name
                        structured_para_dict[structured_name] = para_dict[
                            var_name]
                    para_dict = structured_para_dict
        else:
            # load state dict by `io.save_params/persistables` save format
            # TODO(chenweihang): [ Now only supports loading parameters seperately ]
            # If users save all parameters as one file, the [ variable.name -> variable ]
            # mapping info will lost, so users need to give variable list, but users build 
            # variable list in dygraph mode is difficult, we recommend users to use
            # paddle.static.load_program_state in this case

            # Try to load all the files in the directory in VarBase format, 
            # the file name is used as the name of VarBase
            load_var_list = []

            # 1. load file names
            var_name_list = []
            for root, _, files in os.walk(model_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    tmp_var_name = os.path.relpath(file_path, model_path)
                    var_name = tmp_var_name.replace("\\", "/")
                    var_name_list.append(var_name)

            # 2. create and load VarBase
            with guard():
                for name in var_name_list:
                    new_var = _varbase_creator(name=name, persistable=True)
                    _dygraph_tracer().trace_op(
                        type='load',
                        inputs={},
                        outputs={'Out': new_var},
                        attrs={'file_path': os.path.join(model_path, name)})
                    load_var_list.append(new_var)

            # 3. construct state_dict
            para_dict = dict()
            for var in load_var_list:
                para_dict[var.name] = var.numpy()

    return para_dict, opti_dict
