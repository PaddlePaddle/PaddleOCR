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
"""Defination of Server and Worker."""

from . import ps_pb2 as pslib
# NOTE: reduce removed in fuctools in python3
from functools import reduce


class Server(object):
    """
        A Server basic class
        it's a base class, does not have implementation
    """

    def __init__(self):
        pass


class Worker(object):
    """
        A Worker basic class.
        it's a base class, does not have implementation
    """

    def __init__(self):
        pass


class DownpourServer(Server):
    """
        DownpourServer class is used to generate server program_desc
        Args:
            server: it is pslib.ServerParameter()
        Examples:
            server = DownpourServer()
    """

    def __init__(self):
        self._server = pslib.ServerParameter()
        self._server.downpour_server_param.service_param.server_class = "DownpourBrpcPsServer"
        self._server.downpour_server_param.service_param.client_class = "DownpourBrpcPsClient"
        self._server.downpour_server_param.service_param.service_class = "DownpourPsService"
        self._server.downpour_server_param.service_param.start_server_port = 0
        self._server.downpour_server_param.service_param.server_thread_num = 12

    def add_sparse_table(self, table_id, strategy):
        """
        Args:
            table_id(int): id of sparse params table
            strategy(dict): the config dict.
        Returns:
            return None
        """

        for table in self._server.downpour_server_param.downpour_table_param:
            if table.table_id == table_id:
                if table.type == pslib.PS_SPARSE_TABLE:
                    return
                else:
                    raise ValueError("expect table %s type=%s, but actual type=%s" \
                                     %(table_id, pslib.PS_SPARSE_TABLE, table.type))
        if strategy is None:
            strategy = dict()
        table = self._server.downpour_server_param.downpour_table_param.add()
        table.table_id = table_id
        table.type = pslib.PS_SPARSE_TABLE

        support_sparse_key_list = ['sparse_table_class', 'sparse_compress_in_save', 'sparse_shard_num', \
                                   'sparse_accessor_class', 'sparse_learning_rate', 'sparse_initial_g2sum', 'sparse_initial_range', \
                                   'sparse_weight_bounds', 'sparse_embedx_dim', 'sparse_embedx_threshold', 'sparse_nonclk_coeff', \
                                   'sparse_click_coeff', 'sparse_base_threshold', 'sparse_delta_threshold', 'sparse_delta_keep_days', \
                                   'sparse_delete_after_unseen_days', 'sparse_show_click_decay_rate', 'sparse_delete_threshold', \
                                   'sparse_converter', 'sparse_deconverter', 'sparse_enable_cache', 'sparse_cache_rate', \
                                   'sparse_cache_file_num', 'sparse_beta1_decay_rate', 'sparse_beta2_decay_rate', \
                                   'sparse_ada_epsilon', 'sparse_optimizer', 'sparse_ssd_unseenday_threshold', \
                                   'embed_sparse_optimizer', 'embed_sparse_learning_rate', 'embed_sparse_weight_bounds', \
                                   'embed_sparse_initial_range', 'embed_sparse_initial_g2sum', 'embed_sparse_beta1_decay_rate', \
                                   'embed_sparse_beta2_decay_rate', 'embedx_sparse_optimizer', 'embedx_sparse_learning_rate', \
                                   'embedx_sparse_weight_bounds', 'embedx_sparse_initial_range', 'embedx_sparse_initial_g2sum', \
                                   'embedx_sparse_beta1_decay_rate', 'embedx_sparse_beta2_decay_rate']

        for key in strategy:
            if key not in support_sparse_key_list:
                raise ValueError("strategy key '%s' not support" % (key))

        support_table_calss = ['DownpourSparseTable', 'DownpourSparseSSDTable']
        if strategy.get('sparse_table_class') is not None:
            table_class = strategy.get('sparse_table_class')
            if table_class not in support_table_calss:
                raise ValueError(
                    "support sparse_table_class: [ 'DownpourSparseTable', 'DownpourSparseSSDTable'], \
                        but actual %s" % (table_class))
        else:
            table_class = 'DownpourSparseTable'

        table.table_class = table_class

        if table_class == 'DownpourSparseTable' or table_class == 'DownpourSparseSSDTable':
            table.enable_sparse_table_cache = strategy.get(
                'sparse_enable_cache', True)
            table.sparse_table_cache_rate = strategy.get('sparse_cache_rate',
                                                         0.00055)
            table.sparse_table_cache_file_num = strategy.get(
                'sparse_cache_file_num', 16)
            table.compress_in_save = strategy.get('sparse_compress_in_save',
                                                  True)
            table.shard_num = strategy.get('sparse_shard_num', 1000)
            # DownpourFeatureValueAccessor: for ctr task, has cvm, embedding and sgd info
            # DownpourCtrAccessor         : for ctr task, has cvm, slot, embedding and sgd info
            # DownpourSparseValueAccessor : for general task, has embedding and sgd info
            # DownpourCtrDoubleAccessor   : for ctr task, which show clk are in double
            # DownpourUnitAccessor        : for ctr task, has cvm, slot, embedding and sgd info

            support_accessor_class = [
                'DownpourFeatureValueAccessor', 'DownpourCtrAccessor',
                'DownpourSparseValueAccessor', 'DownpourCtrDoubleAccessor',
                'DownpourUnitAccessor', 'DownpourDoubleUnitAccessor'
            ]
            if strategy.get('sparse_accessor_class') is not None:
                accessor_class = strategy.get('sparse_accessor_class')
                if accessor_class not in support_accessor_class:
                    raise ValueError(
                        "support sparse_accessor_class: ['DownpourFeatureValueAccessor', 'DownpourCtrAccessor', \
                        'DownpourSparseValueAccessor', 'DownpourCtrDoubleAccessor'], \
                            but actual %s" % (accessor_class))
            else:
                accessor_class = 'DownpourCtrAccessor'

            table.accessor.accessor_class = accessor_class

            if accessor_class == 'DownpourFeatureValueAccessor' \
                    or accessor_class == 'DownpourCtrAccessor' \
                    or accessor_class == 'DownpourCtrDoubleAccessor':
                table.accessor.sparse_sgd_param.learning_rate = strategy.get(
                    'sparse_learning_rate', 0.05)
                table.accessor.sparse_sgd_param.initial_g2sum = strategy.get(
                    'sparse_initial_g2sum', 3)
                table.accessor.sparse_sgd_param.initial_range = strategy.get(
                    'sparse_initial_range', 1e-4)
                if strategy.get('sparse_weight_bounds') is None:
                    table.accessor.sparse_sgd_param.weight_bounds.extend(
                        [-10, 10])
                else:
                    table.accessor.sparse_sgd_param.weight_bounds.extend(
                        strategy.get('sparse_weight_bounds'))
                table.accessor.embedx_dim = strategy.get('sparse_embedx_dim', 8)
                table.accessor.embedx_threshold = strategy.get(
                    'sparse_embedx_threshold', 10)
                table.accessor.fea_dim = int(table.accessor.embedx_dim) + 3
                table.accessor.downpour_accessor_param.nonclk_coeff = strategy.get(
                    'sparse_nonclk_coeff', 0.1)
                table.accessor.downpour_accessor_param.click_coeff = strategy.get(
                    'sparse_click_coeff', 1)
                table.accessor.downpour_accessor_param.base_threshold = strategy.get(
                    'sparse_base_threshold', 1.5)
                table.accessor.downpour_accessor_param.delta_threshold = strategy.get(
                    'sparse_delta_threshold', 0.25)
                table.accessor.downpour_accessor_param.delta_keep_days = strategy.get(
                    'sparse_delta_keep_days', 16)
                table.accessor.downpour_accessor_param.delete_after_unseen_days = strategy.get(
                    'sparse_delete_after_unseen_days', 30)
                table.accessor.downpour_accessor_param.ssd_unseenday_threshold = strategy.get(
                    'sparse_ssd_unseenday_threshold', 1)
                table.accessor.downpour_accessor_param.show_click_decay_rate = strategy.get(
                    'sparse_show_click_decay_rate', 0.98)
                table.accessor.downpour_accessor_param.delete_threshold = strategy.get(
                    'sparse_delete_threshold', 0.8)
                converter = strategy.get(
                    'sparse_converter',
                    "(scripts/xbox_compressor_mf.py | bin/xbox_pb_converter)")
                deconverter = strategy.get(
                    'sparse_deconverter',
                    "(bin/xbox_pb_deconverter | scripts/xbox_decompressor_mf.awk)"
                )

                table1 = table.accessor.table_accessor_save_param.add()
                table1.param = 1
                table1.converter = converter
                table1.deconverter = deconverter

                table2 = table.accessor.table_accessor_save_param.add()
                table2.param = 2
                table2.converter = converter
                table2.deconverter = deconverter
            elif accessor_class == 'DownpourSparseValueAccessor':
                optimizer_name = strategy.get("sparse_optimizer", "adam")
                table.accessor.sparse_commonsgd_param.name = optimizer_name
                table.accessor.embedx_dim = strategy.get('sparse_embedx_dim', 8)
                table.accessor.fea_dim = int(table.accessor.embedx_dim)
                if optimizer_name == "naive":
                    table.accessor.sparse_commonsgd_param.naive.learning_rate = \
                        strategy.get('sparse_learning_rate', 0.05)
                    table.accessor.sparse_commonsgd_param.naive.initial_range = \
                        strategy.get('sparse_initial_range', 1e-4)
                    if strategy.get('sparse_weight_bounds') is None:
                        table.accessor.sparse_commonsgd_param.naive.weight_bounds.extend(
                            [-10, 10])
                    else:
                        table.accessor.sparse_commonsgd_param.naive.weight_bounds.extend(
                            strategy.get('sparse_weight_bounds'))
                elif optimizer_name == "adagrad":
                    table.accessor.sparse_commonsgd_param.adagrad.learning_rate = \
                        strategy.get('sparse_learning_rate', 0.05)
                    table.accessor.sparse_commonsgd_param.adagrad.initial_range = \
                        strategy.get('sparse_initial_range', 1e-4)
                    table.accessor.sparse_commonsgd_param.adagrad.initial_g2sum = strategy.get(
                        'sparse_initial_g2sum', 3)
                    if strategy.get('sparse_weight_bounds') is None:
                        table.accessor.sparse_commonsgd_param.adagrad.weight_bounds.extend(
                            [-10, 10])
                    else:
                        table.accessor.sparse_commonsgd_param.adagrad.weight_bounds.extend(
                            strategy.get('sparse_weight_bounds'))
                elif optimizer_name == "adam":
                    table.accessor.sparse_commonsgd_param.adam.learning_rate = \
                        strategy.get('sparse_learning_rate', 0.001)
                    table.accessor.sparse_commonsgd_param.adam.initial_range = \
                        strategy.get('sparse_initial_range', 1e-4)
                    table.accessor.sparse_commonsgd_param.adam.beta1_decay_rate = strategy.get(
                        'sparse_beta1_decay_rate', 0.9)
                    table.accessor.sparse_commonsgd_param.adam.beta2_decay_rate = strategy.get(
                        'sparse_beta2_decay_rate', 0.999)
                    table.accessor.sparse_commonsgd_param.adam.ada_epsilon = strategy.get(
                        'sparse_ada_epsilon', 1e-8)
                    if strategy.get('sparse_weight_bounds') is None:
                        table.accessor.sparse_commonsgd_param.adam.weight_bounds.extend(
                            [-10, 10])
                    else:
                        table.accessor.sparse_commonsgd_param.adam.weight_bounds.extend(
                            strategy.get('sparse_weight_bounds'))
                converter = strategy.get(
                    'sparse_converter',
                    "(scripts/xbox_compressor_mf.py | bin/xbox_pb_converter)")
                deconverter = strategy.get(
                    'sparse_deconverter',
                    "(bin/xbox_pb_deconverter | scripts/xbox_decompressor_mf.awk)"
                )

                table1 = table.accessor.table_accessor_save_param.add()
                table1.param = 1
                table1.converter = converter
                table1.deconverter = deconverter

                table2 = table.accessor.table_accessor_save_param.add()
                table2.param = 2
                table2.converter = converter
                table2.deconverter = deconverter
            elif accessor_class == 'DownpourUnitAccessor' or accessor_class == 'DownpourDoubleUnitAccessor':
                self.add_sparse_table_common_config(table, strategy)
                self.add_sparse_optimizer(table.accessor.embed_sgd_param,
                                          strategy, "embed_")
                self.add_sparse_optimizer(table.accessor.embedx_sgd_param,
                                          strategy, "embedx_")

    def add_dense_table(self, table_id, param_var, grad_var, strategy,
                        sparse_table_names):
        """
        Args:
            table_id(int): id of sparse params table
            param_var(list): param vars
            grad_var(list): param grad vars
            strategy(dict): the dense config dict
            sparse_table_names(list): sparse table names
        Returns:
            return None
        """
        fea_dim = 0
        dense_param_vars = []
        for p in param_var:
            if p.name not in sparse_table_names:
                dense_param_vars.append(p)

        for param in dense_param_vars:
            fea_dim += reduce(lambda x, y: x * y, param.shape, 1)

        for table in self._server.downpour_server_param.downpour_table_param:
            if table.table_id == table_id:
                if table.type == pslib.PS_DENSE_TABLE:
                    table.accessor.fea_dim = fea_dim
                    return
                else:
                    raise ValueError("expect table %s type=%s, but actual type=%s" \
                                     %(table_id, pslib.PS_DENSE_TABLE, table.type))

        if strategy is None:
            strategy = dict()
        table = self._server.downpour_server_param.downpour_table_param.add()
        table.table_id = table_id
        support_dense_key_list = ['dense_table_class', 'dense_compress_in_save', 'dense_accessor_class', \
                                  'dense_optimizer', 'dense_learning_rate', 'dense_avg_decay', 'dense_ada_decay', \
                                  'dense_ada_epsilon', 'dense_mom_decay', 'dense_naive_lr']

        for key in strategy:
            if key not in support_dense_key_list:
                raise ValueError("strategy key '%s' not support" % (key))

        table.table_class = strategy.get('dense_table_class',
                                         "DownpourDenseTable")
        table.type = pslib.PS_DENSE_TABLE
        table.compress_in_save = strategy.get('dense_compress_in_save', True)
        table.accessor.accessor_class = strategy.get(
            'dense_accessor_class', "DownpourDenseValueAccessor")
        table.accessor.dense_sgd_param.name = strategy.get('dense_optimizer',
                                                           "adam")
        table.accessor.dense_sgd_param.adam.learning_rate = strategy.get(
            'dense_learning_rate', 5e-06)
        table.accessor.dense_sgd_param.adam.avg_decay_rate = strategy.get(
            'dense_avg_decay', 0.999993)
        table.accessor.dense_sgd_param.adam.ada_decay_rate = strategy.get(
            'dense_ada_decay', 0.9999)
        table.accessor.dense_sgd_param.adam.ada_epsilon = strategy.get(
            'dense_ada_epsilon', 1e-8)
        table.accessor.dense_sgd_param.adam.mom_decay_rate = strategy.get(
            'dense_mom_decay', 0.99)
        table.accessor.dense_sgd_param.naive.learning_rate = strategy.get(
            'dense_naive_lr', 0.0002)
        table.accessor.fea_dim = fea_dim

    def add_data_norm_table(self, table_id, learning_rate, param_var, grad_var,
                            strategy, sparse_table_names):
        """
        Args:
            table_id(int): id of datanorm table
            learning_rate(float): the learning rate used to update parameters
            param_var(list): param vars
            grad_var(list): param grad vars
            strategy(dict): the datanorm config dict
            sparse_table_names(list): sparse table names
        Returns:
            return None
        """
        fea_dim = 0
        dense_param_vars = []
        for p in param_var:
            if p.name not in sparse_table_names:
                dense_param_vars.append(p)

        for param in dense_param_vars:
            fea_dim += reduce(lambda x, y: x * y, param.shape, 1)

        for table in self._server.downpour_server_param.downpour_table_param:
            if table.table_id == table_id:
                if table.type == pslib.PS_DENSE_TABLE:
                    table.accessor.fea_dim = fea_dim
                    return
                else:
                    raise ValueError("expect table %s type=%s, but actual type=%s" \
                                     %(table_id, pslib.PS_DENSE_TABLE, table.type))
        if strategy is None:
            strategy = dict()

        support_datanorm_key_list = ['datanorm_table_class', 'datanorm_compress_in_save', \
                                     'datanorm_accessor_class', 'datanorm_operation', 'datanorm_decay_rate']

        for key in strategy:
            if key not in support_datanorm_key_list:
                raise ValueError("strategy key '%s' not support" % (key))

        table = self._server.downpour_server_param.downpour_table_param.add()
        table.table_id = table_id
        table.table_class = strategy.get('datanorm_table_class',
                                         'DownpourDenseTable')
        table.type = pslib.PS_DENSE_TABLE
        table.compress_in_save = strategy.get('datanorm_compress_in_save', True)
        table.accessor.accessor_class = strategy.get(
            'datanorm_accessor_class', 'DownpourDenseValueAccessor')
        table.accessor.dense_sgd_param.name = strategy.get('datanorm_operation',
                                                           'summary')
        table.accessor.dense_sgd_param.summary.summary_decay_rate = strategy.get(
            'datanorm_decay_rate', 0.999999)
        table.accessor.fea_dim = fea_dim

    def add_sparse_optimizer(self, sgd, strategy, prefix):
        optimizer_name = strategy.get(prefix + "sparse_optimizer", "adagrad")
        sgd.name = optimizer_name
        if optimizer_name == "naive":
            sgd.naive.learning_rate = \
                strategy.get(prefix + 'sparse_learning_rate', 0.05)
            sgd.naive.initial_range = \
                strategy.get(prefix + 'sparse_initial_range', 1e-4)
            bounds = strategy.get(prefix + 'sparse_weight_bounds', [-10, 10])
            sgd.naive.weight_bounds.extend(bounds)
        elif optimizer_name == "adagrad":
            sgd.adagrad.learning_rate = \
                strategy.get(prefix + 'sparse_learning_rate', 0.05)
            sgd.adagrad.initial_range = \
                strategy.get(prefix + 'sparse_initial_range', 1e-4)
            if prefix == "embed_":
                sgd.adagrad.initial_range = 0
            sgd.adagrad.initial_g2sum = strategy.get(
                prefix + 'sparse_initial_g2sum', 3)
            bounds = strategy.get(prefix + 'sparse_weight_bounds', [-10, 10])
            sgd.adagrad.weight_bounds.extend(bounds)
        elif optimizer_name == "std_adagrad":
            sgd.adagrad.learning_rate = \
                strategy.get(prefix + 'sparse_learning_rate', 0.05)
            sgd.adagrad.initial_range = \
                strategy.get(prefix + 'sparse_initial_range', 1e-4)
            if prefix == "embed_":
                sgd.adagrad.initial_range = 0
            sgd.adagrad.initial_g2sum = strategy.get(
                prefix + 'sparse_initial_g2sum', 3)
            bounds = strategy.get(prefix + 'sparse_weight_bounds', [-10, 10])
            sgd.adagrad.weight_bounds.extend(bounds)
        elif optimizer_name == "adam":
            sgd.adam.learning_rate = \
                strategy.get(prefix + 'sparse_learning_rate', 0.001)
            sgd.adam.initial_range = \
                strategy.get(prefix + 'sparse_initial_range', 1e-4)
            sgd.adam.beta1_decay_rate = strategy.get(
                prefix + 'sparse_beta1_decay_rate', 0.9)
            sgd.adam.beta2_decay_rate = strategy.get(
                prefix + 'sparse_beta2_decay_rate', 0.999)
            sgd.adam.ada_epsilon = strategy.get(prefix + 'sparse_ada_epsilon',
                                                1e-8)
            bounds = strategy.get(prefix + 'sparse_weight_bounds', [-10, 10])
            sgd.adam.weight_bounds.extend(bounds)

    def add_sparse_table_common_config(self, table, strategy):
        table.accessor.embedx_dim = strategy.get('sparse_embedx_dim', 8)
        table.accessor.embedx_threshold = strategy.get(
            'sparse_embedx_threshold', 10)
        table.accessor.fea_dim = int(table.accessor.embedx_dim) + 3
        table.accessor.downpour_accessor_param.nonclk_coeff = strategy.get(
            'sparse_nonclk_coeff', 0.1)
        table.accessor.downpour_accessor_param.click_coeff = strategy.get(
            'sparse_click_coeff', 1)
        table.accessor.downpour_accessor_param.base_threshold = strategy.get(
            'sparse_base_threshold', 1.5)
        table.accessor.downpour_accessor_param.delta_threshold = strategy.get(
            'sparse_delta_threshold', 0.25)
        table.accessor.downpour_accessor_param.delta_keep_days = strategy.get(
            'sparse_delta_keep_days', 16)
        table.accessor.downpour_accessor_param.delete_after_unseen_days = strategy.get(
            'sparse_delete_after_unseen_days', 30)
        table.accessor.downpour_accessor_param.show_click_decay_rate = strategy.get(
            'sparse_show_click_decay_rate', 0.98)
        table.accessor.downpour_accessor_param.delete_threshold = strategy.get(
            'sparse_delete_threshold', 0.8)
        converter = strategy.get(
            'sparse_converter',
            "(scripts/xbox_compressor_mf.py | bin/xbox_pb_converter)")
        deconverter = strategy.get(
            'sparse_deconverter',
            "(bin/xbox_pb_deconverter | scripts/xbox_decompressor_mf.awk)")

        table1 = table.accessor.table_accessor_save_param.add()
        table1.param = 1
        table1.converter = converter
        table1.deconverter = deconverter

        table2 = table.accessor.table_accessor_save_param.add()
        table2.param = 2
        table2.converter = converter
        table2.deconverter = deconverter

    def get_desc(self):
        """
        Return downpour server program_desc
        """
        return self._server


class DownpourWorker(Worker):
    """
        DownpourWorker class is used to generate worker program_desc
        Args:
            window (int): push params frequency
            worker: it is pslib.DownpourTrainerParameter
        Examples:
            worker = DownpourWorker(1)
    """

    def __init__(self, window):
        self.window = window
        self._worker = pslib.DownpourTrainerParameter()

    def add_sparse_table(self,
                         table_id,
                         slot_key_vars,
                         slot_value_vars,
                         slot_value_grads=None):
        """
        Args:
            table_id(int): id of sparse params table
            slot_key_vars(list): slot key id
            slot_value_vars(list): slot key value after embedding
            slot_value_grads(list): grad of all params, default is None
        Returns:
            return None
        """
        if slot_value_grads is None:
            slot_value_grad_names = \
                [var.name + "@GRAD" for var in slot_value_vars]
        else:
            value_to_key = {}
            for i in range(len(slot_key_vars)):
                value_to_key[slot_value_vars[i].name] = slot_key_vars[i]
            slot_value_grad_names = []
            all_grad_names = [var.name for var in slot_value_grads]
            for var in slot_value_vars:
                if var.name + "@GRAD" in all_grad_names:
                    slot_value_grad_names.append(var.name + "@GRAD")
            sorted_slot_value_vars = [i for i in slot_value_vars if \
                                      i.name + "@GRAD" in slot_value_grad_names]
            sorted_slot_value_vars += [i for i in slot_value_vars if \
                                       i.name + "@GRAD" not in slot_value_grad_names]
            sorted_slot_key_vars = \
                [value_to_key[v.name] for v in sorted_slot_value_vars]

        target_table = None
        for table in self._worker.sparse_table:
            if table.table_id == table_id:
                keys = table.slot_key
                key_names = [var.name for var in sorted_slot_key_vars]
                for key_name in key_names:
                    if key_name not in keys:
                        raise ValueError("sparse table %s slot_key error" %
                                         table_id)
                target_table = table
                break

        table = target_table
        if table is not None:
            self._worker.sparse_table.remove(table)
        table = self._worker.sparse_table.add()
        table.table_id = table_id
        table.slot_key.extend([var.name for var in sorted_slot_key_vars])
        table.slot_value.extend([var.name for var in sorted_slot_value_vars])
        table.slot_gradient.extend(slot_value_grad_names)

    def add_dense_table(self, table_id, learning_rate, param_vars, grad_vars,
                        dense_start_table_id, sparse_table_names):
        r"""
        Args:
            table_id(int): id of sparse params table
            learning_rate(float): the learning rate used to update parameters. \
                Can be a float value
            param_vars(list): all dense param. it is a list.
            grad_vars(list): all dense grad parm it is a list.
            dense_start_table_id(int): dense table start index
            sparse_table_names(list): sparse table names
        Returns:
            return None
        """
        sparse_table_name_grad = []
        for name in sparse_table_names:
            sparse_table_name_grad.append(name + "@GRAD")

        dense_param_name = []
        for p in param_vars:
            if p.name not in sparse_table_names:
                dense_param_name.append(p.name)

        dense_grad_name = []
        for g in grad_vars:
            if g.name not in sparse_table_name_grad:
                dense_grad_name.append(g.name)

        dense_param_name.sort()
        dense_grad_name.sort()

        for table in self._worker.dense_table:
            if table.table_id == table_id:
                desc_dense_param_name = list(table.dense_variable_name)
                desc_dense_param_name.sort()

                if dense_param_name == desc_dense_param_name:
                    desc_dense_grad_name = list(
                        table.dense_gradient_variable_name)
                    desc_dense_grad_name.sort()
                    if dense_grad_name == desc_dense_grad_name:
                        return
                    else:
                        raise ValueError(
                            "dense table %s dense_gradient_variable_name "
                            "error" % table_id)
                else:
                    raise ValueError(
                        "dense table %s dense_variable_name error" % table_id)

        table = self._worker.dense_table.add()
        table.table_id = table_id

        #def cmp_fc(x, y):
        #    if x.startswith("fc_") and y.startswith("fc_"):
        #        index_x = x.find('.')
        #        index_y = y.find('.')
        #        if index_x > 0 and index_y > 0:
        #            num_x = x[3:index_x]
        #            num_y = y[3:index_y]
        #            if num_x.isdigit() and num_y.isdigit():
        #                if int(num_x) < int(num_y):
        #                    return -1
        #                if int(num_x) > int(num_y):
        #                    return 1
        #                if x[index_x + 1] == 'w' and y[index_y + 1] == 'b':
        #                    return -1
        #                if x[index_x + 1] == 'b' and y[index_y + 1] == 'w':
        #                    return 1
        #    if x < y:
        #        return -1
        #    else:
        #        return 1

        #table.dense_variable_name.extend(sorted(dense_param_name, cmp_fc))
        #table.dense_gradient_variable_name.extend(
        #    sorted(dense_grad_name, cmp_fc))
        table.dense_variable_name.extend(dense_param_name)
        table.dense_gradient_variable_name.extend(dense_grad_name)

    def get_desc(self):
        """
        Return downpour worker program_desc
        """
        return self._worker
