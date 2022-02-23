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

from ..fluid import core

__all__ = []


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


class OpUpdateInfoHelper(object):
    def __init__(self, info):
        self._info = info

    def verify_key_value(self, name=''):
        result = False
        key_funcs = {
            core.OpAttrInfo: 'name',
            core.OpInputOutputInfo: 'name',
        }
        if name == '':
            result = True
        elif type(self._info) in key_funcs:
            if getattr(self._info, key_funcs[type(self._info)])() == name:
                result = True
        return result


@Singleton
class OpLastCheckpointChecker(object):
    def __init__(self):
        self.raw_version_map = core.get_op_version_map()
        self.checkpoints_map = {}
        self._construct_map()

    def _construct_map(self):
        for op_name in self.raw_version_map:
            last_checkpoint = self.raw_version_map[op_name].checkpoints()[-1]
            infos = last_checkpoint.version_desc().infos()
            self.checkpoints_map[op_name] = infos

    def filter_updates(self, op_name, type=core.OpUpdateType.kInvalid, key=''):
        updates = []
        if op_name in self.checkpoints_map:
            for update in self.checkpoints_map[op_name]:
                if (update.type() == type) or (
                        type == core.OpUpdateType.kInvalid):
                    if OpUpdateInfoHelper(update.info()).verify_key_value(key):
                        updates.append(update.info())
        return updates
