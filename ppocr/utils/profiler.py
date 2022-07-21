# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import sys
import paddle.profiler as profiler

# A global variable to record the number of calling times for profiler
# functions. It is used to specify the tracing range of training steps.
_profiler_step_id = 0

class ProfilerOptions(object):
    '''
    Use a string to initialize a ProfilerOptions.
    The string should be in the format: "key1=value1;key2=value;key3=value3".
    For example:
      "batch_range=[50, 60]"
      "batch_range=[50, 60]; targets=GPU"
      "batch_range=[50, 60]; targets=All; sorted_key='GPUTotal'"
    ProfilerOptions supports following key-value pair:
      batch_range      - an integer list, e.g. [100, 110].
      targets          - a string, the optional values are 'CPU', 'GPU' or 'All'. 
      sorted_key       - a string, the default values are 'CPUTotal'.
      profile_path     - a string, the path to save the serialized profile data.
      exit_on_finished - a boolean.
    '''

    def __init__(self, options_str):
        self._options = {
            'batch_range': [50, 60],
            'targets': [profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            'sorted_key': profiler.SortedKeys['CPUTotal'],
            'profile_path': "",
            'exit_on_finished': True,
            'timer_only': True
        }
        if options_str != None:
            assert isinstance(options_str, str)
            self._options['timer_only'] = False
            if '=' in options_str:
                self._parse_from_string(options_str)

    def _parse_from_string(self, options_str):
        for kv in options_str.replace(' ', '').split(';'):
            key, value = kv.split('=')
            if key == 'batch_range':
                value_list = value.replace('[', '').replace(']', '').split(',')
                value_list = list(map(int, value_list))
                if len(value_list) >= 2 and value_list[0] >= 0 and value_list[
                        1] > value_list[0]:
                    self._options[key] = value_list
            elif key == 'targets':
                if value.lower() == 'all':
                    continue
                elif value.lower() == 'cpu':
                    del self._options[key][1]
                elif value.lower() == 'gpu':
                    del self._options[key][0]
                else:
                    raise ValueError(
                        "Profiler does not have a target named %s." % value)
            elif key == 'exit_on_finished':
                self._options[key] = value.lower() in ("yes", "true", "t", "1")
            elif key == 'sorted_key':
                self._options[key] = profiler.SortedKeys[value]
            elif key in ['profile_path']:
                self._options[key] = value
            else:
                raise ValueError(
                    "ProfilerOptions does not have an option named %s." % key)

class ModelProfiler(object):
    def __init__(self, options_str):
        
        self._profiler_options = ProfilerOptions(options_str)
        self._profiler = profiler.Profiler(targets=self._profiler_options._options['targets'],
                                           scheduler=self._profiler_options._options['batch_range'],
                                           timer_only=self._profiler_options._options['timer_only'])
        self._profiler.start()

    def step(self, batch_size=1):
        global _profiler_step_id

        self._profiler.step(num_samples=batch_size)
        _profiler_step_id += 1
        if not self._profiler_options._options['timer_only']:
            if _profiler_step_id == self._profiler_options._options['batch_range'][1]:
                self.stop()
                if self._profiler_options._options['exit_on_finished']:
                    sys.exit(0)

    def stop(self):
        self._profiler.stop()
        if not self._profiler_options._options['timer_only']:
            self._profiler.summary(sorted_by=self._profiler_options._options['sorted_key'])
            path = self._profiler_options._options['profile_path']
            if path != "":
                self._profiler.export(path=path)

