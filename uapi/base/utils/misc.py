# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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

import os
import contextlib
import subprocess


def run_cmd(cmd, silent=True, wd=None, timeout=None, echo=False):
    """Wrap around `subprocess.run()` to execute a shell command."""
    # XXX: This function is not safe!!!
    cfg = dict(check=True, shell=True, timeout=timeout)
    if silent:
        cfg['stdout'] = subprocess.DEVNULL
    if echo:
        print(cmd)
    if wd is not None:
        with switch_working_dir(wd):
            return subprocess.run(cmd, **cfg)
    else:
        return subprocess.run(cmd, **cfg)


@contextlib.contextmanager
def switch_working_dir(new_wd):
    cwd = os.getcwd()
    os.chdir(new_wd)
    try:
        yield
    finally:
        os.chdir(cwd)


def abspath(path):
    return os.path.abspath(path)


class CachedProperty(object):
    """
    A property that is only computed once per instance and then replaces itself with an ordinary attribute.
    The implementation refers to https://github.com/pydanny/cached-property/blob/master/cached_property.py .
        Note that this implementation does NOT work in multi-thread or coroutine senarios.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.__doc__ = getattr(func, '__doc__', '')

    def __get__(self, obj, cls):
        if obj is None:
            return self
        val = self.func(obj)
        # Hack __dict__ of obj to inject the value
        # Note that this is only executed once
        obj.__dict__[self.func.__name__] = val
        return val
