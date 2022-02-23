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

import collections
from .wrapped_decorator import signature_safe_contextmanager
import six
import sys

__all__ = ['generate', 'switch', 'guard']


class UniqueNameGenerator(object):
    """
    Generate unique name with prefix.

    Args:
        prefix(str): The generated name prefix. All generated name will be
                     started with this prefix.
    """

    def __init__(self, prefix=None):
        self.ids = collections.defaultdict(int)
        if prefix is None:
            prefix = ""
        self.prefix = prefix

    def __call__(self, key):
        """
        Generate unique names with prefix

        Args:
            key(str): The key of return string.

        Returns(str): A unique string with the prefix
        """
        tmp = self.ids[key]
        self.ids[key] += 1
        return self.prefix + "_".join([key, str(tmp)])


class DygraphParameterNameChecker(object):
    """
    Check whether the name of parameter is used.
    """

    def __init__(self):
        self._name_set = set()

    def __call__(self, name):
        '''
        Check whether the name is used. If not used, insert into the _name_set.

        Args:
            name(str): The name of parameter to check.

        Returns(bool): If the name is in name_set,  return True; Otherwise, return False.

        '''
        if name in self._name_set:
            return True
        else:
            self._name_set.add(name)
            return False


dygraph_parameter_name_checker = DygraphParameterNameChecker()

generator = UniqueNameGenerator()


def generate(key):
    """
    Generate unique name with prefix key. Currently, Paddle distinguishes the
    names of the same key by numbering it from zero. For example, when key=fc,
    it continuously generates fc_0, fc_1, fc_2, etc.

    Args: 
        key(str): The prefix of generated name.

    Returns: 
        str: A unique string with the prefix key.

    Examples: 

        .. code-block:: python

            import paddle
            name1 = paddle.utils.unique_name.generate('fc')
            name2 = paddle.utils.unique_name.generate('fc')
            print(name1, name2) # fc_0, fc_1
    """
    return generator(key)


# FIXME(zjl): The previous naming rule in static graph would
# cause memory leak in dygraph mode. It is because the previous
# naming rule would use `conv_0.tmp` as the key, and in dygraph
# mode, `conv_i` increases as batch increases. Thus, keys would
# increase in a way like `conv_0.tmp`, `conv_1.tmp`, .... 
# Not find a better way to fix this bug in dygraph mode. In TF,
# variable name is meaningless in eager execution mode, and in
# PyTorch, there is no variable name at all. Maybe we should
# discard variable name in dygraph mode.
#
# Another concern is that save/load interfaces. Usually, user
# would save model in static graph mode, and load it in dygraph
# mode. Therefore, we keep the variable name of Parameter currently.
# 
# Please fix me if a better method is found.    
# 
# NOTE(zhiqiu): use c++ unique_name_generator in dygraph mode, 
# in order to keep name consistency.
def generate_with_ignorable_key(key):
    from .framework import in_dygraph_mode, _dygraph_tracer
    if in_dygraph_mode():
        return _dygraph_tracer()._generate_unique_name()

    return generator(key)


def switch(new_generator=None, new_para_name_checker=None):
    """
    Switch the namespace of in current context to a new namespace. Though
    :code:`switch()` and :code:`guard()` can both change namespace, 
    :code:`guard()` is recommended since it can manage the context better 
    together with :code:`with` statement.

    Args: 
        new_generator(UniqueNameGenerator, optional): A new UniqueNameGenerator, not
            required normally. Default is None, which means switch to a new anonymous
            namespace.
        new_para_name_checker(DygraphParameterNameChecker, optional): A new DygraphParameterNameChecker,
            not required normally. Default is None, which means  switch to a new parameter name 
            checker.

    Returns: 
        UniqueNameGenerator: The previous UniqueNameGenerator.
        DygraphParameterNameChecker: The previous DygraphParameterNameChecker

    Examples: 

        .. code-block:: python

            import paddle
            name1 = paddle.utils.unique_name.generate('fc')
            name2 = paddle.utils.unique_name.generate('fc')
            print(name1, name2) # fc_0, fc_1

            pre_generator, pre_dygraph_name_checker = paddle.utils.unique_name.switch() # switch to a new anonymous namespace.
            name2 = paddle.utils.unique_name.generate('fc')
            print(name2) # fc_0

            paddle.utils.unique_name.switch(pre_generator, pre_dygraph_name_checker) # switch back to pre_generator.
            name3 = paddle.utils.unique_name.generate('fc')
            print(name3) # fc_2, since pre_generator has generated fc_0, fc_1.
    """
    global generator
    old_generator = generator
    global dygraph_parameter_name_checker
    old_para_name_checker = dygraph_parameter_name_checker
    if new_generator is None:
        generator = UniqueNameGenerator()
    else:
        generator = new_generator

    if new_para_name_checker is None:
        dygraph_parameter_name_checker = DygraphParameterNameChecker()
    else:
        dygraph_parameter_name_checker = new_para_name_checker
    return old_generator, old_para_name_checker


@signature_safe_contextmanager
def guard(new_generator=None):
    """
    Change the namespace of unique name with :code:`with` statement. After calling it,
    a new namespace in the context of :code:`with` will be created, and it will number
    names from zero again when calling :code:`generate()` with same key.

    Args: 
        new_generator(str|bytes, optional): New name of global namespace. Note that str
            in Python2 was spilted into str and bytes in Python3, so here are two 
            types. Default is None. If not None, new_generator will be added into 
            the prefix of unique name generated by :code:`generate()`.
    
    Returns:
        None.

    Examples: 

        .. code-block:: python

            import paddle
            with paddle.utils.unique_name.guard():
                name_1 = paddle.utils.unique_name.generate('fc')
            with paddle.utils.unique_name.guard():
                name_2 = paddle.utils.unique_name.generate('fc')
            print(name_1, name_2) # fc_0, fc_0

            with paddle.utils.unique_name.guard('A'):
                name_1 = paddle.utils.unique_name.generate('fc')
            with paddle.utils.unique_name.guard('B'):
                name_2 = paddle.utils.unique_name.generate('fc')
            print(name_1, name_2) # Afc_0, Bfc_0
    """
    if isinstance(new_generator, six.string_types):
        new_generator = UniqueNameGenerator(new_generator)
    elif isinstance(new_generator, six.binary_type):
        new_generator = UniqueNameGenerator(new_generator.decode())

    old_generator, old_para_name_checker = switch(new_generator)
    try:
        yield
    finally:
        switch(old_generator, old_para_name_checker)
