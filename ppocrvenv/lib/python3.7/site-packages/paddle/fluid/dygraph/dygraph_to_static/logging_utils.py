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

import os
import threading

import six
from paddle.fluid import log_helper
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code

__all__ = ["TranslatorLogger", "set_verbosity", "set_code_level"]

VERBOSITY_ENV_NAME = 'TRANSLATOR_VERBOSITY'
CODE_LEVEL_ENV_NAME = 'TRANSLATOR_CODE_LEVEL'
DEFAULT_VERBOSITY = -1
DEFAULT_CODE_LEVEL = -1

LOG_AllTransformer = 100


def synchronized(func):
    def wrapper(*args, **kwargs):
        with threading.Lock():
            return func(*args, **kwargs)

    return wrapper


class TranslatorLogger(object):
    """
    class for Logging and debugging during the tranformation from dygraph to static graph.
    The object of this class is a singleton.
    """

    @synchronized
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.logger_name = "Dynamic-to-Static"
        self._logger = log_helper.get_logger(
            self.logger_name,
            1,
            fmt='%(asctime)s %(name)s %(levelname)s: %(message)s')
        self._verbosity_level = None
        self._transformed_code_level = None
        self._need_to_echo_log_to_stdout = None
        self._need_to_echo_code_to_stdout = None

    @property
    def logger(self):
        return self._logger

    @property
    def verbosity_level(self):
        if self._verbosity_level is not None:
            return self._verbosity_level
        else:
            return int(os.getenv(VERBOSITY_ENV_NAME, DEFAULT_VERBOSITY))

    @verbosity_level.setter
    def verbosity_level(self, level):
        self.check_level(level)
        self._verbosity_level = level

    @property
    def transformed_code_level(self):
        if self._transformed_code_level is not None:
            return self._transformed_code_level
        else:
            return int(os.getenv(CODE_LEVEL_ENV_NAME, DEFAULT_CODE_LEVEL))

    @transformed_code_level.setter
    def transformed_code_level(self, level):
        self.check_level(level)
        self._transformed_code_level = level

    @property
    def need_to_echo_log_to_stdout(self):
        if self._need_to_echo_log_to_stdout is not None:
            return self._need_to_echo_log_to_stdout
        return False

    @need_to_echo_log_to_stdout.setter
    def need_to_echo_log_to_stdout(self, log_to_stdout):
        assert isinstance(log_to_stdout, (bool, type(None)))
        self._need_to_echo_log_to_stdout = log_to_stdout

    @property
    def need_to_echo_code_to_stdout(self):
        if self._need_to_echo_code_to_stdout is not None:
            return self._need_to_echo_code_to_stdout
        return False

    @need_to_echo_code_to_stdout.setter
    def need_to_echo_code_to_stdout(self, code_to_stdout):
        assert isinstance(code_to_stdout, (bool, type(None)))
        self._need_to_echo_code_to_stdout = code_to_stdout

    def check_level(self, level):
        if isinstance(level, (six.integer_types, type(None))):
            rv = level
        else:
            raise TypeError("Level is not an integer: {}".format(level))
        return rv

    def has_code_level(self, level):
        level = self.check_level(level)
        return level == self.transformed_code_level

    def has_verbosity(self, level):
        """
        Checks whether the verbosity level set by the user is greater than or equal to the log level.
        Args:
            level(int): The level of log.
        Returns:
            True if the verbosity level set by the user is greater than or equal to the log level, otherwise False.
        """
        level = self.check_level(level)
        return self.verbosity_level >= level

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
        if self.need_to_echo_log_to_stdout:
            self._output_to_stdout('ERROR: ' + msg, *args)

    def warn(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
        if self.need_to_echo_log_to_stdout:
            self._output_to_stdout('WARNING: ' + msg, *args)

    def log(self, level, msg, *args, **kwargs):
        if self.has_verbosity(level):
            msg_with_level = '(Level {}) {}'.format(level, msg)
            self.logger.info(msg_with_level, *args, **kwargs)
            if self.need_to_echo_log_to_stdout:
                self._output_to_stdout('INFO: ' + msg_with_level, *args)

    def log_transformed_code(self, level, ast_node, transformer_name, *args,
                             **kwargs):
        if self.has_code_level(level):
            source_code = ast_to_source_code(ast_node)
            if level == LOG_AllTransformer:
                header_msg = "After the last level ast transformer: '{}', the transformed code:\n" \
                    .format(transformer_name)
            else:
                header_msg = "After the level {} ast transformer: '{}', the transformed code:\n"\
                    .format(level, transformer_name)

            msg = header_msg + source_code
            self.logger.info(msg, *args, **kwargs)

            if self.need_to_echo_code_to_stdout:
                self._output_to_stdout('INFO: ' + msg, *args)

    def _output_to_stdout(self, msg, *args):
        msg = self.logger_name + ' ' + msg
        print(msg % args)


_TRANSLATOR_LOGGER = TranslatorLogger()


def set_verbosity(level=0, also_to_stdout=False):
    """
    Sets the verbosity level of log for dygraph to static graph. Logs can be output to stdout by setting `also_to_stdout`.

    There are two means to set the logging verbosity:

    1. Call function `set_verbosity`

    2. Set environment variable `TRANSLATOR_VERBOSITY`


    **Note**:
    `set_verbosity` has a higher priority than the environment variable.

    Args:
        level(int): The verbosity level. The larger value idicates more verbosity.
            The default value is 0, which means no logging.
        also_to_stdout(bool): Whether to also output log messages to `sys.stdout`.

    Examples:
        .. code-block:: python

            import os
            import paddle

            paddle.jit.set_verbosity(1)
            # The verbosity level is now 1

            os.environ['TRANSLATOR_VERBOSITY'] = '3'
            # The verbosity level is now 3, but it has no effect because it has a lower priority than `set_verbosity`
    """
    _TRANSLATOR_LOGGER.verbosity_level = level
    _TRANSLATOR_LOGGER.need_to_echo_log_to_stdout = also_to_stdout


def get_verbosity():
    return _TRANSLATOR_LOGGER.verbosity_level


def set_code_level(level=LOG_AllTransformer, also_to_stdout=False):
    """
    Sets the level to print code from specific level Ast Transformer. Code can be output to stdout by setting `also_to_stdout`.

    There are two means to set the code level:

    1. Call function `set_code_level`

    2. Set environment variable `TRANSLATOR_CODE_LEVEL`


    **Note**:
    `set_code_level` has a higher priority than the environment variable.

    Args:
        level(int): The level to print code. Default is 100, which means to print the code after all AST Transformers.
        also_to_stdout(bool): Whether to also output code to `sys.stdout`.

    Examples:
        .. code-block:: python

            import paddle

            paddle.jit.set_code_level(2)
            # It will print the transformed code at level 2, which means to print the code after second transformer,
            # as the date of August 28, 2020, it is CastTransformer.

            os.environ['TRANSLATOR_CODE_LEVEL'] = '3'
            # The code level is now 3, but it has no effect because it has a lower priority than `set_code_level`

    """
    _TRANSLATOR_LOGGER.transformed_code_level = level
    _TRANSLATOR_LOGGER.need_to_echo_code_to_stdout = also_to_stdout


def get_code_level():
    return _TRANSLATOR_LOGGER.transformed_code_level


def error(msg, *args, **kwargs):
    _TRANSLATOR_LOGGER.error(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    _TRANSLATOR_LOGGER.warn(msg, *args, **kwargs)


def log(level, msg, *args, **kwargs):
    _TRANSLATOR_LOGGER.log(level, msg, *args, **kwargs)


def log_transformed_code(level, ast_node, transformer_name, *args, **kwargs):
    _TRANSLATOR_LOGGER.log_transformed_code(level, ast_node, transformer_name,
                                            *args, **kwargs)
