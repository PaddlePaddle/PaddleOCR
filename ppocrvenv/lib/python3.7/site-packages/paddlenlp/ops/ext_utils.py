# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import subprocess
import textwrap
import inspect
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.dep_util import newer_group

from paddle.utils.cpp_extension import load_op_meta_info_and_register_op
from paddle.utils.cpp_extension.extension_utils import _jit_compile, _import_module_from_library
from paddle.utils.cpp_extension.cpp_extension import (
    CUDA_HOME, CppExtension, BuildExtension as PaddleBuildExtension)
from paddlenlp.utils.env import PPNLP_HOME
from paddlenlp.utils.log import logger

if CUDA_HOME and not os.path.exists(CUDA_HOME):
    # CUDA_HOME is only None for Windows CPU version in paddle `find_cuda_home`.
    # Clear it for other non-CUDA situations.
    CUDA_HOME = None

LOADED_EXT = {}


def _get_files(path):
    """
    Helps to list all files under the given path.
    """
    if os.path.isfile(path):
        return [path]
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        for file in files:
            file = os.path.join(root, file)
            all_files.append(file)
    return all_files


class CMakeExtension(Extension):
    def __init__(self, name, source_dir=None):
        # A CMakeExtension needs a source_dir instead of a file list.
        Extension.__init__(self, name, sources=[])
        if source_dir is None:
            self.source_dir = str(Path(__file__).parent.resolve())
        else:
            self.source_dir = os.path.abspath(os.path.expanduser(source_dir))
        self.sources = _get_files(self.source_dir)

    def build_with_command(self, ext_builder):
        """
        Custom `build_ext.build_extension` in `Extension` instead of `Command`.
        `ext_builder` is the instance of `build_ext` command.
        """
        # refer to https://github.com/pybind/cmake_example/blob/master/setup.py
        if ext_builder.compiler.compiler_type == "msvc":
            raise NotImplementedError
        cmake_args = getattr(self, "cmake_args", []) + [
            "-DCMAKE_BUILD_TYPE={}".format("Debug"
                                           if ext_builder.debug else "Release"),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_builder.build_lib),
        ]
        build_args = []

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(ext_builder, "parallel") and ext_builder.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(ext_builder.parallel)]

        if not os.path.exists(ext_builder.build_temp):
            os.makedirs(ext_builder.build_temp)

        # Redirect stdout/stderr to mute, especially when allowing errors
        stdout = getattr(self, "_std_out_handle", None)
        subprocess.check_call(
            ["cmake", self.source_dir] + cmake_args,
            cwd=ext_builder.build_temp,
            stdout=stdout,
            stderr=stdout)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=ext_builder.build_temp,
            stdout=stdout,
            stderr=stdout)

    def get_target_filename(self):
        raise NotImplementedError


class FasterTransformerExtension(CMakeExtension):
    def __init__(self, name, source_dir=None):
        super(FasterTransformerExtension, self).__init__(name, source_dir)
        self.sources = _get_files(
            os.path.
            join(self.source_dir, "faster_transformer", "src")) + _get_files(
                os.path.join(self.source_dir, "patches", "FasterTransformer"))
        self._std_out_handle = None
        # Env variable may not work as expected, since jit compile by `load`
        # would not re-built if source code is not update.
        # self.sm = os.environ.get("PPNLP_GENERATE_CODE", None)

    def build_with_command(self, ext_builder):
        if CUDA_HOME is None:  # GPU only
            # TODO(guosheng): should we touch a dummy file or add a quick exit
            # method to avoid meaningless process in `load`
            logger.warning(
                "FasterTransformer is not available because CUDA can not be found."
            )
            raise NotImplementedError
        # TODO(guosheng): Multiple -std seems be passed in FasterTransformer,
        # which is not allowed by NVCC. Fix it later.
        self.cmake_args = [f"-DPY_CMD={sys.executable}"]
        # `GetCUDAComputeCapability` is not exposed yet, and detect CUDA/GPU
        # version in cmake file.
        # self.cmake_args += [f"-DSM={self.sm}"] if self.sm is not None else []
        self.cmake_args += [f"-DWITH_GPT=ON"]
        try:
            super(FasterTransformerExtension,
                  self).build_with_command(ext_builder)
            # FasterTransformer cmake file resets `CMAKE_LIBRARY_OUTPUT_DIRECTORY`
            # to `CMAKE_BINARY_DIR/lib`, thus copy the lib back to `build_ext.build_lib`.
            # Maybe move this copy to CMakeList.
            # `copy_tree` or `copy_file`, boost lib might be included
            ext_builder.copy_tree(
                os.path.join(ext_builder.build_temp, "lib"),
                ext_builder.build_lib)
        except Exception as e:
            logger.warning(
                "FasterTransformer is not available due to build errors.")
            raise e

    def get_target_filename(self):
        # CMake file has fixed the name of lib, maybe we can copy it as the name
        # returned by `BuildExtension.get_ext_filename` after build.
        return "libdecoding_op.so"


class BuildExtension(PaddleBuildExtension):
    """
    Support both `CppExtention` of Paddle and custom extensions of PaddleNLP.
    """

    def build_extensions(self):
        custom_exts = []  # for
        no_custom_exts = []  # for normal extentions paddle.utils.cpp_extension
        for ext in self.extensions:
            if hasattr(ext, "build_with_command"):
                # custom build in Extension
                ext.build_with_command(self)
                custom_exts.append(ext)
            else:
                no_custom_exts.append(ext)
        if no_custom_exts:
            # Build CppExtentio/CUDAExtension with `PaddleBuildExtension`
            self.extensions = no_custom_exts
            super(BuildExtension, self).build_extensions()
        self.extensions = custom_exts + no_custom_exts


EXTENSIONS = {"FasterTransformer": FasterTransformerExtension}


def get_extension_maker(name):
    # Use `paddle.utils.cpp_extension.CppExtension` as the default
    # TODO(guosheng): Maybe register extension classes into `Extensions`.
    return EXTENSIONS.get(name, CppExtension)


def _write_setup_file(name, file_path, build_dir, **kwargs):
    """
    Automatically generate setup.py and write it into build directory.
    `kwargws` is arguments for the corresponding Extension initialization.
    Any type extension can be jit build.
    """
    template = textwrap.dedent("""
    from setuptools import setup
    from paddlenlp.ops.ext_utils import get_extension_maker, BuildExtension

    setup(
        name='{name}',
        ext_modules=[
            get_extension_maker('{name}')(
                name='{name}',
                {kwargs_str})],
        cmdclass={{'build_ext' : BuildExtension.with_options(
            output_dir=r'{build_dir}')
        }})""").lstrip()
    kwargs_str = ""
    for key, value in kwargs.items():
        kwargs_str += key + "=" + (f"'{value}'"
                                   if isinstance(value, str) else value) + ","
    content = template.format(
        name=name, kwargs_str=kwargs_str, build_dir=build_dir)

    with open(file_path, 'w') as f:
        f.write(content)


def load(name, build_dir=None, force=False, verbose=False, **kwargs):
    # TODO(guosheng): Need better way to resolve unsupported such as CPU. Currently,
    # raise NotImplementedError and skip `_jit_compile`. Otherwise, `_jit_compile`
    # will output the error to stdout (when verbose is True) and raise `RuntimeError`,
    # which is not friendly for users though no other bad effect.
    if CUDA_HOME is None:
        logger.warning("%s is not available because CUDA can not be found." %
                       name)
        raise NotImplementedError
    if name in LOADED_EXT.keys():
        return LOADED_EXT[name]
    if build_dir is None:
        # Maybe under package dir is better to avoid cmake source path conflict
        # with different source path.
        # build_dir = os.path.join(PPNLP_HOME, 'extenstions')
        build_dir = os.path.join(
            str(Path(__file__).parent.resolve()), 'extenstions')
    build_base_dir = os.path.abspath(
        os.path.expanduser(os.path.join(build_dir, name)))
    if not os.path.exists(build_base_dir):
        os.makedirs(build_base_dir)

    extension = get_extension_maker(name)(name, **kwargs)
    # Check if 'target' is out-of-date with respect to any file to avoid rebuild
    if isinstance(extension, CMakeExtension):
        # `CppExtention/CUDAExtension `has version manager by `PaddleBuildExtension`
        # Maybe move this to CMakeExtension later.
        # TODO(guosheng): flags/args changes may also trigger build, and maybe
        # need version manager like `PaddleBuildExtension`.
        ext_filename = extension.get_target_filename()
        ext_filepath = os.path.join(build_base_dir, ext_filename)
        if not force:
            ext_sources = extension.sources
            if os.path.exists(ext_filepath) and not newer_group(
                    ext_sources, ext_filepath, 'newer'):
                logger.debug("skipping '%s' extension (up-to-date) build" %
                             name)
                ops = load_op_meta_info_and_register_op(ext_filepath)
                LOADED_EXT[name] = ops
                return LOADED_EXT[name]

    # write setup file and jit compile
    file_path = os.path.join(build_dir, "{}_setup.py".format(name))
    _write_setup_file(name, file_path, build_base_dir, **kwargs)
    _jit_compile(file_path, verbose)
    if isinstance(extension, CMakeExtension):
        # Load a shared library (if exists) only to register op.
        if os.path.exists(ext_filepath):
            ops = load_op_meta_info_and_register_op(ext_filepath)
            LOADED_EXT[name] = ops
            return LOADED_EXT[name]
    else:
        # Import as callable python api
        return _import_module_from_library(name, build_base_dir, verbose)
