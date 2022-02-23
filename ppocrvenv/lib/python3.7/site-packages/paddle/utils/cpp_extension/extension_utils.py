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
import re
import sys
import json
import glob
import atexit
import hashlib
import logging
import collections
import textwrap
import warnings
import subprocess
import threading

from importlib import machinery
from contextlib import contextmanager
from setuptools.command import bdist_egg

try:
    from subprocess import DEVNULL  # py3
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

from ...fluid import core
from ...fluid.framework import OpProtoHolder
from ...sysconfig import get_include, get_lib

logger = logging.getLogger("utils.cpp_extension")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

OS_NAME = sys.platform
IS_WINDOWS = OS_NAME.startswith('win')

MSVC_COMPILE_FLAGS = [
    '/MT', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018',
    '/wd4190', '/EHsc', '/w', '/DGOOGLE_GLOG_DLL_DECL',
    '/DBOOST_HAS_STATIC_ASSERT', '/DNDEBUG', '/DPADDLE_USE_DSO'
]
CLANG_COMPILE_FLAGS = [
    '-fno-common', '-dynamic', '-DNDEBUG', '-g', '-fwrapv', '-O3', '-arch',
    'x86_64'
]
CLANG_LINK_FLAGS = [
    '-dynamiclib', '-undefined', 'dynamic_lookup', '-arch', 'x86_64'
]

MSVC_LINK_FLAGS = ['/MACHINE:X64']

COMMON_NVCC_FLAGS = ['-DPADDLE_WITH_CUDA', '-DEIGEN_USE_GPU']

GCC_MINI_VERSION = (5, 4, 0)
MSVC_MINI_VERSION = (19, 0, 24215)
# Give warning if using wrong compiler
WRONG_COMPILER_WARNING = '''
                        *************************************
                        *  Compiler Compatibility WARNING   *
                        *************************************

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Found that your compiler ({user_compiler}) is not compatible with the compiler 
built Paddle for this platform, which is {paddle_compiler} on {platform}. Please
use {paddle_compiler} to compile your custom op. Or you may compile Paddle from
source using {user_compiler}, and then also use it compile your custom op.

See https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html
for help with compiling Paddle from source.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
# Give warning if used compiler version is incompatible
ABI_INCOMPATIBILITY_WARNING = '''
                            **********************************
                            *    ABI Compatibility WARNING   *
                            **********************************

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Found that your compiler ({user_compiler} == {version}) may be ABI-incompatible with pre-installed Paddle!
Please use compiler that is ABI-compatible with GCC >= 5.4 (Recommended 8.2).

See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html for ABI Compatibility
information

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

DEFAULT_OP_ATTR_NAMES = [
    core.op_proto_and_checker_maker.kOpRoleAttrName(),
    core.op_proto_and_checker_maker.kOpRoleVarAttrName(),
    core.op_proto_and_checker_maker.kOpNameScopeAttrName(),
    core.op_proto_and_checker_maker.kOpCreationCallstackAttrName(),
    core.op_proto_and_checker_maker.kOpDeviceAttrName(),
    core.op_proto_and_checker_maker.kOpWithQuantAttrName()
]


@contextmanager
def bootstrap_context():
    """
    Context to manage how to write `__bootstrap__` code in .egg
    """
    origin_write_stub = bdist_egg.write_stub
    bdist_egg.write_stub = custom_write_stub
    yield

    bdist_egg.write_stub = origin_write_stub


def load_op_meta_info_and_register_op(lib_filename):
    core.load_op_meta_info_and_register_op(lib_filename)
    return OpProtoHolder.instance().update_op_proto()


def custom_write_stub(resource, pyfile):
    """
    Customized write_stub function to allow us to inject generated python
    api codes into egg python file.
    """
    _stub_template = textwrap.dedent("""
        import os
        import sys
        import types
        import paddle
        
        def inject_ext_module(module_name, api_names):
            if module_name in sys.modules:
                return sys.modules[module_name]

            new_module = types.ModuleType(module_name)
            for api_name in api_names:
                setattr(new_module, api_name, eval(api_name))

            return new_module

        def __bootstrap__():
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            so_path = os.path.join(cur_dir, "{resource}")

            assert os.path.exists(so_path)

            # load custom op shared library with abs path
            new_custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)
            m = inject_ext_module(__name__, new_custom_ops)
        
        __bootstrap__()

        {custom_api}
        """).lstrip()

    # Parse registerring op information
    _, op_info = CustomOpInfo.instance().last()
    so_path = op_info.so_path

    new_custom_ops = load_op_meta_info_and_register_op(so_path)
    assert len(
        new_custom_ops
    ) > 0, "Required at least one custom operators, but received len(custom_op) =  %d" % len(
        new_custom_ops)

    # NOTE: To avoid importing .so file instead of python file because they have same name,
    # we rename .so shared library to another name, see EasyInstallCommand.
    filename, ext = os.path.splitext(resource)
    resource = filename + "_pd_" + ext

    api_content = []
    for op_name in new_custom_ops:
        api_content.append(_custom_api_content(op_name))

    with open(pyfile, 'w') as f:
        f.write(
            _stub_template.format(
                resource=resource, custom_api='\n\n'.join(api_content)))


OpInfo = collections.namedtuple('OpInfo', ['so_name', 'so_path'])


class CustomOpInfo:
    """
    A global Singleton map to record all compiled custom ops information.
    """

    @classmethod
    def instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        assert not hasattr(
            self.__class__,
            '_instance'), 'Please use `instance()` to get CustomOpInfo object!'
        # NOTE(Aurelius84): Use OrderedDict to save more order information
        self.op_info_map = collections.OrderedDict()

    def add(self, op_name, so_name, so_path=None):
        self.op_info_map[op_name] = OpInfo(so_name, so_path)

    def last(self):
        """
        Return the lastest insert custom op info.
        """
        assert len(self.op_info_map) > 0
        return next(reversed(self.op_info_map.items()))


VersionFields = collections.namedtuple('VersionFields', [
    'sources',
    'extra_compile_args',
    'extra_link_args',
    'library_dirs',
    'runtime_library_dirs',
    'include_dirs',
    'define_macros',
    'undef_macros',
])


class VersionManager:
    def __init__(self, version_field):
        self.version_field = version_field
        self.version = self.hasher(version_field)

    def hasher(self, version_field):
        from paddle.fluid.layers.utils import flatten

        md5 = hashlib.md5()
        for field in version_field._fields:
            elem = getattr(version_field, field)
            if not elem: continue
            if isinstance(elem, (list, tuple, dict)):
                flat_elem = flatten(elem)
                md5 = combine_hash(md5, tuple(flat_elem))
            else:
                raise RuntimeError(
                    "Support types with list, tuple and dict, but received {} with {}.".
                    format(type(elem), elem))

        return md5.hexdigest()

    @property
    def details(self):
        return self.version_field._asdict()


def combine_hash(md5, value):
    """
    Return new hash value.
    DO NOT use `hash()` because it doesn't generate stable value between different process.
    See https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
    """
    md5.update(repr(value).encode())
    return md5


def clean_object_if_change_cflags(so_path, extension):
    """
    If already compiling source before, we should check whether cflags 
    have changed and delete the built object to re-compile the source
    even though source file content keeps unchanaged.
    """

    def serialize(path, version_info):
        assert isinstance(version_info, dict)
        with open(path, 'w') as f:
            f.write(json.dumps(version_info, indent=4, sort_keys=True))

    def deserialize(path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            content = f.read()
            return json.loads(content)

    # version file
    VERSION_FILE = "version.txt"
    base_dir = os.path.dirname(so_path)
    so_name = os.path.basename(so_path)
    version_file = os.path.join(base_dir, VERSION_FILE)

    # version info
    args = [getattr(extension, field, None) for field in VersionFields._fields]
    version_field = VersionFields._make(args)
    versioner = VersionManager(version_field)

    if os.path.exists(so_path) and os.path.exists(version_file):
        old_version_info = deserialize(version_file)
        so_version = old_version_info.get(so_name, None)
        # delete shared library file if version is changed to re-compile it.
        if so_version is not None and so_version != versioner.version:
            log_v(
                "Re-Compiling {}, because specified cflags have been changed. New signature {} has been saved into {}.".
                format(so_name, versioner.version, version_file))
            os.remove(so_path)
            # update new version information
            new_version_info = versioner.details
            new_version_info[so_name] = versioner.version
            serialize(version_file, new_version_info)
    else:
        # If compile at first time, save compiling detail information for debug.
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        details = versioner.details
        details[so_name] = versioner.version
        serialize(version_file, details)


def prepare_unix_cudaflags(cflags):
    """
    Prepare all necessary compiled flags for nvcc compiling CUDA files.
    """
    cflags = COMMON_NVCC_FLAGS + [
        '-ccbin', 'cc', '-Xcompiler', '-fPIC', '--expt-relaxed-constexpr',
        '-DNVCC'
    ] + cflags + get_cuda_arch_flags(cflags)

    return cflags


def prepare_win_cudaflags(cflags):
    """
    Prepare all necessary compiled flags for nvcc compiling CUDA files.
    """
    cflags = COMMON_NVCC_FLAGS + ['-w'] + cflags + get_cuda_arch_flags(cflags)

    return cflags


def add_std_without_repeat(cflags, compiler_type, use_std14=False):
    """
    Append -std=c++11/14 in cflags if without specific it before.
    """
    cpp_flag_prefix = '/std:' if compiler_type == 'msvc' else '-std='
    if not any(cpp_flag_prefix in flag for flag in cflags):
        suffix = 'c++14' if use_std14 else 'c++11'
        cpp_flag = cpp_flag_prefix + suffix
        cflags.append(cpp_flag)


def get_cuda_arch_flags(cflags):
    """
    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.
    """
    # TODO(Aurelius84):
    return []


def _get_fluid_path():
    """
    Return installed fluid dir path.
    """
    import paddle
    return os.path.join(os.path.dirname(paddle.__file__), 'fluid')


def _get_core_name():
    """
    Return pybind DSO module name.
    """
    import paddle
    ext_name = '.pyd' if IS_WINDOWS else '.so'
    if not paddle.fluid.core.load_noavx:
        return 'core_avx' + ext_name
    else:
        return 'core_noavx' + ext_name


def _get_lib_core_path():
    """
    Return real path of libcore_(no)avx.dylib on MacOS.
    """
    raw_core_name = _get_core_name()
    lib_core_name = "lib{}.dylib".format(raw_core_name[:-3])
    return os.path.join(_get_fluid_path(), lib_core_name)


def _get_dll_core_path():
    """
    Return real path of libcore_(no)avx.dylib on Windows.
    """
    raw_core_name = _get_core_name()
    dll_core_name = "paddle_pybind.dll"
    return os.path.join(_get_fluid_path(), dll_core_name)


def _reset_so_rpath(so_path):
    """
    NOTE(Aurelius84): Runtime path of core_(no)avx.so is modified into `@loader_path/../libs`
    in setup.py.in. While loading custom op, `@loader_path` is the dirname of custom op
    instead of `paddle/fluid`. So we modify `@loader_path` from custom dylib into `@rpath`
    to ensure dynamic loader find it correctly.

    Moreover, we will add `-rpath site-packages/paddle/fluid` while linking the dylib so
    that we don't need to set `LD_LIBRARY_PATH` any more.
    """
    assert os.path.exists(so_path)
    if OS_NAME.startswith("darwin"):
        origin_runtime_path = "@loader_path/../libs/"
        rpath = "@rpath/{}".format(_get_core_name())
        cmd = 'install_name_tool -change {} {} {}'.format(origin_runtime_path,
                                                          rpath, so_path)

        run_cmd(cmd)


def normalize_extension_kwargs(kwargs, use_cuda=False):
    """
    Normalize include_dirs, library_dir and other attributes in kwargs.
    """
    assert isinstance(kwargs, dict)
    # append necessary include dir path of paddle
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs.extend(find_paddle_includes(use_cuda))

    kwargs['include_dirs'] = include_dirs

    # append necessary lib path of paddle
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs.extend(find_paddle_libraries(use_cuda))
    kwargs['library_dirs'] = library_dirs

    # append compile flags and check settings of compiler
    extra_compile_args = kwargs.get('extra_compile_args', [])
    if isinstance(extra_compile_args, dict):
        for compiler in ['cxx', 'nvcc']:
            if compiler not in extra_compile_args:
                extra_compile_args[compiler] = []

    if IS_WINDOWS:
        # TODO(zhouwei): may append compile flags in future
        pass
        # append link flags
        extra_link_args = kwargs.get('extra_link_args', [])
        extra_link_args.extend(MSVC_LINK_FLAGS)
        lib_core_name = create_sym_link_if_not_exist()
        extra_link_args.append('{}'.format(lib_core_name))
        if use_cuda:
            extra_link_args.extend(['cudadevrt.lib', 'cudart_static.lib'])
        kwargs['extra_link_args'] = extra_link_args

    else:
        ########################### Linux Platform ###########################
        extra_link_args = kwargs.get('extra_link_args', [])
        # On Linux, GCC support '-l:xxx.so' to specify the library name
        # without `lib` prefix.
        if OS_NAME.startswith('linux'):
            extra_link_args.append('-l:{}'.format(_get_core_name()))
        ########################### MacOS Platform ###########################
        else:
            # See _reset_so_rpath for details.
            extra_link_args.append('-Wl,-rpath,{}'.format(_get_fluid_path()))
            # On MacOS, ld don't support `-l:xx`, so we create a
            # libcore_avx.dylib symbol link.
            lib_core_name = create_sym_link_if_not_exist()
            extra_link_args.append('-l{}'.format(lib_core_name))
        ###########################   -- END --    ###########################

        add_compile_flag(extra_compile_args, ['-w'])  # disable warning

        if use_cuda:
            extra_link_args.append('-lcudart')

        kwargs['extra_link_args'] = extra_link_args

        # add runtime library dirs
        runtime_library_dirs = kwargs.get('runtime_library_dirs', [])
        runtime_library_dirs.extend(find_paddle_libraries(use_cuda))
        kwargs['runtime_library_dirs'] = runtime_library_dirs

    kwargs['extra_compile_args'] = extra_compile_args

    kwargs['language'] = 'c++'
    return kwargs


def create_sym_link_if_not_exist():
    """
    Create soft symbol link of `core_avx.so` or `core_noavx.so`
    """
    assert OS_NAME.startswith('darwin') or IS_WINDOWS

    raw_core_name = _get_core_name()
    core_path = os.path.join(_get_fluid_path(), raw_core_name)
    if IS_WINDOWS:
        new_dll_core_path = _get_dll_core_path()
        # create symbol link on windows
        if not os.path.exists(new_dll_core_path):
            try:
                os.symlink(core_path, new_dll_core_path)
            except Exception:
                warnings.warn(
                    "Failed to create soft symbol link for {}.\n You can run prompt as administrator and execute the "
                    "following command manually: `mklink {} {}`. Now it will create hard link for {} trickly.".
                    format(raw_core_name, new_dll_core_path, core_path,
                           raw_core_name))
                run_cmd('mklink /H {} {}'.format(new_dll_core_path, core_path))
        # core_avx or core_noavx with lib suffix
        assert os.path.exists(new_dll_core_path)
        return raw_core_name[:-4] + ".lib"

    else:
        new_lib_core_path = _get_lib_core_path()
        # create symbol link on mac
        if not os.path.exists(new_lib_core_path):
            try:
                os.symlink(core_path, new_lib_core_path)
                assert os.path.exists(new_lib_core_path)
            except Exception:
                raise RuntimeError(
                    "Failed to create soft symbol link for {}.\n Please execute the following command manually: `ln -s {} {}`".
                    format(raw_core_name, core_path, new_lib_core_path))

        # core_avx or core_noavx without suffix
        return raw_core_name[:-3]


def find_cuda_home():
    """
    Use heuristic method to find cuda path
    """
    # step 1. find in $CUDA_HOME or $CUDA_PATH
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')

    # step 2.  find path by `which nvcc`
    if cuda_home is None:
        which_cmd = 'where' if IS_WINDOWS else 'which'
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc_path = subprocess.check_output(
                    [which_cmd, 'nvcc'], stderr=devnull)
                nvcc_path = nvcc_path.decode()
                # Multi CUDA, select the first
                nvcc_path = nvcc_path.split('\r\n')[0]

                # for example: /usr/local/cuda/bin/nvcc
                cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        except:
            if IS_WINDOWS:
                # search from default NVIDIA GPU path
                candidate_paths = glob.glob(
                    'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*.*'
                )
                if len(candidate_paths) > 0:
                    cuda_home = candidate_paths[0]
            else:
                cuda_home = "/usr/local/cuda"
    # step 3. check whether path is valid
    if cuda_home and not os.path.exists(
            cuda_home) and core.is_compiled_with_cuda():
        cuda_home = None

    return cuda_home


def find_rocm_home():
    """
    Use heuristic method to find rocm path
    """
    # step 1. find in $ROCM_HOME or $ROCM_PATH
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')

    # step 2.  find path by `which nvcc`
    if rocm_home is None:
        which_cmd = 'where' if IS_WINDOWS else 'which'
        try:
            with open(os.devnull, 'w') as devnull:
                hipcc_path = subprocess.check_output(
                    [which_cmd, 'hipcc'], stderr=devnull)
                hipcc_path = hipcc_path.decode()
                hipcc_path = hipcc_path.rstrip('\r\n')

                # for example: /opt/rocm/bin/hipcc
                rocm_home = os.path.dirname(os.path.dirname(hipcc_path))
        except:
            rocm_home = "/opt/rocm"
    # step 3. check whether path is valid
    if rocm_home and not os.path.exists(
            rocm_home) and core.is_compiled_with_rocm():
        rocm_home = None

    return rocm_home


def find_cuda_includes():
    """
    Use heuristic method to find cuda include path
    """
    cuda_home = find_cuda_home()
    if cuda_home is None:
        raise ValueError(
            "Not found CUDA runtime, please use `export CUDA_HOME=XXX` to specific it."
        )

    return [os.path.join(cuda_home, 'include')]


def find_rocm_includes():
    """
    Use heuristic method to find rocm include path
    """
    rocm_home = find_rocm_home()
    if rocm_home is None:
        raise ValueError(
            "Not found ROCM runtime, please use `export ROCM_PATH= XXX` to specific it."
        )

    return [os.path.join(rocm_home, 'include')]


def find_paddle_includes(use_cuda=False):
    """
    Return Paddle necessary include dir path.
    """
    # pythonXX/site-packages/paddle/include
    paddle_include_dir = get_include()
    third_party_dir = os.path.join(paddle_include_dir, 'third_party')
    include_dirs = [paddle_include_dir, third_party_dir]

    if use_cuda:
        if core.is_compiled_with_rocm():
            rocm_include_dir = find_rocm_includes()
            include_dirs.extend(rocm_include_dir)
        else:
            cuda_include_dir = find_cuda_includes()
            include_dirs.extend(cuda_include_dir)

    if OS_NAME.startswith('darwin'):
        # NOTE(Aurelius84): Ensure to find std v1 headers correctly.
        std_v1_includes = find_clang_cpp_include()
        if std_v1_includes is not None and os.path.exists(std_v1_includes):
            include_dirs.append(std_v1_includes)

    return include_dirs


def find_clang_cpp_include(compiler='clang'):
    std_v1_includes = None
    try:
        compiler_version = subprocess.check_output([compiler, "--version"])
        compiler_version = compiler_version.decode()
        infos = compiler_version.split("\n")
        for info in infos:
            if "InstalledDir" in info:
                v1_path = info.split(':')[-1].strip()
                if v1_path and os.path.exists(v1_path):
                    std_v1_includes = os.path.join(
                        os.path.dirname(v1_path), 'include/c++/v1')
    except Exception:
        # Just raise warnings because the include dir is not required.
        warnings.warn(
            "Failed to search `include/c++/v1/` include dirs. Don't worry because it's not required."
        )
    return std_v1_includes


def find_cuda_libraries():
    """
    Use heuristic method to find cuda static lib path
    """
    cuda_home = find_cuda_home()
    if cuda_home is None:
        raise ValueError(
            "Not found CUDA runtime, please use `export CUDA_HOME=XXX` to specific it."
        )
    if IS_WINDOWS:
        cuda_lib_dir = [os.path.join(cuda_home, 'lib', 'x64')]
    else:
        cuda_lib_dir = [os.path.join(cuda_home, 'lib64')]

    return cuda_lib_dir


def find_rocm_libraries():
    """
    Use heuristic method to find rocm dynamic lib path
    """
    rocm_home = find_rocm_home()
    if rocm_home is None:
        raise ValueError(
            "Not found ROCM runtime, please use `export ROCM_PATH=XXX` to specific it."
        )
    rocm_lib_dir = [os.path.join(rocm_home, 'lib')]

    return rocm_lib_dir


def find_paddle_libraries(use_cuda=False):
    """
    Return Paddle necessary library dir path.
    """
    # pythonXX/site-packages/paddle/libs
    paddle_lib_dirs = [get_lib()]

    if use_cuda:
        if core.is_compiled_with_rocm():
            rocm_lib_dir = find_rocm_libraries()
            paddle_lib_dirs.extend(rocm_lib_dir)
        else:
            cuda_lib_dir = find_cuda_libraries()
            paddle_lib_dirs.extend(cuda_lib_dir)

    # add `paddle/fluid` to search `core_avx.so` or `core_noavx.so`
    paddle_lib_dirs.append(_get_fluid_path())

    return paddle_lib_dirs


def add_compile_flag(extra_compile_args, flags):
    assert isinstance(flags, list)
    if isinstance(extra_compile_args, dict):
        for args in extra_compile_args.values():
            args.extend(flags)
    else:
        extra_compile_args.extend(flags)


def is_cuda_file(path):

    cuda_suffix = set(['.cu'])
    items = os.path.splitext(path)
    assert len(items) > 1
    return items[-1] in cuda_suffix


def get_build_directory(verbose=False):
    """
    Return paddle extension root directory to put shared library. It could be specified by
    ``export PADDLE_EXTENSION_DIR=XXX`` . If not set, ``~/.cache/paddle_extension`` will be used
    by default.

    Returns:
        The root directory of compiling customized operators.

    Examples:

    .. code-block:: python

        from paddle.utils.cpp_extension import get_build_directory

        build_dir = get_build_directory()
        print(build_dir)

    """
    root_extensions_directory = os.environ.get('PADDLE_EXTENSION_DIR')
    if root_extensions_directory is None:
        dir_name = "paddle_extensions"
        root_extensions_directory = os.path.join(
            os.path.expanduser('~/.cache'), dir_name)
        if IS_WINDOWS:
            root_extensions_directory = os.path.normpath(
                root_extensions_directory)

        log_v("$PADDLE_EXTENSION_DIR is not set, using path: {} by default.".
              format(root_extensions_directory), verbose)

    if not os.path.exists(root_extensions_directory):
        os.makedirs(root_extensions_directory)

    return root_extensions_directory


def parse_op_info(op_name):
    """
    Parse input names and outpus detail information from registered custom op
    from OpInfoMap.
    """
    if op_name not in OpProtoHolder.instance().op_proto_map:
        raise ValueError(
            "Please load {} shared library file firstly by `paddle.utils.cpp_extension.load_op_meta_info_and_register_op(...)`".
            format(op_name))
    op_proto = OpProtoHolder.instance().get_op_proto(op_name)

    in_names = [x.name for x in op_proto.inputs]
    out_names = [x.name for x in op_proto.outputs]
    attr_names = [
        x.name for x in op_proto.attrs if x.name not in DEFAULT_OP_ATTR_NAMES
    ]

    return in_names, out_names, attr_names


def _import_module_from_library(module_name, build_directory, verbose=False):
    """
    Load shared library and import it as callable python module.
    """
    if IS_WINDOWS:
        dynamic_suffix = '.pyd'
    elif OS_NAME.startswith('darwin'):
        dynamic_suffix = '.dylib'
    else:
        dynamic_suffix = '.so'
    ext_path = os.path.join(build_directory, module_name + dynamic_suffix)
    if not os.path.exists(ext_path):
        raise FileNotFoundError("Extension path: {} does not exist.".format(
            ext_path))

    # load custom op_info and kernels from .so shared library
    log_v('loading shared library from: {}'.format(ext_path), verbose)
    op_names = load_op_meta_info_and_register_op(ext_path)

    # generate Python api in ext_path
    return _generate_python_module(module_name, op_names, build_directory,
                                   verbose)


def _generate_python_module(module_name,
                            op_names,
                            build_directory,
                            verbose=False):
    """
    Automatically generate python file to allow import or load into as module
    """

    def remove_if_exit(filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

    # NOTE: Use unique id as suffix to avoid write same file at same time in
    # both multi-thread and multi-process.
    thread_id = str(threading.currentThread().ident)
    api_file = os.path.join(build_directory,
                            module_name + '_' + thread_id + '.py')
    log_v("generate api file: {}".format(api_file), verbose)

    # delete the temp file before exit python process    
    atexit.register(lambda: remove_if_exit(api_file))

    # write into .py file with RWLock
    api_content = [_custom_api_content(op_name) for op_name in op_names]
    with open(api_file, 'w') as f:
        f.write('\n\n'.join(api_content))

    # load module
    custom_module = _load_module_from_file(api_file, module_name, verbose)
    return custom_module


def _custom_api_content(op_name):
    params_str, ins_str, attrs_str, outs_str = _get_api_inputs_str(op_name)

    API_TEMPLATE = textwrap.dedent("""
        from paddle.fluid.core import VarBase
        from paddle.fluid.framework import in_dygraph_mode, _dygraph_tracer
        from paddle.fluid.layer_helper import LayerHelper

        def {op_name}({inputs}):
            # prepare inputs and outputs
            ins = {ins}
            attrs = {attrs}
            outs = {{}}
            out_names = {out_names}

            # The output variable's dtype use default value 'float32',
            # and the actual dtype of output variable will be inferred in runtime.
            if in_dygraph_mode():
                for out_name in out_names:
                    outs[out_name] = VarBase()
                _dygraph_tracer().trace_op(type="{op_name}", inputs=ins, outputs=outs, attrs=attrs)
            else:
                helper = LayerHelper("{op_name}", **locals())
                for out_name in out_names:
                    outs[out_name] = helper.create_variable(dtype='float32')

                helper.append_op(type="{op_name}", inputs=ins, outputs=outs, attrs=attrs)

            res = [outs[out_name] for out_name in out_names]

            return res[0] if len(res)==1 else res
            """).lstrip()

    # generate python api file
    api_content = API_TEMPLATE.format(
        op_name=op_name,
        inputs=params_str,
        ins=ins_str,
        attrs=attrs_str,
        out_names=outs_str)

    return api_content


def _load_module_from_file(api_file_path, module_name, verbose=False):
    """
    Load module from python file.
    """
    if not os.path.exists(api_file_path):
        raise FileNotFoundError("File : {} does not exist.".format(
            api_file_path))

    # Unique readable module name to place custom api.
    log_v('import module from file: {}'.format(api_file_path), verbose)
    ext_name = "_paddle_cpp_extension_" + module_name

    # load module with RWLock
    loader = machinery.SourceFileLoader(ext_name, api_file_path)
    module = loader.load_module()

    return module


def _get_api_inputs_str(op_name):
    """
    Returns string of api parameters and inputs dict.
    """
    in_names, out_names, attr_names = parse_op_info(op_name)
    # e.g: x, y, z
    param_names = in_names + attr_names
    # NOTE(chenweihang): we add suffix `@VECTOR` for std::vector<Tensor> input,
    # but the string contains `@` cannot used as argument name, so we split
    # input name by `@`, and only use first substr as argument
    params_str = ','.join([p.split("@")[0].lower() for p in param_names])
    # e.g: {'X': x, 'Y': y, 'Z': z}
    ins_str = "{%s}" % ','.join([
        "'{}' : {}".format(in_name, in_name.split("@")[0].lower())
        for in_name in in_names
    ])
    # e.g: {'num': n}
    attrs_str = "{%s}" % ",".join([
        "'{}' : {}".format(attr_name, attr_name.split("@")[0].lower())
        for attr_name in attr_names
    ])
    # e.g: ['Out', 'Index']
    outs_str = "[%s]" % ','.join(["'{}'".format(name) for name in out_names])
    return params_str, ins_str, attrs_str, outs_str


def _write_setup_file(name,
                      sources,
                      file_path,
                      build_dir,
                      include_dirs,
                      extra_cxx_cflags,
                      extra_cuda_cflags,
                      link_args,
                      verbose=False):
    """
    Automatically generate setup.py and write it into build directory.
    """
    template = textwrap.dedent("""
    import os
    from paddle.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, setup
    from paddle.utils.cpp_extension import get_build_directory


    setup(
        name='{name}',
        ext_modules=[
            {prefix}Extension(
                sources={sources},
                include_dirs={include_dirs},
                extra_compile_args={{'cxx':{extra_cxx_cflags}, 'nvcc':{extra_cuda_cflags}}},
                extra_link_args={extra_link_args})],
        cmdclass={{"build_ext" : BuildExtension.with_options(
            output_dir=r'{build_dir}',
            no_python_abi_suffix=True)
        }})""").lstrip()

    with_cuda = False
    if any([is_cuda_file(source) for source in sources]):
        with_cuda = True
    log_v("with_cuda: {}".format(with_cuda), verbose)

    content = template.format(
        name=name,
        prefix='CUDA' if with_cuda else 'Cpp',
        sources=list2str(sources),
        include_dirs=list2str(include_dirs),
        extra_cxx_cflags=list2str(extra_cxx_cflags),
        extra_cuda_cflags=list2str(extra_cuda_cflags),
        extra_link_args=list2str(link_args),
        build_dir=build_dir)

    log_v('write setup.py into {}'.format(file_path), verbose)
    with open(file_path, 'w') as f:
        f.write(content)


def list2str(args):
    """
    Convert list[str] into string. For example: ['x', 'y'] -> "['x', 'y']"
    """
    if args is None: return '[]'
    assert isinstance(args, (list, tuple))
    args = ["{}".format(arg) for arg in args]
    return repr(args)


def _jit_compile(file_path, verbose=False):
    """
    Build shared library in subprocess
    """
    ext_dir = os.path.dirname(file_path)
    setup_file = os.path.basename(file_path)

    # Using interpreter same with current process.
    interpreter = sys.executable

    try:
        py_version = subprocess.check_output([interpreter, '-V'])
        py_version = py_version.decode()
        log_v("Using Python interpreter: {}, version: {}".format(
            interpreter, py_version.strip()), verbose)
    except Exception:
        _, error, _ = sys.exc_info()
        raise RuntimeError(
            'Failed to check Python interpreter with `{}`, errors: {}'.format(
                interpreter, error))

    if IS_WINDOWS:
        compile_cmd = 'cd /d {} && {} {} build'.format(ext_dir, interpreter,
                                                       setup_file)
    else:
        compile_cmd = 'cd {} && {} {} build'.format(ext_dir, interpreter,
                                                    setup_file)

    print("Compiling user custom op, it will cost a few seconds.....")
    run_cmd(compile_cmd, verbose)


def parse_op_name_from(sources):
    """
    Parse registerring custom op name from sources.
    """

    def regex(content):
        pattern = re.compile(r'PD_BUILD_OP\(([^,\)]+)\)')
        content = re.sub(r'\s|\t|\n', '', content)
        op_name = pattern.findall(content)
        op_name = set([re.sub('_grad', '', name) for name in op_name])

        return op_name

    op_names = set()
    for source in sources:
        with open(source, 'r') as f:
            content = f.read()
            op_names |= regex(content)

    return list(op_names)


def run_cmd(command, verbose=False):
    """
    Execute command with subprocess.
    """
    # logging
    log_v("execute command: {}".format(command), verbose)

    # execute command
    try:
        if verbose:
            return subprocess.check_call(
                command, shell=True, stderr=subprocess.STDOUT)
        else:
            return subprocess.check_call(command, shell=True, stdout=DEVNULL)
    except Exception:
        _, error, _ = sys.exc_info()
        raise RuntimeError("Failed to run command: {}, errors: {}".format(
            compile, error))


def check_abi_compatibility(compiler, verbose=False):
    """
    Check whether GCC version on user local machine is compatible with Paddle in
    site-packages.
    """
    if os.environ.get('PADDLE_SKIP_CHECK_ABI') in ['True', 'true', '1']:
        return True

    if not IS_WINDOWS:
        cmd_out = subprocess.check_output(
            ['which', compiler], stderr=subprocess.STDOUT)
        compiler_path = os.path.realpath(cmd_out.decode()).strip()
        # if not found any suitable compiler, raise warning
        if not any(name in compiler_path
                   for name in _expected_compiler_current_platform()):
            warnings.warn(
                WRONG_COMPILER_WARNING.format(
                    user_compiler=compiler,
                    paddle_compiler=_expected_compiler_current_platform()[0],
                    platform=OS_NAME))
            return False

    version = (0, 0, 0)
    # clang++ have no ABI compatibility problem
    if OS_NAME.startswith('darwin'):
        return True
    try:
        if OS_NAME.startswith('linux'):
            mini_required_version = GCC_MINI_VERSION
            version_info = subprocess.check_output(
                [compiler, '-dumpfullversion', '-dumpversion'])
            version_info = version_info.decode()
            version = version_info.strip().split('.')
        elif IS_WINDOWS:
            mini_required_version = MSVC_MINI_VERSION
            compiler_info = subprocess.check_output(
                compiler, stderr=subprocess.STDOUT)
            try:
                compiler_info = compiler_info.decode('UTF-8')
            except UnicodeDecodeError:
                compiler_info = compiler_info.decode('gbk')
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.strip())
            if match is not None:
                version = match.groups()
    except Exception:
        # check compiler version failed
        _, error, _ = sys.exc_info()
        warnings.warn('Failed to check compiler version for {}: {}'.format(
            compiler, error))
        return False

    # check version compatibility
    assert len(version) == 3
    if tuple(map(int, version)) >= mini_required_version:
        return True
    warnings.warn(
        ABI_INCOMPATIBILITY_WARNING.format(
            user_compiler=compiler, version='.'.join(version)))
    return False


def _expected_compiler_current_platform():
    """
    Returns supported compiler string on current platform
    """
    if OS_NAME.startswith('darwin'):
        expect_compilers = ['clang', 'clang++']
    elif OS_NAME.startswith('linux'):
        expect_compilers = ['gcc', 'g++', 'gnu-c++', 'gnu-cc']
    elif IS_WINDOWS:
        expect_compilers = ['cl']
    return expect_compilers


def log_v(info, verbose=True):
    """
    Print log information on stdout.
    """
    if verbose:
        logger.info(info)
