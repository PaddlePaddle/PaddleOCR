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
import six
import copy
import re

import setuptools
from setuptools.command.easy_install import easy_install
from setuptools.command.build_ext import build_ext
from distutils.command.build import build

from .extension_utils import find_cuda_home, find_rocm_home, normalize_extension_kwargs, add_compile_flag, run_cmd
from .extension_utils import is_cuda_file, prepare_unix_cudaflags, prepare_win_cudaflags
from .extension_utils import _import_module_from_library, _write_setup_file, _jit_compile
from .extension_utils import check_abi_compatibility, log_v, CustomOpInfo, parse_op_name_from
from .extension_utils import clean_object_if_change_cflags, _reset_so_rpath, _get_fluid_path
from .extension_utils import bootstrap_context, get_build_directory, add_std_without_repeat

from .extension_utils import IS_WINDOWS, OS_NAME, MSVC_COMPILE_FLAGS, MSVC_COMPILE_FLAGS
from .extension_utils import CLANG_COMPILE_FLAGS, CLANG_LINK_FLAGS

from ...fluid import core

# Note(zhouwei): On windows, it will export function 'PyInit_[name]' by default,
# The solution is: 1.User add function PyInit_[name] 2. set not to export
# refer to https://stackoverflow.com/questions/34689210/error-exporting-symbol-when-building-python-c-extension-in-windows
if IS_WINDOWS and six.PY3:
    from distutils.command.build_ext import build_ext as _du_build_ext
    from unittest.mock import Mock
    _du_build_ext.get_export_symbols = Mock(return_value=None)

CUDA_HOME = find_cuda_home()
if core.is_compiled_with_rocm():
    ROCM_HOME = find_rocm_home()
    CUDA_HOME = ROCM_HOME


def setup(**attr):
    """
    The interface is used to config the process of compiling customized operators,
    mainly includes how to compile shared library, automatically generate python API 
    and install it into site-package. It supports using customized operators directly with
    ``import`` statement.

    It encapsulates the python built-in ``setuptools.setup`` function and keeps arguments
    and usage same as the native interface. Meanwhile, it hiddens Paddle inner framework
    concepts, such as necessary compiling flags, included paths of head files, and linking
    flags. It also will automatically search and valid local environment and versions of 
    ``cc(Linux)`` , ``cl.exe(Windows)`` and ``nvcc`` , then compiles customized operators 
    supporting CPU or GPU device according to the specified Extension type.

    Moreover, `ABI compatibility <https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html>`_ 
    will be checked to ensure that compiler version from ``cc(Linux)`` , ``cl.exe(Windows)``
    on local machine is compatible with pre-installed Paddle whl in python site-packages.

    For Linux, GCC version will be checked . For example if Paddle with CUDA 10.1 is built with GCC 8.2, 
    then the version of user's local machine should satisfy GCC >= 8.2. 
    For Windows, Visual Studio version will be checked, and it should be greater than or equal to that of 
    PaddlePaddle (Visual Studio 2017). 
    If the above conditions are not met, the corresponding warning will be printed, and a fatal error may 
    occur because of ABI compatibility.

    .. note::
        
        1. Currently we support Linux, MacOS and Windows platfrom.
        2. On Linux platform, we recommend to use GCC 8.2 as soft linking condidate of ``/usr/bin/cc`` .
           Then, Use ``which cc`` to ensure location of ``cc`` and using ``cc --version`` to ensure linking 
           GCC version.
        3. On Windows platform, we recommend to install `` Visual Studio`` (>=2017).


    Compared with Just-In-Time ``load`` interface, it only compiles once by executing
    ``python setup.py install`` . Then customized operators API will be available everywhere
    after importing it.

    A simple example of ``setup.py`` as followed: 

    .. code-block:: text

        # setup.py 

        # Case 1: Compiling customized operators supporting CPU and GPU devices
        from paddle.utils.cpp_extension import CUDAExtension, setup

        setup(
            name='custom_op',  # name of package used by "import"
            ext_modules=CUDAExtension(
                sources=['relu_op.cc', 'relu_op.cu', 'tanh_op.cc', 'tanh_op.cu']  # Support for compilation of multiple OPs
            )
        )

        # Case 2: Compiling customized operators supporting only CPU device
        from paddle.utils.cpp_extension import CppExtension, setup

        setup(
            name='custom_op',  # name of package used by "import"
            ext_modules=CppExtension(
                sources=['relu_op.cc', 'tanh_op.cc']  # Support for compilation of multiple OPs
            )
        )


    Applying compilation and installation by executing ``python setup.py install`` under source files directory.
    Then we can use the layer api as followed:

    .. code-block:: text

        import paddle
        from custom_op import relu, tanh

        x = paddle.randn([4, 10], dtype='float32')
        relu_out = relu(x)
        tanh_out = tanh(x)
    

    Args:
        name(str): Specify the name of shared library file and installed python package.
        ext_modules(Extension): Specify the Extension instance including customized operator source files, compiling flags et.al. 
                                If only compile operator supporting CPU device, please use ``CppExtension`` ; If compile operator
                                supporting CPU and GPU devices, please use ``CUDAExtension`` .
        include_dirs(list[str], optional): Specify the extra include directories to search head files. The interface will automatically add
                                 ``site-package/paddle/include`` . Please add the corresponding directory path if including third-party
                                 head files. Default is None.
        extra_compile_args(list[str] | dict, optional): Specify the extra compiling flags such as ``-O3`` . If set ``list[str]`` , all these flags
                                will be applied for ``cc`` and ``nvcc`` compiler. It support specify flags only applied ``cc`` or ``nvcc``
                                compiler using dict type with ``{'cxx': [...], 'nvcc': [...]}`` . Default is None.
        **attr(dict, optional): Specify other arguments same as ``setuptools.setup`` .

    Returns: None

    """
    cmdclass = attr.get('cmdclass', {})
    assert isinstance(cmdclass, dict)
    # if not specific cmdclass in setup, add it automatically.
    if 'build_ext' not in cmdclass:
        cmdclass['build_ext'] = BuildExtension.with_options(
            no_python_abi_suffix=True)
        attr['cmdclass'] = cmdclass

    error_msg = """
    Required to specific `name` argument in paddle.utils.cpp_extension.setup.
    It's used as `import XXX` when you want install and import your custom operators.\n
    For Example:
        # setup.py file
        from paddle.utils.cpp_extension import CUDAExtension, setup
        setup(name='custom_module',
              ext_modules=CUDAExtension(
              sources=['relu_op.cc', 'relu_op.cu'])

        # After running `python setup.py install`
        from custom_module import relu
    """
    # name argument is required
    if 'name' not in attr:
        raise ValueError(error_msg)

    assert not attr['name'].endswith('module'),  \
    "Please don't use 'module' as suffix in `name` argument, "
    "it will be stripped in setuptools.bdist_egg and cause import error."

    ext_modules = attr.get('ext_modules', [])
    if not isinstance(ext_modules, list):
        ext_modules = [ext_modules]
    assert len(
        ext_modules
    ) == 1, "Required only one Extension, but received {}. If you want to compile multi operators, you can include all necessary source files in one Extension.".format(
        len(ext_modules))
    # replace Extension.name with attr['name] to keep consistant with Package name.
    for ext_module in ext_modules:
        ext_module.name = attr['name']

    attr['ext_modules'] = ext_modules

    # Add rename .so hook in easy_install
    assert 'easy_install' not in cmdclass
    cmdclass['easy_install'] = EasyInstallCommand

    # Note(Aurelius84): Add rename build_base directory hook in build command.
    # To avoid using same build directory that will lead to remove the directory
    # by mistake while parallelling execute setup.py, for example on CI.
    assert 'build' not in cmdclass
    build_base = os.path.join('build', attr['name'])
    cmdclass['build'] = BuildCommand.with_options(build_base=build_base)

    # Always set zip_safe=False to make compatible in PY2 and PY3
    # See http://peak.telecommunity.com/DevCenter/setuptools#setting-the-zip-safe-flag
    attr['zip_safe'] = False

    # switch `write_stub` to inject paddle api in .egg
    with bootstrap_context():
        setuptools.setup(**attr)


def CppExtension(sources, *args, **kwargs):
    """
    The interface is used to config source files of customized operators and complies
    Op Kernel only supporting CPU device. Please use ``CUDAExtension`` if you want to
    compile Op Kernel that supports both CPU and GPU devices.

    It further encapsulates python built-in ``setuptools.Extension`` .The arguments and
    usage are same as the native interface, except for no need to explicitly specify
    ``name`` .

    **A simple example:**

    .. code-block:: text

        # setup.py 

        # Compiling customized operators supporting only CPU device
        from paddle.utils.cpp_extension import CppExtension, setup

        setup(
            name='custom_op',
            ext_modules=CppExtension(sources=['relu_op.cc'])
        )


    .. note::
        It is mainly used in ``setup`` and the nama of built shared library keeps same
        as ``name`` argument specified in ``setup`` interface.


    Args:
        sources(list[str]): Specify the C++/CUDA source files of customized operators.
        *args(list[options], optional): Specify other arguments same as ``setuptools.Extension`` .
        **kwargs(dict[option], optional): Specify other arguments same as ``setuptools.Extension`` .

    Returns:
        setuptools.Extension: An instance of ``setuptools.Extension``
    """
    kwargs = normalize_extension_kwargs(kwargs, use_cuda=False)
    # Note(Aurelius84): While using `setup` and `jit`, the Extension `name` will
    # be replaced as `setup.name` to keep consistant with package. Because we allow
    # users can not specific name in Extension.
    # See `paddle.utils.cpp_extension.setup` for details.
    name = kwargs.get('name', None)
    if name is None:
        name = _generate_extension_name(sources)

    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(sources, *args, **kwargs):
    """
    The interface is used to config source files of customized operators and complies
    Op Kernel supporting both CPU and GPU devices. Please use ``CppExtension`` if you want to
    compile Op Kernel that supports only CPU device.

    It further encapsulates python built-in ``setuptools.Extension`` .The arguments and
    usage are same as the native interface, except for no need to explicitly specify
    ``name`` .

    **A simple example:**

    .. code-block:: text

        # setup.py 

        # Compiling customized operators supporting CPU and GPU devices
        from paddle.utils.cpp_extension import CUDAExtension, setup

        setup(
            name='custom_op',
            ext_modules=CUDAExtension(
                sources=['relu_op.cc', 'relu_op.cu']
            )
        )


    .. note::
        It is mainly used in ``setup`` and the nama of built shared library keeps same
        as ``name`` argument specified in ``setup`` interface.


    Args:
        sources(list[str]): Specify the C++/CUDA source files of customized operators.
        *args(list[options], optional): Specify other arguments same as ``setuptools.Extension`` .
        **kwargs(dict[option], optional): Specify other arguments same as ``setuptools.Extension`` .

    Returns:
        setuptools.Extension: An instance of setuptools.Extension
    """
    kwargs = normalize_extension_kwargs(kwargs, use_cuda=True)
    # Note(Aurelius84): While using `setup` and `jit`, the Extension `name` will
    # be replaced as `setup.name` to keep consistant with package. Because we allow
    # users can not specific name in Extension.
    # See `paddle.utils.cpp_extension.setup` for details.
    name = kwargs.get('name', None)
    if name is None:
        name = _generate_extension_name(sources)

    return setuptools.Extension(name, sources, *args, **kwargs)


def _generate_extension_name(sources):
    """
    Generate extension name by source files.
    """
    assert len(sources) > 0, "source files is empty"
    file_prefix = []
    for source in sources:
        source = os.path.basename(source)
        filename, _ = os.path.splitext(source)
        # Use list to generate same order.
        if filename not in file_prefix:
            file_prefix.append(filename)

    return '_'.join(file_prefix)


class BuildExtension(build_ext, object):
    """
    Inherited from setuptools.command.build_ext to customize how to apply
    compilation process with share library.
    """

    @classmethod
    def with_options(cls, **options):
        """
        Returns a BuildExtension subclass containing use-defined options.
        """

        class cls_with_options(cls):
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                cls.__init__(self, *args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs):
        """
        Attributes is initialized with following oreder:

            1. super(self).__init__()
            2. initialize_options(self)
            3. the reset of current __init__()
            4. finalize_options(self)

        So, it is recommended to set attribute value in `finalize_options`.
        """
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", True)
        self.output_dir = kwargs.get("output_dir", None)
        # whether containing cuda source file in Extensions
        self.contain_cuda_file = False

    def initialize_options(self):
        super(BuildExtension, self).initialize_options()

    def finalize_options(self):
        super(BuildExtension, self).finalize_options()
        # NOTE(Aurelius84): Set location of compiled shared library.
        # Carefully to modify this because `setup.py build/install`
        # and `load` interface rely on this attribute.
        if self.output_dir is not None:
            self.build_lib = self.output_dir

    def build_extensions(self):
        if OS_NAME.startswith("darwin"):
            self._valid_clang_compiler()

        self._check_abi()

        # Note(Aurelius84): If already compiling source before, we should check whether
        # cflags have changed and delete the built shared library to re-compile the source
        # even though source file content keep unchanged.
        so_name = self.get_ext_fullpath(self.extensions[0].name)
        clean_object_if_change_cflags(
            os.path.abspath(so_name), self.extensions[0])

        # Consider .cu, .cu.cc as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cu.cc']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def unix_custom_single_compiler(obj, src, ext, cc_args, extra_postargs,
                                        pp_opts):
            """
            Monkey patch machanism to replace inner compiler to custom complie process on Unix platform.
            """
            # use abspath to ensure no warning and don't remove deecopy because modify params
            # with dict type is dangerous.
            src = os.path.abspath(src)
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                # nvcc compile CUDA source
                if is_cuda_file(src):
                    if core.is_compiled_with_rocm():
                        assert ROCM_HOME is not None, "Not found ROCM runtime, \
                            please use `export ROCM_PATH= XXX` to specify it."

                        hipcc_cmd = os.path.join(ROCM_HOME, 'bin', 'hipcc')
                        self.compiler.set_executable('compiler_so', hipcc_cmd)
                        # {'nvcc': {}, 'cxx: {}}
                        if isinstance(cflags, dict):
                            cflags = cflags['hipcc']
                    else:
                        assert CUDA_HOME is not None, "Not found CUDA runtime, \
                            please use `export CUDA_HOME= XXX` to specify it."

                        nvcc_cmd = os.path.join(CUDA_HOME, 'bin', 'nvcc')
                        self.compiler.set_executable('compiler_so', nvcc_cmd)
                        # {'nvcc': {}, 'cxx: {}}
                        if isinstance(cflags, dict):
                            cflags = cflags['nvcc']

                    cflags = prepare_unix_cudaflags(cflags)
                # cxx compile Cpp source
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']

                # NOTE(Aurelius84): Since Paddle 2.0, we require gcc version > 5.x,
                # so we add this flag to ensure the symbol names from user compiled
                # shared library have same ABI suffix with core_(no)avx.so.
                # See https://stackoverflow.com/questions/34571583/understanding-gcc-5s-glibcxx-use-cxx11-abi-or-the-new-abi
                add_compile_flag(cflags, ['-D_GLIBCXX_USE_CXX11_ABI=1'])
                # Append this macor only when jointly compiling .cc with .cu
                if not is_cuda_file(src) and self.contain_cuda_file:
                    cflags.append('-DPADDLE_WITH_CUDA')

                add_std_without_repeat(
                    cflags, self.compiler.compiler_type, use_std14=True)
                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # restore original_compiler
                self.compiler.set_executable('compiler_so', original_compiler)

        def win_custom_single_compiler(sources,
                                       output_dir=None,
                                       macros=None,
                                       include_dirs=None,
                                       debug=0,
                                       extra_preargs=None,
                                       extra_postargs=None,
                                       depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def win_custom_spawn(cmd):
                # Using regex to modify compile options
                compile_options = self.compiler.compile_options
                for i in range(len(cmd)):
                    if re.search('/MD', cmd[i]) is not None:
                        cmd[i] = '/MT'
                    if re.search('/W[1-4]', cmd[i]) is not None:
                        cmd[i] = '/W0'

                # Using regex to match src, obj and include files
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                assert len(src_list) == 1 and len(obj_list) == 1
                src = src_list[0]
                obj = obj_list[0]
                if is_cuda_file(src):
                    assert CUDA_HOME is not None, "Not found CUDA runtime, \
                        please use `export CUDA_HOME= XXX` to specify it."

                    nvcc_cmd = os.path.join(CUDA_HOME, 'bin', 'nvcc')
                    if isinstance(self.cflags, dict):
                        cflags = self.cflags['nvcc']
                    elif isinstance(self.cflags, list):
                        cflags = self.cflags
                    else:
                        cflags = []

                    cflags = prepare_win_cudaflags(cflags) + ['--use-local-env']
                    for flag in MSVC_COMPILE_FLAGS:
                        cflags = ['-Xcompiler', flag] + cflags
                    cmd = [nvcc_cmd, '-c', src, '-o', obj
                           ] + include_list + cflags
                elif isinstance(self.cflags, dict):
                    cflags = MSVC_COMPILE_FLAGS + self.cflags['cxx']
                    cmd += cflags
                elif isinstance(self.cflags, list):
                    cflags = MSVC_COMPILE_FLAGS + self.cflags
                    cmd += cflags
                # Append this macor only when jointly compiling .cc with .cu
                if not is_cuda_file(src) and self.contain_cuda_file:
                    cmd.append('-DPADDLE_WITH_CUDA')

                return original_spawn(cmd)

            try:
                self.compiler.spawn = win_custom_spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        def object_filenames_with_cuda(origina_func, build_directory):
            """
            Decorated the function to add customized naming machanism.
            Originally, both .cc/.cu will have .o object output that will
            bring file override problem. Use .cu.o as CUDA object suffix.
            """

            def wrapper(source_filenames, strip_dir=0, output_dir=''):
                try:
                    objects = origina_func(source_filenames, strip_dir,
                                           output_dir)
                    for i, source in enumerate(source_filenames):
                        # modify xx.o -> xx.cu.o/xx.cu.obj
                        if is_cuda_file(source):
                            old_obj = objects[i]
                            if self.compiler.compiler_type == 'msvc':
                                objects[i] = old_obj[:-3] + 'cu.obj'
                            else:
                                objects[i] = old_obj[:-1] + 'cu.o'
                    # if user set build_directory, output objects there.
                    if build_directory is not None:
                        objects = [
                            os.path.join(build_directory, os.path.basename(obj))
                            for obj in objects
                        ]
                    # ensure to use abspath
                    objects = [os.path.abspath(obj) for obj in objects]
                finally:
                    self.compiler.object_filenames = origina_func

                return objects

            return wrapper

        # customized compile process
        if self.compiler.compiler_type == 'msvc':
            self.compiler.compile = win_custom_single_compiler
        else:
            self.compiler._compile = unix_custom_single_compiler

        self.compiler.object_filenames = object_filenames_with_cuda(
            self.compiler.object_filenames, self.build_lib)
        self._record_op_info()

        print("Compiling user custom op, it will cost a few seconds.....")
        build_ext.build_extensions(self)

        # Reset runtime library path on MacOS platform
        so_path = self.get_ext_fullpath(self.extensions[0]._full_name)
        _reset_so_rpath(so_path)

    def get_ext_filename(self, fullname):
        # for example: custommed_extension.cpython-37m-x86_64-linux-gnu.so
        ext_name = super(BuildExtension, self).get_ext_filename(fullname)
        split_str = '.'
        name_items = ext_name.split(split_str)
        if self.no_python_abi_suffix and six.PY3:
            assert len(
                name_items
            ) > 2, "Expected len(name_items) > 2, but received {}".format(
                len(name_items))
            name_items.pop(-2)
            ext_name = split_str.join(name_items)

        # custommed_extension.dylib
        if OS_NAME.startswith('darwin'):
            name_items[-1] = 'dylib'
            ext_name = split_str.join(name_items)
        return ext_name

    def _valid_clang_compiler(self):
        """
        Make sure to use Clang as compiler on Mac platform
        """
        compiler_infos = ['clang'] + CLANG_COMPILE_FLAGS
        linker_infos = ['clang'] + CLANG_LINK_FLAGS
        self.compiler.set_executables(
            compiler=compiler_infos,
            compiler_so=compiler_infos,
            compiler_cxx=['clang'],
            linker_exe=['clang'],
            linker_so=linker_infos)

    def _check_abi(self):
        """
        Check ABI Compatibility.
        """
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        elif IS_WINDOWS:
            compiler = os.environ.get('CXX', 'cl')
        else:
            compiler = os.environ.get('CXX', 'c++')

        check_abi_compatibility(compiler)
        # Warn user if VC env is activated but `DISTUTILS_USE_SDK` is not set.
        if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and 'DISTUTILS_USE_SDK' not in os.environ:
            msg = (
                'It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.'
                'This may lead to multiple activations of the VC env.'
                'Please run `set DISTUTILS_USE_SDK=1` and try again.')
            raise UserWarning(msg)

    def _record_op_info(self):
        """
        Record custom op information.
        """
        # parse shared library abs path
        outputs = self.get_outputs()
        assert len(outputs) == 1
        # multi operators built into same one .so file
        so_path = os.path.abspath(outputs[0])
        so_name = os.path.basename(so_path)

        for i, extension in enumerate(self.extensions):
            sources = [os.path.abspath(s) for s in extension.sources]
            if not self.contain_cuda_file:
                self.contain_cuda_file = any([is_cuda_file(s) for s in sources])
            op_names = parse_op_name_from(sources)

            for op_name in op_names:
                CustomOpInfo.instance().add(op_name,
                                            so_name=so_name,
                                            so_path=so_path)


class EasyInstallCommand(easy_install, object):
    """
    Extend easy_intall Command to control the behavior of naming shared library
    file.

    NOTE(Aurelius84): This is a hook subclass inherited Command used to rename shared
                    library file after extracting egg-info into site-packages.
    """

    def __init__(self, *args, **kwargs):
        super(EasyInstallCommand, self).__init__(*args, **kwargs)

    # NOTE(Aurelius84): Add args and kwargs to make compatible with PY2/PY3
    def run(self, *args, **kwargs):
        super(EasyInstallCommand, self).run(*args, **kwargs)
        # NOTE: To avoid failing import .so file instead of
        # python file because they have same name, we rename
        # .so shared library to another name.
        for egg_file in self.outputs:
            filename, ext = os.path.splitext(egg_file)
            will_rename = False
            if OS_NAME.startswith('linux') and ext == '.so':
                will_rename = True
            elif OS_NAME.startswith('darwin') and ext == '.dylib':
                will_rename = True
            elif IS_WINDOWS and ext == '.pyd':
                will_rename = True

            if will_rename:
                new_so_path = filename + "_pd_" + ext
                if not os.path.exists(new_so_path):
                    os.rename(r'%s' % egg_file, r'%s' % new_so_path)
                assert os.path.exists(new_so_path)


class BuildCommand(build, object):
    """
    Extend build Command to control the behavior of specifying `build_base` root directory.

    NOTE(Aurelius84): This is a hook subclass inherited Command used to specify customized
                      build_base directory.
    """

    @classmethod
    def with_options(cls, **options):
        """
        Returns a BuildCommand subclass containing use-defined options.
        """

        class cls_with_options(cls):
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                cls.__init__(self, *args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs):
        # Note: shall put before super()
        self._specified_build_base = kwargs.get('build_base', None)

        super(BuildCommand, self).__init__(*args, **kwargs)

    def initialize_options(self):
        """
        build_base is root directory for all sub-command, such as
        build_lib, build_temp. See `distutils.command.build` for details.
        """
        super(BuildCommand, self).initialize_options()
        if self._specified_build_base is not None:
            self.build_base = self._specified_build_base


def load(name,
         sources,
         extra_cxx_cflags=None,
         extra_cuda_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False):
    """
    An Interface to automatically compile C++/CUDA source files Just-In-Time
    and return callable python function as other Paddle layers API. It will
    append user defined custom operators in background while building models.

    It will perform compiling, linking, Python API generation and module loading
    processes under a individual subprocess. It does not require CMake or Ninja 
    environment. On Linux platform, it requires GCC compiler whose version is 
    greater than 5.4 and it should be soft linked to ``/usr/bin/cc`` . On Windows 
    platform, it requires Visual Studio whose version is greater than 2017.
    On MacOS, clang++ is requited. In addition, if compiling Operators supporting 
    GPU device, please make sure ``nvcc`` compiler is installed in local environment.
    
    Moreover, `ABI compatibility <https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html>`_ 
    will be checked to ensure that compiler version from ``cc(Linux)`` , ``cl.exe(Windows)``
    on local machine is compatible with pre-installed Paddle whl in python site-packages.

    For Linux, GCC version will be checked . For example if Paddle with CUDA 10.1 is built with GCC 8.2, 
    then the version of user's local machine should satisfy GCC >= 8.2. 
    For Windows, Visual Studio version will be checked, and it should be greater than or equal to that of 
    PaddlePaddle (Visual Studio 2017). 
    If the above conditions are not met, the corresponding warning will be printed, and a fatal error may 
    occur because of ABI compatibility.

    Compared with ``setup`` interface, it doesn't need extra ``setup.py`` and excute
    ``python setup.py install`` command. The interface contains all compiling and installing
    process underground.

    .. note::

        1. Currently we support Linux, MacOS and Windows platfrom.
        2. On Linux platform, we recommend to use GCC 8.2 as soft linking condidate of ``/usr/bin/cc`` .
           Then, Use ``which cc`` to ensure location of ``cc`` and using ``cc --version`` to ensure linking 
           GCC version.
        3. On Windows platform, we recommend to install `` Visual Studio`` (>=2017).


    **A simple example:**

    .. code-block:: text
    
        import paddle
        from paddle.utils.cpp_extension import load

        custom_op_module = load(
            name="op_shared_libary_name",                # name of shared library
            sources=['relu_op.cc', 'relu_op.cu'],        # source files of customized op
            extra_cxx_cflags=['-g', '-w'],               # optional, specify extra flags to compile .cc/.cpp file
            extra_cuda_cflags=['-O2'],                   # optional, specify extra flags to compile .cu file
            verbose=True                                 # optional, specify to output log information
        )

        x = paddle.randn([4, 10], dtype='float32')
        out = custom_op_module.relu(x)


    Args:
        name(str): Specify the name of generated shared library file name, not including ``.so`` and ``.dll`` suffix.
        sources(list[str]): Specify source files name of customized operators.  Supporting ``.cc`` , ``.cpp`` for CPP file
                            and ``.cu`` for CUDA file.
        extra_cxx_cflags(list[str], optional): Specify additional flags used to compile CPP files. By default
                               all basic and framework related flags have been included.
        extra_cuda_cflags(list[str], optional): Specify additional flags used to compile CUDA files. By default
                               all basic and framework related flags have been included. 
                               See `Cuda Compiler Driver NVCC <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`_
                               for details. Default is None.
        extra_ldflags(list[str], optional): Specify additional flags used to link shared library. See
                                `GCC Link Options <https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html>`_ for details.
                                Default is None.
        extra_include_paths(list[str], optional): Specify additional include path used to search header files. By default
                                all basic headers are included implicitly from ``site-package/paddle/include`` .
                                Default is None.
        build_directory(str, optional): Specify root directory path to put shared library file. If set None,
                            it will use ``PADDLE_EXTENSION_DIR`` from os.environ. Use
                            ``paddle.utils.cpp_extension.get_build_directory()`` to see the location. Default is None.
        verbose(bool, optional): whether to verbose compiled log information. Default is False

    Returns:
        Module: A callable python module contains all CustomOp Layer APIs.

    """

    if build_directory is None:
        build_directory = get_build_directory(verbose)

    # ensure to use abs path
    build_directory = os.path.abspath(build_directory)

    log_v("build_directory: {}".format(build_directory), verbose)

    file_path = os.path.join(build_directory, "{}_setup.py".format(name))
    sources = [os.path.abspath(source) for source in sources]

    if extra_cxx_cflags is None: extra_cxx_cflags = []
    if extra_cuda_cflags is None: extra_cuda_cflags = []
    assert isinstance(
        extra_cxx_cflags, list
    ), "Required type(extra_cxx_cflags) == list[str], but received {}".format(
        extra_cxx_cflags)
    assert isinstance(
        extra_cuda_cflags, list
    ), "Required type(extra_cuda_cflags) == list[str], but received {}".format(
        extra_cuda_cflags)

    log_v("additional extra_cxx_cflags: [{}], extra_cuda_cflags: [{}]".format(
        ' '.join(extra_cxx_cflags), ' '.join(extra_cuda_cflags)), verbose)

    # write setup.py file and compile it
    build_base_dir = os.path.join(build_directory, name)

    _write_setup_file(name, sources, file_path, build_base_dir,
                      extra_include_paths, extra_cxx_cflags, extra_cuda_cflags,
                      extra_ldflags, verbose)
    _jit_compile(file_path, verbose)

    # import as callable python api
    custom_op_api = _import_module_from_library(name, build_base_dir, verbose)

    return custom_op_api
