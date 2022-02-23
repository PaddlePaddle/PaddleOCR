"""
Helpers for detection of compiler features
"""
import tempfile
import os
import sys
from numpy.distutils.system_info import dict_append

def try_compile(compiler, code=None, flags=[], ext=None):
    """Returns True if the compiler is able to compile the given code"""
    from distutils.errors import CompileError
    from numpy.distutils.fcompiler import FCompiler

    if code is None:
        if isinstance(compiler, FCompiler):
            code = "      program main\n      return\n      end"
        else:
            code = 'int main (int argc, char **argv) { return 0; }'

    ext = ext or compiler.src_extensions[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        fname = os.path.join(temp_dir, 'main'+ext)
        with open(fname, 'w') as f:
            f.write(code)

        try:
            compiler.compile([fname], output_dir=temp_dir, extra_postargs=flags)
        except CompileError:
            return False
    return True


def has_flag(compiler, flag, ext=None):
    """Returns True if the compiler supports the given flag"""
    return try_compile(compiler, flags=[flag], ext=ext)


def get_cxx_std_flag(compiler):
    """Detects compiler flag for c++14, c++11, or None if not detected"""
    # GNU C compiler documentation uses single dash:
    #    https://gcc.gnu.org/onlinedocs/gcc/Standards.html
    # but silently understands two dashes, like --std=c++11 too.
    # Other GCC compatible compilers, like Intel C Compiler on Linux do not.
    gnu_flags = ['-std=c++14', '-std=c++11']
    flags_by_cc = {
        'msvc': ['/std:c++14', None],
        'intelw': ['/Qstd=c++14', '/Qstd=c++11'],
        'intelem': ['-std=c++14', '-std=c++11']
    }
    flags = flags_by_cc.get(compiler.compiler_type, gnu_flags)

    for flag in flags:
        if flag is None:
            return None

        if has_flag(compiler, flag, ext='.cpp'):
            return flag

    from numpy.distutils import log
    log.warn('Could not detect c++ standard flag')
    return None


def get_c_std_flag(compiler):
    """Detects compiler flag to enable C99"""
    gnu_flag = '-std=c99'
    flag_by_cc = {
        'msvc': None,
        'intelw': '/Qstd=c99',
        'intelem': '-std=c99'
    }
    flag = flag_by_cc.get(compiler.compiler_type, gnu_flag)

    if flag is None:
        return None

    if has_flag(compiler, flag, ext='.c'):
        return flag

    from numpy.distutils import log
    log.warn('Could not detect c99 standard flag')
    return None


def try_add_flag(args, compiler, flag, ext=None):
    """Appends flag to the list of arguments if supported by the compiler"""
    if try_compile(compiler, flags=args+[flag], ext=ext):
        args.append(flag)


def set_c_flags_hook(build_ext, ext):
    """Sets basic compiler flags for compiling C99 code"""
    std_flag = get_c_std_flag(build_ext.compiler)
    if std_flag is not None:
        ext.extra_compile_args.append(std_flag)


def set_cxx_flags_hook(build_ext, ext):
    """Sets basic compiler flags for compiling C++11 code"""
    cc = build_ext._cxx_compiler
    args = ext.extra_compile_args

    std_flag = get_cxx_std_flag(cc)
    if std_flag is not None:
        args.append(std_flag)

    if sys.platform == 'darwin':
        # Set min macOS version
        min_macos_flag = '-mmacosx-version-min=10.9'
        if has_flag(cc, min_macos_flag):
            args.append(min_macos_flag)
            ext.extra_link_args.append(min_macos_flag)


def set_cxx_flags_clib_hook(build_clib, build_info):
    cc = build_clib.compiler
    new_args = []
    new_link_args = []

    std_flag = get_cxx_std_flag(cc)
    if std_flag is not None:
        new_args.append(std_flag)

    if sys.platform == 'darwin':
        # Set min macOS version
        min_macos_flag = '-mmacosx-version-min=10.9'
        if has_flag(cc, min_macos_flag):
            new_args.append(min_macos_flag)
            new_link_args.append(min_macos_flag)

    dict_append(build_info, extra_compiler_args=new_args,
                extra_link_args=new_link_args)

