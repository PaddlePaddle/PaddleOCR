import re
import os
import sys
from distutils.util import get_platform

import numpy as np

from .system_info import combine_dict


__all__ = ['needs_g77_abi_wrapper', 'get_g77_abi_wrappers',
           'gfortran_legacy_flag_hook', 'blas_ilp64_pre_build_hook',
           'get_f2py_int64_options', 'generic_pre_build_hook',
           'write_file_content', 'ilp64_pre_build_hook']


def get_fcompiler_ilp64_flags():
    """
    Dictionary of compiler flags for switching to 8-byte default integer
    size.
    """
    flags = {
        'absoft': ['-i8'],  # Absoft
        'compaq': ['-i8'],  # Compaq Fortran
        'compaqv': ['/integer_size:64'],  # Compaq Visual Fortran
        'g95': ['-i8'],  # g95
        'gnu95': ['-fdefault-integer-8'],  # GNU gfortran
        'ibm': ['-qintsize=8'],  # IBM XL Fortran
        'intel': ['-i8'],  # Intel Fortran Compiler for 32-bit
        'intele': ['-i8'],  # Intel Fortran Compiler for Itanium
        'intelem': ['-i8'],  # Intel Fortran Compiler for 64-bit
        'intelv': ['-i8'],  # Intel Visual Fortran Compiler for 32-bit
        'intelev': ['-i8'],  # Intel Visual Fortran Compiler for Itanium
        'intelvem': ['-i8'],  # Intel Visual Fortran Compiler for 64-bit
        'lahey': ['--long'],  # Lahey/Fujitsu Fortran 95 Compiler
        'mips': ['-i8'],  # MIPSpro Fortran Compiler
        'nag': ['-i8'],  # NAGWare Fortran 95 compiler
        'nagfor': ['-i8'],  # NAG Fortran compiler
        'pathf95': ['-i8'],  # PathScale Fortran compiler
        'pg': ['-i8'],  # Portland Group Fortran Compiler
        'flang': ['-i8'],  # Portland Group Fortran LLVM Compiler
        'sun': ['-i8'],  # Sun or Forte Fortran 95 Compiler
    }
    # No support for this:
    # - g77
    # - hpux
    # Unknown:
    # - vast
    return flags


def get_fcompiler_macro_include_flags(path):
    """
    Dictionary of compiler flags for cpp-style preprocessing, with
    an #include search path, and safety options necessary for macro
    expansion.
    """
    intel_opts = ['-fpp', '-I' + path]
    nag_opts = ['-fpp', '-I' + path]

    flags = {
        'absoft': ['-W132', '-cpp', '-I' + path],
        'gnu95': ['-cpp', '-ffree-line-length-none',
                  '-ffixed-line-length-none', '-I' + path],
        'intel': intel_opts,
        'intele': intel_opts,
        'intelem': intel_opts,
        'intelv': intel_opts,
        'intelev': intel_opts,
        'intelvem': intel_opts,
        'lahey': ['-Cpp', '--wide', '-I' + path],
        'mips': ['-col120', '-I' + path],
        'nag': nag_opts,
        'nagfor': nag_opts,
        'pathf95': ['-ftpp', '-macro-expand', '-I' + path],
        'flang': ['-Mpreprocess', '-Mextend', '-I' + path],
        'sun': ['-fpp', '-I' + path],
    }
    # No support for this:
    # - ibm (line length option turns on fixed format)
    # TODO:
    # - pg
    return flags


def uses_mkl(info):
    r_mkl = re.compile("mkl")
    libraries = info.get('libraries', '')
    for library in libraries:
        if r_mkl.search(library):
            return True

    return False


def needs_g77_abi_wrapper(info):
    """Returns True if g77 ABI wrapper must be used."""
    try:
        needs_wrapper = int(os.environ["SCIPY_USE_G77_ABI_WRAPPER"]) != 0
    except KeyError:
        needs_wrapper = uses_mkl(info)
    return needs_wrapper


def get_g77_abi_wrappers(info):
    """
    Returns file names of source files containing Fortran ABI wrapper
    routines.
    """
    wrapper_sources = []

    path = os.path.abspath(os.path.dirname(__file__))
    if needs_g77_abi_wrapper(info):
        wrapper_sources += [
            os.path.join(path, 'src', 'wrap_g77_abi_f.f'),
            os.path.join(path, 'src', 'wrap_g77_abi_c.c'),
        ]
    else:
        wrapper_sources += [
            os.path.join(path, 'src', 'wrap_dummy_g77_abi.f'),
        ]
    return wrapper_sources


def gfortran_legacy_flag_hook(cmd, ext):
    """
    Pre-build hook to add dd gfortran legacy flag -fallow-argument-mismatch
    """
    from .compiler_helper import try_add_flag
    from distutils.version import LooseVersion

    if isinstance(ext, dict):
        # build_clib
        compilers = ((cmd._f_compiler, ext.setdefault('extra_f77_compile_args', [])),
                      (cmd._f_compiler, ext.setdefault('extra_f90_compile_args', [])))
    else:
        # build_ext
        compilers = ((cmd._f77_compiler, ext.extra_f77_compile_args),
                     (cmd._f90_compiler, ext.extra_f90_compile_args))

    for compiler, args in compilers:
        if compiler is None:
            continue

        if compiler.compiler_type == "gnu95" and compiler.version >= LooseVersion("10"):
            try_add_flag(args, compiler, "-fallow-argument-mismatch")


def _get_build_src_dir():
    plat_specifier = ".{}-{}.{}".format(get_platform(), *sys.version_info[:2])
    return os.path.join('build', 'src' + plat_specifier)


def get_f2py_int64_options():
    if np.dtype('i') == np.dtype(np.int64):
        int64_name = 'int'
    elif np.dtype('l') == np.dtype(np.int64):
        int64_name = 'long'
    elif np.dtype('q') == np.dtype(np.int64):
        int64_name = 'long_long'
    else:
        raise RuntimeError("No 64-bit integer type available in f2py!")

    f2cmap_fn = os.path.join(_get_build_src_dir(), 'int64.f2cmap')
    text = "{'integer': {'': '%s'}, 'logical': {'': '%s'}}\n" % (
        int64_name, int64_name)

    write_file_content(f2cmap_fn, text)

    return ['--f2cmap', f2cmap_fn]


def ilp64_pre_build_hook(cmd, ext):
    """
    Pre-build hook for adding Fortran compiler flags that change
    default integer size to 64-bit.
    """
    fcompiler_flags = get_fcompiler_ilp64_flags()
    return generic_pre_build_hook(cmd, ext, fcompiler_flags=fcompiler_flags)


def blas_ilp64_pre_build_hook(blas_info):
    """
    Pre-build hook for adding ILP64 BLAS compilation flags, and
    mangling Fortran source files to rename BLAS/LAPACK symbols when
    there are symbol suffixes.

    Examples
    --------
    ::

        from scipy._build_utils import blas_ilp64_pre_build_hook
        ext = config.add_extension(...)
        ext._pre_build_hook = blas_ilp64_pre_build_hook(blas_info)

    """
    return lambda cmd, ext: _blas_ilp64_pre_build_hook(cmd, ext, blas_info)


def _blas_ilp64_pre_build_hook(cmd, ext, blas_info):
    # Determine BLAS symbol suffix/prefix, if any
    macros = dict(blas_info.get('define_macros', []))
    prefix = macros.get('BLAS_SYMBOL_PREFIX', '')
    suffix = macros.get('BLAS_SYMBOL_SUFFIX', '')

    if suffix:
        if not suffix.endswith('_'):
            # Symbol suffix has to end with '_' to be Fortran-compatible
            raise RuntimeError("BLAS/LAPACK has incompatible symbol suffix: "
                               "{!r}".format(suffix))

        suffix = suffix[:-1]

    # When symbol prefix/suffix is present, we have to patch sources
    if prefix or suffix:
        include_dir = os.path.join(_get_build_src_dir(), 'blas64-include')

        fcompiler_flags = combine_dict(get_fcompiler_ilp64_flags(),
                                       get_fcompiler_macro_include_flags(include_dir))

        # Add the include dir for C code
        if isinstance(ext, dict):
            ext.setdefault('include_dirs', [])
            ext['include_dirs'].append(include_dir)
        else:
            ext.include_dirs.append(include_dir)

        # Create name-mapping include files
        include_name_f = 'blas64-prefix-defines.inc'
        include_name_c = 'blas64-prefix-defines.h'
        include_fn_f = os.path.join(include_dir, include_name_f)
        include_fn_c = os.path.join(include_dir, include_name_c)

        text = ""
        for symbol in get_blas_lapack_symbols():
            text += '#define {} {}{}_{}\n'.format(symbol, prefix, symbol, suffix)
            text += '#define {} {}{}_{}\n'.format(symbol.upper(), prefix, symbol, suffix)

            # Code generation may give source codes with mixed-case names
            for j in (1, 2):
                s = symbol[:j].lower() + symbol[j:].upper()
                text += '#define {} {}{}_{}\n'.format(s, prefix, symbol, suffix)
                s = symbol[:j].upper() + symbol[j:].lower()
                text += '#define {} {}{}_{}\n'.format(s, prefix, symbol, suffix)

        write_file_content(include_fn_f, text)

        ctext = re.sub(r'^#define (.*) (.*)$', r'#define \1_ \2_', text, flags=re.M)
        write_file_content(include_fn_c, text + "\n" + ctext)

        # Patch sources to include it
        def patch_source(filename, old_text):
            text = '#include "{}"\n'.format(include_name_f)
            text += old_text
            return text
    else:
        fcompiler_flags = get_fcompiler_ilp64_flags()
        patch_source = None

    return generic_pre_build_hook(cmd, ext,
                                  fcompiler_flags=fcompiler_flags,
                                  patch_source_func=patch_source,
                                  source_fnpart="_blas64")


def generic_pre_build_hook(cmd, ext, fcompiler_flags, patch_source_func=None,
                           source_fnpart=None):
    """
    Pre-build hook for adding compiler flags and patching sources.

    Parameters
    ----------
    cmd : distutils.core.Command
        Hook input. Current distutils command (build_clib or build_ext).
    ext : dict or numpy.distutils.extension.Extension
        Hook input. Configuration information for library (dict, build_clib)
        or extension (numpy.distutils.extension.Extension, build_ext).
    fcompiler_flags : dict
        Dictionary of ``{'compiler_name': ['-flag1', ...]}`` containing
        compiler flags to set.
    patch_source_func : callable, optional
        Function patching sources, see `_generic_patch_sources` below.
    source_fnpart : str, optional
        String to append to the modified file basename before extension.

    """
    is_clib = isinstance(ext, dict)

    if is_clib:
        build_info = ext
        del ext

        # build_clib doesn't have separate f77/f90 compilers
        f77 = cmd._f_compiler
        f90 = cmd._f_compiler
    else:
        f77 = cmd._f77_compiler
        f90 = cmd._f90_compiler

    # Add compiler flags
    if is_clib:
        f77_args = build_info.setdefault('extra_f77_compile_args', [])
        f90_args = build_info.setdefault('extra_f90_compile_args', [])
        compilers = [(f77, f77_args), (f90, f90_args)]
    else:
        compilers = [(f77, ext.extra_f77_compile_args),
                     (f90, ext.extra_f90_compile_args)]

    for compiler, args in compilers:
        if compiler is None:
            continue

        try:
            flags = fcompiler_flags[compiler.compiler_type]
        except KeyError as e:
            raise RuntimeError(
                "Compiler {!r} is not supported in this "
                "configuration.".format(compiler.compiler_type)
            ) from e

        args.extend(flag for flag in flags if flag not in args)

    # Mangle sources
    if patch_source_func is not None:
        if is_clib:
            build_info.setdefault('depends', []).extend(build_info['sources'])
            new_sources = _generic_patch_sources(build_info['sources'], patch_source_func,
                                                 source_fnpart)
            build_info['sources'][:] = new_sources
        else:
            ext.depends.extend(ext.sources)
            new_sources = _generic_patch_sources(ext.sources, patch_source_func,
                                                 source_fnpart)
            ext.sources[:] = new_sources


def _generic_patch_sources(filenames, patch_source_func, source_fnpart, root_dir=None):
    """
    Patch Fortran sources, creating new source files.

    Parameters
    ----------
    filenames : list
        List of Fortran source files to patch.
        Files not ending in ``.f`` or ``.f90`` are left unaltered.
    patch_source_func : callable(filename, old_contents) -> new_contents
        Function to apply to file contents, returning new file contents
        as a string.
    source_fnpart : str
        String to append to the modified file basename before extension.
    root_dir : str, optional
        Source root directory. Default: cwd

    Returns
    -------
    new_filenames : list
        List of names of the newly created patched sources.

    """
    new_filenames = []

    if root_dir is None:
        root_dir = os.getcwd()

    root_dir = os.path.abspath(root_dir)
    src_dir = os.path.join(root_dir, _get_build_src_dir())

    for src in filenames:
        base, ext = os.path.splitext(os.path.basename(src))

        if ext not in ('.f', '.f90'):
            new_filenames.append(src)
            continue

        with open(src, 'r') as fsrc:
            text = patch_source_func(src, fsrc.read())

        # Generate useful target directory name under src_dir
        src_path = os.path.abspath(os.path.dirname(src))

        for basedir in [src_dir, root_dir]:
            if os.path.commonpath([src_path, basedir]) == basedir:
                rel_path = os.path.relpath(src_path, basedir)
                break
        else:
            raise ValueError(f"{src!r} not under {root_dir!r}")

        dst = os.path.join(src_dir, rel_path, base + source_fnpart + ext)
        write_file_content(dst, text)

        new_filenames.append(dst)

    return new_filenames


def write_file_content(filename, content):
    """
    Write content to file, but only if it differs from the current one.
    """
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            old_content = f.read()

        if old_content == content:
            return

    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    with open(filename, 'w') as f:
        f.write(content)


def get_blas_lapack_symbols():
    cached = getattr(get_blas_lapack_symbols, 'cached', None)
    if cached is not None:
        return cached

    # Obtain symbol list from Cython Blas/Lapack interface
    srcdir = os.path.join(os.path.dirname(__file__), os.pardir, 'linalg')

    symbols = []

    # Get symbols from the generated files
    for fn in ['cython_blas_signatures.txt', 'cython_lapack_signatures.txt']:
        with open(os.path.join(srcdir, fn), 'r') as f:
            for line in f:
                m = re.match(r"^\s*[a-z]+\s+([a-z0-9]+)\(", line)
                if m:
                    symbols.append(m.group(1))

    # Get the rest from the generator script
    # (we cannot import it directly here, so use exec)
    sig_fn = os.path.join(srcdir, '_cython_signature_generator.py')
    with open(sig_fn, 'r') as f:
        code = f.read()
    ns = {'__name__': '<module>'}
    exec(code, ns)
    symbols.extend(ns['blas_exclusions'])
    symbols.extend(ns['lapack_exclusions'])

    get_blas_lapack_symbols.cached = tuple(sorted(set(symbols)))
    return get_blas_lapack_symbols.cached
