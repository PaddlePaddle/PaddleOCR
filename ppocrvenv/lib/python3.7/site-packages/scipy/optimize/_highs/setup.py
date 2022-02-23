'''
setup.py for HiGHS scipy interface

Some CMake files are used to create source lists for compilation
'''

import pathlib
from datetime import datetime
import os
from os.path import join


def pre_build_hook(build_ext, ext):
    from scipy._build_utils.compiler_helper import get_cxx_std_flag
    std_flag = get_cxx_std_flag(build_ext._cxx_compiler)
    if std_flag is not None:
        ext.extra_compile_args.append(std_flag)

def basiclu_pre_build_hook(build_clib, build_info):
    from scipy._build_utils.compiler_helper import get_c_std_flag
    c_flag = get_c_std_flag(build_clib.compiler)
    if c_flag is not None:
        if 'extra_compiler_args' not in build_info:
            build_info['extra_compiler_args'] = []
        build_info['extra_compiler_args'].append(c_flag)

def _get_sources(CMakeLists, start_token, end_token):
    # Read in sources from CMakeLists.txt
    CMakeLists = pathlib.Path(__file__).parent / CMakeLists
    with open(CMakeLists, 'r', encoding='utf-8') as f:
        s = f.read()

        # Find block where sources are listed
        start_idx = s.find(start_token) + len(start_token)
        end_idx = s[start_idx:].find(end_token) + len(s[:start_idx])
        sources = s[start_idx:end_idx].split('\n')
        sources = [s.strip() for s in sources if s[0] != '#']

    # Make relative to setup.py
    sources = [str(pathlib.Path('src/' + s)) for s in sources]
    return sources

# Grab some more info about HiGHS from root CMakeLists
def _get_version(CMakeLists, start_token, end_token=')'):
    CMakeLists = pathlib.Path(__file__).parent / CMakeLists
    with open(CMakeLists, 'r', encoding='utf-8') as f:
        s = f.read()
        start_idx = s.find(start_token) + len(start_token) + 1
        end_idx = s[start_idx:].find(end_token) + len(s[:start_idx])
    return s[start_idx:end_idx].strip()


def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration
    config = Configuration('_highs', parent_package, top_path)

    # HiGHS info
    _major_dot_minor = _get_version(
        'CMakeLists.txt', 'project(HIGHS VERSION', 'LANGUAGES CXX C')
    HIGHS_VERSION_MAJOR, HIGHS_VERSION_MINOR = _major_dot_minor.split('.')
    HIGHS_VERSION_PATCH = _get_version(
        'CMakeLists.txt', 'HIGHS_VERSION_PATCH')
    GITHASH = 'n/a'
    HIGHS_DIR = str(pathlib.Path(__file__).parent.resolve())

    # Here are the pound defines that HConfig.h would usually provide;
    # We provide an empty HConfig.h file and do the defs and undefs
    # here:
    TODAY_DATE = datetime.today().strftime('%Y-%m-%d')
    DEFINE_MACROS = [
        ('CMAKE_BUILD_TYPE', '"Release"'),
        ('HiGHSRELEASE', None),
        ('IPX_ON', 'ON'),
        ('HIGHS_GITHASH', '"%s"' % GITHASH),
        ('HIGHS_COMPILATION_DATE', '"' + TODAY_DATE + '"'),
        ('HIGHS_VERSION_MAJOR', HIGHS_VERSION_MAJOR),
        ('HIGHS_VERSION_MINOR', HIGHS_VERSION_MINOR),
        ('HIGHS_VERSION_PATCH', HIGHS_VERSION_PATCH),
        ('HIGHS_DIR', '"' + HIGHS_DIR + '"'),
        # ('NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION', None),
    ]
    UNDEF_MACROS = [
        'OPENMP',  # unconditionally disable openmp
        'EXT_PRESOLVE',
        'SCIP_DEV',
        'HiGHSDEV',
        'OSI_FOUND',
    ]

    # Compile BASICLU as a static library to appease clang:
    # (won't allow -std=c++11/14 option for C sources)
    basiclu_sources = _get_sources('src/CMakeLists.txt',
                                   'set(basiclu_sources\n', ')')
    config.add_library(
        'basiclu',
        sources=basiclu_sources,
        include_dirs=[
            'src',
            join('src', 'ipm', 'basiclu', 'include'),
        ],
        language='c',
        macros=DEFINE_MACROS,
        _pre_build_hook=basiclu_pre_build_hook,
    )

    # highs_wrapper:
    ipx_sources = _get_sources('src/CMakeLists.txt', 'set(ipx_sources\n', ')')
    highs_sources = _get_sources('src/CMakeLists.txt', 'set(sources\n', ')')
    # filter out MIP sources until MIP is officially supported
    highs_sources = [s for s in highs_sources
                     if pathlib.Path(s).parent.name != 'mip']
    ext = config.add_extension(
        '_highs_wrapper',
        sources=[join('cython', 'src', '_highs_wrapper.cxx')] + \
                highs_sources + ipx_sources,
        include_dirs=[
            # highs_wrapper
            'src',
            join('cython', 'src'),
            join('src', 'lp_data'),
            # highs
            join('src', 'io'),
            join('src', 'ipm', 'ipx', 'include'),
            # IPX
            join('src', 'ipm', 'ipx', 'include'),
            join('src', 'ipm', 'basiclu', 'include'),
        ],
        language='c++',
        libraries=['basiclu'],
        define_macros=DEFINE_MACROS,
        undef_macros=UNDEF_MACROS,
    )
    # Add c++11/14 support:
    ext._pre_build_hook = pre_build_hook

    # Export constants and enums from HiGHS:
    ext = config.add_extension(
        '_highs_constants',
        sources=[join('cython', 'src', '_highs_constants.cxx')],
        include_dirs=[
            'src',
            join('cython', 'src'),
            join('src', 'io'),
            join('src', 'lp_data'),
            join('src', 'simplex'),
        ],
        language='c++',
    )
    ext._pre_build_hook = pre_build_hook

    config.add_data_files(os.path.join('cython', 'src', '*.pxd'))

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
