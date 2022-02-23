from os.path import join

from numpy.distutils.misc_util import get_info


def pre_build_hook(build_ext, ext):
    from scipy._build_utils.compiler_helper import get_cxx_std_flag
    std_flag = get_cxx_std_flag(build_ext._cxx_compiler)
    if std_flag is not None:
        ext.extra_compile_args.append(std_flag)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils.compiler_helper import set_cxx_flags_hook
    import numpy as np
    config = Configuration('stats', parent_package, top_path)

    config.add_data_dir('tests')

    statlib_src = [join('statlib', '*.f')]
    config.add_library('statlib', sources=statlib_src)

    # add statlib module
    config.add_extension('statlib',
                         sources=['statlib.pyf'],
                         f2py_options=['--no-wrap-functions'],
                         libraries=['statlib'],
                         depends=statlib_src)

    # add _stats module
    config.add_extension('_stats',
                         sources=['_stats.c'])

    # add mvn module
    config.add_extension('mvn',
                         sources=['mvn.pyf', 'mvndst.f'])

    # add _sobol module
    config.add_extension('_sobol',
                         sources=['_sobol.c'])
    config.add_data_files('_sobol_direction_numbers.npz')

    # add _qmc_cy module
    ext = config.add_extension('_qmc_cy',
                               sources=['_qmc_cy.cxx'])
    ext._pre_build_hook = set_cxx_flags_hook

    # add BiasedUrn module
    config.add_data_files('biasedurn.pxd')
    from _generate_pyx import isNPY_OLD  # type: ignore[import]
    NPY_OLD = isNPY_OLD()

    if NPY_OLD:
        biasedurn_libs = []
        biasedurn_libdirs = []
    else:
        biasedurn_libs = ['npyrandom', 'npymath']
        biasedurn_libdirs = [join(np.get_include(),
                                  '..', '..', 'random', 'lib')]
        biasedurn_libdirs += get_info('npymath')['library_dirs']

    ext = config.add_extension(
        'biasedurn',
        sources=[
            'biasedurn.cxx',
            'biasedurn/impls.cpp',
            'biasedurn/fnchyppr.cpp',
            'biasedurn/wnchyppr.cpp',
            'biasedurn/stoc1.cpp',
            'biasedurn/stoc3.cpp'],
        include_dirs=[np.get_include()],
        library_dirs=biasedurn_libdirs,
        libraries=biasedurn_libs,
        define_macros=[('R_BUILD', None)],
        language='c++',
        depends=['biasedurn/stocR.h'],
    )
    ext._pre_build_hook = pre_build_hook

    # add boost stats distributions
    config.add_subpackage('_boost')

    # Type stubs
    config.add_data_files('*.pyi')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
