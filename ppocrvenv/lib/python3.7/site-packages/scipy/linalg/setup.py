from os.path import join


def configuration(parent_package='', top_path=None):
    from distutils.sysconfig import get_python_inc
    from scipy._build_utils.system_info import get_info, numpy_info
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from scipy._build_utils import (get_g77_abi_wrappers, gfortran_legacy_flag_hook,
                                    blas_ilp64_pre_build_hook, get_f2py_int64_options,
                                    uses_blas64)

    config = Configuration('linalg', parent_package, top_path)

    lapack_opt = get_info('lapack_opt')

    atlas_version = ([v[3:-3] for k, v in lapack_opt.get('define_macros', [])
                      if k == 'ATLAS_INFO']+[None])[0]
    if atlas_version:
        print(('ATLAS version: %s' % atlas_version))

    if uses_blas64():
        lapack_ilp64_opt = get_info('lapack_ilp64_opt', 2)

    # fblas:
    sources = ['fblas.pyf.src']
    sources += get_g77_abi_wrappers(lapack_opt)
    depends = ['fblas_l?.pyf.src']

    config.add_extension('_fblas',
                         sources=sources,
                         depends=depends,
                         extra_info=lapack_opt
                         )

    if uses_blas64():
        sources = ['fblas_64.pyf.src'] + sources[1:]
        ext = config.add_extension('_fblas_64',
                                   sources=sources,
                                   depends=depends,
                                   f2py_options=get_f2py_int64_options(),
                                   extra_info=lapack_ilp64_opt)
        ext._pre_build_hook = blas_ilp64_pre_build_hook(lapack_ilp64_opt)

    # flapack:
    sources = ['flapack.pyf.src']
    sources += get_g77_abi_wrappers(lapack_opt)
    dep_pfx = join('src', 'lapack_deprecations')
    deprecated_lapack_routines = [join(dep_pfx, c + 'gegv.f') for c in 'cdsz']
    sources += deprecated_lapack_routines
    depends = ['flapack_gen.pyf.src',
               'flapack_gen_banded.pyf.src',
               'flapack_gen_tri.pyf.src',
               'flapack_pos_def.pyf.src',
               'flapack_pos_def_tri.pyf.src',
               'flapack_sym_herm.pyf.src',
               'flapack_other.pyf.src',
               'flapack_user.pyf.src']

    config.add_extension('_flapack',
                         sources=sources,
                         depends=depends,
                         extra_info=lapack_opt
                         )

    if uses_blas64():
        sources = ['flapack_64.pyf.src'] + sources[1:]
        ext = config.add_extension('_flapack_64',
                                   sources=sources,
                                   depends=depends,
                                   f2py_options=get_f2py_int64_options(),
                                   extra_info=lapack_ilp64_opt)
        ext._pre_build_hook = blas_ilp64_pre_build_hook(lapack_ilp64_opt)

    if atlas_version is not None:
        # cblas:
        config.add_extension('_cblas',
                             sources=['cblas.pyf.src'],
                             depends=['cblas.pyf.src', 'cblas_l1.pyf.src'],
                             extra_info=lapack_opt
                             )

        # clapack:
        config.add_extension('_clapack',
                             sources=['clapack.pyf.src'],
                             depends=['clapack.pyf.src'],
                             extra_info=lapack_opt
                             )

    # _flinalg:
    config.add_extension('_flinalg',
                         sources=[join('src', 'det.f'), join('src', 'lu.f')],
                         extra_info=lapack_opt
                         )

    # _interpolative:
    ext = config.add_extension('_interpolative',
                               sources=[join('src', 'id_dist', 'src', '*.f'),
                                        "interpolative.pyf"],
                               extra_info=lapack_opt
                               )
    ext._pre_build_hook = gfortran_legacy_flag_hook

    # _solve_toeplitz:
    config.add_extension('_solve_toeplitz',
                         sources=[('_solve_toeplitz.c')],
                         include_dirs=[get_numpy_include_dirs()])

    # _matfuncs_sqrtm_triu:
    config.add_extension('_matfuncs_sqrtm_triu',
                         sources=[('_matfuncs_sqrtm_triu.c')],
                         include_dirs=[get_numpy_include_dirs()])

    config.add_data_dir('tests')

    # Cython BLAS/LAPACK
    config.add_data_files('cython_blas.pxd')
    config.add_data_files('cython_lapack.pxd')

    sources = ['_blas_subroutine_wrappers.f', '_lapack_subroutine_wrappers.f']
    sources += get_g77_abi_wrappers(lapack_opt)
    includes = numpy_info().get_include_dirs() + [get_python_inc()]
    config.add_library('fwrappers', sources=sources, include_dirs=includes)

    config.add_extension('cython_blas',
                         sources=['cython_blas.c'],
                         depends=['cython_blas.pyx', 'cython_blas.pxd',
                                  'fortran_defs.h', '_blas_subroutines.h'],
                         include_dirs=['.'],
                         libraries=['fwrappers'],
                         extra_info=lapack_opt)

    config.add_extension('cython_lapack',
                         sources=['cython_lapack.c'],
                         depends=['cython_lapack.pyx', 'cython_lapack.pxd',
                                  'fortran_defs.h', '_lapack_subroutines.h'],
                         include_dirs=['.'],
                         libraries=['fwrappers'],
                         extra_info=lapack_opt)

    config.add_extension('_decomp_update',
                         sources=['_decomp_update.c'])

    # Add any license files
    config.add_data_files('src/id_dist/doc/doc.tex')
    config.add_data_files('src/lapack_deprecations/LICENSE')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
