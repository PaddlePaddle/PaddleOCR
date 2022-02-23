def configuration(parent_package='', top_path=None):
    from numpy import get_include
    from scipy._build_utils.system_info import get_info
    from scipy._build_utils import uses_blas64
    from numpy.distutils.misc_util import Configuration

    from os.path import join, dirname

    if uses_blas64():
        lapack_opt = get_info('lapack_ilp64_opt')
    else:
        lapack_opt = get_info('lapack_opt')

    lib_inc = join(dirname(dirname(dirname(__file__))), '_lib')
    bld_inc = join(dirname(dirname(dirname(__file__))), '_build_utils', 'src')

    config = Configuration('_trlib', parent_package, top_path)
    config.add_extension('_trlib',
                         sources=['_trlib.c', 'trlib_krylov.c',
                                  'trlib_eigen_inverse.c', 'trlib_leftmost.c',
                                  'trlib_quadratic_zero.c', 'trlib_tri_factor.c'],
                         include_dirs=[get_include(), lib_inc, bld_inc, 'trlib'],
                         extra_info=lapack_opt,
                         )
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
