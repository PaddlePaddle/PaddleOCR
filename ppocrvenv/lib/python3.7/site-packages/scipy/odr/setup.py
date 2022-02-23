from os.path import join
from scipy._build_utils import numpy_nodepr_api


def configuration(parent_package='', top_path=None):
    import warnings
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils.system_info import get_info
    from scipy._build_utils import (uses_blas64, blas_ilp64_pre_build_hook,
                                    combine_dict)

    config = Configuration('odr', parent_package, top_path)

    libodr_files = ['d_odr.f',
                    'd_mprec.f',
                    'dlunoc.f',
                    'd_lpk.f']

    if uses_blas64():
        blas_info = get_info('blas_ilp64_opt')
        pre_build_hook = blas_ilp64_pre_build_hook(blas_info)
    else:
        blas_info = get_info('blas_opt')
        pre_build_hook = None

    odrpack_src = [join('odrpack', x) for x in libodr_files]
    config.add_library('odrpack', sources=odrpack_src,
                       _pre_build_hook=pre_build_hook)

    sources = ['__odrpack.c']

    cfg = combine_dict(blas_info, numpy_nodepr_api,
                       libraries=['odrpack'],
                       include_dirs=['.'])
    ext = config.add_extension('__odrpack',
        sources=sources,
        depends=(['odrpack.h'] + odrpack_src),
        **cfg
    )
    ext._pre_build_hook = pre_build_hook

    config.add_data_dir('tests')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
