from scipy._build_utils import numpy_nodepr_api
from scipy._build_utils import tempita
import os
import sys


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils.compiler_helper import set_c_flags_hook

    config = Configuration('signal', parent_package, top_path)

    config.add_data_dir('tests')

    config.add_subpackage('windows')

    # convert the *.c.in files : `lfilter.c.in -> lfilter.c` etc
    srcdir = os.path.join(os.getcwd(), 'scipy', 'signal')
    tempita.process_tempita(os.path.join(srcdir, 'lfilter.c.in') )
    tempita.process_tempita(os.path.join(srcdir, 'correlate_nd.c.in') )

    sigtools = config.add_extension('sigtools',
                         sources=['sigtoolsmodule.c', 'firfilter.c',
                                  'medianfilter.c', 'lfilter.c',
                                  'correlate_nd.c'],
                         depends=['sigtools.h'],
                         include_dirs=['.'],
                         **numpy_nodepr_api)
    sigtools._pre_build_hook = set_c_flags_hook

    if int(os.environ.get('SCIPY_USE_PYTHRAN', 1)):
        import pythran
        ext = pythran.dist.PythranExtension(
            'scipy.signal._max_len_seq_inner',
            sources=["scipy/signal/_max_len_seq_inner.py"],
            config=['compiler.blas=none'])
        config.ext_modules.append(ext)

        ext = pythran.dist.PythranExtension(
            'scipy.signal._spectral',
            sources=["scipy/signal/_spectral.py"],
            config=['compiler.blas=none'])
        config.ext_modules.append(ext)
    else:
        config.add_extension(
            '_spectral', sources=['_spectral.c'])

        config.add_extension(
            '_max_len_seq_inner', sources=['_max_len_seq_inner.c'])

    config.add_extension(
        '_peak_finding_utils', sources=['_peak_finding_utils.c'])
    config.add_extension(
        '_sosfilt', sources=['_sosfilt.c'])
    config.add_extension(
        '_upfirdn_apply', sources=['_upfirdn_apply.c'])
    spline_src = ['splinemodule.c', 'S_bspline_util.c', 'D_bspline_util.c',
                  'C_bspline_util.c', 'Z_bspline_util.c', 'bspline_util.c']
    config.add_extension('spline', sources=spline_src, **numpy_nodepr_api)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
