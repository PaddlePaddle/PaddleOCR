import os
import sys
import subprocess


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils.compiler_helper import set_cxx_flags_hook

    config = Configuration('sparse',parent_package,top_path)

    config.add_data_dir('tests')

    config.add_subpackage('linalg')
    config.add_subpackage('csgraph')

    config.add_extension('_csparsetools',
                         sources=['_csparsetools.c'])

    def get_sparsetools_sources(ext, build_dir):
        # Defer generation of source files
        subprocess.check_call([sys.executable,
                               os.path.join(os.path.dirname(__file__),
                                            'generate_sparsetools.py'),
                               '--no-force'])
        return []

    depends = ['sparsetools_impl.h',
               'bsr_impl.h',
               'csc_impl.h',
               'csr_impl.h',
               'other_impl.h',
               'bool_ops.h',
               'bsr.h',
               'complex_ops.h',
               'coo.h',
               'csc.h',
               'csgraph.h',
               'csr.h',
               'dense.h',
               'dia.h',
               'sparsetools.h',
               'util.h']
    depends = [os.path.join('sparsetools', hdr) for hdr in depends],
    sparsetools = config.add_extension('_sparsetools',
                         define_macros=[('__STDC_FORMAT_MACROS', 1)],
                         depends=depends,
                         include_dirs=['sparsetools'],
                         sources=[os.path.join('sparsetools', 'sparsetools.cxx'),
                                  os.path.join('sparsetools', 'csr.cxx'),
                                  os.path.join('sparsetools', 'csc.cxx'),
                                  os.path.join('sparsetools', 'bsr.cxx'),
                                  os.path.join('sparsetools', 'other.cxx'),
                                  get_sparsetools_sources]
                         )
    sparsetools._pre_build_hook = set_cxx_flags_hook

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
