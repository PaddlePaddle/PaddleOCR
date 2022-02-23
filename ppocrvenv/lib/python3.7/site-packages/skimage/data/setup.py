#!/usr/bin/env python
from _registry import legacy_datasets


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('data', parent_package, top_path)
    # This minimal dataset was available as part of
    # scikit-image 0.15 and will be retained until
    # further notice.
    # Testing data and additional datasets should only
    # be made available via pooch
    config.add_data_files(*legacy_datasets)
    # It seems hard to create a consistent hash for README.txt since
    # the line endings keep getting converted
    config.add_data_files('README.txt')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          author='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Minimal sample dataset for scikit-image',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
