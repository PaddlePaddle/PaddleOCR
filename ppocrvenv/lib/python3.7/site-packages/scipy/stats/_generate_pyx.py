import pathlib
from shutil import copyfile
import subprocess
import sys


def isNPY_OLD():
    '''
    A new random C API was added in 1.18 and became stable in 1.19.
    Prefer the new random C API when building with recent numpy.
    '''
    import numpy as np
    ver = tuple(int(num) for num in np.__version__.split('.')[:2])
    return ver < (1, 19)


def make_biasedurn():
    '''Substitute True/False values for NPY_OLD Cython build variable.'''
    biasedurn_base = (pathlib.Path(__file__).parent / 'biasedurn').absolute()
    with open(biasedurn_base.with_suffix('.pyx.templ'), 'r') as src:
        contents = src.read()
    with open(biasedurn_base.with_suffix('.pyx'), 'w') as dest:
        dest.write(contents.format(NPY_OLD=str(bool(isNPY_OLD()))))


def make_boost():
    # Call code generator inside _boost directory
    code_gen = pathlib.Path(__file__).parent / '_boost/include/code_gen.py'
    subprocess.run([sys.executable, str(code_gen)], check=True)


if __name__ == '__main__':
    make_biasedurn()
    make_boost()
