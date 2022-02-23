# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.


def run_tests(args=''):
    try:
        from pytest import main
    except ImportError:
        print("Tests require pytest, pytest not installed")
        return 1
    else:
        from shlex import split
        from subprocess import call
        from sys import executable
        cli = [executable, "-m", "pytest", "--pyargs", "h5py"]
        cli.extend(split(args))
        return call(cli)
