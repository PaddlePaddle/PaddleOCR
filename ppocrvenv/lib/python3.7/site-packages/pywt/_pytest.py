"""common test-related code."""
import os
import sys
import multiprocessing
import numpy as np
import pytest


__all__ = ['uses_matlab',   # skip if pymatbridge and Matlab unavailable
           'uses_futures',  # skip if futures unavailable
           'uses_pymatbridge',  # skip if no PYWT_XSLOW environment variable
           'uses_precomputed',  # skip if PYWT_XSLOW environment variable found
           'matlab_result_dict_cwt',   # dict with precomputed Matlab dwt data
           'matlab_result_dict_dwt',   # dict with precomputed Matlab cwt data
           'futures',      # the futures module or None
           'max_workers',  # the number of workers available to futures
           'size_set',     # the set of Matlab tests to run
           ]

try:
    if sys.version_info[0] == 2:
        import futures
    else:
        from concurrent import futures
    max_workers = multiprocessing.cpu_count()
    futures_available = True
except ImportError:
    futures_available = False
    futures = None

# check if pymatbridge + MATLAB tests should be run
matlab_result_dict_dwt = None
matlab_result_dict_cwt = None
matlab_missing = True
use_precomputed = True
size_set = 'reduced'
if 'PYWT_XSLOW' in os.environ:
    try:
        from pymatbridge import Matlab
        mlab = Matlab()
        matlab_missing = False
        use_precomputed = False
        size_set = 'full'
    except ImportError:
        print("To run Matlab compatibility tests you need to have MathWorks "
              "MATLAB, MathWorks Wavelet Toolbox and the pymatbridge Python "
              "package installed.")
if use_precomputed:
    # load dictionaries of precomputed results
    data_dir = os.path.join(os.path.dirname(__file__), 'tests', 'data')
    matlab_data_file_cwt = os.path.join(
        data_dir, 'cwt_matlabR2015b_result.npz')
    matlab_result_dict_cwt = np.load(matlab_data_file_cwt)

    matlab_data_file_dwt = os.path.join(
        data_dir, 'dwt_matlabR2012a_result.npz')
    matlab_result_dict_dwt = np.load(matlab_data_file_dwt)

uses_futures = pytest.mark.skipif(
    not futures_available, reason='futures not available')
uses_matlab = pytest.mark.skipif(
    matlab_missing, reason='pymatbridge and/or Matlab not available')
uses_pymatbridge = pytest.mark.skipif(
    use_precomputed,
    reason='PYWT_XSLOW set: skipping tests against precomputed Matlab results')
uses_precomputed = pytest.mark.skipif(
    not use_precomputed,
    reason='PYWT_XSLOW not set: test against precomputed matlab tests')
