""" This script was used to generate dwt_matlabR2012a_result.npz by storing
the outputs from Matlab R2012a. """

from __future__ import division, print_function, absolute_import

import numpy as np
import pywt

try:
    from pymatbridge import Matlab
    mlab = Matlab()
    _matlab_missing = False
except ImportError:
    print("To run Matlab compatibility tests you need to have MathWorks "
          "MATLAB, MathWorks Wavelet Toolbox and the pymatbridge Python "
          "package installed.")
    _matlab_missing = True

if _matlab_missing:
    raise EnvironmentError("Can't generate matlab data files without MATLAB")

size_set = 'reduced'

# list of mode names in pywt and matlab
modes = [('zero', 'zpd'),
         ('constant', 'sp0'),
         ('symmetric', 'sym'),
         ('periodic', 'ppd'),
         ('smooth', 'sp1'),
         ('periodization', 'per')]

families = ('gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor')
wavelets = sum([pywt.wavelist(name) for name in families], [])

rstate = np.random.RandomState(1234)
mlab.start()
try:
    all_matlab_results = {}
    for wavelet in wavelets:
        w = pywt.ContinuousWavelet(wavelet)
        if np.any((wavelet == np.array(['shan', 'cmor'])),axis=0):
            mlab.set_variable('wavelet', wavelet+str(w.bandwidth_frequency)+'-'+str(w.center_frequency))
        elif wavelet == 'fbsp':
            mlab.set_variable('wavelet', wavelet+str(w.fbsp_order)+'-'+str(w.bandwidth_frequency)+'-'+str(w.center_frequency))
        else:
            mlab.set_variable('wavelet', wavelet)
        if size_set == 'full':
            data_sizes = list(range(100, 101)) + \
                [100, 200, 500, 1000, 50000]
            Scales = (1,np.arange(1,3),np.arange(1,4),np.arange(1,5))
        else:
            data_sizes = (1000, 1000 + 1)
            Scales = (1,np.arange(1,3))
        mlab_code = ("psi = wavefun(wavelet,10)")
        res = mlab.run_code(mlab_code)
        if not res['success']:
            raise RuntimeError(
                "Matlab failed to execute the provided code. "
                "Check that the wavelet toolbox is installed.")
        psi = np.asarray(mlab.get_variable('psi'))
        psi_key = '_'.join([wavelet, 'psi'])
        all_matlab_results[psi_key] = psi
        for N in data_sizes:
            data = rstate.randn(N)
            mlab.set_variable('data', data)

            # Matlab result
            scale_count = 0
            for scales in Scales:
                scale_count += 1
                mlab.set_variable('scales', scales)
                mlab_code = ("coefs = cwt(data, scales, wavelet)")
                res = mlab.run_code(mlab_code)
                if not res['success']:
                    raise RuntimeError(
                        "Matlab failed to execute the provided code. "
                        "Check that the wavelet toolbox is installed.")
                # need np.asarray because sometimes the output is type float
                coefs = np.asarray(mlab.get_variable('coefs'))
                coefs_key = '_'.join([str(scale_count), wavelet, str(N), 'coefs'])
                all_matlab_results[coefs_key] = coefs

finally:
    mlab.stop()

np.savez('cwt_matlabR2015b_result.npz', **all_matlab_results)
