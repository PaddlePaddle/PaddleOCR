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
         ('reflect', 'symw'),
         ('periodic', 'ppd'),
         ('smooth', 'sp1'),
         ('periodization', 'per'),
         ('antisymmetric', 'asym'),
         ('antireflect', 'asymw')]

families = ('db', 'sym', 'coif', 'bior', 'rbio')
wavelets = sum([pywt.wavelist(name) for name in families], [])

rstate = np.random.RandomState(1234)
mlab.start()
try:
    all_matlab_results = {}
    for wavelet in wavelets:
        w = pywt.Wavelet(wavelet)
        mlab.set_variable('wavelet', wavelet)
        if size_set == 'full':
            data_sizes = list(range(w.dec_len, 40)) + \
                [100, 200, 500, 1000, 50000]
        else:
            data_sizes = (w.dec_len, w.dec_len + 1)
        for N in data_sizes:
            data = rstate.randn(N)
            mlab.set_variable('data', data)
            for pmode, mmode in modes:
                # Matlab result
                if np.any((wavelet == np.array(['coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17'])),axis=0):
                    mlab.set_variable('Lo_D', w.dec_lo)
                    mlab.set_variable('Hi_D', w.dec_hi)
                    mlab_code = ("[ma, md] = dwt(data, Lo_D, Hi_D, "
                                 "'mode', '%s');" % mmode)
                else:
                    mlab_code = ("[ma, md] = dwt(data, wavelet, "
                                 "'mode', '%s');" % mmode)
                res = mlab.run_code(mlab_code)
                if not res['success']:
                    raise RuntimeError(
                        "Matlab failed to execute the provided code. "
                        "Check that the wavelet toolbox is installed.")
                # need np.asarray because sometimes the output is type float
                ma = np.asarray(mlab.get_variable('ma'))
                md = np.asarray(mlab.get_variable('md'))
                ma_key = '_'.join([mmode, wavelet, str(N), 'ma'])
                md_key = '_'.join([mmode, wavelet, str(N), 'md'])
                all_matlab_results[ma_key] = ma
                all_matlab_results[md_key] = md

                # Matlab result
                mlab.set_variable('Lo_D', w.dec_lo)
                mlab.set_variable('Hi_D', w.dec_hi)
                mlab_code = ("[ma, md] = dwt(data, Lo_D, Hi_D, "
                             "'mode', '%s');" % mmode)
                res = mlab.run_code(mlab_code)
                if not res['success']:
                    raise RuntimeError(
                        "Matlab failed to execute the provided code. "
                        "Check that the wavelet toolbox is installed.")
                # need np.asarray because sometimes the output is type float
                ma = np.asarray(mlab.get_variable('ma'))
                md = np.asarray(mlab.get_variable('md'))
                ma_key = '_'.join([mmode, wavelet, str(N), 'ma_pywtCoeffs'])
                md_key = '_'.join([mmode, wavelet, str(N), 'md_pywtCoeffs'])
                all_matlab_results[ma_key] = ma
                all_matlab_results[md_key] = md
finally:
    mlab.stop()

np.savez('dwt_matlabR2012a_result.npz', **all_matlab_results)
