import subprocess
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile lanms: {}'.format(BASE_DIR))


def merge_quadrangle_n9(polys, thres=0.3, precision=10000):
    from .adaptor import merge_quadrangle_n9 as nms_impl
    if len(polys) == 0:
        return np.array([], dtype='float32')
    p = polys.copy()
    p[:,:8] *= precision
    ret = np.array(nms_impl(p, thres), dtype='float32')
    ret[:,:8] /= precision
    return ret

