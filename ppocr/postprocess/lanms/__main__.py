import numpy as np


from . import merge_quadrangle_n9

if __name__ == '__main__':
    # unit square with confidence 1
    q = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1], dtype='float32')

    print(merge_quadrangle_n9(np.array([q, q + 0.1, q + 2])))
