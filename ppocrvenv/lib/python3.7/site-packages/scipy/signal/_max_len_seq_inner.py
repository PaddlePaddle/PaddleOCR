# Author: Eric Larson
# 2014

import numpy as np

#pythran export _max_len_seq_inner(intp[], int8[], int, int, int8[])

# Fast inner loop of max_len_seq.
def _max_len_seq_inner(taps, state, nbits, length, seq):
    # Here we compute MLS using a shift register, indexed using a ring buffer
    # technique (faster than using something like np.roll to shift)
    n_taps = taps.shape[0]
    idx = 0
    for i in range(length):
        feedback = state[idx]
        seq[i] = feedback
        for ti in range(n_taps):
            feedback ^= state[(taps[ti] + idx) % nbits]
        state[idx] = feedback
        idx = (idx + 1) % nbits
    # state must be rolled s.t. next run, when idx==0, it's in the right place
    return np.roll(state, -idx, axis=0)
