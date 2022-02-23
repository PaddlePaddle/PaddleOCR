import numpy as np

#pythran export _brief_loop(float32[:,:] or float64[:,:], uint8[:,:], int64[:,2], int32[:,2], int32[:,2])
def _brief_loop(image, descriptors, keypoints, pos0, pos1):
    for p in range(pos0.shape[0]):
        pr0, pc0 = pos0[p]
        pr1, pc1 = pos1[p]
        for k in range(keypoints.shape[0]):
            kr, kc = keypoints[k]
            if image[kr + pr0, kc + pc0] < image[kr + pr1, kc + pc1]:
                descriptors[k, p] = True
