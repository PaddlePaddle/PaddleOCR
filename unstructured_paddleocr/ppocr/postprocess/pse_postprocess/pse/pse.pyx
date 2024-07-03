
import numpy as np
import cv2
cimport numpy as np
cimport cython
cimport libcpp
cimport libcpp.pair
cimport libcpp.queue
from libcpp.pair cimport *
from libcpp.queue  cimport *

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int32_t, ndim=2] _pse(np.ndarray[np.uint8_t, ndim=3] kernels,
                                         np.ndarray[np.int32_t, ndim=2] label,
                                         int kernel_num,
                                         int label_num,
                                         float min_area=0):
    cdef np.ndarray[np.int32_t, ndim=2] pred
    pred = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)

    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0

    cdef libcpp.queue.queue[libcpp.pair.pair[np.int16_t,np.int16_t]] que = \
        queue[libcpp.pair.pair[np.int16_t,np.int16_t]]()
    cdef libcpp.queue.queue[libcpp.pair.pair[np.int16_t,np.int16_t]] nxt_que = \
        queue[libcpp.pair.pair[np.int16_t,np.int16_t]]()
    cdef np.int16_t* dx = [-1, 1, 0, 0]
    cdef np.int16_t* dy = [0, 0, -1, 1]
    cdef np.int16_t tmpx, tmpy

    points = np.array(np.where(label > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        tmpx, tmpy = points[point_idx, 0], points[point_idx, 1]
        que.push(pair[np.int16_t,np.int16_t](tmpx, tmpy))
        pred[tmpx, tmpy] = label[tmpx, tmpy]

    cdef libcpp.pair.pair[np.int16_t,np.int16_t] cur
    cdef int cur_label
    for kernel_idx in range(kernel_num - 1, -1, -1):
        while not que.empty():
            cur = que.front()
            que.pop()
            cur_label = pred[cur.first, cur.second]

            is_edge = True
            for j in range(4):
                tmpx = cur.first + dx[j]
                tmpy = cur.second + dy[j]
                if tmpx < 0 or tmpx >= label.shape[0] or tmpy < 0 or tmpy >= label.shape[1]:
                    continue
                if kernels[kernel_idx, tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue

                que.push(pair[np.int16_t,np.int16_t](tmpx, tmpy))
                pred[tmpx, tmpy] = cur_label
                is_edge = False
            if is_edge:
                nxt_que.push(cur)

        que, nxt_que = nxt_que, que

    return pred

def pse(kernels, min_area):
    kernel_num = kernels.shape[0]
    label_num, label = cv2.connectedComponents(kernels[-1], connectivity=4)
    return _pse(kernels[:-1], label, kernel_num, label_num, min_area)
