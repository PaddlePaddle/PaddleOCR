import time
import random
import skimage.graph.heap as heap

from skimage._shared.testing import test_parallel


@test_parallel()
def test_heap():
    _test_heap(100000, True)
    _test_heap(100000, False)


def _test_heap(n, fast_update):
    # generate random numbers with duplicates
    random.seed(0)
    a = [random.uniform(1.0, 100.0) for i in range(n // 2)]
    a = a + a

    t0 = time.perf_counter()

    # insert in heap with random removals
    if fast_update:
        h = heap.FastUpdateBinaryHeap(128, n)
    else:
        h = heap.BinaryHeap(128)
    for i in range(len(a)):
        h.push(a[i], i)
        if a[i] < 25:
            # double-push same ref sometimes to test fast update codepaths
            h.push(2 * a[i], i)
        if 25 < a[i] < 50:
            # pop some to test random removal
            h.pop()

    # pop from heap
    b = []
    while True:
        try:
            b.append(h.pop()[0])
        except IndexError:
            break

    t1 = time.perf_counter()

    # verify
    for i in range(1, len(b)):
        assert(b[i] >= b[i - 1])

    return t1 - t0
