import itertools
import threading
import time

import numpy as np
from numpy.testing import assert_equal
import pytest
import scipy.interpolate


class TestGIL:
    """Check if the GIL is properly released by scipy.interpolate functions."""

    def setup_method(self):
        self.messages = []

    def log(self, message):
        self.messages.append(message)

    def make_worker_thread(self, target, args):
        log = self.log

        class WorkerThread(threading.Thread):
            def run(self):
                log('interpolation started')
                target(*args)
                log('interpolation complete')

        return WorkerThread()

    @pytest.mark.slow
    @pytest.mark.xfail(reason='race conditions, may depend on system load')
    def test_rectbivariatespline(self):
        def generate_params(n_points):
            x = y = np.linspace(0, 1000, n_points)
            x_grid, y_grid = np.meshgrid(x, y)
            z = x_grid * y_grid
            return x, y, z

        def calibrate_delay(requested_time):
            for n_points in itertools.count(5000, 1000):
                args = generate_params(n_points)
                time_started = time.time()
                interpolate(*args)
                if time.time() - time_started > requested_time:
                    return args

        def interpolate(x, y, z):
            scipy.interpolate.RectBivariateSpline(x, y, z)

        args = calibrate_delay(requested_time=3)
        worker_thread = self.make_worker_thread(interpolate, args)
        worker_thread.start()
        for i in range(3):
            time.sleep(0.5)
            self.log('working')
        worker_thread.join()
        assert_equal(self.messages, [
            'interpolation started',
            'working',
            'working',
            'working',
            'interpolation complete',
        ])

