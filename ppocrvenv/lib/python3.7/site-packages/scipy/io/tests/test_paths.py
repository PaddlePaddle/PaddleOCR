"""
Ensure that we can use pathlib.Path objects in all relevant IO functions.
"""
import sys
from pathlib import Path

import numpy as np

import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse


class TestPaths:
    data = np.arange(5).astype(np.int64)

    def test_savemat(self):
        with tempdir() as temp_dir:
            path = Path(temp_dir) / 'data.mat'
            scipy.io.savemat(path, {'data': self.data})
            assert path.is_file()

    def test_loadmat(self):
        # Save data with string path, load with pathlib.Path
        with tempdir() as temp_dir:
            path = Path(temp_dir) / 'data.mat'
            scipy.io.savemat(str(path), {'data': self.data})

            mat_contents = scipy.io.loadmat(path)
            assert (mat_contents['data'] == self.data).all()

    def test_whosmat(self):
        # Save data with string path, load with pathlib.Path
        with tempdir() as temp_dir:
            path = Path(temp_dir) / 'data.mat'
            scipy.io.savemat(str(path), {'data': self.data})

            contents = scipy.io.whosmat(path)
            assert contents[0] == ('data', (1, 5), 'int64')

    def test_readsav(self):
        path = Path(__file__).parent / 'data/scalar_string.sav'
        scipy.io.readsav(path)

    def test_hb_read(self):
        # Save data with string path, load with pathlib.Path
        with tempdir() as temp_dir:
            data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
            path = Path(temp_dir) / 'data.hb'
            scipy.io.harwell_boeing.hb_write(str(path), data)

            data_new = scipy.io.harwell_boeing.hb_read(path)
            assert (data_new != data).nnz == 0

    def test_hb_write(self):
        with tempdir() as temp_dir:
            data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
            path = Path(temp_dir) / 'data.hb'
            scipy.io.harwell_boeing.hb_write(path, data)
            assert path.is_file()

    def test_mmio_read(self):
        # Save data with string path, load with pathlib.Path
        with tempdir() as temp_dir:
            data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
            path = Path(temp_dir) / 'data.mtx'
            scipy.io.mmwrite(str(path), data)

            data_new = scipy.io.mmread(path)
            assert (data_new != data).nnz == 0

    def test_mmio_write(self):
        with tempdir() as temp_dir:
            data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
            path = Path(temp_dir) / 'data.mtx'
            scipy.io.mmwrite(path, data)

    def test_netcdf_file(self):
        path = Path(__file__).parent / 'data/example_1.nc'
        scipy.io.netcdf.netcdf_file(path)

    def test_wavfile_read(self):
        path = Path(__file__).parent / 'data/test-8000Hz-le-2ch-1byteu.wav'
        scipy.io.wavfile.read(path)

    def test_wavfile_write(self):
        # Read from str path, write to Path
        input_path = Path(__file__).parent / 'data/test-8000Hz-le-2ch-1byteu.wav'
        rate, data = scipy.io.wavfile.read(str(input_path))

        with tempdir() as temp_dir:
            output_path = Path(temp_dir) / input_path.name
            scipy.io.wavfile.write(output_path, rate, data)
