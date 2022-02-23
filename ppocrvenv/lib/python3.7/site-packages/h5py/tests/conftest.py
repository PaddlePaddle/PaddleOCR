import h5py
import pytest


@pytest.fixture()
def writable_file(tmp_path):
    with h5py.File(tmp_path / 'test.h5', 'w') as f:
        yield f
