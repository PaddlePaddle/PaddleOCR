import pytest

import numpy as np
from h5py import File
from .common import TestCase
from .data_files import get_data_file_path


def test_vlen_big_endian():
    with File(get_data_file_path("vlen_string_s390x.h5")) as f:
        assert f.attrs["created_on_s390x"] == 1

        dset = f["DSvariable"]
        assert dset[0] == b"Parting"
        assert dset[1] == b"is such"
        assert dset[2] == b"sweet"
        assert dset[3] == b"sorrow..."

        dset = f["DSLEfloat"]
        assert dset[0] == 3.14
        assert dset[1] == 1.61
        assert dset[2] == 2.71
        assert dset[3] == 2.41
        assert dset[4] == 1.2
        assert dset.dtype == "<f8"

        # Same float values with big endianess
        assert f["DSBEfloat"][0] == 3.14
        assert f["DSBEfloat"].dtype == ">f8"

        assert f["DSLEint"][0] == 1
        assert f["DSLEint"].dtype == "<u8"

        # Same int values with big endianess
        assert f["DSBEint"][0] == 1
        assert f["DSBEint"].dtype == ">i8"


class TestEndianess(TestCase):
    def test_simple_int_be(self):
        fname = self.mktemp()

        arr = np.ndarray(shape=(1,), dtype=">i4", buffer=bytearray([0, 1, 3, 2]))
        be_number = 0 * 256 ** 3 + 1 * 256 ** 2 + 3 * 256 ** 1 + 2 * 256 ** 0

        with File(fname, mode="w") as f:
            f.create_dataset("int", data=arr)

        with File(fname, mode="r") as f:
            assert f["int"][()][0] == be_number
