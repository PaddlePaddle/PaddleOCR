import pytest

from .common import TestCase
from h5py import File


class SampleException(Exception):
    pass

def throwing(name, obj):
    print(name, obj)
    raise SampleException("throwing exception")

class TestVisit(TestCase):
    def test_visit(self):
        fname = self.mktemp()
        fid = File(fname, 'w')
        fid.create_dataset('foo', (100,), dtype='uint8')
        with pytest.raises(SampleException, match='throwing exception'):
            fid.visititems(throwing)
        fid.close()
