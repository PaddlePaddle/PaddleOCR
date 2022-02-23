import h5py
from h5py import h5f, h5p

from .common import ut, TestCase

@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 8, 9), 'file image operations require HDF5 >= 1.8.9')
class TestFileImage(TestCase):
    def test_load_from_image(self):
        from binascii import a2b_base64
        from zlib import decompress

        compressed_image = 'eJzr9HBx4+WS4mIAAQ4OBhYGAQZk8B8KKjhQ+TD5BCjNCKU7oPQKJpg4I1hOAiouCDUfXV1IkKsrSPV/NACzx4AFQnMwjIKRCDxcHQNAdASUD0ulJ5hQ1ZWkFpeAaFh69KDQXkYGNohZjDA+JCUzMkIEmKHqELQAWKkAByytOoBJViAPJM7ExATWyAE0B8RgZkyAJmlYDoEAIahukJoNU6+HMTA0UOgT6oBgP38XUI6G5UMFZrzKR8EoGAUjGMDKYVgxDSsuAHcfMK8='

        image = decompress(a2b_base64(compressed_image))

        fapl = h5p.create(h5py.h5p.FILE_ACCESS)
        fapl.set_fapl_core()
        fapl.set_file_image(image)

        fid = h5f.open(self.mktemp().encode(), h5py.h5f.ACC_RDONLY, fapl=fapl)
        f = h5py.File(fid)

        self.assertTrue('test' in f)

    def test_open_from_image(self):
        from binascii import a2b_base64
        from zlib import decompress

        compressed_image = 'eJzr9HBx4+WS4mIAAQ4OBhYGAQZk8B8KKjhQ+TD5BCjNCKU7oPQKJpg4I1hOAiouCDUfXV1IkKsrSPV/NACzx4AFQnMwjIKRCDxcHQNAdASUD0ulJ5hQ1ZWkFpeAaFh69KDQXkYGNohZjDA+JCUzMkIEmKHqELQAWKkAByytOoBJViAPJM7ExATWyAE0B8RgZkyAJmlYDoEAIahukJoNU6+HMTA0UOgT6oBgP38XUI6G5UMFZrzKR8EoGAUjGMDKYVgxDSsuAHcfMK8='

        image = decompress(a2b_base64(compressed_image))

        fid = h5f.open_file_image(image)
        f = h5py.File(fid)

        self.assertTrue('test' in f)
