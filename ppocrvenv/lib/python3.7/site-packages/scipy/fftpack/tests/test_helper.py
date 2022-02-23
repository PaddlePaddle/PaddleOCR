# Created by Pearu Peterson, September 2002

__usage__ = """
Build fftpack:
  python setup_fftpack.py build
Run tests if scipy is installed:
  python -c 'import scipy;scipy.fftpack.test(<level>)'
Run tests if fftpack is not installed:
  python tests/test_helper.py [<level>]
"""

from numpy.testing import assert_array_almost_equal
from scipy.fftpack import fftshift, ifftshift, fftfreq, rfftfreq

from numpy import pi, random

class TestFFTShift:

    def test_definition(self):
        x = [0,1,2,3,4,-4,-3,-2,-1]
        y = [-4,-3,-2,-1,0,1,2,3,4]
        assert_array_almost_equal(fftshift(x),y)
        assert_array_almost_equal(ifftshift(y),x)
        x = [0,1,2,3,4,-5,-4,-3,-2,-1]
        y = [-5,-4,-3,-2,-1,0,1,2,3,4]
        assert_array_almost_equal(fftshift(x),y)
        assert_array_almost_equal(ifftshift(y),x)

    def test_inverse(self):
        for n in [1,4,9,100,211]:
            x = random.random((n,))
            assert_array_almost_equal(ifftshift(fftshift(x)),x)


class TestFFTFreq:

    def test_definition(self):
        x = [0,1,2,3,4,-4,-3,-2,-1]
        assert_array_almost_equal(9*fftfreq(9),x)
        assert_array_almost_equal(9*pi*fftfreq(9,pi),x)
        x = [0,1,2,3,4,-5,-4,-3,-2,-1]
        assert_array_almost_equal(10*fftfreq(10),x)
        assert_array_almost_equal(10*pi*fftfreq(10,pi),x)


class TestRFFTFreq:

    def test_definition(self):
        x = [0,1,1,2,2,3,3,4,4]
        assert_array_almost_equal(9*rfftfreq(9),x)
        assert_array_almost_equal(9*pi*rfftfreq(9,pi),x)
        x = [0,1,1,2,2,3,3,4,4,5]
        assert_array_almost_equal(10*rfftfreq(10),x)
        assert_array_almost_equal(10*pi*rfftfreq(10,pi),x)
