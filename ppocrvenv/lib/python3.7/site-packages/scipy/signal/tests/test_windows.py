import pickle

import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose,
                           assert_equal, assert_, assert_array_less,
                           suppress_warnings)
from pytest import raises as assert_raises

from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann


window_funcs = [
    ('boxcar', ()),
    ('triang', ()),
    ('parzen', ()),
    ('bohman', ()),
    ('blackman', ()),
    ('nuttall', ()),
    ('blackmanharris', ()),
    ('flattop', ()),
    ('bartlett', ()),
    ('hanning', ()),
    ('barthann', ()),
    ('hamming', ()),
    ('kaiser', (1,)),
    ('dpss', (2,)),
    ('gaussian', (0.5,)),
    ('general_gaussian', (1.5, 2)),
    ('chebwin', (1,)),
    ('cosine', ()),
    ('hann', ()),
    ('exponential', ()),
    ('taylor', ()),
    ('tukey', (0.5,)),
    ]


class TestBartHann:

    def test_basic(self):
        assert_allclose(windows.barthann(6, sym=True),
                        [0, 0.35857354213752, 0.8794264578624801,
                         0.8794264578624801, 0.3585735421375199, 0])
        assert_allclose(windows.barthann(7),
                        [0, 0.27, 0.73, 1.0, 0.73, 0.27, 0])
        assert_allclose(windows.barthann(6, False),
                        [0, 0.27, 0.73, 1.0, 0.73, 0.27])


class TestBartlett:

    def test_basic(self):
        assert_allclose(windows.bartlett(6), [0, 0.4, 0.8, 0.8, 0.4, 0])
        assert_allclose(windows.bartlett(7), [0, 1/3, 2/3, 1.0, 2/3, 1/3, 0])
        assert_allclose(windows.bartlett(6, False),
                        [0, 1/3, 2/3, 1.0, 2/3, 1/3])


class TestBlackman:

    def test_basic(self):
        assert_allclose(windows.blackman(6, sym=False),
                        [0, 0.13, 0.63, 1.0, 0.63, 0.13], atol=1e-14)
        assert_allclose(windows.blackman(7, sym=False),
                        [0, 0.09045342435412804, 0.4591829575459636,
                         0.9203636180999081, 0.9203636180999081,
                         0.4591829575459636, 0.09045342435412804], atol=1e-8)
        assert_allclose(windows.blackman(6),
                        [0, 0.2007701432625305, 0.8492298567374694,
                         0.8492298567374694, 0.2007701432625305, 0],
                        atol=1e-14)
        assert_allclose(windows.blackman(7, True),
                        [0, 0.13, 0.63, 1.0, 0.63, 0.13, 0], atol=1e-14)


class TestBlackmanHarris:

    def test_basic(self):
        assert_allclose(windows.blackmanharris(6, False),
                        [6.0e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645])
        assert_allclose(windows.blackmanharris(7, sym=False),
                        [6.0e-05, 0.03339172347815117, 0.332833504298565,
                         0.8893697722232837, 0.8893697722232838,
                         0.3328335042985652, 0.03339172347815122])
        assert_allclose(windows.blackmanharris(6),
                        [6.0e-05, 0.1030114893456638, 0.7938335106543362,
                         0.7938335106543364, 0.1030114893456638, 6.0e-05])
        assert_allclose(windows.blackmanharris(7, sym=True),
                        [6.0e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645,
                         6.0e-05])


class TestTaylor:

    def test_normalized(self):
        """Tests windows of small length that are normalized to 1. See the
        documentation for the Taylor window for more information on
        normalization.
        """
        assert_allclose(windows.taylor(1, 2, 15), 1.0)
        assert_allclose(
            windows.taylor(5, 2, 15),
            np.array([0.75803341, 0.90757699, 1.0, 0.90757699, 0.75803341])
        )
        assert_allclose(
            windows.taylor(6, 2, 15),
            np.array([
                0.7504082, 0.86624416, 0.98208011, 0.98208011, 0.86624416,
                0.7504082
            ])
        )

    def test_non_normalized(self):
        """Test windows of small length that are not normalized to 1. See
        the documentation for the Taylor window for more information on
        normalization.
        """
        assert_allclose(
            windows.taylor(5, 2, 15, norm=False),
            np.array([
                0.87508054, 1.04771499, 1.15440894, 1.04771499, 0.87508054
            ])
        )
        assert_allclose(
            windows.taylor(6, 2, 15, norm=False),
            np.array([
                0.86627793, 1.0, 1.13372207, 1.13372207, 1.0, 0.86627793
            ])
        )

    def test_correctness(self):
        """This test ensures the correctness of the implemented Taylor
        Windowing function. A Taylor Window of 1024 points is created, its FFT
        is taken, and the Peak Sidelobe Level (PSLL) and 3dB and 18dB bandwidth
        are found and checked.

        A publication from Sandia National Laboratories was used as reference
        for the correctness values [1]_.

        References
        -----
        .. [1] Armin Doerry, "Catalog of Window Taper Functions for
               Sidelobe Control", 2017.
               https://www.researchgate.net/profile/Armin_Doerry/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control/links/58f92cb2a6fdccb121c9d54d/Catalog-of-Window-Taper-Functions-for-Sidelobe-Control.pdf
        """
        M_win = 1024
        N_fft = 131072
        # Set norm=False for correctness as the values obtained from the
        # scientific publication do not normalize the values. Normalizing
        # changes the sidelobe level from the desired value.
        w = windows.taylor(M_win, nbar=4, sll=35, norm=False, sym=False)
        f = fft(w, N_fft)
        spec = 20 * np.log10(np.abs(f / np.amax(f)))

        first_zero = np.argmax(np.diff(spec) > 0)

        PSLL = np.amax(spec[first_zero:-first_zero])

        BW_3dB = 2*np.argmax(spec <= -3.0102999566398121) / N_fft * M_win
        BW_18dB = 2*np.argmax(spec <= -18.061799739838872) / N_fft * M_win

        assert_allclose(PSLL, -35.1672, atol=1)
        assert_allclose(BW_3dB, 1.1822, atol=0.1)
        assert_allclose(BW_18dB, 2.6112, atol=0.1)


class TestBohman:

    def test_basic(self):
        assert_allclose(windows.bohman(6),
                        [0, 0.1791238937062839, 0.8343114522576858,
                         0.8343114522576858, 0.1791238937062838, 0])
        assert_allclose(windows.bohman(7, sym=True),
                        [0, 0.1089977810442293, 0.6089977810442293, 1.0,
                         0.6089977810442295, 0.1089977810442293, 0])
        assert_allclose(windows.bohman(6, False),
                        [0, 0.1089977810442293, 0.6089977810442293, 1.0,
                         0.6089977810442295, 0.1089977810442293])


class TestBoxcar:

    def test_basic(self):
        assert_allclose(windows.boxcar(6), [1, 1, 1, 1, 1, 1])
        assert_allclose(windows.boxcar(7), [1, 1, 1, 1, 1, 1, 1])
        assert_allclose(windows.boxcar(6, False), [1, 1, 1, 1, 1, 1])


cheb_odd_true = array([0.200938, 0.107729, 0.134941, 0.165348,
                       0.198891, 0.235450, 0.274846, 0.316836,
                       0.361119, 0.407338, 0.455079, 0.503883,
                       0.553248, 0.602637, 0.651489, 0.699227,
                       0.745266, 0.789028, 0.829947, 0.867485,
                       0.901138, 0.930448, 0.955010, 0.974482,
                       0.988591, 0.997138, 1.000000, 0.997138,
                       0.988591, 0.974482, 0.955010, 0.930448,
                       0.901138, 0.867485, 0.829947, 0.789028,
                       0.745266, 0.699227, 0.651489, 0.602637,
                       0.553248, 0.503883, 0.455079, 0.407338,
                       0.361119, 0.316836, 0.274846, 0.235450,
                       0.198891, 0.165348, 0.134941, 0.107729,
                       0.200938])

cheb_even_true = array([0.203894, 0.107279, 0.133904,
                        0.163608, 0.196338, 0.231986,
                        0.270385, 0.311313, 0.354493,
                        0.399594, 0.446233, 0.493983,
                        0.542378, 0.590916, 0.639071,
                        0.686302, 0.732055, 0.775783,
                        0.816944, 0.855021, 0.889525,
                        0.920006, 0.946060, 0.967339,
                        0.983557, 0.994494, 1.000000,
                        1.000000, 0.994494, 0.983557,
                        0.967339, 0.946060, 0.920006,
                        0.889525, 0.855021, 0.816944,
                        0.775783, 0.732055, 0.686302,
                        0.639071, 0.590916, 0.542378,
                        0.493983, 0.446233, 0.399594,
                        0.354493, 0.311313, 0.270385,
                        0.231986, 0.196338, 0.163608,
                        0.133904, 0.107279, 0.203894])


class TestChebWin:

    def test_basic(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            assert_allclose(windows.chebwin(6, 100),
                            [0.1046401879356917, 0.5075781475823447, 1.0, 1.0,
                             0.5075781475823447, 0.1046401879356917])
            assert_allclose(windows.chebwin(7, 100),
                            [0.05650405062850233, 0.316608530648474,
                             0.7601208123539079, 1.0, 0.7601208123539079,
                             0.316608530648474, 0.05650405062850233])
            assert_allclose(windows.chebwin(6, 10),
                            [1.0, 0.6071201674458373, 0.6808391469897297,
                             0.6808391469897297, 0.6071201674458373, 1.0])
            assert_allclose(windows.chebwin(7, 10),
                            [1.0, 0.5190521247588651, 0.5864059018130382,
                             0.6101519801307441, 0.5864059018130382,
                             0.5190521247588651, 1.0])
            assert_allclose(windows.chebwin(6, 10, False),
                            [1.0, 0.5190521247588651, 0.5864059018130382,
                             0.6101519801307441, 0.5864059018130382,
                             0.5190521247588651])

    def test_cheb_odd_high_attenuation(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            cheb_odd = windows.chebwin(53, at=-40)
        assert_array_almost_equal(cheb_odd, cheb_odd_true, decimal=4)

    def test_cheb_even_high_attenuation(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            cheb_even = windows.chebwin(54, at=40)
        assert_array_almost_equal(cheb_even, cheb_even_true, decimal=4)

    def test_cheb_odd_low_attenuation(self):
        cheb_odd_low_at_true = array([1.000000, 0.519052, 0.586405,
                                      0.610151, 0.586405, 0.519052,
                                      1.000000])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            cheb_odd = windows.chebwin(7, at=10)
        assert_array_almost_equal(cheb_odd, cheb_odd_low_at_true, decimal=4)

    def test_cheb_even_low_attenuation(self):
        cheb_even_low_at_true = array([1.000000, 0.451924, 0.51027,
                                       0.541338, 0.541338, 0.51027,
                                       0.451924, 1.000000])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            cheb_even = windows.chebwin(8, at=-10)
        assert_array_almost_equal(cheb_even, cheb_even_low_at_true, decimal=4)


exponential_data = {
    (4, None, 0.2, False):
        array([4.53999297624848542e-05,
               6.73794699908546700e-03, 1.00000000000000000e+00,
               6.73794699908546700e-03]),
    (4, None, 0.2, True): array([0.00055308437014783, 0.0820849986238988,
                                 0.0820849986238988, 0.00055308437014783]),
    (4, None, 1.0, False): array([0.1353352832366127, 0.36787944117144233, 1.,
                                  0.36787944117144233]),
    (4, None, 1.0, True): array([0.22313016014842982, 0.60653065971263342,
                                 0.60653065971263342, 0.22313016014842982]),
    (4, 2, 0.2, False):
        array([4.53999297624848542e-05, 6.73794699908546700e-03,
               1.00000000000000000e+00, 6.73794699908546700e-03]),
    (4, 2, 0.2, True): None,
    (4, 2, 1.0, False): array([0.1353352832366127, 0.36787944117144233, 1.,
                               0.36787944117144233]),
    (4, 2, 1.0, True): None,
    (5, None, 0.2, True):
        array([4.53999297624848542e-05,
               6.73794699908546700e-03, 1.00000000000000000e+00,
               6.73794699908546700e-03, 4.53999297624848542e-05]),
    (5, None, 1.0, True): array([0.1353352832366127, 0.36787944117144233, 1.,
                                 0.36787944117144233, 0.1353352832366127]),
    (5, 2, 0.2, True): None,
    (5, 2, 1.0, True): None
}


def test_exponential():
    for k, v in exponential_data.items():
        if v is None:
            assert_raises(ValueError, windows.exponential, *k)
        else:
            win = windows.exponential(*k)
            assert_allclose(win, v, rtol=1e-14)


class TestFlatTop:

    def test_basic(self):
        assert_allclose(windows.flattop(6, sym=False),
                        [-0.000421051, -0.051263156, 0.19821053, 1.0,
                         0.19821053, -0.051263156])
        assert_allclose(windows.flattop(7, sym=False),
                        [-0.000421051, -0.03684078115492348,
                         0.01070371671615342, 0.7808739149387698,
                         0.7808739149387698, 0.01070371671615342,
                         -0.03684078115492348])
        assert_allclose(windows.flattop(6),
                        [-0.000421051, -0.0677142520762119, 0.6068721525762117,
                         0.6068721525762117, -0.0677142520762119,
                         -0.000421051])
        assert_allclose(windows.flattop(7, True),
                        [-0.000421051, -0.051263156, 0.19821053, 1.0,
                         0.19821053, -0.051263156, -0.000421051])


class TestGaussian:

    def test_basic(self):
        assert_allclose(windows.gaussian(6, 1.0),
                        [0.04393693362340742, 0.3246524673583497,
                         0.8824969025845955, 0.8824969025845955,
                         0.3246524673583497, 0.04393693362340742])
        assert_allclose(windows.gaussian(7, 1.2),
                        [0.04393693362340742, 0.2493522087772962,
                         0.7066482778577162, 1.0, 0.7066482778577162,
                         0.2493522087772962, 0.04393693362340742])
        assert_allclose(windows.gaussian(7, 3),
                        [0.6065306597126334, 0.8007374029168081,
                         0.9459594689067654, 1.0, 0.9459594689067654,
                         0.8007374029168081, 0.6065306597126334])
        assert_allclose(windows.gaussian(6, 3, False),
                        [0.6065306597126334, 0.8007374029168081,
                         0.9459594689067654, 1.0, 0.9459594689067654,
                         0.8007374029168081])


class TestGeneralCosine:

    def test_basic(self):
        assert_allclose(windows.general_cosine(5, [0.5, 0.3, 0.2]),
                        [0.4, 0.3, 1, 0.3, 0.4])
        assert_allclose(windows.general_cosine(4, [0.5, 0.3, 0.2], sym=False),
                        [0.4, 0.3, 1, 0.3])


class TestGeneralHamming:

    def test_basic(self):
        assert_allclose(windows.general_hamming(5, 0.7),
                        [0.4, 0.7, 1.0, 0.7, 0.4])
        assert_allclose(windows.general_hamming(5, 0.75, sym=False),
                        [0.5, 0.6727457514, 0.9522542486,
                         0.9522542486, 0.6727457514])
        assert_allclose(windows.general_hamming(6, 0.75, sym=True),
                        [0.5, 0.6727457514, 0.9522542486,
                        0.9522542486, 0.6727457514, 0.5])


class TestHamming:

    def test_basic(self):
        assert_allclose(windows.hamming(6, False),
                        [0.08, 0.31, 0.77, 1.0, 0.77, 0.31])
        assert_allclose(windows.hamming(7, sym=False),
                        [0.08, 0.2531946911449826, 0.6423596296199047,
                         0.9544456792351128, 0.9544456792351128,
                         0.6423596296199047, 0.2531946911449826])
        assert_allclose(windows.hamming(6),
                        [0.08, 0.3978521825875242, 0.9121478174124757,
                         0.9121478174124757, 0.3978521825875242, 0.08])
        assert_allclose(windows.hamming(7, sym=True),
                        [0.08, 0.31, 0.77, 1.0, 0.77, 0.31, 0.08])


class TestHann:

    def test_basic(self):
        assert_allclose(windows.hann(6, sym=False),
                        [0, 0.25, 0.75, 1.0, 0.75, 0.25])
        assert_allclose(windows.hann(7, sym=False),
                        [0, 0.1882550990706332, 0.6112604669781572,
                         0.9504844339512095, 0.9504844339512095,
                         0.6112604669781572, 0.1882550990706332])
        assert_allclose(windows.hann(6, True),
                        [0, 0.3454915028125263, 0.9045084971874737,
                         0.9045084971874737, 0.3454915028125263, 0])
        assert_allclose(windows.hann(7),
                        [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0])


class TestKaiser:

    def test_basic(self):
        assert_allclose(windows.kaiser(6, 0.5),
                        [0.9403061933191572, 0.9782962393705389,
                         0.9975765035372042, 0.9975765035372042,
                         0.9782962393705389, 0.9403061933191572])
        assert_allclose(windows.kaiser(7, 0.5),
                        [0.9403061933191572, 0.9732402256999829,
                         0.9932754654413773, 1.0, 0.9932754654413773,
                         0.9732402256999829, 0.9403061933191572])
        assert_allclose(windows.kaiser(6, 2.7),
                        [0.2603047507678832, 0.6648106293528054,
                         0.9582099802511439, 0.9582099802511439,
                         0.6648106293528054, 0.2603047507678832])
        assert_allclose(windows.kaiser(7, 2.7),
                        [0.2603047507678832, 0.5985765418119844,
                         0.8868495172060835, 1.0, 0.8868495172060835,
                         0.5985765418119844, 0.2603047507678832])
        assert_allclose(windows.kaiser(6, 2.7, False),
                        [0.2603047507678832, 0.5985765418119844,
                         0.8868495172060835, 1.0, 0.8868495172060835,
                         0.5985765418119844])


class TestNuttall:

    def test_basic(self):
        assert_allclose(windows.nuttall(6, sym=False),
                        [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298,
                         0.0613345])
        assert_allclose(windows.nuttall(7, sym=False),
                        [0.0003628, 0.03777576895352025, 0.3427276199688195,
                         0.8918518610776603, 0.8918518610776603,
                         0.3427276199688196, 0.0377757689535203])
        assert_allclose(windows.nuttall(6),
                        [0.0003628, 0.1105152530498718, 0.7982580969501282,
                         0.7982580969501283, 0.1105152530498719, 0.0003628])
        assert_allclose(windows.nuttall(7, True),
                        [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298,
                         0.0613345, 0.0003628])


class TestParzen:

    def test_basic(self):
        assert_allclose(windows.parzen(6),
                        [0.009259259259259254, 0.25, 0.8611111111111112,
                         0.8611111111111112, 0.25, 0.009259259259259254])
        assert_allclose(windows.parzen(7, sym=True),
                        [0.00583090379008747, 0.1574344023323616,
                         0.6501457725947521, 1.0, 0.6501457725947521,
                         0.1574344023323616, 0.00583090379008747])
        assert_allclose(windows.parzen(6, False),
                        [0.00583090379008747, 0.1574344023323616,
                         0.6501457725947521, 1.0, 0.6501457725947521,
                         0.1574344023323616])


class TestTriang:

    def test_basic(self):

        assert_allclose(windows.triang(6, True),
                        [1/6, 1/2, 5/6, 5/6, 1/2, 1/6])
        assert_allclose(windows.triang(7),
                        [1/4, 1/2, 3/4, 1, 3/4, 1/2, 1/4])
        assert_allclose(windows.triang(6, sym=False),
                        [1/4, 1/2, 3/4, 1, 3/4, 1/2])


tukey_data = {
    (4, 0.5, True): array([0.0, 1.0, 1.0, 0.0]),
    (4, 0.9, True): array([0.0, 0.84312081893436686,
                           0.84312081893436686, 0.0]),
    (4, 1.0, True): array([0.0, 0.75, 0.75, 0.0]),
    (4, 0.5, False): array([0.0, 1.0, 1.0, 1.0]),
    (4, 0.9, False): array([0.0, 0.58682408883346526,
                            1.0, 0.58682408883346526]),
    (4, 1.0, False): array([0.0, 0.5, 1.0, 0.5]),
    (5, 0.0, True): array([1.0, 1.0, 1.0, 1.0, 1.0]),
    (5, 0.8, True): array([0.0, 0.69134171618254492,
                           1.0, 0.69134171618254492, 0.0]),
    (5, 1.0, True): array([0.0, 0.5, 1.0, 0.5, 0.0]),

    (6, 0): [1, 1, 1, 1, 1, 1],
    (7, 0): [1, 1, 1, 1, 1, 1, 1],
    (6, .25): [0, 1, 1, 1, 1, 0],
    (7, .25): [0, 1, 1, 1, 1, 1, 0],
    (6,): [0, 0.9045084971874737, 1.0, 1.0, 0.9045084971874735, 0],
    (7,): [0, 0.75, 1.0, 1.0, 1.0, 0.75, 0],
    (6, .75): [0, 0.5522642316338269, 1.0, 1.0, 0.5522642316338267, 0],
    (7, .75): [0, 0.4131759111665348, 0.9698463103929542, 1.0,
               0.9698463103929542, 0.4131759111665347, 0],
    (6, 1): [0, 0.3454915028125263, 0.9045084971874737, 0.9045084971874737,
             0.3454915028125263, 0],
    (7, 1): [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0],
}


class TestTukey:

    def test_basic(self):
        # Test against hardcoded data
        for k, v in tukey_data.items():
            if v is None:
                assert_raises(ValueError, windows.tukey, *k)
            else:
                win = windows.tukey(*k)
                assert_allclose(win, v, rtol=1e-14)

    def test_extremes(self):
        # Test extremes of alpha correspond to boxcar and hann
        tuk0 = windows.tukey(100, 0)
        box0 = windows.boxcar(100)
        assert_array_almost_equal(tuk0, box0)

        tuk1 = windows.tukey(100, 1)
        han1 = windows.hann(100)
        assert_array_almost_equal(tuk1, han1)


dpss_data = {
    # All values from MATLAB:
    # * taper[1] of (3, 1.4, 3) sign-flipped
    # * taper[3] of (5, 1.5, 5) sign-flipped
    (4, 0.1, 2): ([[0.497943898, 0.502047681, 0.502047681, 0.497943898], [0.670487993, 0.224601537, -0.224601537, -0.670487993]], [0.197961815, 0.002035474]),  # noqa
    (3, 1.4, 3): ([[0.410233151, 0.814504464, 0.410233151], [0.707106781, 0.0, -0.707106781], [0.575941629, -0.580157287, 0.575941629]], [0.999998093, 0.998067480, 0.801934426]),  # noqa
    (5, 1.5, 5): ([[0.1745071052, 0.4956749177, 0.669109327, 0.495674917, 0.174507105], [0.4399493348, 0.553574369, 0.0, -0.553574369, -0.439949334], [0.631452756, 0.073280238, -0.437943884, 0.073280238, 0.631452756], [0.553574369, -0.439949334, 0.0, 0.439949334, -0.553574369], [0.266110290, -0.498935248, 0.600414741, -0.498935248, 0.266110290147157]], [0.999728571, 0.983706916, 0.768457889, 0.234159338, 0.013947282907567]),  # noqa: E501
    (100, 2, 4): ([[0.0030914414, 0.0041266922, 0.005315076, 0.006665149, 0.008184854, 0.0098814158, 0.011761239, 0.013829809, 0.016091597, 0.018549973, 0.02120712, 0.02406396, 0.027120092, 0.030373728, 0.033821651, 0.037459181, 0.041280145, 0.045276872, 0.049440192, 0.053759447, 0.058222524, 0.062815894, 0.067524661, 0.072332638, 0.077222418, 0.082175473, 0.087172252, 0.092192299, 0.097214376, 0.1022166, 0.10717657, 0.11207154, 0.11687856, 0.12157463, 0.12613686, 0.13054266, 0.13476986, 0.13879691, 0.14260302, 0.14616832, 0.14947401, 0.1525025, 0.15523755, 0.15766438, 0.15976981, 0.16154233, 0.16297223, 0.16405162, 0.16477455, 0.16513702, 0.16513702, 0.16477455, 0.16405162, 0.16297223, 0.16154233, 0.15976981, 0.15766438, 0.15523755, 0.1525025, 0.14947401, 0.14616832, 0.14260302, 0.13879691, 0.13476986, 0.13054266, 0.12613686, 0.12157463, 0.11687856, 0.11207154, 0.10717657, 0.1022166, 0.097214376, 0.092192299, 0.087172252, 0.082175473, 0.077222418, 0.072332638, 0.067524661, 0.062815894, 0.058222524, 0.053759447, 0.049440192, 0.045276872, 0.041280145, 0.037459181, 0.033821651, 0.030373728, 0.027120092, 0.02406396, 0.02120712, 0.018549973, 0.016091597, 0.013829809, 0.011761239, 0.0098814158, 0.008184854, 0.006665149, 0.005315076, 0.0041266922, 0.0030914414], [0.018064449, 0.022040342, 0.026325013, 0.030905288, 0.035764398, 0.040881982, 0.046234148, 0.051793558, 0.057529559, 0.063408356, 0.069393216, 0.075444716, 0.081521022, 0.087578202, 0.093570567, 0.099451049, 0.10517159, 0.11068356, 0.11593818, 0.12088699, 0.12548227, 0.12967752, 0.1334279, 0.13669069, 0.13942569, 0.1415957, 0.14316686, 0.14410905, 0.14439626, 0.14400686, 0.14292389, 0.1411353, 0.13863416, 0.13541876, 0.13149274, 0.12686516, 0.12155045, 0.1155684, 0.10894403, 0.10170748, 0.093893752, 0.08554251, 0.076697768, 0.067407559, 0.057723559, 0.04770068, 0.037396627, 0.026871428, 0.016186944, 0.0054063557, -0.0054063557, -0.016186944, -0.026871428, -0.037396627, -0.04770068, -0.057723559, -0.067407559, -0.076697768, -0.08554251, -0.093893752, -0.10170748, -0.10894403, -0.1155684, -0.12155045, -0.12686516, -0.13149274, -0.13541876, -0.13863416, -0.1411353, -0.14292389, -0.14400686, -0.14439626, -0.14410905, -0.14316686, -0.1415957, -0.13942569, -0.13669069, -0.1334279, -0.12967752, -0.12548227, -0.12088699, -0.11593818, -0.11068356, -0.10517159, -0.099451049, -0.093570567, -0.087578202, -0.081521022, -0.075444716, -0.069393216, -0.063408356, -0.057529559, -0.051793558, -0.046234148, -0.040881982, -0.035764398, -0.030905288, -0.026325013, -0.022040342, -0.018064449], [0.064817553, 0.072567801, 0.080292992, 0.087918235, 0.095367076, 0.10256232, 0.10942687, 0.1158846, 0.12186124, 0.12728523, 0.13208858, 0.13620771, 0.13958427, 0.14216587, 0.14390678, 0.14476863, 0.1447209, 0.14374148, 0.14181704, 0.13894336, 0.13512554, 0.13037812, 0.1247251, 0.11819984, 0.11084487, 0.10271159, 0.093859853, 0.084357497, 0.074279719, 0.063708406, 0.052731374, 0.041441525, 0.029935953, 0.018314987, 0.0066811877, -0.0048616765, -0.016209689, -0.027259848, -0.037911124, -0.048065512, -0.05762905, -0.066512804, -0.0746338, -0.081915903, -0.088290621, -0.09369783, -0.098086416, -0.10141482, -0.10365146, -0.10477512, -0.10477512, -0.10365146, -0.10141482, -0.098086416, -0.09369783, -0.088290621, -0.081915903, -0.0746338, -0.066512804, -0.05762905, -0.048065512, -0.037911124, -0.027259848, -0.016209689, -0.0048616765, 0.0066811877, 0.018314987, 0.029935953, 0.041441525, 0.052731374, 0.063708406, 0.074279719, 0.084357497, 0.093859853, 0.10271159, 0.11084487, 0.11819984, 0.1247251, 0.13037812, 0.13512554, 0.13894336, 0.14181704, 0.14374148, 0.1447209, 0.14476863, 0.14390678, 0.14216587, 0.13958427, 0.13620771, 0.13208858, 0.12728523, 0.12186124, 0.1158846, 0.10942687, 0.10256232, 0.095367076, 0.087918235, 0.080292992, 0.072567801, 0.064817553], [0.14985551, 0.15512305, 0.15931467, 0.16236806, 0.16423291, 0.16487165, 0.16426009, 0.1623879, 0.1592589, 0.15489114, 0.14931693, 0.14258255, 0.13474785, 0.1258857, 0.11608124, 0.10543095, 0.094041635, 0.082029213, 0.069517411, 0.056636348, 0.043521028, 0.030309756, 0.017142511, 0.0041592774, -0.0085016282, -0.020705223, -0.032321494, -0.043226982, -0.053306291, -0.062453515, -0.070573544, -0.077583253, -0.083412547, -0.088005244, -0.091319802, -0.093329861, -0.094024602, -0.093408915, -0.091503383, -0.08834406, -0.08398207, -0.078483012, -0.071926192, -0.064403681, -0.056019215, -0.046886954, -0.037130106, -0.026879442, -0.016271713, -0.005448, 0.005448, 0.016271713, 0.026879442, 0.037130106, 0.046886954, 0.056019215, 0.064403681, 0.071926192, 0.078483012, 0.08398207, 0.08834406, 0.091503383, 0.093408915, 0.094024602, 0.093329861, 0.091319802, 0.088005244, 0.083412547, 0.077583253, 0.070573544, 0.062453515, 0.053306291, 0.043226982, 0.032321494, 0.020705223, 0.0085016282, -0.0041592774, -0.017142511, -0.030309756, -0.043521028, -0.056636348, -0.069517411, -0.082029213, -0.094041635, -0.10543095, -0.11608124, -0.1258857, -0.13474785, -0.14258255, -0.14931693, -0.15489114, -0.1592589, -0.1623879, -0.16426009, -0.16487165, -0.16423291, -0.16236806, -0.15931467, -0.15512305, -0.14985551]], [0.999943140, 0.997571533, 0.959465463, 0.721862496]),  # noqa: E501
}


class TestDPSS:

    def test_basic(self):
        # Test against hardcoded data
        for k, v in dpss_data.items():
            win, ratios = windows.dpss(*k, return_ratios=True)
            assert_allclose(win, v[0], atol=1e-7, err_msg=k)
            assert_allclose(ratios, v[1], rtol=1e-5, atol=1e-7, err_msg=k)

    def test_unity(self):
        # Test unity value handling (gh-2221)
        for M in range(1, 21):
            # corrected w/approximation (default)
            win = windows.dpss(M, M / 2.1)
            expected = M % 2  # one for odd, none for even
            assert_equal(np.isclose(win, 1.).sum(), expected,
                         err_msg='%s' % (win,))
            # corrected w/subsample delay (slower)
            win_sub = windows.dpss(M, M / 2.1, norm='subsample')
            if M > 2:
                # @M=2 the subsample doesn't do anything
                assert_equal(np.isclose(win_sub, 1.).sum(), expected,
                             err_msg='%s' % (win_sub,))
                assert_allclose(win, win_sub, rtol=0.03)  # within 3%
            # not the same, l2-norm
            win_2 = windows.dpss(M, M / 2.1, norm=2)
            expected = 1 if M == 1 else 0
            assert_equal(np.isclose(win_2, 1.).sum(), expected,
                         err_msg='%s' % (win_2,))

    def test_extremes(self):
        # Test extremes of alpha
        lam = windows.dpss(31, 6, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.)
        lam = windows.dpss(31, 7, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.)
        lam = windows.dpss(31, 8, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.)

    def test_degenerate(self):
        # Test failures
        assert_raises(ValueError, windows.dpss, 4, 1.5, -1)  # Bad Kmax
        assert_raises(ValueError, windows.dpss, 4, 1.5, -5)
        assert_raises(TypeError, windows.dpss, 4, 1.5, 1.1)
        assert_raises(ValueError, windows.dpss, 3, 1.5, 3)  # NW must be < N/2.
        assert_raises(ValueError, windows.dpss, 3, -1, 3)  # NW must be pos
        assert_raises(ValueError, windows.dpss, 3, 0, 3)
        assert_raises(ValueError, windows.dpss, -1, 1, 3)  # negative M


class TestGetWindow:

    def test_boxcar(self):
        w = windows.get_window('boxcar', 12)
        assert_array_equal(w, np.ones_like(w))

        # window is a tuple of len 1
        w = windows.get_window(('boxcar',), 16)
        assert_array_equal(w, np.ones_like(w))

    def test_cheb_odd(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            w = windows.get_window(('chebwin', -40), 53, fftbins=False)
        assert_array_almost_equal(w, cheb_odd_true, decimal=4)

    def test_cheb_even(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            w = windows.get_window(('chebwin', 40), 54, fftbins=False)
        assert_array_almost_equal(w, cheb_even_true, decimal=4)

    def test_dpss(self):
        win1 = windows.get_window(('dpss', 3), 64, fftbins=False)
        win2 = windows.dpss(64, 3)
        assert_array_almost_equal(win1, win2, decimal=4)

    def test_kaiser_float(self):
        win1 = windows.get_window(7.2, 64)
        win2 = windows.kaiser(64, 7.2, False)
        assert_allclose(win1, win2)

    def test_invalid_inputs(self):
        # Window is not a float, tuple, or string
        assert_raises(ValueError, windows.get_window, set('hann'), 8)

        # Unknown window type error
        assert_raises(ValueError, windows.get_window, 'broken', 4)

    def test_array_as_window(self):
        # github issue 3603
        osfactor = 128
        sig = np.arange(128)

        win = windows.get_window(('kaiser', 8.0), osfactor // 2)
        with assert_raises(ValueError, match='must have the same length'):
            resample(sig, len(sig) * osfactor, window=win)

    def test_general_cosine(self):
        assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4),
                        [0.4, 0.3, 1, 0.3])
        assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4,
                                   fftbins=False),
                        [0.4, 0.55, 0.55, 0.4])

    def test_general_hamming(self):
        assert_allclose(get_window(('general_hamming', 0.7), 5),
                        [0.4, 0.6072949, 0.9427051, 0.9427051, 0.6072949])
        assert_allclose(get_window(('general_hamming', 0.7), 5, fftbins=False),
                        [0.4, 0.7, 1.0, 0.7, 0.4])


def test_windowfunc_basics():
    for window_name, params in window_funcs:
        window = getattr(windows, window_name)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            if window_name in ('hanning',):
                sup.filter(DeprecationWarning)
            # Check symmetry for odd and even lengths
            w1 = window(8, *params, sym=True)
            w2 = window(7, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)

            w1 = window(9, *params, sym=True)
            w2 = window(8, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)

            # Check that functions run and output lengths are correct
            assert_equal(len(window(6, *params, sym=True)), 6)
            assert_equal(len(window(6, *params, sym=False)), 6)
            assert_equal(len(window(7, *params, sym=True)), 7)
            assert_equal(len(window(7, *params, sym=False)), 7)

            # Check invalid lengths
            assert_raises(ValueError, window, 5.5, *params)
            assert_raises(ValueError, window, -7, *params)

            # Check degenerate cases
            assert_array_equal(window(0, *params, sym=True), [])
            assert_array_equal(window(0, *params, sym=False), [])
            assert_array_equal(window(1, *params, sym=True), [1])
            assert_array_equal(window(1, *params, sym=False), [1])

            # Check dtype
            assert_(window(0, *params, sym=True).dtype == 'float')
            assert_(window(0, *params, sym=False).dtype == 'float')
            assert_(window(1, *params, sym=True).dtype == 'float')
            assert_(window(1, *params, sym=False).dtype == 'float')
            assert_(window(6, *params, sym=True).dtype == 'float')
            assert_(window(6, *params, sym=False).dtype == 'float')

            # Check normalization
            assert_array_less(window(10, *params, sym=True), 1.01)
            assert_array_less(window(10, *params, sym=False), 1.01)
            assert_array_less(window(9, *params, sym=True), 1.01)
            assert_array_less(window(9, *params, sym=False), 1.01)

            # Check that DFT-even spectrum is purely real for odd and even
            assert_allclose(fft(window(10, *params, sym=False)).imag,
                            0, atol=1e-14)
            assert_allclose(fft(window(11, *params, sym=False)).imag,
                            0, atol=1e-14)


def test_needs_params():
    for winstr in ['kaiser', 'ksr', 'gaussian', 'gauss', 'gss',
                   'general gaussian', 'general_gaussian',
                   'general gauss', 'general_gauss', 'ggs',
                   'dss', 'dpss', 'general cosine', 'general_cosine',
                   'chebwin', 'cheb', 'general hamming', 'general_hamming',
                   ]:
        assert_raises(ValueError, get_window, winstr, 7)


def test_not_needs_params():
    for winstr in ['barthann',
                   'bartlett',
                   'blackman',
                   'blackmanharris',
                   'bohman',
                   'boxcar',
                   'cosine',
                   'flattop',
                   'hamming',
                   'hanning',
                   'nuttall',
                   'parzen',
                   'taylor',
                   'exponential',
                   'poisson',
                   'tukey',
                   'tuk',
                   'triangle']:
        win = get_window(winstr, 7)
        assert_equal(len(win), 7)


def test_deprecation():
    if dep_hann.__doc__ is not None:  # can be None with `-OO` mode
        assert_('signal.hann is deprecated' in dep_hann.__doc__)
        assert_('deprecated' not in windows.hann.__doc__)


def test_deprecated_pickleable():
    dep_hann2 = pickle.loads(pickle.dumps(dep_hann))
    assert_(dep_hann2 is dep_hann)
