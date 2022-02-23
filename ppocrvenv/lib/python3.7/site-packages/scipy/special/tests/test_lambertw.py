#
# Tests for the lambertw function,
# Adapted from the MPMath tests [1] by Yosef Meller, mellerf@netvision.net.il
# Distributed under the same license as SciPy itself.
#
# [1] mpmath source code, Subversion revision 992
#     http://code.google.com/p/mpmath/source/browse/trunk/mpmath/tests/test_functions2.py?spec=svn994&r=992

import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from scipy.special import lambertw
from numpy import nan, inf, pi, e, isnan, log, r_, array, complex_

from scipy.special._testutils import FuncData


def test_values():
    assert_(isnan(lambertw(nan)))
    assert_equal(lambertw(inf,1).real, inf)
    assert_equal(lambertw(inf,1).imag, 2*pi)
    assert_equal(lambertw(-inf,1).real, inf)
    assert_equal(lambertw(-inf,1).imag, 3*pi)

    assert_equal(lambertw(1.), lambertw(1., 0))

    data = [
        (0,0, 0),
        (0+0j,0, 0),
        (inf,0, inf),
        (0,-1, -inf),
        (0,1, -inf),
        (0,3, -inf),
        (e,0, 1),
        (1,0, 0.567143290409783873),
        (-pi/2,0, 1j*pi/2),
        (-log(2)/2,0, -log(2)),
        (0.25,0, 0.203888354702240164),
        (-0.25,0, -0.357402956181388903),
        (-1./10000,0, -0.000100010001500266719),
        (-0.25,-1, -2.15329236411034965),
        (0.25,-1, -3.00899800997004620-4.07652978899159763j),
        (-0.25,-1, -2.15329236411034965),
        (0.25,1, -3.00899800997004620+4.07652978899159763j),
        (-0.25,1, -3.48973228422959210+7.41405453009603664j),
        (-4,0, 0.67881197132094523+1.91195078174339937j),
        (-4,1, -0.66743107129800988+7.76827456802783084j),
        (-4,-1, 0.67881197132094523-1.91195078174339937j),
        (1000,0, 5.24960285240159623),
        (1000,1, 4.91492239981054535+5.44652615979447070j),
        (1000,-1, 4.91492239981054535-5.44652615979447070j),
        (1000,5, 3.5010625305312892+29.9614548941181328j),
        (3+4j,0, 1.281561806123775878+0.533095222020971071j),
        (-0.4+0.4j,0, -0.10396515323290657+0.61899273315171632j),
        (3+4j,1, -0.11691092896595324+5.61888039871282334j),
        (3+4j,-1, 0.25856740686699742-3.85211668616143559j),
        (-0.5,-1, -0.794023632344689368-0.770111750510379110j),
        (-1./10000,1, -11.82350837248724344+6.80546081842002101j),
        (-1./10000,-1, -11.6671145325663544),
        (-1./10000,-2, -11.82350837248724344-6.80546081842002101j),
        (-1./100000,4, -14.9186890769540539+26.1856750178782046j),
        (-1./100000,5, -15.0931437726379218666+32.5525721210262290086j),
        ((2+1j)/10,0, 0.173704503762911669+0.071781336752835511j),
        ((2+1j)/10,1, -3.21746028349820063+4.56175438896292539j),
        ((2+1j)/10,-1, -3.03781405002993088-3.53946629633505737j),
        ((2+1j)/10,4, -4.6878509692773249+23.8313630697683291j),
        (-(2+1j)/10,0, -0.226933772515757933-0.164986470020154580j),
        (-(2+1j)/10,1, -2.43569517046110001+0.76974067544756289j),
        (-(2+1j)/10,-1, -3.54858738151989450-6.91627921869943589j),
        (-(2+1j)/10,4, -4.5500846928118151+20.6672982215434637j),
        (pi,0, 1.073658194796149172092178407024821347547745350410314531),

        # Former bug in generated branch,
        (-0.5+0.002j,0, -0.78917138132659918344 + 0.76743539379990327749j),
        (-0.5-0.002j,0, -0.78917138132659918344 - 0.76743539379990327749j),
        (-0.448+0.4j,0, -0.11855133765652382241 + 0.66570534313583423116j),
        (-0.448-0.4j,0, -0.11855133765652382241 - 0.66570534313583423116j),
    ]
    data = array(data, dtype=complex_)

    def w(x, y):
        return lambertw(x, y.real.astype(int))
    with np.errstate(all='ignore'):
        FuncData(w, data, (0,1), 2, rtol=1e-10, atol=1e-13).check()


def test_ufunc():
    assert_array_almost_equal(
        lambertw(r_[0., e, 1.]), r_[0., 1., 0.567143290409783873])


def test_lambertw_ufunc_loop_selection():
    # see https://github.com/scipy/scipy/issues/4895
    dt = np.dtype(np.complex128)
    assert_equal(lambertw(0, 0, 0).dtype, dt)
    assert_equal(lambertw([0], 0, 0).dtype, dt)
    assert_equal(lambertw(0, [0], 0).dtype, dt)
    assert_equal(lambertw(0, 0, [0]).dtype, dt)
    assert_equal(lambertw([0], [0], [0]).dtype, dt)
