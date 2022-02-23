from __future__ import annotations
from typing import List, Tuple, Callable, Optional

import pytest
from itertools import product
from numpy.testing import assert_allclose, suppress_warnings
from scipy import special
from scipy.special import cython_special


bint_points = [True, False]
int_points = [-10, -1, 1, 10]
real_points = [-10.0, -1.0, 1.0, 10.0]
complex_points = [complex(*tup) for tup in product(real_points, repeat=2)]


CYTHON_SIGNATURE_MAP = {
    'b': 'bint',
    'f': 'float',
    'd': 'double',
    'g': 'long double',
    'F': 'float complex',
    'D': 'double complex',
    'G': 'long double complex',
    'i': 'int',
    'l': 'long'
}


TEST_POINTS = {
    'b': bint_points,
    'f': real_points,
    'd': real_points,
    'g': real_points,
    'F': complex_points,
    'D': complex_points,
    'G': complex_points,
    'i': int_points,
    'l': int_points,
}


PARAMS: List[Tuple[Callable, Callable, Tuple[str, ...], Optional[str]]] = [
    (special.agm, cython_special.agm, ('dd',), None),
    (special.airy, cython_special._airy_pywrap, ('d', 'D'), None),
    (special.airye, cython_special._airye_pywrap, ('d', 'D'), None),
    (special.bdtr, cython_special.bdtr, ('dld', 'ddd'), None),
    (special.bdtrc, cython_special.bdtrc, ('dld', 'ddd'), None),
    (special.bdtri, cython_special.bdtri, ('dld', 'ddd'), None),
    (special.bdtrik, cython_special.bdtrik, ('ddd',), None),
    (special.bdtrin, cython_special.bdtrin, ('ddd',), None),
    (special.bei, cython_special.bei, ('d',), None),
    (special.beip, cython_special.beip, ('d',), None),
    (special.ber, cython_special.ber, ('d',), None),
    (special.berp, cython_special.berp, ('d',), None),
    (special.besselpoly, cython_special.besselpoly, ('ddd',), None),
    (special.beta, cython_special.beta, ('dd',), None),
    (special.betainc, cython_special.betainc, ('ddd',), None),
    (special.betaincinv, cython_special.betaincinv, ('ddd',), None),
    (special.betaln, cython_special.betaln, ('dd',), None),
    (special.binom, cython_special.binom, ('dd',), None),
    (special.boxcox, cython_special.boxcox, ('dd',), None),
    (special.boxcox1p, cython_special.boxcox1p, ('dd',), None),
    (special.btdtr, cython_special.btdtr, ('ddd',), None),
    (special.btdtri, cython_special.btdtri, ('ddd',), None),
    (special.btdtria, cython_special.btdtria, ('ddd',), None),
    (special.btdtrib, cython_special.btdtrib, ('ddd',), None),
    (special.cbrt, cython_special.cbrt, ('d',), None),
    (special.chdtr, cython_special.chdtr, ('dd',), None),
    (special.chdtrc, cython_special.chdtrc, ('dd',), None),
    (special.chdtri, cython_special.chdtri, ('dd',), None),
    (special.chdtriv, cython_special.chdtriv, ('dd',), None),
    (special.chndtr, cython_special.chndtr, ('ddd',), None),
    (special.chndtridf, cython_special.chndtridf, ('ddd',), None),
    (special.chndtrinc, cython_special.chndtrinc, ('ddd',), None),
    (special.chndtrix, cython_special.chndtrix, ('ddd',), None),
    (special.cosdg, cython_special.cosdg, ('d',), None),
    (special.cosm1, cython_special.cosm1, ('d',), None),
    (special.cotdg, cython_special.cotdg, ('d',), None),
    (special.dawsn, cython_special.dawsn, ('d', 'D'), None),
    (special.ellipe, cython_special.ellipe, ('d',), None),
    (special.ellipeinc, cython_special.ellipeinc, ('dd',), None),
    (special.ellipj, cython_special._ellipj_pywrap, ('dd',), None),
    (special.ellipkinc, cython_special.ellipkinc, ('dd',), None),
    (special.ellipkm1, cython_special.ellipkm1, ('d',), None),
    (special.ellipk, cython_special.ellipk, ('d',), None),
    (special.entr, cython_special.entr, ('d',), None),
    (special.erf, cython_special.erf, ('d', 'D'), None),
    (special.erfc, cython_special.erfc, ('d', 'D'), None),
    (special.erfcx, cython_special.erfcx, ('d', 'D'), None),
    (special.erfi, cython_special.erfi, ('d', 'D'), None),
    (special.erfinv, cython_special.erfinv, ('d',), None),
    (special.erfcinv, cython_special.erfcinv, ('d',), None),
    (special.eval_chebyc, cython_special.eval_chebyc, ('dd', 'dD', 'ld'), None),
    (special.eval_chebys, cython_special.eval_chebys, ('dd', 'dD', 'ld'),
     'd and l differ for negative int'),
    (special.eval_chebyt, cython_special.eval_chebyt, ('dd', 'dD', 'ld'),
     'd and l differ for negative int'),
    (special.eval_chebyu, cython_special.eval_chebyu, ('dd', 'dD', 'ld'),
     'd and l differ for negative int'),
    (special.eval_gegenbauer, cython_special.eval_gegenbauer, ('ddd', 'ddD', 'ldd'),
     'd and l differ for negative int'),
    (special.eval_genlaguerre, cython_special.eval_genlaguerre, ('ddd', 'ddD', 'ldd'),
     'd and l differ for negative int'),
    (special.eval_hermite, cython_special.eval_hermite, ('ld',), None),
    (special.eval_hermitenorm, cython_special.eval_hermitenorm, ('ld',), None),
    (special.eval_jacobi, cython_special.eval_jacobi, ('dddd', 'dddD', 'lddd'),
     'd and l differ for negative int'),
    (special.eval_laguerre, cython_special.eval_laguerre, ('dd', 'dD', 'ld'),
     'd and l differ for negative int'),
    (special.eval_legendre, cython_special.eval_legendre, ('dd', 'dD', 'ld'), None),
    (special.eval_sh_chebyt, cython_special.eval_sh_chebyt, ('dd', 'dD', 'ld'), None),
    (special.eval_sh_chebyu, cython_special.eval_sh_chebyu, ('dd', 'dD', 'ld'),
     'd and l differ for negative int'),
    (special.eval_sh_jacobi, cython_special.eval_sh_jacobi, ('dddd', 'dddD', 'lddd'),
     'd and l differ for negative int'),
    (special.eval_sh_legendre, cython_special.eval_sh_legendre, ('dd', 'dD', 'ld'), None),
    (special.exp1, cython_special.exp1, ('d', 'D'), None),
    (special.exp10, cython_special.exp10, ('d',), None),
    (special.exp2, cython_special.exp2, ('d',), None),
    (special.expi, cython_special.expi, ('d', 'D'), None),
    (special.expit, cython_special.expit, ('f', 'd', 'g'), None),
    (special.expm1, cython_special.expm1, ('d', 'D'), None),
    (special.expn, cython_special.expn, ('ld', 'dd'), None),
    (special.exprel, cython_special.exprel, ('d',), None),
    (special.fdtr, cython_special.fdtr, ('ddd',), None),
    (special.fdtrc, cython_special.fdtrc, ('ddd',), None),
    (special.fdtri, cython_special.fdtri, ('ddd',), None),
    (special.fdtridfd, cython_special.fdtridfd, ('ddd',), None),
    (special.fresnel, cython_special._fresnel_pywrap, ('d', 'D'), None),
    (special.gamma, cython_special.gamma, ('d', 'D'), None),
    (special.gammainc, cython_special.gammainc, ('dd',), None),
    (special.gammaincc, cython_special.gammaincc, ('dd',), None),
    (special.gammainccinv, cython_special.gammainccinv, ('dd',), None),
    (special.gammaincinv, cython_special.gammaincinv, ('dd',), None),
    (special.gammaln, cython_special.gammaln, ('d',), None),
    (special.gammasgn, cython_special.gammasgn, ('d',), None),
    (special.gdtr, cython_special.gdtr, ('ddd',), None),
    (special.gdtrc, cython_special.gdtrc, ('ddd',), None),
    (special.gdtria, cython_special.gdtria, ('ddd',), None),
    (special.gdtrib, cython_special.gdtrib, ('ddd',), None),
    (special.gdtrix, cython_special.gdtrix, ('ddd',), None),
    (special.hankel1, cython_special.hankel1, ('dD',), None),
    (special.hankel1e, cython_special.hankel1e, ('dD',), None),
    (special.hankel2, cython_special.hankel2, ('dD',), None),
    (special.hankel2e, cython_special.hankel2e, ('dD',), None),
    (special.huber, cython_special.huber, ('dd',), None),
    (special.hyp0f1, cython_special.hyp0f1, ('dd', 'dD'), None),
    (special.hyp1f1, cython_special.hyp1f1, ('ddd', 'ddD'), None),
    (special.hyp2f1, cython_special.hyp2f1, ('dddd', 'dddD'), None),
    (special.hyperu, cython_special.hyperu, ('ddd',), None),
    (special.i0, cython_special.i0, ('d',), None),
    (special.i0e, cython_special.i0e, ('d',), None),
    (special.i1, cython_special.i1, ('d',), None),
    (special.i1e, cython_special.i1e, ('d',), None),
    (special.inv_boxcox, cython_special.inv_boxcox, ('dd',), None),
    (special.inv_boxcox1p, cython_special.inv_boxcox1p, ('dd',), None),
    (special.it2i0k0, cython_special._it2i0k0_pywrap, ('d',), None),
    (special.it2j0y0, cython_special._it2j0y0_pywrap, ('d',), None),
    (special.it2struve0, cython_special.it2struve0, ('d',), None),
    (special.itairy, cython_special._itairy_pywrap, ('d',), None),
    (special.iti0k0, cython_special._iti0k0_pywrap, ('d',), None),
    (special.itj0y0, cython_special._itj0y0_pywrap, ('d',), None),
    (special.itmodstruve0, cython_special.itmodstruve0, ('d',), None),
    (special.itstruve0, cython_special.itstruve0, ('d',), None),
    (special.iv, cython_special.iv, ('dd', 'dD'), None),
    (special.ive, cython_special.ive, ('dd', 'dD'), None),
    (special.j0, cython_special.j0, ('d',), None),
    (special.j1, cython_special.j1, ('d',), None),
    (special.jv, cython_special.jv, ('dd', 'dD'), None),
    (special.jve, cython_special.jve, ('dd', 'dD'), None),
    (special.k0, cython_special.k0, ('d',), None),
    (special.k0e, cython_special.k0e, ('d',), None),
    (special.k1, cython_special.k1, ('d',), None),
    (special.k1e, cython_special.k1e, ('d',), None),
    (special.kei, cython_special.kei, ('d',), None),
    (special.keip, cython_special.keip, ('d',), None),
    (special.kelvin, cython_special._kelvin_pywrap, ('d',), None),
    (special.ker, cython_special.ker, ('d',), None),
    (special.kerp, cython_special.kerp, ('d',), None),
    (special.kl_div, cython_special.kl_div, ('dd',), None),
    (special.kn, cython_special.kn, ('ld', 'dd'), None),
    (special.kolmogi, cython_special.kolmogi, ('d',), None),
    (special.kolmogorov, cython_special.kolmogorov, ('d',), None),
    (special.kv, cython_special.kv, ('dd', 'dD'), None),
    (special.kve, cython_special.kve, ('dd', 'dD'), None),
    (special.log1p, cython_special.log1p, ('d', 'D'), None),
    (special.log_ndtr, cython_special.log_ndtr, ('d', 'D'), None),
    (special.ndtri_exp, cython_special.ndtri_exp, ('d',), None),
    (special.loggamma, cython_special.loggamma, ('D',), None),
    (special.logit, cython_special.logit, ('f', 'd', 'g'), None),
    (special.lpmv, cython_special.lpmv, ('ddd',), None),
    (special.mathieu_a, cython_special.mathieu_a, ('dd',), None),
    (special.mathieu_b, cython_special.mathieu_b, ('dd',), None),
    (special.mathieu_cem, cython_special._mathieu_cem_pywrap, ('ddd',), None),
    (special.mathieu_modcem1, cython_special._mathieu_modcem1_pywrap, ('ddd',), None),
    (special.mathieu_modcem2, cython_special._mathieu_modcem2_pywrap, ('ddd',), None),
    (special.mathieu_modsem1, cython_special._mathieu_modsem1_pywrap, ('ddd',), None),
    (special.mathieu_modsem2, cython_special._mathieu_modsem2_pywrap, ('ddd',), None),
    (special.mathieu_sem, cython_special._mathieu_sem_pywrap, ('ddd',), None),
    (special.modfresnelm, cython_special._modfresnelm_pywrap, ('d',), None),
    (special.modfresnelp, cython_special._modfresnelp_pywrap, ('d',), None),
    (special.modstruve, cython_special.modstruve, ('dd',), None),
    (special.nbdtr, cython_special.nbdtr, ('lld', 'ddd'), None),
    (special.nbdtrc, cython_special.nbdtrc, ('lld', 'ddd'), None),
    (special.nbdtri, cython_special.nbdtri, ('lld', 'ddd'), None),
    (special.nbdtrik, cython_special.nbdtrik, ('ddd',), None),
    (special.nbdtrin, cython_special.nbdtrin, ('ddd',), None),
    (special.ncfdtr, cython_special.ncfdtr, ('dddd',), None),
    (special.ncfdtri, cython_special.ncfdtri, ('dddd',), None),
    (special.ncfdtridfd, cython_special.ncfdtridfd, ('dddd',), None),
    (special.ncfdtridfn, cython_special.ncfdtridfn, ('dddd',), None),
    (special.ncfdtrinc, cython_special.ncfdtrinc, ('dddd',), None),
    (special.nctdtr, cython_special.nctdtr, ('ddd',), None),
    (special.nctdtridf, cython_special.nctdtridf, ('ddd',), None),
    (special.nctdtrinc, cython_special.nctdtrinc, ('ddd',), None),
    (special.nctdtrit, cython_special.nctdtrit, ('ddd',), None),
    (special.ndtr, cython_special.ndtr, ('d', 'D'), None),
    (special.ndtri, cython_special.ndtri, ('d',), None),
    (special.nrdtrimn, cython_special.nrdtrimn, ('ddd',), None),
    (special.nrdtrisd, cython_special.nrdtrisd, ('ddd',), None),
    (special.obl_ang1, cython_special._obl_ang1_pywrap, ('dddd',), None),
    (special.obl_ang1_cv, cython_special._obl_ang1_cv_pywrap, ('ddddd',), None),
    (special.obl_cv, cython_special.obl_cv, ('ddd',), None),
    (special.obl_rad1, cython_special._obl_rad1_pywrap, ('dddd',), "see gh-6211"),
    (special.obl_rad1_cv, cython_special._obl_rad1_cv_pywrap, ('ddddd',), "see gh-6211"),
    (special.obl_rad2, cython_special._obl_rad2_pywrap, ('dddd',), "see gh-6211"),
    (special.obl_rad2_cv, cython_special._obl_rad2_cv_pywrap, ('ddddd',), "see gh-6211"),
    (special.pbdv, cython_special._pbdv_pywrap, ('dd',), None),
    (special.pbvv, cython_special._pbvv_pywrap, ('dd',), None),
    (special.pbwa, cython_special._pbwa_pywrap, ('dd',), None),
    (special.pdtr, cython_special.pdtr, ('dd', 'dd'), None),
    (special.pdtrc, cython_special.pdtrc, ('dd', 'dd'), None),
    (special.pdtri, cython_special.pdtri, ('ld', 'dd'), None),
    (special.pdtrik, cython_special.pdtrik, ('dd',), None),
    (special.poch, cython_special.poch, ('dd',), None),
    (special.pro_ang1, cython_special._pro_ang1_pywrap, ('dddd',), None),
    (special.pro_ang1_cv, cython_special._pro_ang1_cv_pywrap, ('ddddd',), None),
    (special.pro_cv, cython_special.pro_cv, ('ddd',), None),
    (special.pro_rad1, cython_special._pro_rad1_pywrap, ('dddd',), "see gh-6211"),
    (special.pro_rad1_cv, cython_special._pro_rad1_cv_pywrap, ('ddddd',), "see gh-6211"),
    (special.pro_rad2, cython_special._pro_rad2_pywrap, ('dddd',), "see gh-6211"),
    (special.pro_rad2_cv, cython_special._pro_rad2_cv_pywrap, ('ddddd',), "see gh-6211"),
    (special.pseudo_huber, cython_special.pseudo_huber, ('dd',), None),
    (special.psi, cython_special.psi, ('d', 'D'), None),
    (special.radian, cython_special.radian, ('ddd',), None),
    (special.rel_entr, cython_special.rel_entr, ('dd',), None),
    (special.rgamma, cython_special.rgamma, ('d', 'D'), None),
    (special.round, cython_special.round, ('d',), None),
    (special.spherical_jn, cython_special.spherical_jn, ('ld', 'ldb', 'lD', 'lDb'), None),
    (special.spherical_yn, cython_special.spherical_yn, ('ld', 'ldb', 'lD', 'lDb'), None),
    (special.spherical_in, cython_special.spherical_in, ('ld', 'ldb', 'lD', 'lDb'), None),
    (special.spherical_kn, cython_special.spherical_kn, ('ld', 'ldb', 'lD', 'lDb'), None),
    (special.shichi, cython_special._shichi_pywrap, ('d', 'D'), None),
    (special.sici, cython_special._sici_pywrap, ('d', 'D'), None),
    (special.sindg, cython_special.sindg, ('d',), None),
    (special.smirnov, cython_special.smirnov, ('ld', 'dd'), None),
    (special.smirnovi, cython_special.smirnovi, ('ld', 'dd'), None),
    (special.spence, cython_special.spence, ('d', 'D'), None),
    (special.sph_harm, cython_special.sph_harm, ('lldd', 'dddd'), None),
    (special.stdtr, cython_special.stdtr, ('dd',), None),
    (special.stdtridf, cython_special.stdtridf, ('dd',), None),
    (special.stdtrit, cython_special.stdtrit, ('dd',), None),
    (special.struve, cython_special.struve, ('dd',), None),
    (special.tandg, cython_special.tandg, ('d',), None),
    (special.tklmbda, cython_special.tklmbda, ('dd',), None),
    (special.voigt_profile, cython_special.voigt_profile, ('ddd',), None),
    (special.wofz, cython_special.wofz, ('D',), None),
    (special.wright_bessel, cython_special.wright_bessel, ('ddd',), None),
    (special.wrightomega, cython_special.wrightomega, ('D',), None),
    (special.xlog1py, cython_special.xlog1py, ('dd', 'DD'), None),
    (special.xlogy, cython_special.xlogy, ('dd', 'DD'), None),
    (special.y0, cython_special.y0, ('d',), None),
    (special.y1, cython_special.y1, ('d',), None),
    (special.yn, cython_special.yn, ('ld', 'dd'), None),
    (special.yv, cython_special.yv, ('dd', 'dD'), None),
    (special.yve, cython_special.yve, ('dd', 'dD'), None),
    (special.zetac, cython_special.zetac, ('d',), None),
    (special.owens_t, cython_special.owens_t, ('dd',), None)
]


IDS = [x[0].__name__ for x in PARAMS]


def _generate_test_points(typecodes):
    axes = tuple(TEST_POINTS[x] for x in typecodes)
    pts = list(product(*axes))
    return pts


def test_cython_api_completeness():
    # Check that everything is tested
    for name in dir(cython_special):
        func = getattr(cython_special, name)
        if callable(func) and not name.startswith('_'):
            for _, cyfun, _, _ in PARAMS:
                if cyfun is func:
                    break
            else:
                raise RuntimeError(f"{name} missing from tests!")


@pytest.mark.parametrize("param", PARAMS, ids=IDS)
def test_cython_api(param):
    pyfunc, cyfunc, specializations, knownfailure = param
    if knownfailure:
        pytest.xfail(reason=knownfailure)

    # Check which parameters are expected to be fused types
    max_params = max(len(spec) for spec in specializations)
    values = [set() for _ in range(max_params)]
    for typecodes in specializations:
        for j, v in enumerate(typecodes):
            values[j].add(v)
    seen = set()
    is_fused_code = [False] * len(values)
    for j, v in enumerate(values):
        vv = tuple(sorted(v))
        if vv in seen:
            continue
        is_fused_code[j] = (len(v) > 1)
        seen.add(vv)

    # Check results
    for typecodes in specializations:
        # Pick the correct specialized function
        signature = [CYTHON_SIGNATURE_MAP[code]
                     for j, code in enumerate(typecodes)
                     if is_fused_code[j]]

        if signature:
            cy_spec_func = cyfunc[tuple(signature)]
        else:
            signature = None
            cy_spec_func = cyfunc

        # Test it
        pts = _generate_test_points(typecodes)
        for pt in pts:
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning)
                pyval = pyfunc(*pt)
                cyval = cy_spec_func(*pt)
            assert_allclose(cyval, pyval, err_msg="{} {} {}".format(pt, typecodes, signature))
