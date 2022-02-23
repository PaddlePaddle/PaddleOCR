# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import warnings

from cryptography import utils, x509


# This exists for pyOpenSSL compatibility and SHOULD NOT BE USED
# WE WILL REMOVE THIS VERY SOON.
def _Certificate(backend, x509) -> x509.Certificate:  # noqa: N802
    warnings.warn(
        "This version of cryptography contains a temporary pyOpenSSL "
        "fallback path. Upgrade pyOpenSSL now.",
        utils.DeprecatedIn35,
    )
    return backend._ossl2cert(x509)


# This exists for pyOpenSSL compatibility and SHOULD NOT BE USED
# WE WILL REMOVE THIS VERY SOON.
def _CertificateSigningRequest(  # noqa: N802
    backend, x509_req
) -> x509.CertificateSigningRequest:
    warnings.warn(
        "This version of cryptography contains a temporary pyOpenSSL "
        "fallback path. Upgrade pyOpenSSL now.",
        utils.DeprecatedIn35,
    )
    return backend._ossl2csr(x509_req)


# This exists for pyOpenSSL compatibility and SHOULD NOT BE USED
# WE WILL REMOVE THIS VERY SOON.
def _CertificateRevocationList(  # noqa: N802
    backend, x509_crl
) -> x509.CertificateRevocationList:
    warnings.warn(
        "This version of cryptography contains a temporary pyOpenSSL "
        "fallback path. Upgrade pyOpenSSL now.",
        utils.DeprecatedIn35,
    )
    return backend._ossl2crl(x509_crl)
