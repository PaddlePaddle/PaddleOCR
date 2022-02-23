# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import typing

from cryptography import utils
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends.openssl.utils import (
    _calculate_digest_and_algorithm,
    _check_not_prehashed,
    _warn_sign_verify_deprecated,
)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
    AsymmetricSignatureContext,
    AsymmetricVerificationContext,
    dsa,
    utils as asym_utils,
)


def _dsa_sig_sign(backend, private_key, data):
    sig_buf_len = backend._lib.DSA_size(private_key._dsa_cdata)
    sig_buf = backend._ffi.new("unsigned char[]", sig_buf_len)
    buflen = backend._ffi.new("unsigned int *")

    # The first parameter passed to DSA_sign is unused by OpenSSL but
    # must be an integer.
    res = backend._lib.DSA_sign(
        0, data, len(data), sig_buf, buflen, private_key._dsa_cdata
    )
    backend.openssl_assert(res == 1)
    backend.openssl_assert(buflen[0])

    return backend._ffi.buffer(sig_buf)[: buflen[0]]


def _dsa_sig_verify(backend, public_key, signature, data):
    # The first parameter passed to DSA_verify is unused by OpenSSL but
    # must be an integer.
    res = backend._lib.DSA_verify(
        0, data, len(data), signature, len(signature), public_key._dsa_cdata
    )

    if res != 1:
        backend._consume_errors()
        raise InvalidSignature


class _DSAVerificationContext(AsymmetricVerificationContext):
    def __init__(self, backend, public_key, signature, algorithm):
        self._backend = backend
        self._public_key = public_key
        self._signature = signature
        self._algorithm = algorithm

        self._hash_ctx = hashes.Hash(self._algorithm, self._backend)

    def update(self, data: bytes):
        self._hash_ctx.update(data)

    def verify(self) -> None:
        data_to_verify = self._hash_ctx.finalize()

        _dsa_sig_verify(
            self._backend, self._public_key, self._signature, data_to_verify
        )


class _DSASignatureContext(AsymmetricSignatureContext):
    def __init__(
        self,
        backend,
        private_key: dsa.DSAPrivateKey,
        algorithm: hashes.HashAlgorithm,
    ):
        self._backend = backend
        self._private_key = private_key
        self._algorithm = algorithm
        self._hash_ctx = hashes.Hash(self._algorithm, self._backend)

    def update(self, data: bytes) -> None:
        self._hash_ctx.update(data)

    def finalize(self) -> bytes:
        data_to_sign = self._hash_ctx.finalize()
        return _dsa_sig_sign(self._backend, self._private_key, data_to_sign)


class _DSAParameters(dsa.DSAParameters):
    def __init__(self, backend, dsa_cdata):
        self._backend = backend
        self._dsa_cdata = dsa_cdata

    def parameter_numbers(self) -> dsa.DSAParameterNumbers:
        p = self._backend._ffi.new("BIGNUM **")
        q = self._backend._ffi.new("BIGNUM **")
        g = self._backend._ffi.new("BIGNUM **")
        self._backend._lib.DSA_get0_pqg(self._dsa_cdata, p, q, g)
        self._backend.openssl_assert(p[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(q[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(g[0] != self._backend._ffi.NULL)
        return dsa.DSAParameterNumbers(
            p=self._backend._bn_to_int(p[0]),
            q=self._backend._bn_to_int(q[0]),
            g=self._backend._bn_to_int(g[0]),
        )

    def generate_private_key(self) -> dsa.DSAPrivateKey:
        return self._backend.generate_dsa_private_key(self)


class _DSAPrivateKey(dsa.DSAPrivateKey):
    def __init__(self, backend, dsa_cdata, evp_pkey):
        self._backend = backend
        self._dsa_cdata = dsa_cdata
        self._evp_pkey = evp_pkey

        p = self._backend._ffi.new("BIGNUM **")
        self._backend._lib.DSA_get0_pqg(
            dsa_cdata, p, self._backend._ffi.NULL, self._backend._ffi.NULL
        )
        self._backend.openssl_assert(p[0] != backend._ffi.NULL)
        self._key_size = self._backend._lib.BN_num_bits(p[0])

    key_size = utils.read_only_property("_key_size")

    def signer(
        self, signature_algorithm: hashes.HashAlgorithm
    ) -> AsymmetricSignatureContext:
        _warn_sign_verify_deprecated()
        _check_not_prehashed(signature_algorithm)
        return _DSASignatureContext(self._backend, self, signature_algorithm)

    def private_numbers(self) -> dsa.DSAPrivateNumbers:
        p = self._backend._ffi.new("BIGNUM **")
        q = self._backend._ffi.new("BIGNUM **")
        g = self._backend._ffi.new("BIGNUM **")
        pub_key = self._backend._ffi.new("BIGNUM **")
        priv_key = self._backend._ffi.new("BIGNUM **")
        self._backend._lib.DSA_get0_pqg(self._dsa_cdata, p, q, g)
        self._backend.openssl_assert(p[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(q[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(g[0] != self._backend._ffi.NULL)
        self._backend._lib.DSA_get0_key(self._dsa_cdata, pub_key, priv_key)
        self._backend.openssl_assert(pub_key[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(priv_key[0] != self._backend._ffi.NULL)
        return dsa.DSAPrivateNumbers(
            public_numbers=dsa.DSAPublicNumbers(
                parameter_numbers=dsa.DSAParameterNumbers(
                    p=self._backend._bn_to_int(p[0]),
                    q=self._backend._bn_to_int(q[0]),
                    g=self._backend._bn_to_int(g[0]),
                ),
                y=self._backend._bn_to_int(pub_key[0]),
            ),
            x=self._backend._bn_to_int(priv_key[0]),
        )

    def public_key(self) -> dsa.DSAPublicKey:
        dsa_cdata = self._backend._lib.DSAparams_dup(self._dsa_cdata)
        self._backend.openssl_assert(dsa_cdata != self._backend._ffi.NULL)
        dsa_cdata = self._backend._ffi.gc(
            dsa_cdata, self._backend._lib.DSA_free
        )
        pub_key = self._backend._ffi.new("BIGNUM **")
        self._backend._lib.DSA_get0_key(
            self._dsa_cdata, pub_key, self._backend._ffi.NULL
        )
        self._backend.openssl_assert(pub_key[0] != self._backend._ffi.NULL)
        pub_key_dup = self._backend._lib.BN_dup(pub_key[0])
        res = self._backend._lib.DSA_set0_key(
            dsa_cdata, pub_key_dup, self._backend._ffi.NULL
        )
        self._backend.openssl_assert(res == 1)
        evp_pkey = self._backend._dsa_cdata_to_evp_pkey(dsa_cdata)
        return _DSAPublicKey(self._backend, dsa_cdata, evp_pkey)

    def parameters(self) -> dsa.DSAParameters:
        dsa_cdata = self._backend._lib.DSAparams_dup(self._dsa_cdata)
        self._backend.openssl_assert(dsa_cdata != self._backend._ffi.NULL)
        dsa_cdata = self._backend._ffi.gc(
            dsa_cdata, self._backend._lib.DSA_free
        )
        return _DSAParameters(self._backend, dsa_cdata)

    def private_bytes(
        self,
        encoding: serialization.Encoding,
        format: serialization.PrivateFormat,
        encryption_algorithm: serialization.KeySerializationEncryption,
    ) -> bytes:
        return self._backend._private_key_bytes(
            encoding,
            format,
            encryption_algorithm,
            self,
            self._evp_pkey,
            self._dsa_cdata,
        )

    def sign(
        self,
        data: bytes,
        algorithm: typing.Union[asym_utils.Prehashed, hashes.HashAlgorithm],
    ) -> bytes:
        data, algorithm = _calculate_digest_and_algorithm(
            self._backend, data, algorithm
        )
        return _dsa_sig_sign(self._backend, self, data)


class _DSAPublicKey(dsa.DSAPublicKey):
    def __init__(self, backend, dsa_cdata, evp_pkey):
        self._backend = backend
        self._dsa_cdata = dsa_cdata
        self._evp_pkey = evp_pkey
        p = self._backend._ffi.new("BIGNUM **")
        self._backend._lib.DSA_get0_pqg(
            dsa_cdata, p, self._backend._ffi.NULL, self._backend._ffi.NULL
        )
        self._backend.openssl_assert(p[0] != backend._ffi.NULL)
        self._key_size = self._backend._lib.BN_num_bits(p[0])

    key_size = utils.read_only_property("_key_size")

    def verifier(
        self,
        signature: bytes,
        signature_algorithm: hashes.HashAlgorithm,
    ) -> AsymmetricVerificationContext:
        _warn_sign_verify_deprecated()
        utils._check_bytes("signature", signature)

        _check_not_prehashed(signature_algorithm)
        return _DSAVerificationContext(
            self._backend, self, signature, signature_algorithm
        )

    def public_numbers(self) -> dsa.DSAPublicNumbers:
        p = self._backend._ffi.new("BIGNUM **")
        q = self._backend._ffi.new("BIGNUM **")
        g = self._backend._ffi.new("BIGNUM **")
        pub_key = self._backend._ffi.new("BIGNUM **")
        self._backend._lib.DSA_get0_pqg(self._dsa_cdata, p, q, g)
        self._backend.openssl_assert(p[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(q[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(g[0] != self._backend._ffi.NULL)
        self._backend._lib.DSA_get0_key(
            self._dsa_cdata, pub_key, self._backend._ffi.NULL
        )
        self._backend.openssl_assert(pub_key[0] != self._backend._ffi.NULL)
        return dsa.DSAPublicNumbers(
            parameter_numbers=dsa.DSAParameterNumbers(
                p=self._backend._bn_to_int(p[0]),
                q=self._backend._bn_to_int(q[0]),
                g=self._backend._bn_to_int(g[0]),
            ),
            y=self._backend._bn_to_int(pub_key[0]),
        )

    def parameters(self) -> dsa.DSAParameters:
        dsa_cdata = self._backend._lib.DSAparams_dup(self._dsa_cdata)
        dsa_cdata = self._backend._ffi.gc(
            dsa_cdata, self._backend._lib.DSA_free
        )
        return _DSAParameters(self._backend, dsa_cdata)

    def public_bytes(
        self,
        encoding: serialization.Encoding,
        format: serialization.PublicFormat,
    ) -> bytes:
        return self._backend._public_key_bytes(
            encoding, format, self, self._evp_pkey, None
        )

    def verify(
        self,
        signature: bytes,
        data: bytes,
        algorithm: typing.Union[asym_utils.Prehashed, hashes.HashAlgorithm],
    ) -> None:
        data, algorithm = _calculate_digest_and_algorithm(
            self._backend, data, algorithm
        )
        return _dsa_sig_verify(self._backend, self, signature, data)
