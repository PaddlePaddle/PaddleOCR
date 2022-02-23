# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


from cryptography.hazmat.backends.openssl.utils import _evp_pkey_derive
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)


_X25519_KEY_SIZE = 32


class _X25519PublicKey(X25519PublicKey):
    def __init__(self, backend, evp_pkey):
        self._backend = backend
        self._evp_pkey = evp_pkey

    def public_bytes(
        self,
        encoding: serialization.Encoding,
        format: serialization.PublicFormat,
    ) -> bytes:
        if (
            encoding is serialization.Encoding.Raw
            or format is serialization.PublicFormat.Raw
        ):
            if (
                encoding is not serialization.Encoding.Raw
                or format is not serialization.PublicFormat.Raw
            ):
                raise ValueError(
                    "When using Raw both encoding and format must be Raw"
                )

            return self._raw_public_bytes()

        return self._backend._public_key_bytes(
            encoding, format, self, self._evp_pkey, None
        )

    def _raw_public_bytes(self) -> bytes:
        ucharpp = self._backend._ffi.new("unsigned char **")
        res = self._backend._lib.EVP_PKEY_get1_tls_encodedpoint(
            self._evp_pkey, ucharpp
        )
        self._backend.openssl_assert(res == 32)
        self._backend.openssl_assert(ucharpp[0] != self._backend._ffi.NULL)
        data = self._backend._ffi.gc(
            ucharpp[0], self._backend._lib.OPENSSL_free
        )
        return self._backend._ffi.buffer(data, res)[:]


class _X25519PrivateKey(X25519PrivateKey):
    def __init__(self, backend, evp_pkey):
        self._backend = backend
        self._evp_pkey = evp_pkey

    def public_key(self) -> X25519PublicKey:
        bio = self._backend._create_mem_bio_gc()
        res = self._backend._lib.i2d_PUBKEY_bio(bio, self._evp_pkey)
        self._backend.openssl_assert(res == 1)
        evp_pkey = self._backend._lib.d2i_PUBKEY_bio(
            bio, self._backend._ffi.NULL
        )
        self._backend.openssl_assert(evp_pkey != self._backend._ffi.NULL)
        evp_pkey = self._backend._ffi.gc(
            evp_pkey, self._backend._lib.EVP_PKEY_free
        )
        return _X25519PublicKey(self._backend, evp_pkey)

    def exchange(self, peer_public_key: X25519PublicKey) -> bytes:
        if not isinstance(peer_public_key, X25519PublicKey):
            raise TypeError("peer_public_key must be X25519PublicKey.")

        return _evp_pkey_derive(self._backend, self._evp_pkey, peer_public_key)

    def private_bytes(
        self,
        encoding: serialization.Encoding,
        format: serialization.PrivateFormat,
        encryption_algorithm: serialization.KeySerializationEncryption,
    ) -> bytes:
        if (
            encoding is serialization.Encoding.Raw
            or format is serialization.PublicFormat.Raw
        ):
            if (
                format is not serialization.PrivateFormat.Raw
                or encoding is not serialization.Encoding.Raw
                or not isinstance(
                    encryption_algorithm, serialization.NoEncryption
                )
            ):
                raise ValueError(
                    "When using Raw both encoding and format must be Raw "
                    "and encryption_algorithm must be NoEncryption()"
                )

            return self._raw_private_bytes()

        return self._backend._private_key_bytes(
            encoding, format, encryption_algorithm, self, self._evp_pkey, None
        )

    def _raw_private_bytes(self) -> bytes:
        # When we drop support for CRYPTOGRAPHY_OPENSSL_LESS_THAN_111 we can
        # switch this to EVP_PKEY_new_raw_private_key
        # The trick we use here is serializing to a PKCS8 key and just
        # using the last 32 bytes, which is the key itself.
        bio = self._backend._create_mem_bio_gc()
        res = self._backend._lib.i2d_PKCS8PrivateKey_bio(
            bio,
            self._evp_pkey,
            self._backend._ffi.NULL,
            self._backend._ffi.NULL,
            0,
            self._backend._ffi.NULL,
            self._backend._ffi.NULL,
        )
        self._backend.openssl_assert(res == 1)
        pkcs8 = self._backend._read_mem_bio(bio)
        self._backend.openssl_assert(len(pkcs8) == 48)
        return pkcs8[-_X25519_KEY_SIZE:]
