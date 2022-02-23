# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import collections
import contextlib
import itertools
import typing
import warnings
from contextlib import contextmanager

from cryptography import utils, x509
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.backends.interfaces import Backend as BackendInterface
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.ciphers import _CipherContext
from cryptography.hazmat.backends.openssl.cmac import _CMACContext
from cryptography.hazmat.backends.openssl.dh import (
    _DHParameters,
    _DHPrivateKey,
    _DHPublicKey,
    _dh_params_dup,
)
from cryptography.hazmat.backends.openssl.dsa import (
    _DSAParameters,
    _DSAPrivateKey,
    _DSAPublicKey,
)
from cryptography.hazmat.backends.openssl.ec import (
    _EllipticCurvePrivateKey,
    _EllipticCurvePublicKey,
)
from cryptography.hazmat.backends.openssl.ed25519 import (
    _Ed25519PrivateKey,
    _Ed25519PublicKey,
)
from cryptography.hazmat.backends.openssl.ed448 import (
    _ED448_KEY_SIZE,
    _Ed448PrivateKey,
    _Ed448PublicKey,
)
from cryptography.hazmat.backends.openssl.hashes import _HashContext
from cryptography.hazmat.backends.openssl.hmac import _HMACContext
from cryptography.hazmat.backends.openssl.poly1305 import (
    _POLY1305_KEY_SIZE,
    _Poly1305Context,
)
from cryptography.hazmat.backends.openssl.rsa import (
    _RSAPrivateKey,
    _RSAPublicKey,
)
from cryptography.hazmat.backends.openssl.x25519 import (
    _X25519PrivateKey,
    _X25519PublicKey,
)
from cryptography.hazmat.backends.openssl.x448 import (
    _X448PrivateKey,
    _X448PublicKey,
)
from cryptography.hazmat.bindings._rust import (
    x509 as rust_x509,
)
from cryptography.hazmat.bindings.openssl import binding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
    dh,
    dsa,
    ec,
    ed25519,
    rsa,
)
from cryptography.hazmat.primitives.asymmetric.padding import (
    MGF1,
    OAEP,
    PKCS1v15,
    PSS,
)
from cryptography.hazmat.primitives.ciphers.algorithms import (
    AES,
    ARC4,
    Blowfish,
    CAST5,
    Camellia,
    ChaCha20,
    IDEA,
    SEED,
    SM4,
    TripleDES,
)
from cryptography.hazmat.primitives.ciphers.modes import (
    CBC,
    CFB,
    CFB8,
    CTR,
    ECB,
    GCM,
    OFB,
    XTS,
)
from cryptography.hazmat.primitives.kdf import scrypt
from cryptography.hazmat.primitives.serialization import pkcs7, ssh
from cryptography.hazmat.primitives.serialization.pkcs12 import (
    PKCS12Certificate,
    PKCS12KeyAndCertificates,
)
from cryptography.x509.base import PUBLIC_KEY_TYPES


_MemoryBIO = collections.namedtuple("_MemoryBIO", ["bio", "char_ptr"])


# Not actually supported, just used as a marker for some serialization tests.
class _RC2(object):
    pass


class Backend(BackendInterface):
    """
    OpenSSL API binding interfaces.
    """

    name = "openssl"

    # FIPS has opinions about acceptable algorithms and key sizes, but the
    # disallowed algorithms are still present in OpenSSL. They just error if
    # you try to use them. To avoid that we allowlist the algorithms in
    # FIPS 140-3. This isn't ideal, but FIPS 140-3 is trash so here we are.
    _fips_aead = {
        b"aes-128-ccm",
        b"aes-192-ccm",
        b"aes-256-ccm",
        b"aes-128-gcm",
        b"aes-192-gcm",
        b"aes-256-gcm",
    }
    _fips_ciphers = (AES, TripleDES)
    # Sometimes SHA1 is still permissible. That logic is contained
    # within the various *_supported methods.
    _fips_hashes = (
        hashes.SHA224,
        hashes.SHA256,
        hashes.SHA384,
        hashes.SHA512,
        hashes.SHA512_224,
        hashes.SHA512_256,
        hashes.SHA3_224,
        hashes.SHA3_256,
        hashes.SHA3_384,
        hashes.SHA3_512,
        hashes.SHAKE128,
        hashes.SHAKE256,
    )
    _fips_ecdh_curves = (
        ec.SECP224R1,
        ec.SECP256R1,
        ec.SECP384R1,
        ec.SECP521R1,
    )
    _fips_rsa_min_key_size = 2048
    _fips_rsa_min_public_exponent = 65537
    _fips_dsa_min_modulus = 1 << 2048
    _fips_dh_min_key_size = 2048
    _fips_dh_min_modulus = 1 << _fips_dh_min_key_size

    def __init__(self):
        self._binding = binding.Binding()
        self._ffi = self._binding.ffi
        self._lib = self._binding.lib
        self._rsa_skip_check_key = False
        self._fips_enabled = self._is_fips_enabled()

        self._cipher_registry = {}
        self._register_default_ciphers()
        if self._fips_enabled and self._lib.CRYPTOGRAPHY_NEEDS_OSRANDOM_ENGINE:
            warnings.warn(
                "OpenSSL FIPS mode is enabled. Can't enable DRBG fork safety.",
                UserWarning,
            )
        else:
            self.activate_osrandom_engine()
        self._dh_types = [self._lib.EVP_PKEY_DH]
        if self._lib.Cryptography_HAS_EVP_PKEY_DHX:
            self._dh_types.append(self._lib.EVP_PKEY_DHX)

    def __repr__(self):
        return "<OpenSSLBackend(version: {}, FIPS: {})>".format(
            self.openssl_version_text(), self._fips_enabled
        )

    def openssl_assert(self, ok, errors=None):
        return binding._openssl_assert(self._lib, ok, errors=errors)

    def _is_fips_enabled(self):
        if self._lib.Cryptography_HAS_300_FIPS:
            mode = self._lib.EVP_default_properties_is_fips_enabled(
                self._ffi.NULL
            )
        else:
            mode = getattr(self._lib, "FIPS_mode", lambda: 0)()

        if mode == 0:
            # OpenSSL without FIPS pushes an error on the error stack
            self._lib.ERR_clear_error()
        return bool(mode)

    def _enable_fips(self):
        # This function enables FIPS mode for OpenSSL 3.0.0 on installs that
        # have the FIPS provider installed properly.
        self._binding._enable_fips()
        assert self._is_fips_enabled()
        self._fips_enabled = self._is_fips_enabled()

    def activate_builtin_random(self):
        if self._lib.CRYPTOGRAPHY_NEEDS_OSRANDOM_ENGINE:
            # Obtain a new structural reference.
            e = self._lib.ENGINE_get_default_RAND()
            if e != self._ffi.NULL:
                self._lib.ENGINE_unregister_RAND(e)
                # Reset the RNG to use the built-in.
                res = self._lib.RAND_set_rand_method(self._ffi.NULL)
                self.openssl_assert(res == 1)
                # decrement the structural reference from get_default_RAND
                res = self._lib.ENGINE_finish(e)
                self.openssl_assert(res == 1)

    @contextlib.contextmanager
    def _get_osurandom_engine(self):
        # Fetches an engine by id and returns it. This creates a structural
        # reference.
        e = self._lib.ENGINE_by_id(self._lib.Cryptography_osrandom_engine_id)
        self.openssl_assert(e != self._ffi.NULL)
        # Initialize the engine for use. This adds a functional reference.
        res = self._lib.ENGINE_init(e)
        self.openssl_assert(res == 1)

        try:
            yield e
        finally:
            # Decrement the structural ref incremented by ENGINE_by_id.
            res = self._lib.ENGINE_free(e)
            self.openssl_assert(res == 1)
            # Decrement the functional ref incremented by ENGINE_init.
            res = self._lib.ENGINE_finish(e)
            self.openssl_assert(res == 1)

    def activate_osrandom_engine(self):
        if self._lib.CRYPTOGRAPHY_NEEDS_OSRANDOM_ENGINE:
            # Unregister and free the current engine.
            self.activate_builtin_random()
            with self._get_osurandom_engine() as e:
                # Set the engine as the default RAND provider.
                res = self._lib.ENGINE_set_default_RAND(e)
                self.openssl_assert(res == 1)
            # Reset the RNG to use the engine
            res = self._lib.RAND_set_rand_method(self._ffi.NULL)
            self.openssl_assert(res == 1)

    def osrandom_engine_implementation(self):
        buf = self._ffi.new("char[]", 64)
        with self._get_osurandom_engine() as e:
            res = self._lib.ENGINE_ctrl_cmd(
                e, b"get_implementation", len(buf), buf, self._ffi.NULL, 0
            )
            self.openssl_assert(res > 0)
        return self._ffi.string(buf).decode("ascii")

    def openssl_version_text(self):
        """
        Friendly string name of the loaded OpenSSL library. This is not
        necessarily the same version as it was compiled against.

        Example: OpenSSL 1.1.1d  10 Sep 2019
        """
        return self._ffi.string(
            self._lib.OpenSSL_version(self._lib.OPENSSL_VERSION)
        ).decode("ascii")

    def openssl_version_number(self):
        return self._lib.OpenSSL_version_num()

    def create_hmac_ctx(self, key, algorithm):
        return _HMACContext(self, key, algorithm)

    def _evp_md_from_algorithm(self, algorithm):
        if algorithm.name == "blake2b" or algorithm.name == "blake2s":
            alg = "{}{}".format(
                algorithm.name, algorithm.digest_size * 8
            ).encode("ascii")
        else:
            alg = algorithm.name.encode("ascii")

        evp_md = self._lib.EVP_get_digestbyname(alg)
        return evp_md

    def _evp_md_non_null_from_algorithm(self, algorithm):
        evp_md = self._evp_md_from_algorithm(algorithm)
        self.openssl_assert(evp_md != self._ffi.NULL)
        return evp_md

    def hash_supported(self, algorithm):
        if self._fips_enabled and not isinstance(algorithm, self._fips_hashes):
            return False

        evp_md = self._evp_md_from_algorithm(algorithm)
        return evp_md != self._ffi.NULL

    def scrypt_supported(self):
        if self._fips_enabled:
            return False
        else:
            return self._lib.Cryptography_HAS_SCRYPT == 1

    def hmac_supported(self, algorithm):
        # FIPS mode still allows SHA1 for HMAC
        if self._fips_enabled and isinstance(algorithm, hashes.SHA1):
            return True

        return self.hash_supported(algorithm)

    def create_hash_ctx(self, algorithm):
        return _HashContext(self, algorithm)

    def cipher_supported(self, cipher, mode):
        if self._fips_enabled:
            # FIPS mode requires AES or TripleDES, but only CBC/ECB allowed
            # in TripleDES mode.
            if not isinstance(cipher, self._fips_ciphers) or (
                isinstance(cipher, TripleDES)
                and not isinstance(mode, (CBC, ECB))
            ):
                return False

        try:
            adapter = self._cipher_registry[type(cipher), type(mode)]
        except KeyError:
            return False
        evp_cipher = adapter(self, cipher, mode)
        return self._ffi.NULL != evp_cipher

    def register_cipher_adapter(self, cipher_cls, mode_cls, adapter):
        if (cipher_cls, mode_cls) in self._cipher_registry:
            raise ValueError(
                "Duplicate registration for: {} {}.".format(
                    cipher_cls, mode_cls
                )
            )
        self._cipher_registry[cipher_cls, mode_cls] = adapter

    def _register_default_ciphers(self):
        for mode_cls in [CBC, CTR, ECB, OFB, CFB, CFB8, GCM]:
            self.register_cipher_adapter(
                AES,
                mode_cls,
                GetCipherByName("{cipher.name}-{cipher.key_size}-{mode.name}"),
            )
        for mode_cls in [CBC, CTR, ECB, OFB, CFB]:
            self.register_cipher_adapter(
                Camellia,
                mode_cls,
                GetCipherByName("{cipher.name}-{cipher.key_size}-{mode.name}"),
            )
        for mode_cls in [CBC, CFB, CFB8, OFB]:
            self.register_cipher_adapter(
                TripleDES, mode_cls, GetCipherByName("des-ede3-{mode.name}")
            )
        self.register_cipher_adapter(
            TripleDES, ECB, GetCipherByName("des-ede3")
        )
        for mode_cls in [CBC, CFB, OFB, ECB]:
            self.register_cipher_adapter(
                Blowfish, mode_cls, GetCipherByName("bf-{mode.name}")
            )
        for mode_cls in [CBC, CFB, OFB, ECB]:
            self.register_cipher_adapter(
                SEED, mode_cls, GetCipherByName("seed-{mode.name}")
            )
        for cipher_cls, mode_cls in itertools.product(
            [CAST5, IDEA],
            [CBC, OFB, CFB, ECB],
        ):
            self.register_cipher_adapter(
                cipher_cls,
                mode_cls,
                GetCipherByName("{cipher.name}-{mode.name}"),
            )
        self.register_cipher_adapter(ARC4, type(None), GetCipherByName("rc4"))
        # We don't actually support RC2, this is just used by some tests.
        self.register_cipher_adapter(_RC2, type(None), GetCipherByName("rc2"))
        self.register_cipher_adapter(
            ChaCha20, type(None), GetCipherByName("chacha20")
        )
        self.register_cipher_adapter(AES, XTS, _get_xts_cipher)
        for mode_cls in [ECB, CBC, OFB, CFB, CTR]:
            self.register_cipher_adapter(
                SM4, mode_cls, GetCipherByName("sm4-{mode.name}")
            )

    def create_symmetric_encryption_ctx(self, cipher, mode):
        return _CipherContext(self, cipher, mode, _CipherContext._ENCRYPT)

    def create_symmetric_decryption_ctx(self, cipher, mode):
        return _CipherContext(self, cipher, mode, _CipherContext._DECRYPT)

    def pbkdf2_hmac_supported(self, algorithm):
        return self.hmac_supported(algorithm)

    def derive_pbkdf2_hmac(
        self, algorithm, length, salt, iterations, key_material
    ):
        buf = self._ffi.new("unsigned char[]", length)
        evp_md = self._evp_md_non_null_from_algorithm(algorithm)
        key_material_ptr = self._ffi.from_buffer(key_material)
        res = self._lib.PKCS5_PBKDF2_HMAC(
            key_material_ptr,
            len(key_material),
            salt,
            len(salt),
            iterations,
            evp_md,
            length,
            buf,
        )
        self.openssl_assert(res == 1)
        return self._ffi.buffer(buf)[:]

    def _consume_errors(self):
        return binding._consume_errors(self._lib)

    def _consume_errors_with_text(self):
        return binding._consume_errors_with_text(self._lib)

    def _bn_to_int(self, bn):
        assert bn != self._ffi.NULL
        self.openssl_assert(not self._lib.BN_is_negative(bn))

        bn_num_bytes = self._lib.BN_num_bytes(bn)
        bin_ptr = self._ffi.new("unsigned char[]", bn_num_bytes)
        bin_len = self._lib.BN_bn2bin(bn, bin_ptr)
        # A zero length means the BN has value 0
        self.openssl_assert(bin_len >= 0)
        val = int.from_bytes(self._ffi.buffer(bin_ptr)[:bin_len], "big")
        return val

    def _int_to_bn(self, num, bn=None):
        """
        Converts a python integer to a BIGNUM. The returned BIGNUM will not
        be garbage collected (to support adding them to structs that take
        ownership of the object). Be sure to register it for GC if it will
        be discarded after use.
        """
        assert bn is None or bn != self._ffi.NULL

        if bn is None:
            bn = self._ffi.NULL

        binary = num.to_bytes(int(num.bit_length() / 8.0 + 1), "big")
        bn_ptr = self._lib.BN_bin2bn(binary, len(binary), bn)
        self.openssl_assert(bn_ptr != self._ffi.NULL)
        return bn_ptr

    def generate_rsa_private_key(self, public_exponent, key_size):
        rsa._verify_rsa_parameters(public_exponent, key_size)

        rsa_cdata = self._lib.RSA_new()
        self.openssl_assert(rsa_cdata != self._ffi.NULL)
        rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)

        bn = self._int_to_bn(public_exponent)
        bn = self._ffi.gc(bn, self._lib.BN_free)

        res = self._lib.RSA_generate_key_ex(
            rsa_cdata, key_size, bn, self._ffi.NULL
        )
        self.openssl_assert(res == 1)
        evp_pkey = self._rsa_cdata_to_evp_pkey(rsa_cdata)

        return _RSAPrivateKey(
            self, rsa_cdata, evp_pkey, self._rsa_skip_check_key
        )

    def generate_rsa_parameters_supported(self, public_exponent, key_size):
        return (
            public_exponent >= 3
            and public_exponent & 1 != 0
            and key_size >= 512
        )

    def load_rsa_private_numbers(self, numbers):
        rsa._check_private_key_components(
            numbers.p,
            numbers.q,
            numbers.d,
            numbers.dmp1,
            numbers.dmq1,
            numbers.iqmp,
            numbers.public_numbers.e,
            numbers.public_numbers.n,
        )
        rsa_cdata = self._lib.RSA_new()
        self.openssl_assert(rsa_cdata != self._ffi.NULL)
        rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
        p = self._int_to_bn(numbers.p)
        q = self._int_to_bn(numbers.q)
        d = self._int_to_bn(numbers.d)
        dmp1 = self._int_to_bn(numbers.dmp1)
        dmq1 = self._int_to_bn(numbers.dmq1)
        iqmp = self._int_to_bn(numbers.iqmp)
        e = self._int_to_bn(numbers.public_numbers.e)
        n = self._int_to_bn(numbers.public_numbers.n)
        res = self._lib.RSA_set0_factors(rsa_cdata, p, q)
        self.openssl_assert(res == 1)
        res = self._lib.RSA_set0_key(rsa_cdata, n, e, d)
        self.openssl_assert(res == 1)
        res = self._lib.RSA_set0_crt_params(rsa_cdata, dmp1, dmq1, iqmp)
        self.openssl_assert(res == 1)
        evp_pkey = self._rsa_cdata_to_evp_pkey(rsa_cdata)

        return _RSAPrivateKey(
            self, rsa_cdata, evp_pkey, self._rsa_skip_check_key
        )

    def load_rsa_public_numbers(self, numbers):
        rsa._check_public_key_components(numbers.e, numbers.n)
        rsa_cdata = self._lib.RSA_new()
        self.openssl_assert(rsa_cdata != self._ffi.NULL)
        rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
        e = self._int_to_bn(numbers.e)
        n = self._int_to_bn(numbers.n)
        res = self._lib.RSA_set0_key(rsa_cdata, n, e, self._ffi.NULL)
        self.openssl_assert(res == 1)
        evp_pkey = self._rsa_cdata_to_evp_pkey(rsa_cdata)

        return _RSAPublicKey(self, rsa_cdata, evp_pkey)

    def _create_evp_pkey_gc(self):
        evp_pkey = self._lib.EVP_PKEY_new()
        self.openssl_assert(evp_pkey != self._ffi.NULL)
        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)
        return evp_pkey

    def _rsa_cdata_to_evp_pkey(self, rsa_cdata):
        evp_pkey = self._create_evp_pkey_gc()
        res = self._lib.EVP_PKEY_set1_RSA(evp_pkey, rsa_cdata)
        self.openssl_assert(res == 1)
        return evp_pkey

    def _bytes_to_bio(self, data):
        """
        Return a _MemoryBIO namedtuple of (BIO, char*).

        The char* is the storage for the BIO and it must stay alive until the
        BIO is finished with.
        """
        data_ptr = self._ffi.from_buffer(data)
        bio = self._lib.BIO_new_mem_buf(data_ptr, len(data))
        self.openssl_assert(bio != self._ffi.NULL)

        return _MemoryBIO(self._ffi.gc(bio, self._lib.BIO_free), data_ptr)

    def _create_mem_bio_gc(self):
        """
        Creates an empty memory BIO.
        """
        bio_method = self._lib.BIO_s_mem()
        self.openssl_assert(bio_method != self._ffi.NULL)
        bio = self._lib.BIO_new(bio_method)
        self.openssl_assert(bio != self._ffi.NULL)
        bio = self._ffi.gc(bio, self._lib.BIO_free)
        return bio

    def _read_mem_bio(self, bio):
        """
        Reads a memory BIO. This only works on memory BIOs.
        """
        buf = self._ffi.new("char **")
        buf_len = self._lib.BIO_get_mem_data(bio, buf)
        self.openssl_assert(buf_len > 0)
        self.openssl_assert(buf[0] != self._ffi.NULL)
        bio_data = self._ffi.buffer(buf[0], buf_len)[:]
        return bio_data

    def _evp_pkey_to_private_key(self, evp_pkey):
        """
        Return the appropriate type of PrivateKey given an evp_pkey cdata
        pointer.
        """

        key_type = self._lib.EVP_PKEY_id(evp_pkey)

        if key_type == self._lib.EVP_PKEY_RSA:
            rsa_cdata = self._lib.EVP_PKEY_get1_RSA(evp_pkey)
            self.openssl_assert(rsa_cdata != self._ffi.NULL)
            rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
            return _RSAPrivateKey(
                self, rsa_cdata, evp_pkey, self._rsa_skip_check_key
            )
        elif key_type == self._lib.EVP_PKEY_DSA:
            dsa_cdata = self._lib.EVP_PKEY_get1_DSA(evp_pkey)
            self.openssl_assert(dsa_cdata != self._ffi.NULL)
            dsa_cdata = self._ffi.gc(dsa_cdata, self._lib.DSA_free)
            return _DSAPrivateKey(self, dsa_cdata, evp_pkey)
        elif key_type == self._lib.EVP_PKEY_EC:
            ec_cdata = self._lib.EVP_PKEY_get1_EC_KEY(evp_pkey)
            self.openssl_assert(ec_cdata != self._ffi.NULL)
            ec_cdata = self._ffi.gc(ec_cdata, self._lib.EC_KEY_free)
            return _EllipticCurvePrivateKey(self, ec_cdata, evp_pkey)
        elif key_type in self._dh_types:
            dh_cdata = self._lib.EVP_PKEY_get1_DH(evp_pkey)
            self.openssl_assert(dh_cdata != self._ffi.NULL)
            dh_cdata = self._ffi.gc(dh_cdata, self._lib.DH_free)
            return _DHPrivateKey(self, dh_cdata, evp_pkey)
        elif key_type == getattr(self._lib, "EVP_PKEY_ED25519", None):
            # EVP_PKEY_ED25519 is not present in OpenSSL < 1.1.1
            return _Ed25519PrivateKey(self, evp_pkey)
        elif key_type == getattr(self._lib, "EVP_PKEY_X448", None):
            # EVP_PKEY_X448 is not present in OpenSSL < 1.1.1
            return _X448PrivateKey(self, evp_pkey)
        elif key_type == getattr(self._lib, "EVP_PKEY_X25519", None):
            # EVP_PKEY_X25519 is not present in OpenSSL < 1.1.0
            return _X25519PrivateKey(self, evp_pkey)
        elif key_type == getattr(self._lib, "EVP_PKEY_ED448", None):
            # EVP_PKEY_ED448 is not present in OpenSSL < 1.1.1
            return _Ed448PrivateKey(self, evp_pkey)
        else:
            raise UnsupportedAlgorithm("Unsupported key type.")

    def _evp_pkey_to_public_key(self, evp_pkey):
        """
        Return the appropriate type of PublicKey given an evp_pkey cdata
        pointer.
        """

        key_type = self._lib.EVP_PKEY_id(evp_pkey)

        if key_type == self._lib.EVP_PKEY_RSA:
            rsa_cdata = self._lib.EVP_PKEY_get1_RSA(evp_pkey)
            self.openssl_assert(rsa_cdata != self._ffi.NULL)
            rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
            return _RSAPublicKey(self, rsa_cdata, evp_pkey)
        elif key_type == self._lib.EVP_PKEY_DSA:
            dsa_cdata = self._lib.EVP_PKEY_get1_DSA(evp_pkey)
            self.openssl_assert(dsa_cdata != self._ffi.NULL)
            dsa_cdata = self._ffi.gc(dsa_cdata, self._lib.DSA_free)
            return _DSAPublicKey(self, dsa_cdata, evp_pkey)
        elif key_type == self._lib.EVP_PKEY_EC:
            ec_cdata = self._lib.EVP_PKEY_get1_EC_KEY(evp_pkey)
            self.openssl_assert(ec_cdata != self._ffi.NULL)
            ec_cdata = self._ffi.gc(ec_cdata, self._lib.EC_KEY_free)
            return _EllipticCurvePublicKey(self, ec_cdata, evp_pkey)
        elif key_type in self._dh_types:
            dh_cdata = self._lib.EVP_PKEY_get1_DH(evp_pkey)
            self.openssl_assert(dh_cdata != self._ffi.NULL)
            dh_cdata = self._ffi.gc(dh_cdata, self._lib.DH_free)
            return _DHPublicKey(self, dh_cdata, evp_pkey)
        elif key_type == getattr(self._lib, "EVP_PKEY_ED25519", None):
            # EVP_PKEY_ED25519 is not present in OpenSSL < 1.1.1
            return _Ed25519PublicKey(self, evp_pkey)
        elif key_type == getattr(self._lib, "EVP_PKEY_X448", None):
            # EVP_PKEY_X448 is not present in OpenSSL < 1.1.1
            return _X448PublicKey(self, evp_pkey)
        elif key_type == getattr(self._lib, "EVP_PKEY_X25519", None):
            # EVP_PKEY_X25519 is not present in OpenSSL < 1.1.0
            return _X25519PublicKey(self, evp_pkey)
        elif key_type == getattr(self._lib, "EVP_PKEY_ED448", None):
            # EVP_PKEY_X25519 is not present in OpenSSL < 1.1.1
            return _Ed448PublicKey(self, evp_pkey)
        else:
            raise UnsupportedAlgorithm("Unsupported key type.")

    def _oaep_hash_supported(self, algorithm):
        if self._lib.Cryptography_HAS_RSA_OAEP_MD:
            return isinstance(
                algorithm,
                (
                    hashes.SHA1,
                    hashes.SHA224,
                    hashes.SHA256,
                    hashes.SHA384,
                    hashes.SHA512,
                ),
            )
        else:
            return isinstance(algorithm, hashes.SHA1)

    def rsa_padding_supported(self, padding):
        if isinstance(padding, PKCS1v15):
            return True
        elif isinstance(padding, PSS) and isinstance(padding._mgf, MGF1):
            # SHA1 is permissible in MGF1 in FIPS
            if self._fips_enabled and isinstance(
                padding._mgf._algorithm, hashes.SHA1
            ):
                return True
            else:
                return self.hash_supported(padding._mgf._algorithm)
        elif isinstance(padding, OAEP) and isinstance(padding._mgf, MGF1):
            return (
                self._oaep_hash_supported(padding._mgf._algorithm)
                and self._oaep_hash_supported(padding._algorithm)
                and (
                    (padding._label is None or len(padding._label) == 0)
                    or self._lib.Cryptography_HAS_RSA_OAEP_LABEL == 1
                )
            )
        else:
            return False

    def generate_dsa_parameters(self, key_size):
        if key_size not in (1024, 2048, 3072, 4096):
            raise ValueError(
                "Key size must be 1024, 2048, 3072, or 4096 bits."
            )

        ctx = self._lib.DSA_new()
        self.openssl_assert(ctx != self._ffi.NULL)
        ctx = self._ffi.gc(ctx, self._lib.DSA_free)

        res = self._lib.DSA_generate_parameters_ex(
            ctx,
            key_size,
            self._ffi.NULL,
            0,
            self._ffi.NULL,
            self._ffi.NULL,
            self._ffi.NULL,
        )

        self.openssl_assert(res == 1)

        return _DSAParameters(self, ctx)

    def generate_dsa_private_key(self, parameters):
        ctx = self._lib.DSAparams_dup(parameters._dsa_cdata)
        self.openssl_assert(ctx != self._ffi.NULL)
        ctx = self._ffi.gc(ctx, self._lib.DSA_free)
        self._lib.DSA_generate_key(ctx)
        evp_pkey = self._dsa_cdata_to_evp_pkey(ctx)

        return _DSAPrivateKey(self, ctx, evp_pkey)

    def generate_dsa_private_key_and_parameters(self, key_size):
        parameters = self.generate_dsa_parameters(key_size)
        return self.generate_dsa_private_key(parameters)

    def _dsa_cdata_set_values(self, dsa_cdata, p, q, g, pub_key, priv_key):
        res = self._lib.DSA_set0_pqg(dsa_cdata, p, q, g)
        self.openssl_assert(res == 1)
        res = self._lib.DSA_set0_key(dsa_cdata, pub_key, priv_key)
        self.openssl_assert(res == 1)

    def load_dsa_private_numbers(self, numbers):
        dsa._check_dsa_private_numbers(numbers)
        parameter_numbers = numbers.public_numbers.parameter_numbers

        dsa_cdata = self._lib.DSA_new()
        self.openssl_assert(dsa_cdata != self._ffi.NULL)
        dsa_cdata = self._ffi.gc(dsa_cdata, self._lib.DSA_free)

        p = self._int_to_bn(parameter_numbers.p)
        q = self._int_to_bn(parameter_numbers.q)
        g = self._int_to_bn(parameter_numbers.g)
        pub_key = self._int_to_bn(numbers.public_numbers.y)
        priv_key = self._int_to_bn(numbers.x)
        self._dsa_cdata_set_values(dsa_cdata, p, q, g, pub_key, priv_key)

        evp_pkey = self._dsa_cdata_to_evp_pkey(dsa_cdata)

        return _DSAPrivateKey(self, dsa_cdata, evp_pkey)

    def load_dsa_public_numbers(self, numbers):
        dsa._check_dsa_parameters(numbers.parameter_numbers)
        dsa_cdata = self._lib.DSA_new()
        self.openssl_assert(dsa_cdata != self._ffi.NULL)
        dsa_cdata = self._ffi.gc(dsa_cdata, self._lib.DSA_free)

        p = self._int_to_bn(numbers.parameter_numbers.p)
        q = self._int_to_bn(numbers.parameter_numbers.q)
        g = self._int_to_bn(numbers.parameter_numbers.g)
        pub_key = self._int_to_bn(numbers.y)
        priv_key = self._ffi.NULL
        self._dsa_cdata_set_values(dsa_cdata, p, q, g, pub_key, priv_key)

        evp_pkey = self._dsa_cdata_to_evp_pkey(dsa_cdata)

        return _DSAPublicKey(self, dsa_cdata, evp_pkey)

    def load_dsa_parameter_numbers(self, numbers):
        dsa._check_dsa_parameters(numbers)
        dsa_cdata = self._lib.DSA_new()
        self.openssl_assert(dsa_cdata != self._ffi.NULL)
        dsa_cdata = self._ffi.gc(dsa_cdata, self._lib.DSA_free)

        p = self._int_to_bn(numbers.p)
        q = self._int_to_bn(numbers.q)
        g = self._int_to_bn(numbers.g)
        res = self._lib.DSA_set0_pqg(dsa_cdata, p, q, g)
        self.openssl_assert(res == 1)

        return _DSAParameters(self, dsa_cdata)

    def _dsa_cdata_to_evp_pkey(self, dsa_cdata):
        evp_pkey = self._create_evp_pkey_gc()
        res = self._lib.EVP_PKEY_set1_DSA(evp_pkey, dsa_cdata)
        self.openssl_assert(res == 1)
        return evp_pkey

    def dsa_hash_supported(self, algorithm):
        return self.hash_supported(algorithm)

    def dsa_parameters_supported(self, p, q, g):
        return True

    def cmac_algorithm_supported(self, algorithm):
        return self.cipher_supported(
            algorithm, CBC(b"\x00" * algorithm.block_size)
        )

    def create_cmac_ctx(self, algorithm):
        return _CMACContext(self, algorithm)

    def load_pem_private_key(self, data, password):
        return self._load_key(
            self._lib.PEM_read_bio_PrivateKey,
            self._evp_pkey_to_private_key,
            data,
            password,
        )

    def load_pem_public_key(self, data):
        mem_bio = self._bytes_to_bio(data)
        evp_pkey = self._lib.PEM_read_bio_PUBKEY(
            mem_bio.bio, self._ffi.NULL, self._ffi.NULL, self._ffi.NULL
        )
        if evp_pkey != self._ffi.NULL:
            evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)
            return self._evp_pkey_to_public_key(evp_pkey)
        else:
            # It's not a (RSA/DSA/ECDSA) subjectPublicKeyInfo, but we still
            # need to check to see if it is a pure PKCS1 RSA public key (not
            # embedded in a subjectPublicKeyInfo)
            self._consume_errors()
            res = self._lib.BIO_reset(mem_bio.bio)
            self.openssl_assert(res == 1)
            rsa_cdata = self._lib.PEM_read_bio_RSAPublicKey(
                mem_bio.bio, self._ffi.NULL, self._ffi.NULL, self._ffi.NULL
            )
            if rsa_cdata != self._ffi.NULL:
                rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
                evp_pkey = self._rsa_cdata_to_evp_pkey(rsa_cdata)
                return _RSAPublicKey(self, rsa_cdata, evp_pkey)
            else:
                self._handle_key_loading_error()

    def load_pem_parameters(self, data):
        mem_bio = self._bytes_to_bio(data)
        # only DH is supported currently
        dh_cdata = self._lib.PEM_read_bio_DHparams(
            mem_bio.bio, self._ffi.NULL, self._ffi.NULL, self._ffi.NULL
        )
        if dh_cdata != self._ffi.NULL:
            dh_cdata = self._ffi.gc(dh_cdata, self._lib.DH_free)
            return _DHParameters(self, dh_cdata)
        else:
            self._handle_key_loading_error()

    def load_der_private_key(self, data, password):
        # OpenSSL has a function called d2i_AutoPrivateKey that in theory
        # handles this automatically, however it doesn't handle encrypted
        # private keys. Instead we try to load the key two different ways.
        # First we'll try to load it as a traditional key.
        bio_data = self._bytes_to_bio(data)
        key = self._evp_pkey_from_der_traditional_key(bio_data, password)
        if key:
            return self._evp_pkey_to_private_key(key)
        else:
            # Finally we try to load it with the method that handles encrypted
            # PKCS8 properly.
            return self._load_key(
                self._lib.d2i_PKCS8PrivateKey_bio,
                self._evp_pkey_to_private_key,
                data,
                password,
            )

    def _evp_pkey_from_der_traditional_key(self, bio_data, password):
        key = self._lib.d2i_PrivateKey_bio(bio_data.bio, self._ffi.NULL)
        if key != self._ffi.NULL:
            # In OpenSSL 3.0.0-alpha15 there exist scenarios where the key will
            # successfully load but errors are still put on the stack. Tracked
            # as https://github.com/openssl/openssl/issues/14996
            self._consume_errors()

            key = self._ffi.gc(key, self._lib.EVP_PKEY_free)
            if password is not None:
                raise TypeError(
                    "Password was given but private key is not encrypted."
                )

            return key
        else:
            self._consume_errors()
            return None

    def load_der_public_key(self, data):
        mem_bio = self._bytes_to_bio(data)
        evp_pkey = self._lib.d2i_PUBKEY_bio(mem_bio.bio, self._ffi.NULL)
        if evp_pkey != self._ffi.NULL:
            evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)
            return self._evp_pkey_to_public_key(evp_pkey)
        else:
            # It's not a (RSA/DSA/ECDSA) subjectPublicKeyInfo, but we still
            # need to check to see if it is a pure PKCS1 RSA public key (not
            # embedded in a subjectPublicKeyInfo)
            self._consume_errors()
            res = self._lib.BIO_reset(mem_bio.bio)
            self.openssl_assert(res == 1)
            rsa_cdata = self._lib.d2i_RSAPublicKey_bio(
                mem_bio.bio, self._ffi.NULL
            )
            if rsa_cdata != self._ffi.NULL:
                rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
                evp_pkey = self._rsa_cdata_to_evp_pkey(rsa_cdata)
                return _RSAPublicKey(self, rsa_cdata, evp_pkey)
            else:
                self._handle_key_loading_error()

    def load_der_parameters(self, data):
        mem_bio = self._bytes_to_bio(data)
        dh_cdata = self._lib.d2i_DHparams_bio(mem_bio.bio, self._ffi.NULL)
        if dh_cdata != self._ffi.NULL:
            dh_cdata = self._ffi.gc(dh_cdata, self._lib.DH_free)
            return _DHParameters(self, dh_cdata)
        elif self._lib.Cryptography_HAS_EVP_PKEY_DHX:
            # We check to see if the is dhx.
            self._consume_errors()
            res = self._lib.BIO_reset(mem_bio.bio)
            self.openssl_assert(res == 1)
            dh_cdata = self._lib.Cryptography_d2i_DHxparams_bio(
                mem_bio.bio, self._ffi.NULL
            )
            if dh_cdata != self._ffi.NULL:
                dh_cdata = self._ffi.gc(dh_cdata, self._lib.DH_free)
                return _DHParameters(self, dh_cdata)

        self._handle_key_loading_error()

    def _cert2ossl(self, cert: x509.Certificate) -> typing.Any:
        data = cert.public_bytes(serialization.Encoding.DER)
        mem_bio = self._bytes_to_bio(data)
        x509 = self._lib.d2i_X509_bio(mem_bio.bio, self._ffi.NULL)
        self.openssl_assert(x509 != self._ffi.NULL)
        x509 = self._ffi.gc(x509, self._lib.X509_free)
        return x509

    def _ossl2cert(self, x509: typing.Any) -> x509.Certificate:
        bio = self._create_mem_bio_gc()
        res = self._lib.i2d_X509_bio(bio, x509)
        self.openssl_assert(res == 1)
        return rust_x509.load_der_x509_certificate(self._read_mem_bio(bio))

    def _csr2ossl(self, csr: x509.CertificateSigningRequest) -> typing.Any:
        data = csr.public_bytes(serialization.Encoding.DER)
        mem_bio = self._bytes_to_bio(data)
        x509_req = self._lib.d2i_X509_REQ_bio(mem_bio.bio, self._ffi.NULL)
        self.openssl_assert(x509_req != self._ffi.NULL)
        x509_req = self._ffi.gc(x509_req, self._lib.X509_REQ_free)
        return x509_req

    def _ossl2csr(
        self, x509_req: typing.Any
    ) -> x509.CertificateSigningRequest:
        bio = self._create_mem_bio_gc()
        res = self._lib.i2d_X509_REQ_bio(bio, x509_req)
        self.openssl_assert(res == 1)
        return rust_x509.load_der_x509_csr(self._read_mem_bio(bio))

    def _crl2ossl(self, crl: x509.CertificateRevocationList) -> typing.Any:
        data = crl.public_bytes(serialization.Encoding.DER)
        mem_bio = self._bytes_to_bio(data)
        x509_crl = self._lib.d2i_X509_CRL_bio(mem_bio.bio, self._ffi.NULL)
        self.openssl_assert(x509_crl != self._ffi.NULL)
        x509_crl = self._ffi.gc(x509_crl, self._lib.X509_CRL_free)
        return x509_crl

    def _ossl2crl(
        self, x509_crl: typing.Any
    ) -> x509.CertificateRevocationList:
        bio = self._create_mem_bio_gc()
        res = self._lib.i2d_X509_CRL_bio(bio, x509_crl)
        self.openssl_assert(res == 1)
        return rust_x509.load_der_x509_crl(self._read_mem_bio(bio))

    def _crl_is_signature_valid(
        self, crl: x509.CertificateRevocationList, public_key: PUBLIC_KEY_TYPES
    ) -> bool:
        if not isinstance(
            public_key,
            (
                _DSAPublicKey,
                _RSAPublicKey,
                _EllipticCurvePublicKey,
            ),
        ):
            raise TypeError(
                "Expecting one of DSAPublicKey, RSAPublicKey,"
                " or EllipticCurvePublicKey."
            )
        x509_crl = self._crl2ossl(crl)
        res = self._lib.X509_CRL_verify(x509_crl, public_key._evp_pkey)

        if res != 1:
            self._consume_errors()
            return False

        return True

    def _csr_is_signature_valid(
        self, csr: x509.CertificateSigningRequest
    ) -> bool:
        x509_req = self._csr2ossl(csr)
        pkey = self._lib.X509_REQ_get_pubkey(x509_req)
        self.openssl_assert(pkey != self._ffi.NULL)
        pkey = self._ffi.gc(pkey, self._lib.EVP_PKEY_free)
        res = self._lib.X509_REQ_verify(x509_req, pkey)

        if res != 1:
            self._consume_errors()
            return False

        return True

    def _check_keys_correspond(self, key1, key2):
        if self._lib.EVP_PKEY_cmp(key1._evp_pkey, key2._evp_pkey) != 1:
            raise ValueError("Keys do not correspond")

    def _load_key(self, openssl_read_func, convert_func, data, password):
        mem_bio = self._bytes_to_bio(data)

        userdata = self._ffi.new("CRYPTOGRAPHY_PASSWORD_DATA *")
        if password is not None:
            utils._check_byteslike("password", password)
            password_ptr = self._ffi.from_buffer(password)
            userdata.password = password_ptr
            userdata.length = len(password)

        evp_pkey = openssl_read_func(
            mem_bio.bio,
            self._ffi.NULL,
            self._ffi.addressof(
                self._lib._original_lib, "Cryptography_pem_password_cb"
            ),
            userdata,
        )

        if evp_pkey == self._ffi.NULL:
            if userdata.error != 0:
                self._consume_errors()
                if userdata.error == -1:
                    raise TypeError(
                        "Password was not given but private key is encrypted"
                    )
                else:
                    assert userdata.error == -2
                    raise ValueError(
                        "Passwords longer than {} bytes are not supported "
                        "by this backend.".format(userdata.maxsize - 1)
                    )
            else:
                self._handle_key_loading_error()

        # In OpenSSL 3.0.0-alpha15 there exist scenarios where the key will
        # successfully load but errors are still put on the stack. Tracked
        # as https://github.com/openssl/openssl/issues/14996
        self._consume_errors()

        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)

        if password is not None and userdata.called == 0:
            raise TypeError(
                "Password was given but private key is not encrypted."
            )

        assert (
            password is not None and userdata.called == 1
        ) or password is None

        return convert_func(evp_pkey)

    def _handle_key_loading_error(self):
        errors = self._consume_errors()

        if not errors:
            raise ValueError(
                "Could not deserialize key data. The data may be in an "
                "incorrect format or it may be encrypted with an unsupported "
                "algorithm."
            )

        elif (
            errors[0]._lib_reason_match(
                self._lib.ERR_LIB_EVP, self._lib.EVP_R_BAD_DECRYPT
            )
            or errors[0]._lib_reason_match(
                self._lib.ERR_LIB_PKCS12,
                self._lib.PKCS12_R_PKCS12_CIPHERFINAL_ERROR,
            )
            or (
                self._lib.Cryptography_HAS_PROVIDERS
                and errors[0]._lib_reason_match(
                    self._lib.ERR_LIB_PROV,
                    self._lib.PROV_R_BAD_DECRYPT,
                )
            )
        ):
            raise ValueError("Bad decrypt. Incorrect password?")

        elif any(
            error._lib_reason_match(
                self._lib.ERR_LIB_EVP,
                self._lib.EVP_R_UNSUPPORTED_PRIVATE_KEY_ALGORITHM,
            )
            for error in errors
        ):
            raise ValueError("Unsupported public key algorithm.")

        else:
            errors = binding._errors_with_text(errors)
            raise ValueError(
                "Could not deserialize key data. The data may be in an "
                "incorrect format, it may be encrypted with an unsupported "
                "algorithm, or it may be an unsupported key type (e.g. EC "
                "curves with explicit parameters).",
                errors,
            )

    def elliptic_curve_supported(self, curve):
        try:
            curve_nid = self._elliptic_curve_to_nid(curve)
        except UnsupportedAlgorithm:
            curve_nid = self._lib.NID_undef

        group = self._lib.EC_GROUP_new_by_curve_name(curve_nid)

        if group == self._ffi.NULL:
            self._consume_errors()
            return False
        else:
            self.openssl_assert(curve_nid != self._lib.NID_undef)
            self._lib.EC_GROUP_free(group)
            return True

    def elliptic_curve_signature_algorithm_supported(
        self, signature_algorithm, curve
    ):
        # We only support ECDSA right now.
        if not isinstance(signature_algorithm, ec.ECDSA):
            return False

        return self.elliptic_curve_supported(curve)

    def generate_elliptic_curve_private_key(self, curve):
        """
        Generate a new private key on the named curve.
        """

        if self.elliptic_curve_supported(curve):
            ec_cdata = self._ec_key_new_by_curve(curve)

            res = self._lib.EC_KEY_generate_key(ec_cdata)
            self.openssl_assert(res == 1)

            evp_pkey = self._ec_cdata_to_evp_pkey(ec_cdata)

            return _EllipticCurvePrivateKey(self, ec_cdata, evp_pkey)
        else:
            raise UnsupportedAlgorithm(
                "Backend object does not support {}.".format(curve.name),
                _Reasons.UNSUPPORTED_ELLIPTIC_CURVE,
            )

    def load_elliptic_curve_private_numbers(self, numbers):
        public = numbers.public_numbers

        ec_cdata = self._ec_key_new_by_curve(public.curve)

        private_value = self._ffi.gc(
            self._int_to_bn(numbers.private_value), self._lib.BN_clear_free
        )
        res = self._lib.EC_KEY_set_private_key(ec_cdata, private_value)
        if res != 1:
            self._consume_errors()
            raise ValueError("Invalid EC key.")

        ec_cdata = self._ec_key_set_public_key_affine_coordinates(
            ec_cdata, public.x, public.y
        )

        evp_pkey = self._ec_cdata_to_evp_pkey(ec_cdata)

        return _EllipticCurvePrivateKey(self, ec_cdata, evp_pkey)

    def load_elliptic_curve_public_numbers(self, numbers):
        ec_cdata = self._ec_key_new_by_curve(numbers.curve)
        ec_cdata = self._ec_key_set_public_key_affine_coordinates(
            ec_cdata, numbers.x, numbers.y
        )
        evp_pkey = self._ec_cdata_to_evp_pkey(ec_cdata)

        return _EllipticCurvePublicKey(self, ec_cdata, evp_pkey)

    def load_elliptic_curve_public_bytes(self, curve, point_bytes):
        ec_cdata = self._ec_key_new_by_curve(curve)
        group = self._lib.EC_KEY_get0_group(ec_cdata)
        self.openssl_assert(group != self._ffi.NULL)
        point = self._lib.EC_POINT_new(group)
        self.openssl_assert(point != self._ffi.NULL)
        point = self._ffi.gc(point, self._lib.EC_POINT_free)
        with self._tmp_bn_ctx() as bn_ctx:
            res = self._lib.EC_POINT_oct2point(
                group, point, point_bytes, len(point_bytes), bn_ctx
            )
            if res != 1:
                self._consume_errors()
                raise ValueError("Invalid public bytes for the given curve")

        res = self._lib.EC_KEY_set_public_key(ec_cdata, point)
        self.openssl_assert(res == 1)
        evp_pkey = self._ec_cdata_to_evp_pkey(ec_cdata)
        return _EllipticCurvePublicKey(self, ec_cdata, evp_pkey)

    def derive_elliptic_curve_private_key(self, private_value, curve):
        ec_cdata = self._ec_key_new_by_curve(curve)

        get_func, group = self._ec_key_determine_group_get_func(ec_cdata)

        point = self._lib.EC_POINT_new(group)
        self.openssl_assert(point != self._ffi.NULL)
        point = self._ffi.gc(point, self._lib.EC_POINT_free)

        value = self._int_to_bn(private_value)
        value = self._ffi.gc(value, self._lib.BN_clear_free)

        with self._tmp_bn_ctx() as bn_ctx:
            res = self._lib.EC_POINT_mul(
                group, point, value, self._ffi.NULL, self._ffi.NULL, bn_ctx
            )
            self.openssl_assert(res == 1)

            bn_x = self._lib.BN_CTX_get(bn_ctx)
            bn_y = self._lib.BN_CTX_get(bn_ctx)

            res = get_func(group, point, bn_x, bn_y, bn_ctx)
            if res != 1:
                self._consume_errors()
                raise ValueError("Unable to derive key from private_value")

        res = self._lib.EC_KEY_set_public_key(ec_cdata, point)
        self.openssl_assert(res == 1)
        private = self._int_to_bn(private_value)
        private = self._ffi.gc(private, self._lib.BN_clear_free)
        res = self._lib.EC_KEY_set_private_key(ec_cdata, private)
        self.openssl_assert(res == 1)

        evp_pkey = self._ec_cdata_to_evp_pkey(ec_cdata)

        return _EllipticCurvePrivateKey(self, ec_cdata, evp_pkey)

    def _ec_key_new_by_curve(self, curve):
        curve_nid = self._elliptic_curve_to_nid(curve)
        return self._ec_key_new_by_curve_nid(curve_nid)

    def _ec_key_new_by_curve_nid(self, curve_nid):
        ec_cdata = self._lib.EC_KEY_new_by_curve_name(curve_nid)
        self.openssl_assert(ec_cdata != self._ffi.NULL)
        return self._ffi.gc(ec_cdata, self._lib.EC_KEY_free)

    def elliptic_curve_exchange_algorithm_supported(self, algorithm, curve):
        if self._fips_enabled and not isinstance(
            curve, self._fips_ecdh_curves
        ):
            return False

        return self.elliptic_curve_supported(curve) and isinstance(
            algorithm, ec.ECDH
        )

    def _ec_cdata_to_evp_pkey(self, ec_cdata):
        evp_pkey = self._create_evp_pkey_gc()
        res = self._lib.EVP_PKEY_set1_EC_KEY(evp_pkey, ec_cdata)
        self.openssl_assert(res == 1)
        return evp_pkey

    def _elliptic_curve_to_nid(self, curve):
        """
        Get the NID for a curve name.
        """

        curve_aliases = {"secp192r1": "prime192v1", "secp256r1": "prime256v1"}

        curve_name = curve_aliases.get(curve.name, curve.name)

        curve_nid = self._lib.OBJ_sn2nid(curve_name.encode())
        if curve_nid == self._lib.NID_undef:
            raise UnsupportedAlgorithm(
                "{} is not a supported elliptic curve".format(curve.name),
                _Reasons.UNSUPPORTED_ELLIPTIC_CURVE,
            )
        return curve_nid

    @contextmanager
    def _tmp_bn_ctx(self):
        bn_ctx = self._lib.BN_CTX_new()
        self.openssl_assert(bn_ctx != self._ffi.NULL)
        bn_ctx = self._ffi.gc(bn_ctx, self._lib.BN_CTX_free)
        self._lib.BN_CTX_start(bn_ctx)
        try:
            yield bn_ctx
        finally:
            self._lib.BN_CTX_end(bn_ctx)

    def _ec_key_determine_group_get_func(self, ctx):
        """
        Given an EC_KEY determine the group and what function is required to
        get point coordinates.
        """
        self.openssl_assert(ctx != self._ffi.NULL)

        nid_two_field = self._lib.OBJ_sn2nid(b"characteristic-two-field")
        self.openssl_assert(nid_two_field != self._lib.NID_undef)

        group = self._lib.EC_KEY_get0_group(ctx)
        self.openssl_assert(group != self._ffi.NULL)

        method = self._lib.EC_GROUP_method_of(group)
        self.openssl_assert(method != self._ffi.NULL)

        nid = self._lib.EC_METHOD_get_field_type(method)
        self.openssl_assert(nid != self._lib.NID_undef)

        if nid == nid_two_field and self._lib.Cryptography_HAS_EC2M:
            get_func = self._lib.EC_POINT_get_affine_coordinates_GF2m
        else:
            get_func = self._lib.EC_POINT_get_affine_coordinates_GFp

        assert get_func

        return get_func, group

    def _ec_key_set_public_key_affine_coordinates(self, ctx, x, y):
        """
        Sets the public key point in the EC_KEY context to the affine x and y
        values.
        """

        if x < 0 or y < 0:
            raise ValueError(
                "Invalid EC key. Both x and y must be non-negative."
            )

        x = self._ffi.gc(self._int_to_bn(x), self._lib.BN_free)
        y = self._ffi.gc(self._int_to_bn(y), self._lib.BN_free)
        res = self._lib.EC_KEY_set_public_key_affine_coordinates(ctx, x, y)
        if res != 1:
            self._consume_errors()
            raise ValueError("Invalid EC key.")

        return ctx

    def _private_key_bytes(
        self, encoding, format, encryption_algorithm, key, evp_pkey, cdata
    ):
        # validate argument types
        if not isinstance(encoding, serialization.Encoding):
            raise TypeError("encoding must be an item from the Encoding enum")
        if not isinstance(format, serialization.PrivateFormat):
            raise TypeError(
                "format must be an item from the PrivateFormat enum"
            )
        if not isinstance(
            encryption_algorithm, serialization.KeySerializationEncryption
        ):
            raise TypeError(
                "Encryption algorithm must be a KeySerializationEncryption "
                "instance"
            )

        # validate password
        if isinstance(encryption_algorithm, serialization.NoEncryption):
            password = b""
        elif isinstance(
            encryption_algorithm, serialization.BestAvailableEncryption
        ):
            password = encryption_algorithm.password
            if len(password) > 1023:
                raise ValueError(
                    "Passwords longer than 1023 bytes are not supported by "
                    "this backend"
                )
        else:
            raise ValueError("Unsupported encryption type")

        # PKCS8 + PEM/DER
        if format is serialization.PrivateFormat.PKCS8:
            if encoding is serialization.Encoding.PEM:
                write_bio = self._lib.PEM_write_bio_PKCS8PrivateKey
            elif encoding is serialization.Encoding.DER:
                write_bio = self._lib.i2d_PKCS8PrivateKey_bio
            else:
                raise ValueError("Unsupported encoding for PKCS8")
            return self._private_key_bytes_via_bio(
                write_bio, evp_pkey, password
            )

        # TraditionalOpenSSL + PEM/DER
        if format is serialization.PrivateFormat.TraditionalOpenSSL:
            if self._fips_enabled and not isinstance(
                encryption_algorithm, serialization.NoEncryption
            ):
                raise ValueError(
                    "Encrypted traditional OpenSSL format is not "
                    "supported in FIPS mode."
                )
            key_type = self._lib.EVP_PKEY_id(evp_pkey)

            if encoding is serialization.Encoding.PEM:
                if key_type == self._lib.EVP_PKEY_RSA:
                    write_bio = self._lib.PEM_write_bio_RSAPrivateKey
                elif key_type == self._lib.EVP_PKEY_DSA:
                    write_bio = self._lib.PEM_write_bio_DSAPrivateKey
                elif key_type == self._lib.EVP_PKEY_EC:
                    write_bio = self._lib.PEM_write_bio_ECPrivateKey
                else:
                    raise ValueError(
                        "Unsupported key type for TraditionalOpenSSL"
                    )
                return self._private_key_bytes_via_bio(
                    write_bio, cdata, password
                )

            if encoding is serialization.Encoding.DER:
                if password:
                    raise ValueError(
                        "Encryption is not supported for DER encoded "
                        "traditional OpenSSL keys"
                    )
                if key_type == self._lib.EVP_PKEY_RSA:
                    write_bio = self._lib.i2d_RSAPrivateKey_bio
                elif key_type == self._lib.EVP_PKEY_EC:
                    write_bio = self._lib.i2d_ECPrivateKey_bio
                elif key_type == self._lib.EVP_PKEY_DSA:
                    write_bio = self._lib.i2d_DSAPrivateKey_bio
                else:
                    raise ValueError(
                        "Unsupported key type for TraditionalOpenSSL"
                    )
                return self._bio_func_output(write_bio, cdata)

            raise ValueError("Unsupported encoding for TraditionalOpenSSL")

        # OpenSSH + PEM
        if format is serialization.PrivateFormat.OpenSSH:
            if encoding is serialization.Encoding.PEM:
                return ssh.serialize_ssh_private_key(key, password)

            raise ValueError(
                "OpenSSH private key format can only be used"
                " with PEM encoding"
            )

        # Anything that key-specific code was supposed to handle earlier,
        # like Raw.
        raise ValueError("format is invalid with this key")

    def _private_key_bytes_via_bio(self, write_bio, evp_pkey, password):
        if not password:
            evp_cipher = self._ffi.NULL
        else:
            # This is a curated value that we will update over time.
            evp_cipher = self._lib.EVP_get_cipherbyname(b"aes-256-cbc")

        return self._bio_func_output(
            write_bio,
            evp_pkey,
            evp_cipher,
            password,
            len(password),
            self._ffi.NULL,
            self._ffi.NULL,
        )

    def _bio_func_output(self, write_bio, *args):
        bio = self._create_mem_bio_gc()
        res = write_bio(bio, *args)
        self.openssl_assert(res == 1)
        return self._read_mem_bio(bio)

    def _public_key_bytes(self, encoding, format, key, evp_pkey, cdata):
        if not isinstance(encoding, serialization.Encoding):
            raise TypeError("encoding must be an item from the Encoding enum")
        if not isinstance(format, serialization.PublicFormat):
            raise TypeError(
                "format must be an item from the PublicFormat enum"
            )

        # SubjectPublicKeyInfo + PEM/DER
        if format is serialization.PublicFormat.SubjectPublicKeyInfo:
            if encoding is serialization.Encoding.PEM:
                write_bio = self._lib.PEM_write_bio_PUBKEY
            elif encoding is serialization.Encoding.DER:
                write_bio = self._lib.i2d_PUBKEY_bio
            else:
                raise ValueError(
                    "SubjectPublicKeyInfo works only with PEM or DER encoding"
                )
            return self._bio_func_output(write_bio, evp_pkey)

        # PKCS1 + PEM/DER
        if format is serialization.PublicFormat.PKCS1:
            # Only RSA is supported here.
            key_type = self._lib.EVP_PKEY_id(evp_pkey)
            if key_type != self._lib.EVP_PKEY_RSA:
                raise ValueError("PKCS1 format is supported only for RSA keys")

            if encoding is serialization.Encoding.PEM:
                write_bio = self._lib.PEM_write_bio_RSAPublicKey
            elif encoding is serialization.Encoding.DER:
                write_bio = self._lib.i2d_RSAPublicKey_bio
            else:
                raise ValueError("PKCS1 works only with PEM or DER encoding")
            return self._bio_func_output(write_bio, cdata)

        # OpenSSH + OpenSSH
        if format is serialization.PublicFormat.OpenSSH:
            if encoding is serialization.Encoding.OpenSSH:
                return ssh.serialize_ssh_public_key(key)

            raise ValueError(
                "OpenSSH format must be used with OpenSSH encoding"
            )

        # Anything that key-specific code was supposed to handle earlier,
        # like Raw, CompressedPoint, UncompressedPoint
        raise ValueError("format is invalid with this key")

    def _parameter_bytes(self, encoding, format, cdata):
        if encoding is serialization.Encoding.OpenSSH:
            raise TypeError("OpenSSH encoding is not supported")

        # Only DH is supported here currently.
        q = self._ffi.new("BIGNUM **")
        self._lib.DH_get0_pqg(cdata, self._ffi.NULL, q, self._ffi.NULL)
        if encoding is serialization.Encoding.PEM:
            if q[0] != self._ffi.NULL:
                write_bio = self._lib.PEM_write_bio_DHxparams
            else:
                write_bio = self._lib.PEM_write_bio_DHparams
        elif encoding is serialization.Encoding.DER:
            if q[0] != self._ffi.NULL:
                write_bio = self._lib.Cryptography_i2d_DHxparams_bio
            else:
                write_bio = self._lib.i2d_DHparams_bio
        else:
            raise TypeError("encoding must be an item from the Encoding enum")

        bio = self._create_mem_bio_gc()
        res = write_bio(bio, cdata)
        self.openssl_assert(res == 1)
        return self._read_mem_bio(bio)

    def dh_supported(self):
        return not self._lib.CRYPTOGRAPHY_IS_BORINGSSL

    def generate_dh_parameters(self, generator, key_size):
        if key_size < dh._MIN_MODULUS_SIZE:
            raise ValueError(
                "DH key_size must be at least {} bits".format(
                    dh._MIN_MODULUS_SIZE
                )
            )

        if generator not in (2, 5):
            raise ValueError("DH generator must be 2 or 5")

        dh_param_cdata = self._lib.DH_new()
        self.openssl_assert(dh_param_cdata != self._ffi.NULL)
        dh_param_cdata = self._ffi.gc(dh_param_cdata, self._lib.DH_free)

        res = self._lib.DH_generate_parameters_ex(
            dh_param_cdata, key_size, generator, self._ffi.NULL
        )
        self.openssl_assert(res == 1)

        return _DHParameters(self, dh_param_cdata)

    def _dh_cdata_to_evp_pkey(self, dh_cdata):
        evp_pkey = self._create_evp_pkey_gc()
        res = self._lib.EVP_PKEY_set1_DH(evp_pkey, dh_cdata)
        self.openssl_assert(res == 1)
        return evp_pkey

    def generate_dh_private_key(self, parameters):
        dh_key_cdata = _dh_params_dup(parameters._dh_cdata, self)

        res = self._lib.DH_generate_key(dh_key_cdata)
        self.openssl_assert(res == 1)

        evp_pkey = self._dh_cdata_to_evp_pkey(dh_key_cdata)

        return _DHPrivateKey(self, dh_key_cdata, evp_pkey)

    def generate_dh_private_key_and_parameters(self, generator, key_size):
        return self.generate_dh_private_key(
            self.generate_dh_parameters(generator, key_size)
        )

    def load_dh_private_numbers(self, numbers):
        parameter_numbers = numbers.public_numbers.parameter_numbers

        dh_cdata = self._lib.DH_new()
        self.openssl_assert(dh_cdata != self._ffi.NULL)
        dh_cdata = self._ffi.gc(dh_cdata, self._lib.DH_free)

        p = self._int_to_bn(parameter_numbers.p)
        g = self._int_to_bn(parameter_numbers.g)

        if parameter_numbers.q is not None:
            q = self._int_to_bn(parameter_numbers.q)
        else:
            q = self._ffi.NULL

        pub_key = self._int_to_bn(numbers.public_numbers.y)
        priv_key = self._int_to_bn(numbers.x)

        res = self._lib.DH_set0_pqg(dh_cdata, p, q, g)
        self.openssl_assert(res == 1)

        res = self._lib.DH_set0_key(dh_cdata, pub_key, priv_key)
        self.openssl_assert(res == 1)

        codes = self._ffi.new("int[]", 1)
        res = self._lib.Cryptography_DH_check(dh_cdata, codes)
        self.openssl_assert(res == 1)

        # DH_check will return DH_NOT_SUITABLE_GENERATOR if p % 24 does not
        # equal 11 when the generator is 2 (a quadratic nonresidue).
        # We want to ignore that error because p % 24 == 23 is also fine.
        # Specifically, g is then a quadratic residue. Within the context of
        # Diffie-Hellman this means it can only generate half the possible
        # values. That sounds bad, but quadratic nonresidues leak a bit of
        # the key to the attacker in exchange for having the full key space
        # available. See: https://crypto.stackexchange.com/questions/12961
        if codes[0] != 0 and not (
            parameter_numbers.g == 2
            and codes[0] ^ self._lib.DH_NOT_SUITABLE_GENERATOR == 0
        ):
            raise ValueError("DH private numbers did not pass safety checks.")

        evp_pkey = self._dh_cdata_to_evp_pkey(dh_cdata)

        return _DHPrivateKey(self, dh_cdata, evp_pkey)

    def load_dh_public_numbers(self, numbers):
        dh_cdata = self._lib.DH_new()
        self.openssl_assert(dh_cdata != self._ffi.NULL)
        dh_cdata = self._ffi.gc(dh_cdata, self._lib.DH_free)

        parameter_numbers = numbers.parameter_numbers

        p = self._int_to_bn(parameter_numbers.p)
        g = self._int_to_bn(parameter_numbers.g)

        if parameter_numbers.q is not None:
            q = self._int_to_bn(parameter_numbers.q)
        else:
            q = self._ffi.NULL

        pub_key = self._int_to_bn(numbers.y)

        res = self._lib.DH_set0_pqg(dh_cdata, p, q, g)
        self.openssl_assert(res == 1)

        res = self._lib.DH_set0_key(dh_cdata, pub_key, self._ffi.NULL)
        self.openssl_assert(res == 1)

        evp_pkey = self._dh_cdata_to_evp_pkey(dh_cdata)

        return _DHPublicKey(self, dh_cdata, evp_pkey)

    def load_dh_parameter_numbers(self, numbers):
        dh_cdata = self._lib.DH_new()
        self.openssl_assert(dh_cdata != self._ffi.NULL)
        dh_cdata = self._ffi.gc(dh_cdata, self._lib.DH_free)

        p = self._int_to_bn(numbers.p)
        g = self._int_to_bn(numbers.g)

        if numbers.q is not None:
            q = self._int_to_bn(numbers.q)
        else:
            q = self._ffi.NULL

        res = self._lib.DH_set0_pqg(dh_cdata, p, q, g)
        self.openssl_assert(res == 1)

        return _DHParameters(self, dh_cdata)

    def dh_parameters_supported(self, p, g, q=None):
        dh_cdata = self._lib.DH_new()
        self.openssl_assert(dh_cdata != self._ffi.NULL)
        dh_cdata = self._ffi.gc(dh_cdata, self._lib.DH_free)

        p = self._int_to_bn(p)
        g = self._int_to_bn(g)

        if q is not None:
            q = self._int_to_bn(q)
        else:
            q = self._ffi.NULL

        res = self._lib.DH_set0_pqg(dh_cdata, p, q, g)
        self.openssl_assert(res == 1)

        codes = self._ffi.new("int[]", 1)
        res = self._lib.Cryptography_DH_check(dh_cdata, codes)
        self.openssl_assert(res == 1)

        return codes[0] == 0

    def dh_x942_serialization_supported(self):
        return self._lib.Cryptography_HAS_EVP_PKEY_DHX == 1

    def x25519_load_public_bytes(self, data):
        # When we drop support for CRYPTOGRAPHY_OPENSSL_LESS_THAN_111 we can
        # switch this to EVP_PKEY_new_raw_public_key
        if len(data) != 32:
            raise ValueError("An X25519 public key is 32 bytes long")

        evp_pkey = self._create_evp_pkey_gc()
        res = self._lib.EVP_PKEY_set_type(evp_pkey, self._lib.NID_X25519)
        self.openssl_assert(res == 1)
        res = self._lib.EVP_PKEY_set1_tls_encodedpoint(
            evp_pkey, data, len(data)
        )
        self.openssl_assert(res == 1)
        return _X25519PublicKey(self, evp_pkey)

    def x25519_load_private_bytes(self, data):
        # When we drop support for CRYPTOGRAPHY_OPENSSL_LESS_THAN_111 we can
        # switch this to EVP_PKEY_new_raw_private_key and drop the
        # zeroed_bytearray garbage.
        # OpenSSL only has facilities for loading PKCS8 formatted private
        # keys using the algorithm identifiers specified in
        # https://tools.ietf.org/html/draft-ietf-curdle-pkix-09.
        # This is the standard PKCS8 prefix for a 32 byte X25519 key.
        # The form is:
        #    0:d=0  hl=2 l=  46 cons: SEQUENCE
        #    2:d=1  hl=2 l=   1 prim: INTEGER           :00
        #    5:d=1  hl=2 l=   5 cons: SEQUENCE
        #    7:d=2  hl=2 l=   3 prim: OBJECT            :1.3.101.110
        #    12:d=1  hl=2 l=  34 prim: OCTET STRING      (the key)
        # Of course there's a bit more complexity. In reality OCTET STRING
        # contains an OCTET STRING of length 32! So the last two bytes here
        # are \x04\x20, which is an OCTET STRING of length 32.
        if len(data) != 32:
            raise ValueError("An X25519 private key is 32 bytes long")

        pkcs8_prefix = b'0.\x02\x01\x000\x05\x06\x03+en\x04"\x04 '
        with self._zeroed_bytearray(48) as ba:
            ba[0:16] = pkcs8_prefix
            ba[16:] = data
            bio = self._bytes_to_bio(ba)
            evp_pkey = self._lib.d2i_PrivateKey_bio(bio.bio, self._ffi.NULL)

        self.openssl_assert(evp_pkey != self._ffi.NULL)
        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)
        self.openssl_assert(
            self._lib.EVP_PKEY_id(evp_pkey) == self._lib.EVP_PKEY_X25519
        )
        return _X25519PrivateKey(self, evp_pkey)

    def _evp_pkey_keygen_gc(self, nid):
        evp_pkey_ctx = self._lib.EVP_PKEY_CTX_new_id(nid, self._ffi.NULL)
        self.openssl_assert(evp_pkey_ctx != self._ffi.NULL)
        evp_pkey_ctx = self._ffi.gc(evp_pkey_ctx, self._lib.EVP_PKEY_CTX_free)
        res = self._lib.EVP_PKEY_keygen_init(evp_pkey_ctx)
        self.openssl_assert(res == 1)
        evp_ppkey = self._ffi.new("EVP_PKEY **")
        res = self._lib.EVP_PKEY_keygen(evp_pkey_ctx, evp_ppkey)
        self.openssl_assert(res == 1)
        self.openssl_assert(evp_ppkey[0] != self._ffi.NULL)
        evp_pkey = self._ffi.gc(evp_ppkey[0], self._lib.EVP_PKEY_free)
        return evp_pkey

    def x25519_generate_key(self):
        evp_pkey = self._evp_pkey_keygen_gc(self._lib.NID_X25519)
        return _X25519PrivateKey(self, evp_pkey)

    def x25519_supported(self):
        if self._fips_enabled:
            return False
        return not self._lib.CRYPTOGRAPHY_IS_LIBRESSL

    def x448_load_public_bytes(self, data):
        if len(data) != 56:
            raise ValueError("An X448 public key is 56 bytes long")

        evp_pkey = self._lib.EVP_PKEY_new_raw_public_key(
            self._lib.NID_X448, self._ffi.NULL, data, len(data)
        )
        self.openssl_assert(evp_pkey != self._ffi.NULL)
        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)
        return _X448PublicKey(self, evp_pkey)

    def x448_load_private_bytes(self, data):
        if len(data) != 56:
            raise ValueError("An X448 private key is 56 bytes long")

        data_ptr = self._ffi.from_buffer(data)
        evp_pkey = self._lib.EVP_PKEY_new_raw_private_key(
            self._lib.NID_X448, self._ffi.NULL, data_ptr, len(data)
        )
        self.openssl_assert(evp_pkey != self._ffi.NULL)
        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)
        return _X448PrivateKey(self, evp_pkey)

    def x448_generate_key(self):
        evp_pkey = self._evp_pkey_keygen_gc(self._lib.NID_X448)
        return _X448PrivateKey(self, evp_pkey)

    def x448_supported(self):
        if self._fips_enabled:
            return False
        return (
            not self._lib.CRYPTOGRAPHY_OPENSSL_LESS_THAN_111
            and not self._lib.CRYPTOGRAPHY_IS_BORINGSSL
        )

    def ed25519_supported(self):
        if self._fips_enabled:
            return False
        return not self._lib.CRYPTOGRAPHY_OPENSSL_LESS_THAN_111B

    def ed25519_load_public_bytes(self, data):
        utils._check_bytes("data", data)

        if len(data) != ed25519._ED25519_KEY_SIZE:
            raise ValueError("An Ed25519 public key is 32 bytes long")

        evp_pkey = self._lib.EVP_PKEY_new_raw_public_key(
            self._lib.NID_ED25519, self._ffi.NULL, data, len(data)
        )
        self.openssl_assert(evp_pkey != self._ffi.NULL)
        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)

        return _Ed25519PublicKey(self, evp_pkey)

    def ed25519_load_private_bytes(self, data):
        if len(data) != ed25519._ED25519_KEY_SIZE:
            raise ValueError("An Ed25519 private key is 32 bytes long")

        utils._check_byteslike("data", data)
        data_ptr = self._ffi.from_buffer(data)
        evp_pkey = self._lib.EVP_PKEY_new_raw_private_key(
            self._lib.NID_ED25519, self._ffi.NULL, data_ptr, len(data)
        )
        self.openssl_assert(evp_pkey != self._ffi.NULL)
        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)

        return _Ed25519PrivateKey(self, evp_pkey)

    def ed25519_generate_key(self):
        evp_pkey = self._evp_pkey_keygen_gc(self._lib.NID_ED25519)
        return _Ed25519PrivateKey(self, evp_pkey)

    def ed448_supported(self):
        if self._fips_enabled:
            return False
        return (
            not self._lib.CRYPTOGRAPHY_OPENSSL_LESS_THAN_111B
            and not self._lib.CRYPTOGRAPHY_IS_BORINGSSL
        )

    def ed448_load_public_bytes(self, data):
        utils._check_bytes("data", data)
        if len(data) != _ED448_KEY_SIZE:
            raise ValueError("An Ed448 public key is 57 bytes long")

        evp_pkey = self._lib.EVP_PKEY_new_raw_public_key(
            self._lib.NID_ED448, self._ffi.NULL, data, len(data)
        )
        self.openssl_assert(evp_pkey != self._ffi.NULL)
        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)

        return _Ed448PublicKey(self, evp_pkey)

    def ed448_load_private_bytes(self, data):
        utils._check_byteslike("data", data)
        if len(data) != _ED448_KEY_SIZE:
            raise ValueError("An Ed448 private key is 57 bytes long")

        data_ptr = self._ffi.from_buffer(data)
        evp_pkey = self._lib.EVP_PKEY_new_raw_private_key(
            self._lib.NID_ED448, self._ffi.NULL, data_ptr, len(data)
        )
        self.openssl_assert(evp_pkey != self._ffi.NULL)
        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)

        return _Ed448PrivateKey(self, evp_pkey)

    def ed448_generate_key(self):
        evp_pkey = self._evp_pkey_keygen_gc(self._lib.NID_ED448)
        return _Ed448PrivateKey(self, evp_pkey)

    def derive_scrypt(self, key_material, salt, length, n, r, p):
        buf = self._ffi.new("unsigned char[]", length)
        key_material_ptr = self._ffi.from_buffer(key_material)
        res = self._lib.EVP_PBE_scrypt(
            key_material_ptr,
            len(key_material),
            salt,
            len(salt),
            n,
            r,
            p,
            scrypt._MEM_LIMIT,
            buf,
            length,
        )
        if res != 1:
            errors = self._consume_errors_with_text()
            # memory required formula explained here:
            # https://blog.filippo.io/the-scrypt-parameters/
            min_memory = 128 * n * r // (1024 ** 2)
            raise MemoryError(
                "Not enough memory to derive key. These parameters require"
                " {} MB of memory.".format(min_memory),
                errors,
            )
        return self._ffi.buffer(buf)[:]

    def aead_cipher_supported(self, cipher):
        cipher_name = aead._aead_cipher_name(cipher)
        if self._fips_enabled and cipher_name not in self._fips_aead:
            return False
        return self._lib.EVP_get_cipherbyname(cipher_name) != self._ffi.NULL

    @contextlib.contextmanager
    def _zeroed_bytearray(self, length):
        """
        This method creates a bytearray, which we copy data into (hopefully
        also from a mutable buffer that can be dynamically erased!), and then
        zero when we're done.
        """
        ba = bytearray(length)
        try:
            yield ba
        finally:
            self._zero_data(ba, length)

    def _zero_data(self, data, length):
        # We clear things this way because at the moment we're not
        # sure of a better way that can guarantee it overwrites the
        # memory of a bytearray and doesn't just replace the underlying char *.
        for i in range(length):
            data[i] = 0

    @contextlib.contextmanager
    def _zeroed_null_terminated_buf(self, data):
        """
        This method takes bytes, which can be a bytestring or a mutable
        buffer like a bytearray, and yields a null-terminated version of that
        data. This is required because PKCS12_parse doesn't take a length with
        its password char * and ffi.from_buffer doesn't provide null
        termination. So, to support zeroing the data via bytearray we
        need to build this ridiculous construct that copies the memory, but
        zeroes it after use.
        """
        if data is None:
            yield self._ffi.NULL
        else:
            data_len = len(data)
            buf = self._ffi.new("char[]", data_len + 1)
            self._ffi.memmove(buf, data, data_len)
            try:
                yield buf
            finally:
                # Cast to a uint8_t * so we can assign by integer
                self._zero_data(self._ffi.cast("uint8_t *", buf), data_len)

    def load_key_and_certificates_from_pkcs12(self, data, password):
        pkcs12 = self.load_pkcs12(data, password)
        return (
            pkcs12.key,
            pkcs12.cert.certificate if pkcs12.cert else None,
            [cert.certificate for cert in pkcs12.additional_certs],
        )

    def load_pkcs12(self, data, password):
        if password is not None:
            utils._check_byteslike("password", password)

        bio = self._bytes_to_bio(data)
        p12 = self._lib.d2i_PKCS12_bio(bio.bio, self._ffi.NULL)
        if p12 == self._ffi.NULL:
            self._consume_errors()
            raise ValueError("Could not deserialize PKCS12 data")

        p12 = self._ffi.gc(p12, self._lib.PKCS12_free)
        evp_pkey_ptr = self._ffi.new("EVP_PKEY **")
        x509_ptr = self._ffi.new("X509 **")
        sk_x509_ptr = self._ffi.new("Cryptography_STACK_OF_X509 **")
        with self._zeroed_null_terminated_buf(password) as password_buf:
            res = self._lib.PKCS12_parse(
                p12, password_buf, evp_pkey_ptr, x509_ptr, sk_x509_ptr
            )

        # Workaround for
        # https://github.com/libressl-portable/portable/issues/659
        # TODO: Once 3.4.0 is released, add branch.
        if self._lib.CRYPTOGRAPHY_IS_LIBRESSL:
            self._consume_errors()

        if res == 0:
            self._consume_errors()
            raise ValueError("Invalid password or PKCS12 data")

        cert = None
        key = None
        additional_certificates = []

        if evp_pkey_ptr[0] != self._ffi.NULL:
            evp_pkey = self._ffi.gc(evp_pkey_ptr[0], self._lib.EVP_PKEY_free)
            key = self._evp_pkey_to_private_key(evp_pkey)

        if x509_ptr[0] != self._ffi.NULL:
            x509 = self._ffi.gc(x509_ptr[0], self._lib.X509_free)
            cert_obj = self._ossl2cert(x509)
            name = None
            maybe_name = self._lib.X509_alias_get0(x509, self._ffi.NULL)
            if maybe_name != self._ffi.NULL:
                name = self._ffi.string(maybe_name)
            cert = PKCS12Certificate(cert_obj, name)

        if sk_x509_ptr[0] != self._ffi.NULL:
            sk_x509 = self._ffi.gc(sk_x509_ptr[0], self._lib.sk_X509_free)
            num = self._lib.sk_X509_num(sk_x509_ptr[0])

            # In OpenSSL < 3.0.0 PKCS12 parsing reverses the order of the
            # certificates.
            indices: typing.Iterable[int]
            if (
                self._lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER
                or self._lib.CRYPTOGRAPHY_IS_BORINGSSL
            ):
                indices = range(num)
            else:
                indices = reversed(range(num))

            for i in indices:
                x509 = self._lib.sk_X509_value(sk_x509, i)
                self.openssl_assert(x509 != self._ffi.NULL)
                x509 = self._ffi.gc(x509, self._lib.X509_free)
                addl_cert = self._ossl2cert(x509)
                addl_name = None
                maybe_name = self._lib.X509_alias_get0(x509, self._ffi.NULL)
                if maybe_name != self._ffi.NULL:
                    addl_name = self._ffi.string(maybe_name)
                additional_certificates.append(
                    PKCS12Certificate(addl_cert, addl_name)
                )

        return PKCS12KeyAndCertificates(key, cert, additional_certificates)

    def serialize_key_and_certificates_to_pkcs12(
        self, name, key, cert, cas, encryption_algorithm
    ):
        password = None
        if name is not None:
            utils._check_bytes("name", name)

        if isinstance(encryption_algorithm, serialization.NoEncryption):
            nid_cert = -1
            nid_key = -1
            pkcs12_iter = 0
            mac_iter = 0
        elif isinstance(
            encryption_algorithm, serialization.BestAvailableEncryption
        ):
            # PKCS12 encryption is hopeless trash and can never be fixed.
            # This is the least terrible option.
            nid_cert = self._lib.NID_pbe_WithSHA1And3_Key_TripleDES_CBC
            nid_key = self._lib.NID_pbe_WithSHA1And3_Key_TripleDES_CBC
            # At least we can set this higher than OpenSSL's default
            pkcs12_iter = 20000
            # mac_iter chosen for compatibility reasons, see:
            # https://www.openssl.org/docs/man1.1.1/man3/PKCS12_create.html
            # Did we mention how lousy PKCS12 encryption is?
            mac_iter = 1
            password = encryption_algorithm.password
        else:
            raise ValueError("Unsupported key encryption type")

        if cas is None or len(cas) == 0:
            sk_x509 = self._ffi.NULL
        else:
            sk_x509 = self._lib.sk_X509_new_null()
            sk_x509 = self._ffi.gc(sk_x509, self._lib.sk_X509_free)

            # This list is to keep the x509 values alive until end of function
            ossl_cas = []
            for ca in cas:
                ossl_ca = self._cert2ossl(ca)
                ossl_cas.append(ossl_ca)
                res = self._lib.sk_X509_push(sk_x509, ossl_ca)
                backend.openssl_assert(res >= 1)

        with self._zeroed_null_terminated_buf(password) as password_buf:
            with self._zeroed_null_terminated_buf(name) as name_buf:
                ossl_cert = self._cert2ossl(cert) if cert else self._ffi.NULL
                evp_pkey = key._evp_pkey if key else self._ffi.NULL
                p12 = self._lib.PKCS12_create(
                    password_buf,
                    name_buf,
                    evp_pkey,
                    ossl_cert,
                    sk_x509,
                    nid_key,
                    nid_cert,
                    pkcs12_iter,
                    mac_iter,
                    0,
                )

        self.openssl_assert(p12 != self._ffi.NULL)
        p12 = self._ffi.gc(p12, self._lib.PKCS12_free)

        bio = self._create_mem_bio_gc()
        res = self._lib.i2d_PKCS12_bio(bio, p12)
        self.openssl_assert(res > 0)
        return self._read_mem_bio(bio)

    def poly1305_supported(self):
        if self._fips_enabled:
            return False
        return self._lib.Cryptography_HAS_POLY1305 == 1

    def create_poly1305_ctx(self, key):
        utils._check_byteslike("key", key)
        if len(key) != _POLY1305_KEY_SIZE:
            raise ValueError("A poly1305 key is 32 bytes long")

        return _Poly1305Context(self, key)

    def pkcs7_supported(self):
        return not self._lib.CRYPTOGRAPHY_IS_BORINGSSL

    def load_pem_pkcs7_certificates(self, data):
        utils._check_bytes("data", data)
        bio = self._bytes_to_bio(data)
        p7 = self._lib.PEM_read_bio_PKCS7(
            bio.bio, self._ffi.NULL, self._ffi.NULL, self._ffi.NULL
        )
        if p7 == self._ffi.NULL:
            self._consume_errors()
            raise ValueError("Unable to parse PKCS7 data")

        p7 = self._ffi.gc(p7, self._lib.PKCS7_free)
        return self._load_pkcs7_certificates(p7)

    def load_der_pkcs7_certificates(self, data):
        utils._check_bytes("data", data)
        bio = self._bytes_to_bio(data)
        p7 = self._lib.d2i_PKCS7_bio(bio.bio, self._ffi.NULL)
        if p7 == self._ffi.NULL:
            self._consume_errors()
            raise ValueError("Unable to parse PKCS7 data")

        p7 = self._ffi.gc(p7, self._lib.PKCS7_free)
        return self._load_pkcs7_certificates(p7)

    def _load_pkcs7_certificates(self, p7):
        nid = self._lib.OBJ_obj2nid(p7.type)
        self.openssl_assert(nid != self._lib.NID_undef)
        if nid != self._lib.NID_pkcs7_signed:
            raise UnsupportedAlgorithm(
                "Only basic signed structures are currently supported. NID"
                " for this data was {}".format(nid),
                _Reasons.UNSUPPORTED_SERIALIZATION,
            )

        sk_x509 = p7.d.sign.cert
        num = self._lib.sk_X509_num(sk_x509)
        certs = []
        for i in range(num):
            x509 = self._lib.sk_X509_value(sk_x509, i)
            self.openssl_assert(x509 != self._ffi.NULL)
            res = self._lib.X509_up_ref(x509)
            # When OpenSSL is less than 1.1.0 up_ref returns the current
            # refcount. On 1.1.0+ it returns 1 for success.
            self.openssl_assert(res >= 1)
            x509 = self._ffi.gc(x509, self._lib.X509_free)
            cert = self._ossl2cert(x509)
            certs.append(cert)

        return certs

    def pkcs7_sign(self, builder, encoding, options):
        bio = self._bytes_to_bio(builder._data)
        init_flags = self._lib.PKCS7_PARTIAL
        final_flags = 0

        if len(builder._additional_certs) == 0:
            certs = self._ffi.NULL
        else:
            certs = self._lib.sk_X509_new_null()
            certs = self._ffi.gc(certs, self._lib.sk_X509_free)
            # This list is to keep the x509 values alive until end of function
            ossl_certs = []
            for cert in builder._additional_certs:
                ossl_cert = self._cert2ossl(cert)
                ossl_certs.append(ossl_cert)
                res = self._lib.sk_X509_push(certs, ossl_cert)
                self.openssl_assert(res >= 1)

        if pkcs7.PKCS7Options.DetachedSignature in options:
            # Don't embed the data in the PKCS7 structure
            init_flags |= self._lib.PKCS7_DETACHED
            final_flags |= self._lib.PKCS7_DETACHED

        # This just inits a structure for us. However, there
        # are flags we need to set, joy.
        p7 = self._lib.PKCS7_sign(
            self._ffi.NULL,
            self._ffi.NULL,
            certs,
            self._ffi.NULL,
            init_flags,
        )
        self.openssl_assert(p7 != self._ffi.NULL)
        p7 = self._ffi.gc(p7, self._lib.PKCS7_free)
        signer_flags = 0
        # These flags are configurable on a per-signature basis
        # but we've deliberately chosen to make the API only allow
        # setting it across all signatures for now.
        if pkcs7.PKCS7Options.NoCapabilities in options:
            signer_flags |= self._lib.PKCS7_NOSMIMECAP
        elif pkcs7.PKCS7Options.NoAttributes in options:
            signer_flags |= self._lib.PKCS7_NOATTR

        if pkcs7.PKCS7Options.NoCerts in options:
            signer_flags |= self._lib.PKCS7_NOCERTS

        for certificate, private_key, hash_algorithm in builder._signers:
            ossl_cert = self._cert2ossl(certificate)
            md = self._evp_md_non_null_from_algorithm(hash_algorithm)
            p7signerinfo = self._lib.PKCS7_sign_add_signer(
                p7, ossl_cert, private_key._evp_pkey, md, signer_flags
            )
            self.openssl_assert(p7signerinfo != self._ffi.NULL)

        for option in options:
            # DetachedSignature, NoCapabilities, and NoAttributes are already
            # handled so we just need to check these last two options.
            if option is pkcs7.PKCS7Options.Text:
                final_flags |= self._lib.PKCS7_TEXT
            elif option is pkcs7.PKCS7Options.Binary:
                final_flags |= self._lib.PKCS7_BINARY

        bio_out = self._create_mem_bio_gc()
        if encoding is serialization.Encoding.SMIME:
            # This finalizes the structure
            res = self._lib.SMIME_write_PKCS7(
                bio_out, p7, bio.bio, final_flags
            )
        elif encoding is serialization.Encoding.PEM:
            res = self._lib.PKCS7_final(p7, bio.bio, final_flags)
            self.openssl_assert(res == 1)
            res = self._lib.PEM_write_bio_PKCS7_stream(
                bio_out, p7, bio.bio, final_flags
            )
        else:
            assert encoding is serialization.Encoding.DER
            # We need to call finalize here becauase i2d_PKCS7_bio does not
            # finalize.
            res = self._lib.PKCS7_final(p7, bio.bio, final_flags)
            self.openssl_assert(res == 1)
            # OpenSSL 3.0 leaves a random bio error on the stack:
            # https://github.com/openssl/openssl/issues/16681
            if self._lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER:
                self._consume_errors()
            res = self._lib.i2d_PKCS7_bio(bio_out, p7)
        self.openssl_assert(res == 1)
        return self._read_mem_bio(bio_out)


class GetCipherByName(object):
    def __init__(self, fmt):
        self._fmt = fmt

    def __call__(self, backend, cipher, mode):
        cipher_name = self._fmt.format(cipher=cipher, mode=mode).lower()
        return backend._lib.EVP_get_cipherbyname(cipher_name.encode("ascii"))


def _get_xts_cipher(backend, cipher, mode):
    cipher_name = "aes-{}-xts".format(cipher.key_size // 2)
    return backend._lib.EVP_get_cipherbyname(cipher_name.encode("ascii"))


backend = Backend()
