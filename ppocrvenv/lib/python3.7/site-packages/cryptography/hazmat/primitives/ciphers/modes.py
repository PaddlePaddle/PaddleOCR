# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import abc
import typing

from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
    BlockCipherAlgorithm,
    CipherAlgorithm,
)


class Mode(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def name(self) -> str:
        """
        A string naming this mode (e.g. "ECB", "CBC").
        """

    @abc.abstractmethod
    def validate_for_algorithm(self, algorithm: CipherAlgorithm) -> None:
        """
        Checks that all the necessary invariants of this (mode, algorithm)
        combination are met.
        """


class ModeWithInitializationVector(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def initialization_vector(self) -> bytes:
        """
        The value of the initialization vector for this mode as bytes.
        """


class ModeWithTweak(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def tweak(self) -> bytes:
        """
        The value of the tweak for this mode as bytes.
        """


class ModeWithNonce(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def nonce(self) -> bytes:
        """
        The value of the nonce for this mode as bytes.
        """


class ModeWithAuthenticationTag(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def tag(self) -> typing.Optional[bytes]:
        """
        The value of the tag supplied to the constructor of this mode.
        """


def _check_aes_key_length(self, algorithm):
    if algorithm.key_size > 256 and algorithm.name == "AES":
        raise ValueError(
            "Only 128, 192, and 256 bit keys are allowed for this AES mode"
        )


def _check_iv_length(self, algorithm):
    if len(self.initialization_vector) * 8 != algorithm.block_size:
        raise ValueError(
            "Invalid IV size ({}) for {}.".format(
                len(self.initialization_vector), self.name
            )
        )


def _check_nonce_length(nonce: bytes, name: str, algorithm) -> None:
    if len(nonce) * 8 != algorithm.block_size:
        raise ValueError(
            "Invalid nonce size ({}) for {}.".format(len(nonce), name)
        )


def _check_iv_and_key_length(self, algorithm):
    _check_aes_key_length(self, algorithm)
    _check_iv_length(self, algorithm)


class CBC(Mode, ModeWithInitializationVector):
    name = "CBC"

    def __init__(self, initialization_vector: bytes):
        utils._check_byteslike("initialization_vector", initialization_vector)
        self._initialization_vector = initialization_vector

    @property
    def initialization_vector(self) -> bytes:
        return self._initialization_vector

    validate_for_algorithm = _check_iv_and_key_length


class XTS(Mode, ModeWithTweak):
    name = "XTS"

    def __init__(self, tweak: bytes):
        utils._check_byteslike("tweak", tweak)

        if len(tweak) != 16:
            raise ValueError("tweak must be 128-bits (16 bytes)")

        self._tweak = tweak

    @property
    def tweak(self) -> bytes:
        return self._tweak

    def validate_for_algorithm(self, algorithm: CipherAlgorithm) -> None:
        if algorithm.key_size not in (256, 512):
            raise ValueError(
                "The XTS specification requires a 256-bit key for AES-128-XTS"
                " and 512-bit key for AES-256-XTS"
            )


class ECB(Mode):
    name = "ECB"

    validate_for_algorithm = _check_aes_key_length


class OFB(Mode, ModeWithInitializationVector):
    name = "OFB"

    def __init__(self, initialization_vector: bytes):
        utils._check_byteslike("initialization_vector", initialization_vector)
        self._initialization_vector = initialization_vector

    @property
    def initialization_vector(self) -> bytes:
        return self._initialization_vector

    validate_for_algorithm = _check_iv_and_key_length


class CFB(Mode, ModeWithInitializationVector):
    name = "CFB"

    def __init__(self, initialization_vector: bytes):
        utils._check_byteslike("initialization_vector", initialization_vector)
        self._initialization_vector = initialization_vector

    @property
    def initialization_vector(self) -> bytes:
        return self._initialization_vector

    validate_for_algorithm = _check_iv_and_key_length


class CFB8(Mode, ModeWithInitializationVector):
    name = "CFB8"

    def __init__(self, initialization_vector: bytes):
        utils._check_byteslike("initialization_vector", initialization_vector)
        self._initialization_vector = initialization_vector

    @property
    def initialization_vector(self) -> bytes:
        return self._initialization_vector

    validate_for_algorithm = _check_iv_and_key_length


class CTR(Mode, ModeWithNonce):
    name = "CTR"

    def __init__(self, nonce: bytes):
        utils._check_byteslike("nonce", nonce)
        self._nonce = nonce

    @property
    def nonce(self) -> bytes:
        return self._nonce

    def validate_for_algorithm(self, algorithm: CipherAlgorithm) -> None:
        _check_aes_key_length(self, algorithm)
        _check_nonce_length(self.nonce, self.name, algorithm)


class GCM(Mode, ModeWithInitializationVector, ModeWithAuthenticationTag):
    name = "GCM"
    _MAX_ENCRYPTED_BYTES = (2 ** 39 - 256) // 8
    _MAX_AAD_BYTES = (2 ** 64) // 8

    def __init__(
        self,
        initialization_vector: bytes,
        tag: typing.Optional[bytes] = None,
        min_tag_length: int = 16,
    ):
        # OpenSSL 3.0.0 constrains GCM IVs to [64, 1024] bits inclusive
        # This is a sane limit anyway so we'll enforce it here.
        utils._check_byteslike("initialization_vector", initialization_vector)
        if len(initialization_vector) < 8 or len(initialization_vector) > 128:
            raise ValueError(
                "initialization_vector must be between 8 and 128 bytes (64 "
                "and 1024 bits)."
            )
        self._initialization_vector = initialization_vector
        if tag is not None:
            utils._check_bytes("tag", tag)
            if min_tag_length < 4:
                raise ValueError("min_tag_length must be >= 4")
            if len(tag) < min_tag_length:
                raise ValueError(
                    "Authentication tag must be {} bytes or longer.".format(
                        min_tag_length
                    )
                )
        self._tag = tag
        self._min_tag_length = min_tag_length

    @property
    def tag(self) -> typing.Optional[bytes]:
        return self._tag

    @property
    def initialization_vector(self) -> bytes:
        return self._initialization_vector

    def validate_for_algorithm(self, algorithm: CipherAlgorithm) -> None:
        _check_aes_key_length(self, algorithm)
        if not isinstance(algorithm, BlockCipherAlgorithm):
            raise UnsupportedAlgorithm(
                "GCM requires a block cipher algorithm",
                _Reasons.UNSUPPORTED_CIPHER,
            )
        block_size_bytes = algorithm.block_size // 8
        if self._tag is not None and len(self._tag) > block_size_bytes:
            raise ValueError(
                "Authentication tag cannot be more than {} bytes.".format(
                    block_size_bytes
                )
            )
