# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import abc


class AsymmetricSignatureContext(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, data: bytes) -> None:
        """
        Processes the provided bytes and returns nothing.
        """

    @abc.abstractmethod
    def finalize(self) -> bytes:
        """
        Returns the signature as bytes.
        """


class AsymmetricVerificationContext(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, data: bytes) -> None:
        """
        Processes the provided bytes and returns nothing.
        """

    @abc.abstractmethod
    def verify(self) -> None:
        """
        Raises an exception if the bytes provided to update do not match the
        signature or the signature does not match the public key.
        """
