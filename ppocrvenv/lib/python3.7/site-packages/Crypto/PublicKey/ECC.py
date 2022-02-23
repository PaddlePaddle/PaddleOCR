# ===================================================================
#
# Copyright (c) 2015, Legrandin <helderijs@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===================================================================

from __future__ import print_function

import re
import struct
import binascii
from collections import namedtuple

from Crypto.Util.py3compat import bord, tobytes, tostr, bchr, is_string
from Crypto.Util.number import bytes_to_long, long_to_bytes

from Crypto.Math.Numbers import Integer
from Crypto.Util.asn1 import (DerObjectId, DerOctetString, DerSequence,
                              DerBitString)

from Crypto.PublicKey import (_expand_subject_public_key_info,
                              _create_subject_public_key_info,
                              _extract_subject_public_key_info)

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
                                  SmartPointer, c_size_t, c_uint8_ptr,
                                  c_ulonglong)

from Crypto.Random import get_random_bytes
from Crypto.Random.random import getrandbits


_ec_lib = load_pycryptodome_raw_lib("Crypto.PublicKey._ec_ws", """
typedef void EcContext;
typedef void EcPoint;
int ec_ws_new_context(EcContext **pec_ctx,
                      const uint8_t *modulus,
                      const uint8_t *b,
                      const uint8_t *order,
                      size_t len,
                      uint64_t seed);
void ec_free_context(EcContext *ec_ctx);
int ec_ws_new_point(EcPoint **pecp,
                    const uint8_t *x,
                    const uint8_t *y,
                    size_t len,
                    const EcContext *ec_ctx);
void ec_free_point(EcPoint *ecp);
int ec_ws_get_xy(uint8_t *x,
                 uint8_t *y,
                 size_t len,
                 const EcPoint *ecp);
int ec_ws_double(EcPoint *p);
int ec_ws_add(EcPoint *ecpa, EcPoint *ecpb);
int ec_ws_scalar(EcPoint *ecp,
                 const uint8_t *k,
                 size_t len,
                 uint64_t seed);
int ec_ws_clone(EcPoint **pecp2, const EcPoint *ecp);
int ec_ws_copy(EcPoint *ecp1, const EcPoint *ecp2);
int ec_ws_cmp(const EcPoint *ecp1, const EcPoint *ecp2);
int ec_ws_neg(EcPoint *p);
int ec_ws_normalize(EcPoint *ecp);
int ec_ws_is_pai(EcPoint *ecp);
""")

_Curve = namedtuple("_Curve", "p b order Gx Gy G modulus_bits oid context desc openssh")
_curves = {}


p192_names = ["p192", "NIST P-192", "P-192", "prime192v1", "secp192r1",
              "nistp192"]


def init_p192():
    p = 0xfffffffffffffffffffffffffffffffeffffffffffffffff
    b = 0x64210519e59c80e70fa7e9ab72243049feb8deecc146b9b1
    order = 0xffffffffffffffffffffffff99def836146bc9b1b4d22831
    Gx = 0x188da80eb03090f67cbf20eb43a18800f4ff0afd82ff1012
    Gy = 0x07192b95ffc8da78631011ed6b24cdd573f977a11e794811

    p192_modulus = long_to_bytes(p, 24)
    p192_b = long_to_bytes(b, 24)
    p192_order = long_to_bytes(order, 24)

    ec_p192_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p192_context.address_of(),
                                       c_uint8_ptr(p192_modulus),
                                       c_uint8_ptr(p192_b),
                                       c_uint8_ptr(p192_order),
                                       c_size_t(len(p192_modulus)),
                                       c_ulonglong(getrandbits(64))
                                       )
    if result:
        raise ImportError("Error %d initializing P-192 context" % result)

    context = SmartPointer(ec_p192_context.get(), _ec_lib.ec_free_context)
    p192 = _Curve(Integer(p),
                  Integer(b),
                  Integer(order),
                  Integer(Gx),
                  Integer(Gy),
                  None,
                  192,
                  "1.2.840.10045.3.1.1",    # ANSI X9.62 / SEC2
                  context,
                  "NIST P-192",
                  "ecdsa-sha2-nistp192")
    global p192_names
    _curves.update(dict.fromkeys(p192_names, p192))


init_p192()
del init_p192


p224_names = ["p224", "NIST P-224", "P-224", "prime224v1", "secp224r1",
              "nistp224"]


def init_p224():
    p = 0xffffffffffffffffffffffffffffffff000000000000000000000001
    b = 0xb4050a850c04b3abf54132565044b0b7d7bfd8ba270b39432355ffb4
    order = 0xffffffffffffffffffffffffffff16a2e0b8f03e13dd29455c5c2a3d
    Gx = 0xb70e0cbd6bb4bf7f321390b94a03c1d356c21122343280d6115c1d21
    Gy = 0xbd376388b5f723fb4c22dfe6cd4375a05a07476444d5819985007e34

    p224_modulus = long_to_bytes(p, 28)
    p224_b = long_to_bytes(b, 28)
    p224_order = long_to_bytes(order, 28)

    ec_p224_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p224_context.address_of(),
                                       c_uint8_ptr(p224_modulus),
                                       c_uint8_ptr(p224_b),
                                       c_uint8_ptr(p224_order),
                                       c_size_t(len(p224_modulus)),
                                       c_ulonglong(getrandbits(64))
                                       )
    if result:
        raise ImportError("Error %d initializing P-224 context" % result)

    context = SmartPointer(ec_p224_context.get(), _ec_lib.ec_free_context)
    p224 = _Curve(Integer(p),
                  Integer(b),
                  Integer(order),
                  Integer(Gx),
                  Integer(Gy),
                  None,
                  224,
                  "1.3.132.0.33",    # SEC 2
                  context,
                  "NIST P-224",
                  "ecdsa-sha2-nistp224")
    global p224_names
    _curves.update(dict.fromkeys(p224_names, p224))


init_p224()
del init_p224


p256_names = ["p256", "NIST P-256", "P-256", "prime256v1", "secp256r1",
              "nistp256"]


def init_p256():
    p = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
    b = 0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
    order = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
    Gx = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296
    Gy = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5

    p256_modulus = long_to_bytes(p, 32)
    p256_b = long_to_bytes(b, 32)
    p256_order = long_to_bytes(order, 32)

    ec_p256_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p256_context.address_of(),
                                       c_uint8_ptr(p256_modulus),
                                       c_uint8_ptr(p256_b),
                                       c_uint8_ptr(p256_order),
                                       c_size_t(len(p256_modulus)),
                                       c_ulonglong(getrandbits(64))
                                       )
    if result:
        raise ImportError("Error %d initializing P-256 context" % result)

    context = SmartPointer(ec_p256_context.get(), _ec_lib.ec_free_context)
    p256 = _Curve(Integer(p),
                  Integer(b),
                  Integer(order),
                  Integer(Gx),
                  Integer(Gy),
                  None,
                  256,
                  "1.2.840.10045.3.1.7",    # ANSI X9.62 / SEC2
                  context,
                  "NIST P-256",
                  "ecdsa-sha2-nistp256")
    global p256_names
    _curves.update(dict.fromkeys(p256_names, p256))


init_p256()
del init_p256


p384_names = ["p384", "NIST P-384", "P-384", "prime384v1", "secp384r1",
              "nistp384"]


def init_p384():
    p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000ffffffff
    b = 0xb3312fa7e23ee7e4988e056be3f82d19181d9c6efe8141120314088f5013875ac656398d8a2ed19d2a85c8edd3ec2aef
    order = 0xffffffffffffffffffffffffffffffffffffffffffffffffc7634d81f4372ddf581a0db248b0a77aecec196accc52973
    Gx = 0xaa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b9859f741e082542a385502f25dbf55296c3a545e3872760aB7
    Gy = 0x3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147ce9da3113b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5F

    p384_modulus = long_to_bytes(p, 48)
    p384_b = long_to_bytes(b, 48)
    p384_order = long_to_bytes(order, 48)

    ec_p384_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p384_context.address_of(),
                                       c_uint8_ptr(p384_modulus),
                                       c_uint8_ptr(p384_b),
                                       c_uint8_ptr(p384_order),
                                       c_size_t(len(p384_modulus)),
                                       c_ulonglong(getrandbits(64))
                                       )
    if result:
        raise ImportError("Error %d initializing P-384 context" % result)

    context = SmartPointer(ec_p384_context.get(), _ec_lib.ec_free_context)
    p384 = _Curve(Integer(p),
                  Integer(b),
                  Integer(order),
                  Integer(Gx),
                  Integer(Gy),
                  None,
                  384,
                  "1.3.132.0.34",   # SEC 2
                  context,
                  "NIST P-384",
                  "ecdsa-sha2-nistp384")
    global p384_names
    _curves.update(dict.fromkeys(p384_names, p384))


init_p384()
del init_p384


p521_names = ["p521", "NIST P-521", "P-521", "prime521v1", "secp521r1",
              "nistp521"]


def init_p521():
    p = 0x000001ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
    b = 0x00000051953eb9618e1c9a1f929a21a0b68540eea2da725b99b315f3b8b489918ef109e156193951ec7e937b1652c0bd3bb1bf073573df883d2c34f1ef451fd46b503f00
    order = 0x000001fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa51868783bf2f966b7fcc0148f709a5d03bb5c9b8899c47aebb6fb71e91386409
    Gx = 0x000000c6858e06b70404e9cd9e3ecb662395b4429c648139053fb521f828af606b4d3dbaa14b5e77efe75928fe1dc127a2ffa8de3348b3c1856a429bf97e7e31c2e5bd66
    Gy = 0x0000011839296a789a3bc0045c8a5fb42c7d1bd998f54449579b446817afbd17273e662c97ee72995ef42640c550b9013fad0761353c7086a272c24088be94769fd16650

    p521_modulus = long_to_bytes(p, 66)
    p521_b = long_to_bytes(b, 66)
    p521_order = long_to_bytes(order, 66)

    ec_p521_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p521_context.address_of(),
                                       c_uint8_ptr(p521_modulus),
                                       c_uint8_ptr(p521_b),
                                       c_uint8_ptr(p521_order),
                                       c_size_t(len(p521_modulus)),
                                       c_ulonglong(getrandbits(64))
                                       )
    if result:
        raise ImportError("Error %d initializing P-521 context" % result)

    context = SmartPointer(ec_p521_context.get(), _ec_lib.ec_free_context)
    p521 = _Curve(Integer(p),
                  Integer(b),
                  Integer(order),
                  Integer(Gx),
                  Integer(Gy),
                  None,
                  521,
                  "1.3.132.0.35",   # SEC 2
                  context,
                  "NIST P-521",
                  "ecdsa-sha2-nistp521")
    global p521_names
    _curves.update(dict.fromkeys(p521_names, p521))


init_p521()
del init_p521


class UnsupportedEccFeature(ValueError):
    pass


class EccPoint(object):
    """A class to abstract a point over an Elliptic Curve.

    The class support special methods for:

    * Adding two points: ``R = S + T``
    * In-place addition: ``S += T``
    * Negating a point: ``R = -T``
    * Comparing two points: ``if S == T: ...``
    * Multiplying a point by a scalar: ``R = S*k``
    * In-place multiplication by a scalar: ``T *= k``

    :ivar x: The affine X-coordinate of the ECC point
    :vartype x: integer

    :ivar y: The affine Y-coordinate of the ECC point
    :vartype y: integer

    :ivar xy: The tuple with X- and Y- coordinates
    """

    def __init__(self, x, y, curve="p256"):

        try:
            self._curve = _curves[curve]
        except KeyError:
            raise ValueError("Unknown curve name %s" % str(curve))
        self._curve_name = curve

        modulus_bytes = self.size_in_bytes()
        context = self._curve.context

        xb = long_to_bytes(x, modulus_bytes)
        yb = long_to_bytes(y, modulus_bytes)
        if len(xb) != modulus_bytes or len(yb) != modulus_bytes:
            raise ValueError("Incorrect coordinate length")

        self._point = VoidPointer()
        result = _ec_lib.ec_ws_new_point(self._point.address_of(),
                                         c_uint8_ptr(xb),
                                         c_uint8_ptr(yb),
                                         c_size_t(modulus_bytes),
                                         context.get())
        if result:
            if result == 15:
                raise ValueError("The EC point does not belong to the curve")
            raise ValueError("Error %d while instantiating an EC point" % result)

        # Ensure that object disposal of this Python object will (eventually)
        # free the memory allocated by the raw library for the EC point
        self._point = SmartPointer(self._point.get(),
                                   _ec_lib.ec_free_point)

    def set(self, point):
        self._point = VoidPointer()
        result = _ec_lib.ec_ws_clone(self._point.address_of(),
                                     point._point.get())
        if result:
            raise ValueError("Error %d while cloning an EC point" % result)

        self._point = SmartPointer(self._point.get(),
                                   _ec_lib.ec_free_point)
        return self

    def __eq__(self, point):
        return 0 == _ec_lib.ec_ws_cmp(self._point.get(), point._point.get())

    def __neg__(self):
        np = self.copy()
        result = _ec_lib.ec_ws_neg(np._point.get())
        if result:
            raise ValueError("Error %d while inverting an EC point" % result)
        return np

    def copy(self):
        """Return a copy of this point."""
        x, y = self.xy
        np = EccPoint(x, y, self._curve_name)
        return np

    def is_point_at_infinity(self):
        """``True`` if this is the point-at-infinity."""
        return self.xy == (0, 0)

    def point_at_infinity(self):
        """Return the point-at-infinity for the curve this point is on."""
        return EccPoint(0, 0, self._curve_name)

    @property
    def x(self):
        return self.xy[0]

    @property
    def y(self):
        return self.xy[1]

    @property
    def xy(self):
        modulus_bytes = self.size_in_bytes()
        xb = bytearray(modulus_bytes)
        yb = bytearray(modulus_bytes)
        result = _ec_lib.ec_ws_get_xy(c_uint8_ptr(xb),
                                      c_uint8_ptr(yb),
                                      c_size_t(modulus_bytes),
                                      self._point.get())
        if result:
            raise ValueError("Error %d while encoding an EC point" % result)

        return (Integer(bytes_to_long(xb)), Integer(bytes_to_long(yb)))

    def size_in_bytes(self):
        """Size of each coordinate, in bytes."""
        return (self.size_in_bits() + 7) // 8

    def size_in_bits(self):
        """Size of each coordinate, in bits."""
        return self._curve.modulus_bits

    def double(self):
        """Double this point (in-place operation).

        :Return:
            :class:`EccPoint` : this same object (to enable chaining)
        """

        result = _ec_lib.ec_ws_double(self._point.get())
        if result:
            raise ValueError("Error %d while doubling an EC point" % result)
        return self

    def __iadd__(self, point):
        """Add a second point to this one"""

        result = _ec_lib.ec_ws_add(self._point.get(), point._point.get())
        if result:
            if result == 16:
                raise ValueError("EC points are not on the same curve")
            raise ValueError("Error %d while adding two EC points" % result)
        return self

    def __add__(self, point):
        """Return a new point, the addition of this one and another"""

        np = self.copy()
        np += point
        return np

    def __imul__(self, scalar):
        """Multiply this point by a scalar"""

        if scalar < 0:
            raise ValueError("Scalar multiplication is only defined for non-negative integers")
        sb = long_to_bytes(scalar)
        result = _ec_lib.ec_ws_scalar(self._point.get(),
                                      c_uint8_ptr(sb),
                                      c_size_t(len(sb)),
                                      c_ulonglong(getrandbits(64)))
        if result:
            raise ValueError("Error %d during scalar multiplication" % result)
        return self

    def __mul__(self, scalar):
        """Return a new point, the scalar product of this one"""

        np = self.copy()
        np *= scalar
        return np

    def __rmul__(self, left_hand):
        return self.__mul__(left_hand)


# Last piece of initialization
p192_G = EccPoint(_curves['p192'].Gx, _curves['p192'].Gy, "p192")
p192 = _curves['p192']._replace(G=p192_G)
_curves.update(dict.fromkeys(p192_names, p192))
del p192_G, p192, p192_names

p224_G = EccPoint(_curves['p224'].Gx, _curves['p224'].Gy, "p224")
p224 = _curves['p224']._replace(G=p224_G)
_curves.update(dict.fromkeys(p224_names, p224))
del p224_G, p224, p224_names

p256_G = EccPoint(_curves['p256'].Gx, _curves['p256'].Gy, "p256")
p256 = _curves['p256']._replace(G=p256_G)
_curves.update(dict.fromkeys(p256_names, p256))
del p256_G, p256, p256_names

p384_G = EccPoint(_curves['p384'].Gx, _curves['p384'].Gy, "p384")
p384 = _curves['p384']._replace(G=p384_G)
_curves.update(dict.fromkeys(p384_names, p384))
del p384_G, p384, p384_names

p521_G = EccPoint(_curves['p521'].Gx, _curves['p521'].Gy, "p521")
p521 = _curves['p521']._replace(G=p521_G)
_curves.update(dict.fromkeys(p521_names, p521))
del p521_G, p521, p521_names


class EccKey(object):
    r"""Class defining an ECC key.
    Do not instantiate directly.
    Use :func:`generate`, :func:`construct` or :func:`import_key` instead.

    :ivar curve: The name of the ECC as defined in :numref:`curve_names`.
    :vartype curve: string

    :ivar pointQ: an ECC point representating the public component
    :vartype pointQ: :class:`EccPoint`

    :ivar d: A scalar representating the private component
    :vartype d: integer
    """

    def __init__(self, **kwargs):
        """Create a new ECC key

        Keywords:
          curve : string
            It must be *"p256"*, *"P-256"*, *"prime256v1"* or *"secp256r1"*.
          d : integer
            Only for a private key. It must be in the range ``[1..order-1]``.
          point : EccPoint
            Mandatory for a public key. If provided for a private key,
            the implementation will NOT check whether it matches ``d``.
        """

        kwargs_ = dict(kwargs)
        curve_name = kwargs_.pop("curve", None)
        self._d = kwargs_.pop("d", None)
        self._point = kwargs_.pop("point", None)
        if kwargs_:
            raise TypeError("Unknown parameters: " + str(kwargs_))

        if curve_name not in _curves:
            raise ValueError("Unsupported curve (%s)", curve_name)
        self._curve = _curves[curve_name]

        if self._d is None:
            if self._point is None:
                raise ValueError("Either private or public ECC component must be specified, not both")
        else:
            self._d = Integer(self._d)
            if not 1 <= self._d < self._curve.order:
                raise ValueError("Invalid ECC private component")

        self.curve = self._curve.desc

    def __eq__(self, other):
        if other.has_private() != self.has_private():
            return False

        return other.pointQ == self.pointQ

    def __repr__(self):
        if self.has_private():
            extra = ", d=%d" % int(self._d)
        else:
            extra = ""
        x, y = self.pointQ.xy
        return "EccKey(curve='%s', point_x=%d, point_y=%d%s)" % (self._curve.desc, x, y, extra)

    def has_private(self):
        """``True`` if this key can be used for making signatures or decrypting data."""

        return self._d is not None

    def _sign(self, z, k):
        assert 0 < k < self._curve.order

        order = self._curve.order
        blind = Integer.random_range(min_inclusive=1,
                                     max_exclusive=order)

        blind_d = self._d * blind
        inv_blind_k = (blind * k).inverse(order)

        r = (self._curve.G * k).x % order
        s = inv_blind_k * (blind * z + blind_d * r) % order
        return (r, s)

    def _verify(self, z, rs):
        order = self._curve.order
        sinv = rs[1].inverse(order)
        point1 = self._curve.G * ((sinv * z) % order)
        point2 = self.pointQ * ((sinv * rs[0]) % order)
        return (point1 + point2).x == rs[0]

    @property
    def d(self):
        if not self.has_private():
            raise ValueError("This is not a private ECC key")
        return self._d

    @property
    def pointQ(self):
        if self._point is None:
            self._point = self._curve.G * self._d
        return self._point

    def public_key(self):
        """A matching ECC public key.

        Returns:
            a new :class:`EccKey` object
        """

        return EccKey(curve=self._curve.desc, point=self.pointQ)

    def _export_SEC1(self, compress):
        # See 2.2 in RFC5480 and 2.3.3 in SEC1
        #
        # The first byte is:
        # - 0x02:   compressed, only X-coordinate, Y-coordinate is even
        # - 0x03:   compressed, only X-coordinate, Y-coordinate is odd
        # - 0x04:   uncompressed, X-coordinate is followed by Y-coordinate
        #
        # PAI is in theory encoded as 0x00.

        modulus_bytes = self.pointQ.size_in_bytes()

        if compress:
            if self.pointQ.y.is_odd():
                first_byte = b'\x03'
            else:
                first_byte = b'\x02'
            public_key = (first_byte +
                          self.pointQ.x.to_bytes(modulus_bytes))
        else:
            public_key = (b'\x04' +
                          self.pointQ.x.to_bytes(modulus_bytes) +
                          self.pointQ.y.to_bytes(modulus_bytes))
        return public_key

    def _export_subjectPublicKeyInfo(self, compress):

        public_key = self._export_SEC1(compress)
        unrestricted_oid = "1.2.840.10045.2.1"
        return _create_subject_public_key_info(unrestricted_oid,
                                               public_key,
                                               DerObjectId(self._curve.oid))

    def _export_private_der(self, include_ec_params=True):

        assert self.has_private()

        # ECPrivateKey ::= SEQUENCE {
        #           version        INTEGER { ecPrivkeyVer1(1) } (ecPrivkeyVer1),
        #           privateKey     OCTET STRING,
        #           parameters [0] ECParameters {{ NamedCurve }} OPTIONAL,
        #           publicKey  [1] BIT STRING OPTIONAL
        #    }

        # Public key - uncompressed form
        modulus_bytes = self.pointQ.size_in_bytes()
        public_key = (b'\x04' +
                      self.pointQ.x.to_bytes(modulus_bytes) +
                      self.pointQ.y.to_bytes(modulus_bytes))

        seq = [1,
               DerOctetString(self.d.to_bytes(modulus_bytes)),
               DerObjectId(self._curve.oid, explicit=0),
               DerBitString(public_key, explicit=1)]

        if not include_ec_params:
            del seq[2]

        return DerSequence(seq).encode()

    def _export_pkcs8(self, **kwargs):
        from Crypto.IO import PKCS8

        if kwargs.get('passphrase', None) is not None and 'protection' not in kwargs:
            raise ValueError("At least the 'protection' parameter should be present")

        unrestricted_oid = "1.2.840.10045.2.1"
        private_key = self._export_private_der(include_ec_params=False)
        result = PKCS8.wrap(private_key,
                            unrestricted_oid,
                            key_params=DerObjectId(self._curve.oid),
                            **kwargs)
        return result

    def _export_public_pem(self, compress):
        from Crypto.IO import PEM

        encoded_der = self._export_subjectPublicKeyInfo(compress)
        return PEM.encode(encoded_der, "PUBLIC KEY")

    def _export_private_pem(self, passphrase, **kwargs):
        from Crypto.IO import PEM

        encoded_der = self._export_private_der()
        return PEM.encode(encoded_der, "EC PRIVATE KEY", passphrase, **kwargs)

    def _export_private_clear_pkcs8_in_clear_pem(self):
        from Crypto.IO import PEM

        encoded_der = self._export_pkcs8()
        return PEM.encode(encoded_der, "PRIVATE KEY")

    def _export_private_encrypted_pkcs8_in_clear_pem(self, passphrase, **kwargs):
        from Crypto.IO import PEM

        assert passphrase
        if 'protection' not in kwargs:
            raise ValueError("At least the 'protection' parameter should be present")
        encoded_der = self._export_pkcs8(passphrase=passphrase, **kwargs)
        return PEM.encode(encoded_der, "ENCRYPTED PRIVATE KEY")

    def _export_openssh(self, compress):
        if self.has_private():
            raise ValueError("Cannot export OpenSSH private keys")

        desc = self._curve.openssh
        modulus_bytes = self.pointQ.size_in_bytes()

        if compress:
            first_byte = 2 + self.pointQ.y.is_odd()
            public_key = (bchr(first_byte) +
                          self.pointQ.x.to_bytes(modulus_bytes))
        else:
            public_key = (b'\x04' +
                          self.pointQ.x.to_bytes(modulus_bytes) +
                          self.pointQ.y.to_bytes(modulus_bytes))

        middle = desc.split("-")[2]
        comps = (tobytes(desc), tobytes(middle), public_key)
        blob = b"".join([struct.pack(">I", len(x)) + x for x in comps])
        return desc + " " + tostr(binascii.b2a_base64(blob))

    def export_key(self, **kwargs):
        """Export this ECC key.

        Args:
          format (string):
            The format to use for encoding the key:

            - ``'DER'``. The key will be encoded in ASN.1 DER format (binary).
              For a public key, the ASN.1 ``subjectPublicKeyInfo`` structure
              defined in `RFC5480`_ will be used.
              For a private key, the ASN.1 ``ECPrivateKey`` structure defined
              in `RFC5915`_ is used instead (possibly within a PKCS#8 envelope,
              see the ``use_pkcs8`` flag below).
            - ``'PEM'``. The key will be encoded in a PEM_ envelope (ASCII).
            - ``'OpenSSH'``. The key will be encoded in the OpenSSH_ format
              (ASCII, public keys only).
            - ``'SEC1'``. The public key (i.e., the EC point) will be encoded
              into ``bytes`` according to Section 2.3.3 of `SEC1`_
              (which is a subset of the older X9.62 ITU standard).

          passphrase (byte string or string):
            The passphrase to use for protecting the private key.

          use_pkcs8 (boolean):
            Only relevant for private keys.

            If ``True`` (default and recommended), the `PKCS#8`_ representation
            will be used.

            If ``False``, the much weaker `PEM encryption`_ mechanism will be used.

          protection (string):
            When a private key is exported with password-protection
            and PKCS#8 (both ``DER`` and ``PEM`` formats), this parameter MUST be
            present and be a valid algorithm supported by :mod:`Crypto.IO.PKCS8`.
            It is recommended to use ``PBKDF2WithHMAC-SHA1AndAES128-CBC``.

          compress (boolean):
            If ``True``, the method returns a more compact representation
            of the public key, with the X-coordinate only.

            If ``False`` (default), the method returns the full public key.

        .. warning::
            If you don't provide a passphrase, the private key will be
            exported in the clear!

        .. note::
            When exporting a private key with password-protection and `PKCS#8`_
            (both ``DER`` and ``PEM`` formats), any extra parameters
            to ``export_key()`` will be passed to :mod:`Crypto.IO.PKCS8`.

        .. _PEM:        http://www.ietf.org/rfc/rfc1421.txt
        .. _`PEM encryption`: http://www.ietf.org/rfc/rfc1423.txt
        .. _`PKCS#8`:   http://www.ietf.org/rfc/rfc5208.txt
        .. _OpenSSH:    http://www.openssh.com/txt/rfc5656.txt
        .. _RFC5480:    https://tools.ietf.org/html/rfc5480
        .. _RFC5915:    http://www.ietf.org/rfc/rfc5915.txt
        .. _SEC1:       https://www.secg.org/sec1-v2.pdf

        Returns:
            A multi-line string (for PEM and OpenSSH) or
            ``bytes`` (for DER and SEC1) with the encoded key.
        """

        args = kwargs.copy()
        ext_format = args.pop("format")
        if ext_format not in ("PEM", "DER", "OpenSSH", "SEC1"):
            raise ValueError("Unknown format '%s'" % ext_format)

        compress = args.pop("compress", False)

        if self.has_private():
            passphrase = args.pop("passphrase", None)
            if is_string(passphrase):
                passphrase = tobytes(passphrase)
                if not passphrase:
                    raise ValueError("Empty passphrase")
            use_pkcs8 = args.pop("use_pkcs8", True)
            if ext_format == "PEM":
                if use_pkcs8:
                    if passphrase:
                        return self._export_private_encrypted_pkcs8_in_clear_pem(passphrase, **args)
                    else:
                        return self._export_private_clear_pkcs8_in_clear_pem()
                else:
                    return self._export_private_pem(passphrase, **args)
            elif ext_format == "DER":
                # DER
                if passphrase and not use_pkcs8:
                    raise ValueError("Private keys can only be encrpyted with DER using PKCS#8")
                if use_pkcs8:
                    return self._export_pkcs8(passphrase=passphrase, **args)
                else:
                    return self._export_private_der()
            else:
                raise ValueError("Private keys cannot be exported "
                                 "in the '%s' format" % ext_format)
        else:  # Public key
            if args:
                raise ValueError("Unexpected parameters: '%s'" % args)
            if ext_format == "PEM":
                return self._export_public_pem(compress)
            elif ext_format == "DER":
                return self._export_subjectPublicKeyInfo(compress)
            elif ext_format == "SEC1":
                return self._export_SEC1(compress)
            else:
                return self._export_openssh(compress)


def generate(**kwargs):
    """Generate a new private key on the given curve.

    Args:

      curve (string):
        Mandatory. It must be a curve name defined in :numref:`curve_names`.

      randfunc (callable):
        Optional. The RNG to read randomness from.
        If ``None``, :func:`Crypto.Random.get_random_bytes` is used.
    """

    curve_name = kwargs.pop("curve")
    curve = _curves[curve_name]
    randfunc = kwargs.pop("randfunc", get_random_bytes)
    if kwargs:
        raise TypeError("Unknown parameters: " + str(kwargs))

    d = Integer.random_range(min_inclusive=1,
                             max_exclusive=curve.order,
                             randfunc=randfunc)

    return EccKey(curve=curve_name, d=d)


def construct(**kwargs):
    """Build a new ECC key (private or public) starting
    from some base components.

    Args:

      curve (string):
        Mandatory. It must be a curve name defined in :numref:`curve_names`.

      d (integer):
        Only for a private key. It must be in the range ``[1..order-1]``.

      point_x (integer):
        Mandatory for a public key. X coordinate (affine) of the ECC point.

      point_y (integer):
        Mandatory for a public key. Y coordinate (affine) of the ECC point.

    Returns:
      :class:`EccKey` : a new ECC key object
    """

    curve_name = kwargs["curve"]
    curve = _curves[curve_name]
    point_x = kwargs.pop("point_x", None)
    point_y = kwargs.pop("point_y", None)

    if "point" in kwargs:
        raise TypeError("Unknown keyword: point")

    if None not in (point_x, point_y):
        # ValueError is raised if the point is not on the curve
        kwargs["point"] = EccPoint(point_x, point_y, curve_name)

    # Validate that the private key matches the public one
    d = kwargs.get("d", None)
    if d is not None and "point" in kwargs:
        pub_key = curve.G * d
        if pub_key.xy != (point_x, point_y):
            raise ValueError("Private and public ECC keys do not match")

    return EccKey(**kwargs)


def _import_public_der(ec_point, curve_oid=None, curve_name=None):
    """Convert an encoded EC point into an EccKey object

    ec_point: byte string with the EC point (SEC1-encoded)
    curve_oid: string with the name the curve
    curve_name: string with the OID of the curve

    Either curve_id or curve_name must be specified

    """

    for _curve_name, curve in _curves.items():
        if curve_oid and curve.oid == curve_oid:
            break
        if curve_name == _curve_name:
            break
    else:
        if curve_oid:
            raise UnsupportedEccFeature("Unsupported ECC curve (OID: %s)" % curve_oid)
        else:
            raise UnsupportedEccFeature("Unsupported ECC curve (%s)" % curve_name)

    # See 2.2 in RFC5480 and 2.3.3 in SEC1
    # The first byte is:
    # - 0x02:   compressed, only X-coordinate, Y-coordinate is even
    # - 0x03:   compressed, only X-coordinate, Y-coordinate is odd
    # - 0x04:   uncompressed, X-coordinate is followed by Y-coordinate
    #
    # PAI is in theory encoded as 0x00.

    modulus_bytes = curve.p.size_in_bytes()
    point_type = bord(ec_point[0])

    # Uncompressed point
    if point_type == 0x04:
        if len(ec_point) != (1 + 2 * modulus_bytes):
            raise ValueError("Incorrect EC point length")
        x = Integer.from_bytes(ec_point[1:modulus_bytes+1])
        y = Integer.from_bytes(ec_point[modulus_bytes+1:])
    # Compressed point
    elif point_type in (0x02, 0x03):
        if len(ec_point) != (1 + modulus_bytes):
            raise ValueError("Incorrect EC point length")
        x = Integer.from_bytes(ec_point[1:])
        # Right now, we only support Short Weierstrass curves
        y = (x**3 - x*3 + curve.b).sqrt(curve.p)
        if point_type == 0x02 and y.is_odd():
            y = curve.p - y
        if point_type == 0x03 and y.is_even():
            y = curve.p - y
    else:
        raise ValueError("Incorrect EC point encoding")

    return construct(curve=_curve_name, point_x=x, point_y=y)


def _import_subjectPublicKeyInfo(encoded, *kwargs):
    """Convert a subjectPublicKeyInfo into an EccKey object"""

    # See RFC5480

    # Parse the generic subjectPublicKeyInfo structure
    oid, ec_point, params = _expand_subject_public_key_info(encoded)

    # ec_point must be an encoded OCTET STRING
    # params is encoded ECParameters

    # We accept id-ecPublicKey, id-ecDH, id-ecMQV without making any
    # distiction for now.

    # Restrictions can be captured in the key usage certificate
    # extension
    unrestricted_oid = "1.2.840.10045.2.1"
    ecdh_oid = "1.3.132.1.12"
    ecmqv_oid = "1.3.132.1.13"

    if oid not in (unrestricted_oid, ecdh_oid, ecmqv_oid):
        raise UnsupportedEccFeature("Unsupported ECC purpose (OID: %s)" % oid)

    # Parameters are mandatory for all three types
    if not params:
        raise ValueError("Missing ECC parameters")

    # ECParameters ::= CHOICE {
    #   namedCurve         OBJECT IDENTIFIER
    #   -- implicitCurve   NULL
    #   -- specifiedCurve  SpecifiedECDomain
    # }
    #
    # implicitCurve and specifiedCurve are not supported (as per RFC)
    curve_oid = DerObjectId().decode(params).value

    return _import_public_der(ec_point, curve_oid=curve_oid)


def _import_private_der(encoded, passphrase, curve_oid=None):

    # See RFC5915 https://tools.ietf.org/html/rfc5915
    #
    # ECPrivateKey ::= SEQUENCE {
    #           version        INTEGER { ecPrivkeyVer1(1) } (ecPrivkeyVer1),
    #           privateKey     OCTET STRING,
    #           parameters [0] ECParameters {{ NamedCurve }} OPTIONAL,
    #           publicKey  [1] BIT STRING OPTIONAL
    #    }

    private_key = DerSequence().decode(encoded, nr_elements=(3, 4))
    if private_key[0] != 1:
        raise ValueError("Incorrect ECC private key version")

    try:
        parameters = DerObjectId(explicit=0).decode(private_key[2]).value
        if curve_oid is not None and parameters != curve_oid:
            raise ValueError("Curve mismatch")
        curve_oid = parameters
    except ValueError:
        pass

    if curve_oid is None:
        raise ValueError("No curve found")

    for curve_name, curve in _curves.items():
        if curve.oid == curve_oid:
            break
    else:
        raise UnsupportedEccFeature("Unsupported ECC curve (OID: %s)" % curve_oid)

    scalar_bytes = DerOctetString().decode(private_key[1]).payload
    modulus_bytes = curve.p.size_in_bytes()
    if len(scalar_bytes) != modulus_bytes:
        raise ValueError("Private key is too small")
    d = Integer.from_bytes(scalar_bytes)

    # Decode public key (if any)
    if len(private_key) == 4:
        public_key_enc = DerBitString(explicit=1).decode(private_key[3]).value
        public_key = _import_public_der(public_key_enc, curve_oid=curve_oid)
        point_x = public_key.pointQ.x
        point_y = public_key.pointQ.y
    else:
        point_x = point_y = None

    return construct(curve=curve_name, d=d, point_x=point_x, point_y=point_y)


def _import_pkcs8(encoded, passphrase):
    from Crypto.IO import PKCS8

    # From RFC5915, Section 1:
    #
    # Distributing an EC private key with PKCS#8 [RFC5208] involves including:
    # a) id-ecPublicKey, id-ecDH, or id-ecMQV (from [RFC5480]) with the
    #    namedCurve as the parameters in the privateKeyAlgorithm field; and
    # b) ECPrivateKey in the PrivateKey field, which is an OCTET STRING.

    algo_oid, private_key, params = PKCS8.unwrap(encoded, passphrase)

    # We accept id-ecPublicKey, id-ecDH, id-ecMQV without making any
    # distiction for now.
    unrestricted_oid = "1.2.840.10045.2.1"
    ecdh_oid = "1.3.132.1.12"
    ecmqv_oid = "1.3.132.1.13"

    if algo_oid not in (unrestricted_oid, ecdh_oid, ecmqv_oid):
        raise UnsupportedEccFeature("Unsupported ECC purpose (OID: %s)" % algo_oid)

    curve_oid = DerObjectId().decode(params).value

    return _import_private_der(private_key, passphrase, curve_oid)


def _import_x509_cert(encoded, *kwargs):

    sp_info = _extract_subject_public_key_info(encoded)
    return _import_subjectPublicKeyInfo(sp_info)


def _import_der(encoded, passphrase):

    try:
        return _import_subjectPublicKeyInfo(encoded, passphrase)
    except UnsupportedEccFeature as err:
        raise err
    except (ValueError, TypeError, IndexError):
        pass

    try:
        return _import_x509_cert(encoded, passphrase)
    except UnsupportedEccFeature as err:
        raise err
    except (ValueError, TypeError, IndexError):
        pass

    try:
        return _import_private_der(encoded, passphrase)
    except UnsupportedEccFeature as err:
        raise err
    except (ValueError, TypeError, IndexError):
        pass

    try:
        return _import_pkcs8(encoded, passphrase)
    except UnsupportedEccFeature as err:
        raise err
    except (ValueError, TypeError, IndexError):
        pass

    raise ValueError("Not an ECC DER key")


def _import_openssh_public(encoded):
    keystring = binascii.a2b_base64(encoded.split(b' ')[1])

    keyparts = []
    while len(keystring) > 4:
        lk = struct.unpack(">I", keystring[:4])[0]
        keyparts.append(keystring[4:4 + lk])
        keystring = keystring[4 + lk:]

    for curve_name, curve in _curves.items():
        middle = tobytes(curve.openssh.split("-")[2])
        if keyparts[1] == middle:
            break
    else:
        raise ValueError("Unsupported ECC curve")

    return _import_public_der(keyparts[2], curve_oid=curve.oid)


def _import_openssh_private_ecc(data, password):

    from ._openssh import (import_openssh_private_generic,
                           read_bytes, read_string, check_padding)

    ssh_name, decrypted = import_openssh_private_generic(data, password)

    name, decrypted = read_string(decrypted)
    if name not in _curves:
        raise UnsupportedEccFeature("Unsupported ECC curve %s" % name)
    curve = _curves[name]
    modulus_bytes = (curve.modulus_bits + 7) // 8

    public_key, decrypted = read_bytes(decrypted)

    if bord(public_key[0]) != 4:
        raise ValueError("Only uncompressed OpenSSH EC keys are supported")
    if len(public_key) != 2 * modulus_bytes + 1:
        raise ValueError("Incorrect public key length")

    point_x = Integer.from_bytes(public_key[1:1+modulus_bytes])
    point_y = Integer.from_bytes(public_key[1+modulus_bytes:])
    point = EccPoint(point_x, point_y, curve=name)

    private_key, decrypted = read_bytes(decrypted)
    d = Integer.from_bytes(private_key)

    _, padded = read_string(decrypted)  # Comment
    check_padding(padded)

    return EccKey(curve=name, d=d, point=point)


def import_key(encoded, passphrase=None, curve_name=None):
    """Import an ECC key (public or private).

    Args:
      encoded (bytes or multi-line string):
        The ECC key to import.

        An ECC **public** key can be:

        - An X.509 certificate, binary (DER) or ASCII (PEM)
        - An X.509 ``subjectPublicKeyInfo``, binary (DER) or ASCII (PEM)
        - A SEC1_ (or X9.62) byte string. You must also provide the
          ``curve_name``.
        - An OpenSSH line (e.g. the content of ``~/.ssh/id_ecdsa``, ASCII)

        An ECC **private** key can be:

        - In binary format (DER, see section 3 of `RFC5915`_ or `PKCS#8`_)
        - In ASCII format (PEM or `OpenSSH 6.5+`_)

        Private keys can be in the clear or password-protected.

        For details about the PEM encoding, see `RFC1421`_/`RFC1423`_.

      passphrase (byte string):
        The passphrase to use for decrypting a private key.
        Encryption may be applied protected at the PEM level or at the PKCS#8 level.
        This parameter is ignored if the key in input is not encrypted.

      curve_name (string):
        For a SEC1 byte string only. This is the name of the ECC curve,
        as defined in :numref:`curve_names`.

    Returns:
      :class:`EccKey` : a new ECC key object

    Raises:
      ValueError: when the given key cannot be parsed (possibly because
        the pass phrase is wrong).

    .. _RFC1421: http://www.ietf.org/rfc/rfc1421.txt
    .. _RFC1423: http://www.ietf.org/rfc/rfc1423.txt
    .. _RFC5915: http://www.ietf.org/rfc/rfc5915.txt
    .. _`PKCS#8`: http://www.ietf.org/rfc/rfc5208.txt
    .. _`OpenSSH 6.5+`: https://flak.tedunangst.com/post/new-openssh-key-format-and-bcrypt-pbkdf
    .. _SEC1: https://www.secg.org/sec1-v2.pdf
    """

    from Crypto.IO import PEM

    encoded = tobytes(encoded)
    if passphrase is not None:
        passphrase = tobytes(passphrase)

    # PEM
    if encoded.startswith(b'-----BEGIN OPENSSH PRIVATE KEY'):
        text_encoded = tostr(encoded)
        openssh_encoded, marker, enc_flag = PEM.decode(text_encoded, passphrase)
        result = _import_openssh_private_ecc(openssh_encoded, passphrase)
        return result

    elif encoded.startswith(b'-----'):

        text_encoded = tostr(encoded)

        # Remove any EC PARAMETERS section
        # Ignore its content because the curve type must be already given in the key
        ecparams_start = "-----BEGIN EC PARAMETERS-----"
        ecparams_end = "-----END EC PARAMETERS-----"
        text_encoded = re.sub(ecparams_start + ".*?" + ecparams_end, "",
                                text_encoded,
                                flags=re.DOTALL)

        der_encoded, marker, enc_flag = PEM.decode(text_encoded, passphrase)
        if enc_flag:
            passphrase = None
        try:
            result = _import_der(der_encoded, passphrase)
        except UnsupportedEccFeature as uef:
            raise uef
        except ValueError:
            raise ValueError("Invalid DER encoding inside the PEM file")
        return result

    # OpenSSH
    if encoded.startswith(b'ecdsa-sha2-'):
        return _import_openssh_public(encoded)

    # DER
    if len(encoded) > 0 and bord(encoded[0]) == 0x30:
        return _import_der(encoded, passphrase)

    # SEC1
    if len(encoded) > 0 and bord(encoded[0]) in b'\x02\x03\x04':
        if curve_name is None:
            raise ValueError("No curve name was provided")
        return _import_public_der(encoded, curve_name=curve_name)

    raise ValueError("ECC key format is not supported")


if __name__ == "__main__":

    import time

    d = 0xc51e4753afdec1e6b6c6a5b992f43f8dd0c7a8933072708b6522468b2ffb06fd

    point = _curves['p256'].G.copy()
    count = 3000

    start = time.time()
    for x in range(count):
        pointX = point * d
    print("(P-256 G)", (time.time() - start) / count * 1000, "ms")

    start = time.time()
    for x in range(count):
        pointX = pointX * d
    print("(P-256 arbitrary point)", (time.time() - start) / count * 1000, "ms")
