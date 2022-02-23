"""
This is a copy of the OpenSimplex library,
based on commit d861cb290531ad15825f21dc4cc35c5d4f407259 from 20.07.2017.
"""

# Based on: https://gist.github.com/KdotJPG/b1270127455a94ac5d19

import sys
from ctypes import c_long
from math import floor as _floor


if sys.version_info[0] < 3:
    def floor(num):
        return int(_floor(num))
else:
    floor = _floor


STRETCH_CONSTANT_2D = -0.211324865405187    # (1/Math.sqrt(2+1)-1)/2
SQUISH_CONSTANT_2D = 0.366025403784439      # (Math.sqrt(2+1)-1)/2
STRETCH_CONSTANT_3D = -1.0 / 6              # (1/Math.sqrt(3+1)-1)/3
SQUISH_CONSTANT_3D = 1.0 / 3                # (Math.sqrt(3+1)-1)/3
STRETCH_CONSTANT_4D = -0.138196601125011    # (1/Math.sqrt(4+1)-1)/4
SQUISH_CONSTANT_4D = 0.309016994374947      # (Math.sqrt(4+1)-1)/4

NORM_CONSTANT_2D = 47
NORM_CONSTANT_3D = 103
NORM_CONSTANT_4D = 30

DEFAULT_SEED = 0


# Gradients for 2D. They approximate the directions to the
# vertices of an octagon from the center.
GRADIENTS_2D = (
     5,  2,    2,  5,
    -5,  2,   -2,  5,
     5, -2,    2, -5,
    -5, -2,   -2, -5,
)

# Gradients for 3D. They approximate the directions to the
# vertices of a rhombicuboctahedron from the center, skewed so
# that the triangular and square facets can be inscribed inside
# circles of the same radius.
GRADIENTS_3D = (
    -11,  4,  4,     -4,  11,  4,    -4,  4,  11,
     11,  4,  4,      4,  11,  4,     4,  4,  11,
    -11, -4,  4,     -4, -11,  4,    -4, -4,  11,
     11, -4,  4,      4, -11,  4,     4, -4,  11,
    -11,  4, -4,     -4,  11, -4,    -4,  4, -11,
     11,  4, -4,      4,  11, -4,     4,  4, -11,
    -11, -4, -4,     -4, -11, -4,    -4, -4, -11,
     11, -4, -4,      4, -11, -4,     4, -4, -11,
)

# Gradients for 4D. They approximate the directions to the
# vertices of a disprismatotesseractihexadecachoron from the center,
# skewed so that the tetrahedral and cubic facets can be inscribed inside
# spheres of the same radius.
GRADIENTS_4D = (
     3,  1,  1,  1,      1,  3,  1,  1,      1,  1,  3,  1,      1,  1,  1,  3,
    -3,  1,  1,  1,     -1,  3,  1,  1,     -1,  1,  3,  1,     -1,  1,  1,  3,
     3, -1,  1,  1,      1, -3,  1,  1,      1, -1,  3,  1,      1, -1,  1,  3,
    -3, -1,  1,  1,     -1, -3,  1,  1,     -1, -1,  3,  1,     -1, -1,  1,  3,
     3,  1, -1,  1,      1,  3, -1,  1,      1,  1, -3,  1,      1,  1, -1,  3,
    -3,  1, -1,  1,     -1,  3, -1,  1,     -1,  1, -3,  1,     -1,  1, -1,  3,
     3, -1, -1,  1,      1, -3, -1,  1,      1, -1, -3,  1,      1, -1, -1,  3,
    -3, -1, -1,  1,     -1, -3, -1,  1,     -1, -1, -3,  1,     -1, -1, -1,  3,
     3,  1,  1, -1,      1,  3,  1, -1,      1,  1,  3, -1,      1,  1,  1, -3,
    -3,  1,  1, -1,     -1,  3,  1, -1,     -1,  1,  3, -1,     -1,  1,  1, -3,
     3, -1,  1, -1,      1, -3,  1, -1,      1, -1,  3, -1,      1, -1,  1, -3,
    -3, -1,  1, -1,     -1, -3,  1, -1,     -1, -1,  3, -1,     -1, -1,  1, -3,
     3,  1, -1, -1,      1,  3, -1, -1,      1,  1, -3, -1,      1,  1, -1, -3,
    -3,  1, -1, -1,     -1,  3, -1, -1,     -1,  1, -3, -1,     -1,  1, -1, -3,
     3, -1, -1, -1,      1, -3, -1, -1,      1, -1, -3, -1,      1, -1, -1, -3,
    -3, -1, -1, -1,     -1, -3, -1, -1,     -1, -1, -3, -1,     -1, -1, -1, -3,
)


def overflow(x):
    # Since normal python ints and longs can be quite humongous we have to use
    # this hack to make them be able to overflow
    return c_long(x).value


class OpenSimplex(object):
    """
    OpenSimplex n-dimensional gradient noise functions.
    """

    def __init__(self, seed=DEFAULT_SEED):
        """
        Initiate the class and generate permutation arrays from a seed number.
        """
        # Initializes the class using a permutation array generated from a 64-bit seed.
        # Generates a proper permutation (i.e. doesn't merely perform N
        # successive pair swaps on a base array)
        perm = self._perm = [0] * 256 # Have to zero fill so we can properly loop over it later
        perm_grad_index_3D = self._perm_grad_index_3D = [0] * 256
        source = [i for i in range(0, 256)]
        seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
        seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
        seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
        for i in range(255, -1, -1):
            seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
            r = int((seed + 31) % (i + 1))
            if r < 0:
                r += i + 1
            perm[i] = source[r]
            perm_grad_index_3D[i] = int((perm[i] % (len(GRADIENTS_3D) / 3)) * 3)
            source[r] = source[i]

    def _extrapolate2d(self, xsb, ysb, dx, dy):
        perm = self._perm
        index = perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E

        g1, g2 = GRADIENTS_2D[index:index + 2]
        return g1 * dx + g2 * dy

    def _extrapolate3d(self, xsb, ysb, zsb, dx, dy, dz):
        perm = self._perm
        index = self._perm_grad_index_3D[
            (perm[(perm[xsb & 0xFF] + ysb) & 0xFF] + zsb) & 0xFF
            ]

        g1, g2, g3 = GRADIENTS_3D[index:index + 3]
        return g1 * dx + g2 * dy + g3 * dz

    def _extrapolate4d(self, xsb, ysb, zsb, wsb, dx, dy, dz, dw):
        perm = self._perm
        index = perm[(
                         perm[(
                                  perm[(perm[xsb & 0xFF] + ysb) & 0xFF] + zsb
                ) & 0xFF] + wsb
        ) & 0xFF] & 0xFC

        g1, g2, g3, g4 = GRADIENTS_4D[index:index + 4]
        return g1 * dx + g2 * dy + g3 * dz + g4 * dw


    def noise2d(self, x, y):
        """
        Generate 2D OpenSimplex noise from X,Y coordinates.
        """
        # Place input coordinates onto grid.
        stretch_offset = (x + y) * STRETCH_CONSTANT_2D
        xs = x + stretch_offset
        ys = y + stretch_offset

        # Floor to get grid coordinates of rhombus (stretched square) super-cell origin.
        xsb = floor(xs)
        ysb = floor(ys)

        # Skew out to get actual coordinates of rhombus origin. We'll need these later.
        squish_offset = (xsb + ysb) * SQUISH_CONSTANT_2D
        xb = xsb + squish_offset
        yb = ysb + squish_offset

        # Compute grid coordinates relative to rhombus origin.
        xins = xs - xsb
        yins = ys - ysb

        # Sum those together to get a value that determines which region we're in.
        in_sum = xins + yins

        # Positions relative to origin point.
        dx0 = x - xb
        dy0 = y - yb

        value = 0

        # Contribution (1,0)
        dx1 = dx0 - 1 - SQUISH_CONSTANT_2D
        dy1 = dy0 - 0 - SQUISH_CONSTANT_2D
        attn1 = 2 - dx1 * dx1 - dy1 * dy1
        extrapolate = self._extrapolate2d
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate(xsb + 1, ysb + 0, dx1, dy1)

        # Contribution (0,1)
        dx2 = dx0 - 0 - SQUISH_CONSTANT_2D
        dy2 = dy0 - 1 - SQUISH_CONSTANT_2D
        attn2 = 2 - dx2 * dx2 - dy2 * dy2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate(xsb + 0, ysb + 1, dx2, dy2)

        if in_sum <= 1: # We're inside the triangle (2-Simplex) at (0,0)
            zins = 1 - in_sum
            if zins > xins or zins > yins: # (0,0) is one of the closest two triangular vertices
                if xins > yins:
                    xsv_ext = xsb + 1
                    ysv_ext = ysb - 1
                    dx_ext = dx0 - 1
                    dy_ext = dy0 + 1
                else:
                    xsv_ext = xsb - 1
                    ysv_ext = ysb + 1
                    dx_ext = dx0 + 1
                    dy_ext = dy0 - 1
            else: # (1,0) and (0,1) are the closest two vertices.
                xsv_ext = xsb + 1
                ysv_ext = ysb + 1
                dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
                dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT_2D
        else: # We're inside the triangle (2-Simplex) at (1,1)
            zins = 2 - in_sum
            if zins < xins or zins < yins: # (0,0) is one of the closest two triangular vertices
                if xins > yins:
                    xsv_ext = xsb + 2
                    ysv_ext = ysb + 0
                    dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT_2D
                    dy_ext = dy0 + 0 - 2 * SQUISH_CONSTANT_2D
                else:
                    xsv_ext = xsb + 0
                    ysv_ext = ysb + 2
                    dx_ext = dx0 + 0 - 2 * SQUISH_CONSTANT_2D
                    dy_ext = dy0 - 2 - 2 * SQUISH_CONSTANT_2D
            else: # (1,0) and (0,1) are the closest two vertices.
                dx_ext = dx0
                dy_ext = dy0
                xsv_ext = xsb
                ysv_ext = ysb
            xsb += 1
            ysb += 1
            dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
            dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT_2D

        # Contribution (0,0) or (1,1)
        attn0 = 2 - dx0 * dx0 - dy0 * dy0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * extrapolate(xsb, ysb, dx0, dy0)

        # Extra Vertex
        attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
        if attn_ext > 0:
            attn_ext *= attn_ext
            value += attn_ext * attn_ext * extrapolate(xsv_ext, ysv_ext, dx_ext, dy_ext)

        return value / NORM_CONSTANT_2D


    def noise3d(self, x, y, z):
        """
        Generate 3D OpenSimplex noise from X,Y,Z coordinates.
        """
        # Place input coordinates on simplectic honeycomb.
        stretch_offset = (x + y + z) * STRETCH_CONSTANT_3D
        xs = x + stretch_offset
        ys = y + stretch_offset
        zs = z + stretch_offset

        # Floor to get simplectic honeycomb coordinates of rhombohedron (stretched cube) super-cell origin.
        xsb = floor(xs)
        ysb = floor(ys)
        zsb = floor(zs)

        # Skew out to get actual coordinates of rhombohedron origin. We'll need these later.
        squish_offset = (xsb + ysb + zsb) * SQUISH_CONSTANT_3D
        xb = xsb + squish_offset
        yb = ysb + squish_offset
        zb = zsb + squish_offset

        # Compute simplectic honeycomb coordinates relative to rhombohedral origin.
        xins = xs - xsb
        yins = ys - ysb
        zins = zs - zsb

        # Sum those together to get a value that determines which region we're in.
        in_sum = xins + yins + zins

        # Positions relative to origin point.
        dx0 = x - xb
        dy0 = y - yb
        dz0 = z - zb

        value = 0
        extrapolate = self._extrapolate3d
        if in_sum <= 1: # We're inside the tetrahedron (3-Simplex) at (0,0,0)

            # Determine which two of (0,0,1), (0,1,0), (1,0,0) are closest.
            a_point = 0x01
            a_score = xins
            b_point = 0x02
            b_score = yins
            if a_score >= b_score and zins > b_score:
                b_score = zins
                b_point = 0x04
            elif a_score < b_score and zins > a_score:
                a_score = zins
                a_point = 0x04

            # Now we determine the two lattice points not part of the tetrahedron that may contribute.
            # This depends on the closest two tetrahedral vertices, including (0,0,0)
            wins = 1 - in_sum
            if wins > a_score or wins > b_score: # (0,0,0) is one of the closest two tetrahedral vertices.
                c = b_point if (b_score > a_score) else a_point # Our other closest vertex is the closest out of a and b.

                if (c & 0x01) == 0:
                    xsv_ext0 = xsb - 1
                    xsv_ext1 = xsb
                    dx_ext0 = dx0 + 1
                    dx_ext1 = dx0
                else:
                    xsv_ext0 = xsv_ext1 = xsb + 1
                    dx_ext0 = dx_ext1 = dx0 - 1

                if (c & 0x02) == 0:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy_ext1 = dy0
                    if (c & 0x01) == 0:
                        ysv_ext1 -= 1
                        dy_ext1 += 1
                    else:
                        ysv_ext0 -= 1
                        dy_ext0 += 1
                else:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy_ext1 = dy0 - 1

                if (c & 0x04) == 0:
                    zsv_ext0 = zsb
                    zsv_ext1 = zsb - 1
                    dz_ext0 = dz0
                    dz_ext1 = dz0 + 1
                else:
                    zsv_ext0 = zsv_ext1 = zsb + 1
                    dz_ext0 = dz_ext1 = dz0 - 1
            else: # (0,0,0) is not one of the closest two tetrahedral vertices.
                c = (a_point | b_point) # Our two extra vertices are determined by the closest two.

                if (c & 0x01) == 0:
                    xsv_ext0 = xsb
                    xsv_ext1 = xsb - 1
                    dx_ext0 = dx0 - 2 * SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 + 1 - SQUISH_CONSTANT_3D
                else:
                    xsv_ext0 = xsv_ext1 = xsb + 1
                    dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_3D

                if (c & 0x02) == 0:
                    ysv_ext0 = ysb
                    ysv_ext1 = ysb - 1
                    dy_ext0 = dy0 - 2 * SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 + 1 - SQUISH_CONSTANT_3D
                else:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_3D

                if (c & 0x04) == 0:
                    zsv_ext0 = zsb
                    zsv_ext1 = zsb - 1
                    dz_ext0 = dz0 - 2 * SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 + 1 - SQUISH_CONSTANT_3D
                else:
                    zsv_ext0 = zsv_ext1 = zsb + 1
                    dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_3D

            # Contribution (0,0,0)
            attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
            if attn0 > 0:
                attn0 *= attn0
                value += attn0 * attn0 * extrapolate(xsb + 0, ysb + 0, zsb + 0, dx0, dy0, dz0)

            # Contribution (1,0,0)
            dx1 = dx0 - 1 - SQUISH_CONSTANT_3D
            dy1 = dy0 - 0 - SQUISH_CONSTANT_3D
            dz1 = dz0 - 0 - SQUISH_CONSTANT_3D
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb + 1, ysb + 0, zsb + 0, dx1, dy1, dz1)

            # Contribution (0,1,0)
            dx2 = dx0 - 0 - SQUISH_CONSTANT_3D
            dy2 = dy0 - 1 - SQUISH_CONSTANT_3D
            dz2 = dz1
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb + 0, ysb + 1, zsb + 0, dx2, dy2, dz2)

            # Contribution (0,0,1)
            dx3 = dx2
            dy3 = dy1
            dz3 = dz0 - 1 - SQUISH_CONSTANT_3D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb + 0, ysb + 0, zsb + 1, dx3, dy3, dz3)
        elif in_sum >= 2: # We're inside the tetrahedron (3-Simplex) at (1,1,1)

            # Determine which two tetrahedral vertices are the closest, out of (1,1,0), (1,0,1), (0,1,1) but not (1,1,1).
            a_point = 0x06
            a_score = xins
            b_point = 0x05
            b_score = yins
            if a_score <= b_score and zins < b_score:
                b_score = zins
                b_point = 0x03
            elif a_score > b_score and zins < a_score:
                a_score = zins
                a_point = 0x03

            # Now we determine the two lattice points not part of the tetrahedron that may contribute.
            # This depends on the closest two tetrahedral vertices, including (1,1,1)
            wins = 3 - in_sum
            if wins < a_score or wins < b_score: # (1,1,1) is one of the closest two tetrahedral vertices.
                c = b_point if (b_score < a_score) else a_point # Our other closest vertex is the closest out of a and b.

                if (c & 0x01) != 0:
                    xsv_ext0 = xsb + 2
                    xsv_ext1 = xsb + 1
                    dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT_3D
                else:
                    xsv_ext0 = xsv_ext1 = xsb
                    dx_ext0 = dx_ext1 = dx0 - 3 * SQUISH_CONSTANT_3D

                if (c & 0x02) != 0:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT_3D
                    if (c & 0x01) != 0:
                        ysv_ext1 += 1
                        dy_ext1 -= 1
                    else:
                        ysv_ext0 += 1
                        dy_ext0 -= 1
                else:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy_ext1 = dy0 - 3 * SQUISH_CONSTANT_3D

                if (c & 0x04) != 0:
                    zsv_ext0 = zsb + 1
                    zsv_ext1 = zsb + 2
                    dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 - 3 * SQUISH_CONSTANT_3D
                else:
                    zsv_ext0 = zsv_ext1 = zsb
                    dz_ext0 = dz_ext1 = dz0 - 3 * SQUISH_CONSTANT_3D
            else: # (1,1,1) is not one of the closest two tetrahedral vertices.
                c = (a_point & b_point) # Our two extra vertices are determined by the closest two.

                if (c & 0x01) != 0:
                    xsv_ext0 = xsb + 1
                    xsv_ext1 = xsb + 2
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    xsv_ext0 = xsv_ext1 = xsb
                    dx_ext0 = dx0 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D

                if (c & 0x02) != 0:
                    ysv_ext0 = ysb + 1
                    ysv_ext1 = ysb + 2
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy0 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D

                if (c & 0x04) != 0:
                    zsv_ext0 = zsb + 1
                    zsv_ext1 = zsb + 2
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    zsv_ext0 = zsv_ext1 = zsb
                    dz_ext0 = dz0 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D

            # Contribution (1,1,0)
            dx3 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
            dy3 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
            dz3 = dz0 - 0 - 2 * SQUISH_CONSTANT_3D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb + 1, ysb + 1, zsb + 0, dx3, dy3, dz3)

            # Contribution (1,0,1)
            dx2 = dx3
            dy2 = dy0 - 0 - 2 * SQUISH_CONSTANT_3D
            dz2 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb + 1, ysb + 0, zsb + 1, dx2, dy2, dz2)

            # Contribution (0,1,1)
            dx1 = dx0 - 0 - 2 * SQUISH_CONSTANT_3D
            dy1 = dy3
            dz1 = dz2
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb + 0, ysb + 1, zsb + 1, dx1, dy1, dz1)

            # Contribution (1,1,1)
            dx0 = dx0 - 1 - 3 * SQUISH_CONSTANT_3D
            dy0 = dy0 - 1 - 3 * SQUISH_CONSTANT_3D
            dz0 = dz0 - 1 - 3 * SQUISH_CONSTANT_3D
            attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
            if attn0 > 0:
                attn0 *= attn0
                value += attn0 * attn0 * extrapolate(xsb + 1, ysb + 1, zsb + 1, dx0, dy0, dz0)
        else: # We're inside the octahedron (Rectified 3-Simplex) in between.
            # Decide between point (0,0,1) and (1,1,0) as closest
            p1 = xins + yins
            if p1 > 1:
                a_score = p1 - 1
                a_point = 0x03
                a_is_further_side = True
            else:
                a_score = 1 - p1
                a_point = 0x04
                a_is_further_side = False

            # Decide between point (0,1,0) and (1,0,1) as closest
            p2 = xins + zins
            if p2 > 1:
                b_score = p2 - 1
                b_point = 0x05
                b_is_further_side = True
            else:
                b_score = 1 - p2
                b_point = 0x02
                b_is_further_side = False

            # The closest out of the two (1,0,0) and (0,1,1) will replace the furthest out of the two decided above, if closer.
            p3 = yins + zins
            if p3 > 1:
                score = p3 - 1
                if a_score <= b_score and a_score < score:
                    a_point = 0x06
                    a_is_further_side = True
                elif a_score > b_score and b_score < score:
                    b_point = 0x06
                    b_is_further_side = True
            else:
                score = 1 - p3
                if a_score <= b_score and a_score < score:
                    a_point = 0x01
                    a_is_further_side = False
                elif a_score > b_score and b_score < score:
                    b_point = 0x01
                    b_is_further_side = False

            # Where each of the two closest points are determines how the extra two vertices are calculated.
            if a_is_further_side == b_is_further_side:
                if a_is_further_side: # Both closest points on (1,1,1) side

                    # One of the two extra points is (1,1,1)
                    dx_ext0 = dx0 - 1 - 3 * SQUISH_CONSTANT_3D
                    dy_ext0 = dy0 - 1 - 3 * SQUISH_CONSTANT_3D
                    dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT_3D
                    xsv_ext0 = xsb + 1
                    ysv_ext0 = ysb + 1
                    zsv_ext0 = zsb + 1

                    # Other extra point is based on the shared axis.
                    c = (a_point & b_point)
                    if (c & 0x01) != 0:
                        dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb + 2
                        ysv_ext1 = ysb
                        zsv_ext1 = zsb
                    elif (c & 0x02) != 0:
                        dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb
                        ysv_ext1 = ysb + 2
                        zsv_ext1 = zsb
                    else:
                        dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb
                        ysv_ext1 = ysb
                        zsv_ext1 = zsb + 2
                else:# Both closest points on (0,0,0) side

                    # One of the two extra points is (0,0,0)
                    dx_ext0 = dx0
                    dy_ext0 = dy0
                    dz_ext0 = dz0
                    xsv_ext0 = xsb
                    ysv_ext0 = ysb
                    zsv_ext0 = zsb

                    # Other extra point is based on the omitted axis.
                    c = (a_point | b_point)
                    if (c & 0x01) == 0:
                        dx_ext1 = dx0 + 1 - SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb - 1
                        ysv_ext1 = ysb + 1
                        zsv_ext1 = zsb + 1
                    elif (c & 0x02) == 0:
                        dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 + 1 - SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb + 1
                        ysv_ext1 = ysb - 1
                        zsv_ext1 = zsb + 1
                    else:
                        dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 + 1 - SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb + 1
                        ysv_ext1 = ysb + 1
                        zsv_ext1 = zsb - 1
            else: # One point on (0,0,0) side, one point on (1,1,1) side
                if a_is_further_side:
                    c1 = a_point
                    c2 = b_point
                else:
                    c1 = b_point
                    c2 = a_point

                # One contribution is a _permutation of (1,1,-1)
                if (c1 & 0x01) == 0:
                    dx_ext0 = dx0 + 1 - SQUISH_CONSTANT_3D
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                    xsv_ext0 = xsb - 1
                    ysv_ext0 = ysb + 1
                    zsv_ext0 = zsb + 1
                elif (c1 & 0x02) == 0:
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                    dy_ext0 = dy0 + 1 - SQUISH_CONSTANT_3D
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                    xsv_ext0 = xsb + 1
                    ysv_ext0 = ysb - 1
                    zsv_ext0 = zsb + 1
                else:
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                    dz_ext0 = dz0 + 1 - SQUISH_CONSTANT_3D
                    xsv_ext0 = xsb + 1
                    ysv_ext0 = ysb + 1
                    zsv_ext0 = zsb - 1

                # One contribution is a _permutation of (0,0,2)
                dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D
                dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D
                dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D
                xsv_ext1 = xsb
                ysv_ext1 = ysb
                zsv_ext1 = zsb
                if (c2 & 0x01) != 0:
                    dx_ext1 -= 2
                    xsv_ext1 += 2
                elif (c2 & 0x02) != 0:
                    dy_ext1 -= 2
                    ysv_ext1 += 2
                else:
                    dz_ext1 -= 2
                    zsv_ext1 += 2

            # Contribution (1,0,0)
            dx1 = dx0 - 1 - SQUISH_CONSTANT_3D
            dy1 = dy0 - 0 - SQUISH_CONSTANT_3D
            dz1 = dz0 - 0 - SQUISH_CONSTANT_3D
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb + 1, ysb + 0, zsb + 0, dx1, dy1, dz1)

            # Contribution (0,1,0)
            dx2 = dx0 - 0 - SQUISH_CONSTANT_3D
            dy2 = dy0 - 1 - SQUISH_CONSTANT_3D
            dz2 = dz1
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb + 0, ysb + 1, zsb + 0, dx2, dy2, dz2)

            # Contribution (0,0,1)
            dx3 = dx2
            dy3 = dy1
            dz3 = dz0 - 1 - SQUISH_CONSTANT_3D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb + 0, ysb + 0, zsb + 1, dx3, dy3, dz3)

            # Contribution (1,1,0)
            dx4 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
            dy4 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
            dz4 = dz0 - 0 - 2 * SQUISH_CONSTANT_3D
            attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4
            if attn4 > 0:
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb + 1, ysb + 1, zsb + 0, dx4, dy4, dz4)

            # Contribution (1,0,1)
            dx5 = dx4
            dy5 = dy0 - 0 - 2 * SQUISH_CONSTANT_3D
            dz5 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D
            attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5
            if attn5 > 0:
                attn5 *= attn5
                value += attn5 * attn5 * extrapolate(xsb + 1, ysb + 0, zsb + 1, dx5, dy5, dz5)

            # Contribution (0,1,1)
            dx6 = dx0 - 0 - 2 * SQUISH_CONSTANT_3D
            dy6 = dy4
            dz6 = dz5
            attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6
            if attn6 > 0:
                attn6 *= attn6
                value += attn6 * attn6 * extrapolate(xsb + 0, ysb + 1, zsb + 1, dx6, dy6, dz6)

        # First extra vertex
        attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0
        if attn_ext0 > 0:
            attn_ext0 *= attn_ext0
            value += attn_ext0 * attn_ext0 * extrapolate(xsv_ext0, ysv_ext0, zsv_ext0, dx_ext0, dy_ext0, dz_ext0)

        # Second extra vertex
        attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1
        if attn_ext1 > 0:
            attn_ext1 *= attn_ext1
            value += attn_ext1 * attn_ext1 * extrapolate(xsv_ext1, ysv_ext1, zsv_ext1, dx_ext1, dy_ext1, dz_ext1)

        return value / NORM_CONSTANT_3D


    def noise4d(self, x, y, z, w):
        """
        Generate 4D OpenSimplex noise from X,Y,Z,W coordinates.
        """
        # Place input coordinates on simplectic honeycomb.
        stretch_offset = (x + y + z + w) * STRETCH_CONSTANT_4D
        xs = x + stretch_offset
        ys = y + stretch_offset
        zs = z + stretch_offset
        ws = w + stretch_offset

        # Floor to get simplectic honeycomb coordinates of rhombo-hypercube super-cell origin.
        xsb = floor(xs)
        ysb = floor(ys)
        zsb = floor(zs)
        wsb = floor(ws)

        # Skew out to get actual coordinates of stretched rhombo-hypercube origin. We'll need these later.
        squish_offset = (xsb + ysb + zsb + wsb) * SQUISH_CONSTANT_4D
        xb = xsb + squish_offset
        yb = ysb + squish_offset
        zb = zsb + squish_offset
        wb = wsb + squish_offset

        # Compute simplectic honeycomb coordinates relative to rhombo-hypercube origin.
        xins = xs - xsb
        yins = ys - ysb
        zins = zs - zsb
        wins = ws - wsb

        # Sum those together to get a value that determines which region we're in.
        in_sum = xins + yins + zins + wins

        # Positions relative to origin po.
        dx0 = x - xb
        dy0 = y - yb
        dz0 = z - zb
        dw0 = w - wb

        value = 0
        extrapolate = self._extrapolate4d
        if in_sum <= 1: # We're inside the pentachoron (4-Simplex) at (0,0,0,0)

            # Determine which two of (0,0,0,1), (0,0,1,0), (0,1,0,0), (1,0,0,0) are closest.
            a_po = 0x01
            a_score = xins
            b_po = 0x02
            b_score = yins
            if a_score >= b_score and zins > b_score:
                b_score = zins
                b_po = 0x04
            elif a_score < b_score and zins > a_score:
                a_score = zins
                a_po = 0x04

            if a_score >= b_score and wins > b_score:
                b_score = wins
                b_po = 0x08
            elif a_score < b_score and wins > a_score:
                a_score = wins
                a_po = 0x08

            # Now we determine the three lattice pos not part of the pentachoron that may contribute.
            # This depends on the closest two pentachoron vertices, including (0,0,0,0)
            uins = 1 - in_sum
            if uins > a_score or uins > b_score: # (0,0,0,0) is one of the closest two pentachoron vertices.
                c = b_po if (b_score > a_score) else a_po # Our other closest vertex is the closest out of a and b.
                if (c & 0x01) == 0:
                    xsv_ext0 = xsb - 1
                    xsv_ext1 = xsv_ext2 = xsb
                    dx_ext0 = dx0 + 1
                    dx_ext1 = dx_ext2 = dx0
                else:
                    xsv_ext0 = xsv_ext1 = xsv_ext2 = xsb + 1
                    dx_ext0 = dx_ext1 = dx_ext2 = dx0 - 1

                if (c & 0x02) == 0:
                    ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb
                    dy_ext0 = dy_ext1 = dy_ext2 = dy0
                    if (c & 0x01) == 0x01:
                        ysv_ext0 -= 1
                        dy_ext0 += 1
                    else:
                        ysv_ext1 -= 1
                        dy_ext1 += 1

                else:
                    ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb + 1
                    dy_ext0 = dy_ext1 = dy_ext2 = dy0 - 1

                if (c & 0x04) == 0:
                    zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb
                    dz_ext0 = dz_ext1 = dz_ext2 = dz0
                    if (c & 0x03) != 0:
                        if (c & 0x03) == 0x03:
                            zsv_ext0 -= 1
                            dz_ext0 += 1
                        else:
                            zsv_ext1 -= 1
                            dz_ext1 += 1

                    else:
                        zsv_ext2 -= 1
                        dz_ext2 += 1

                else:
                    zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb + 1
                    dz_ext0 = dz_ext1 = dz_ext2 = dz0 - 1


                if (c & 0x08) == 0:
                    wsv_ext0 = wsv_ext1 = wsb
                    wsv_ext2 = wsb - 1
                    dw_ext0 = dw_ext1 = dw0
                    dw_ext2 = dw0 + 1
                else:
                    wsv_ext0 = wsv_ext1 = wsv_ext2 = wsb + 1
                    dw_ext0 = dw_ext1 = dw_ext2 = dw0 - 1

            else: # (0,0,0,0) is not one of the closest two pentachoron vertices.
                c = (a_po | b_po) # Our three extra vertices are determined by the closest two.

                if (c & 0x01) == 0:
                    xsv_ext0 = xsv_ext2 = xsb
                    xsv_ext1 = xsb - 1
                    dx_ext0 = dx0 - 2 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 + 1 - SQUISH_CONSTANT_4D
                    dx_ext2 = dx0 - SQUISH_CONSTANT_4D
                else:
                    xsv_ext0 = xsv_ext1 = xsv_ext2 = xsb + 1
                    dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx_ext2 = dx0 - 1 - SQUISH_CONSTANT_4D

                if (c & 0x02) == 0:
                    ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb
                    dy_ext0 = dy0 - 2 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2 = dy0 - SQUISH_CONSTANT_4D
                    if (c & 0x01) == 0x01:
                        ysv_ext1 -= 1
                        dy_ext1 += 1
                    else:
                        ysv_ext2 -= 1
                        dy_ext2 += 1

                else:
                    ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb + 1
                    dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2 = dy0 - 1 - SQUISH_CONSTANT_4D

                if (c & 0x04) == 0:
                    zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb
                    dz_ext0 = dz0 - 2 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2 = dz0 - SQUISH_CONSTANT_4D
                    if (c & 0x03) == 0x03:
                        zsv_ext1 -= 1
                        dz_ext1 += 1
                    else:
                        zsv_ext2 -= 1
                        dz_ext2 += 1

                else:
                    zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb + 1
                    dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2 = dz0 - 1 - SQUISH_CONSTANT_4D


                if (c & 0x08) == 0:
                    wsv_ext0 = wsv_ext1 = wsb
                    wsv_ext2 = wsb - 1
                    dw_ext0 = dw0 - 2 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 - SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 + 1 - SQUISH_CONSTANT_4D
                else:
                    wsv_ext0 = wsv_ext1 = wsv_ext2 = wsb + 1
                    dw_ext0 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw_ext2 = dw0 - 1 - SQUISH_CONSTANT_4D

            # Contribution (0,0,0,0)
            attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0 - dw0 * dw0
            if attn0 > 0:
                attn0 *= attn0
                value += attn0 * attn0 * extrapolate(xsb + 0, ysb + 0, zsb + 0, wsb + 0, dx0, dy0, dz0, dw0)

            # Contribution (1,0,0,0)
            dx1 = dx0 - 1 - SQUISH_CONSTANT_4D
            dy1 = dy0 - 0 - SQUISH_CONSTANT_4D
            dz1 = dz0 - 0 - SQUISH_CONSTANT_4D
            dw1 = dw0 - 0 - SQUISH_CONSTANT_4D
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb + 1, ysb + 0, zsb + 0, wsb + 0, dx1, dy1, dz1, dw1)

            # Contribution (0,1,0,0)
            dx2 = dx0 - 0 - SQUISH_CONSTANT_4D
            dy2 = dy0 - 1 - SQUISH_CONSTANT_4D
            dz2 = dz1
            dw2 = dw1
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb + 0, ysb + 1, zsb + 0, wsb + 0, dx2, dy2, dz2, dw2)

            # Contribution (0,0,1,0)
            dx3 = dx2
            dy3 = dy1
            dz3 = dz0 - 1 - SQUISH_CONSTANT_4D
            dw3 = dw1
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb + 0, ysb + 0, zsb + 1, wsb + 0, dx3, dy3, dz3, dw3)

            # Contribution (0,0,0,1)
            dx4 = dx2
            dy4 = dy1
            dz4 = dz1
            dw4 = dw0 - 1 - SQUISH_CONSTANT_4D
            attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
            if attn4 > 0:
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb + 0, ysb + 0, zsb + 0, wsb + 1, dx4, dy4, dz4, dw4)

        elif in_sum >= 3: # We're inside the pentachoron (4-Simplex) at (1,1,1,1)
            # Determine which two of (1,1,1,0), (1,1,0,1), (1,0,1,1), (0,1,1,1) are closest.
            a_po = 0x0E
            a_score = xins
            b_po = 0x0D
            b_score = yins
            if a_score <= b_score and zins < b_score:
                b_score = zins
                b_po = 0x0B
            elif a_score > b_score and zins < a_score:
                a_score = zins
                a_po = 0x0B

            if a_score <= b_score and wins < b_score:
                b_score = wins
                b_po = 0x07
            elif a_score > b_score and wins < a_score:
                a_score = wins
                a_po = 0x07

            # Now we determine the three lattice pos not part of the pentachoron that may contribute.
            # This depends on the closest two pentachoron vertices, including (0,0,0,0)
            uins = 4 - in_sum
            if uins < a_score or uins < b_score: # (1,1,1,1) is one of the closest two pentachoron vertices.
                c = b_po if (b_score < a_score) else a_po # Our other closest vertex is the closest out of a and b.

                if (c & 0x01) != 0:
                    xsv_ext0 = xsb + 2
                    xsv_ext1 = xsv_ext2 = xsb + 1
                    dx_ext0 = dx0 - 2 - 4 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx_ext2 = dx0 - 1 - 4 * SQUISH_CONSTANT_4D
                else:
                    xsv_ext0 = xsv_ext1 = xsv_ext2 = xsb
                    dx_ext0 = dx_ext1 = dx_ext2 = dx0 - 4 * SQUISH_CONSTANT_4D

                if (c & 0x02) != 0:
                    ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb + 1
                    dy_ext0 = dy_ext1 = dy_ext2 = dy0 - 1 - 4 * SQUISH_CONSTANT_4D
                    if (c & 0x01) != 0:
                        ysv_ext1 += 1
                        dy_ext1 -= 1
                    else:
                        ysv_ext0 += 1
                        dy_ext0 -= 1

                else:
                    ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb
                    dy_ext0 = dy_ext1 = dy_ext2 = dy0 - 4 * SQUISH_CONSTANT_4D

                if (c & 0x04) != 0:
                    zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb + 1
                    dz_ext0 = dz_ext1 = dz_ext2 = dz0 - 1 - 4 * SQUISH_CONSTANT_4D
                    if (c & 0x03) != 0x03:
                        if (c & 0x03) == 0:
                            zsv_ext0 += 1
                            dz_ext0 -= 1
                        else:
                            zsv_ext1 += 1
                            dz_ext1 -= 1

                    else:
                        zsv_ext2 += 1
                        dz_ext2 -= 1

                else:
                    zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb
                    dz_ext0 = dz_ext1 = dz_ext2 = dz0 - 4 * SQUISH_CONSTANT_4D

                if (c & 0x08) != 0:
                    wsv_ext0 = wsv_ext1 = wsb + 1
                    wsv_ext2 = wsb + 2
                    dw_ext0 = dw_ext1 = dw0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 2 - 4 * SQUISH_CONSTANT_4D
                else:
                    wsv_ext0 = wsv_ext1 = wsv_ext2 = wsb
                    dw_ext0 = dw_ext1 = dw_ext2 = dw0 - 4 * SQUISH_CONSTANT_4D

            else: # (1,1,1,1) is not one of the closest two pentachoron vertices.
                c = (a_po & b_po) # Our three extra vertices are determined by the closest two.

                if (c & 0x01) != 0:
                    xsv_ext0 = xsv_ext2 = xsb + 1
                    xsv_ext1 = xsb + 2
                    dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 - 2 - 3 * SQUISH_CONSTANT_4D
                    dx_ext2 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
                else:
                    xsv_ext0 = xsv_ext1 = xsv_ext2 = xsb
                    dx_ext0 = dx0 - 2 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx_ext2 = dx0 - 3 * SQUISH_CONSTANT_4D

                if (c & 0x02) != 0:
                    ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb + 1
                    dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
                    if (c & 0x01) != 0:
                        ysv_ext2 += 1
                        dy_ext2 -= 1
                    else:
                        ysv_ext1 += 1
                        dy_ext1 -= 1

                else:
                    ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb
                    dy_ext0 = dy0 - 2 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2 = dy0 - 3 * SQUISH_CONSTANT_4D

                if (c & 0x04) != 0:
                    zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb + 1
                    dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
                    if (c & 0x03) != 0:
                        zsv_ext2 += 1
                        dz_ext2 -= 1
                    else:
                        zsv_ext1 += 1
                        dz_ext1 -= 1

                else:
                    zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb
                    dz_ext0 = dz0 - 2 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2 = dz0 - 3 * SQUISH_CONSTANT_4D

                if (c & 0x08) != 0:
                    wsv_ext0 = wsv_ext1 = wsb + 1
                    wsv_ext2 = wsb + 2
                    dw_ext0 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 2 - 3 * SQUISH_CONSTANT_4D
                else:
                    wsv_ext0 = wsv_ext1 = wsv_ext2 = wsb
                    dw_ext0 = dw0 - 2 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw_ext2 = dw0 - 3 * SQUISH_CONSTANT_4D

            # Contribution (1,1,1,0)
            dx4 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
            dy4 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
            dz4 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
            dw4 = dw0 - 3 * SQUISH_CONSTANT_4D
            attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
            if attn4 > 0:
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb + 1, ysb + 1, zsb + 1, wsb + 0, dx4, dy4, dz4, dw4)

            # Contribution (1,1,0,1)
            dx3 = dx4
            dy3 = dy4
            dz3 = dz0 - 3 * SQUISH_CONSTANT_4D
            dw3 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb + 1, ysb + 1, zsb + 0, wsb + 1, dx3, dy3, dz3, dw3)

            # Contribution (1,0,1,1)
            dx2 = dx4
            dy2 = dy0 - 3 * SQUISH_CONSTANT_4D
            dz2 = dz4
            dw2 = dw3
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb + 1, ysb + 0, zsb + 1, wsb + 1, dx2, dy2, dz2, dw2)

            # Contribution (0,1,1,1)
            dx1 = dx0 - 3 * SQUISH_CONSTANT_4D
            dz1 = dz4
            dy1 = dy4
            dw1 = dw3
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb + 0, ysb + 1, zsb + 1, wsb + 1, dx1, dy1, dz1, dw1)

            # Contribution (1,1,1,1)
            dx0 = dx0 - 1 - 4 * SQUISH_CONSTANT_4D
            dy0 = dy0 - 1 - 4 * SQUISH_CONSTANT_4D
            dz0 = dz0 - 1 - 4 * SQUISH_CONSTANT_4D
            dw0 = dw0 - 1 - 4 * SQUISH_CONSTANT_4D
            attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0 - dw0 * dw0
            if attn0 > 0:
                attn0 *= attn0
                value += attn0 * attn0 * extrapolate(xsb + 1, ysb + 1, zsb + 1, wsb + 1, dx0, dy0, dz0, dw0)

        elif in_sum <= 2: # We're inside the first dispentachoron (Rectified 4-Simplex)
            a_is_bigger_side = True
            b_is_bigger_side = True

            # Decide between (1,1,0,0) and (0,0,1,1)
            if xins + yins > zins + wins:
                a_score = xins + yins
                a_po = 0x03
            else:
                a_score = zins + wins
                a_po = 0x0C

            # Decide between (1,0,1,0) and (0,1,0,1)
            if xins + zins > yins + wins:
                b_score = xins + zins
                b_po = 0x05
            else:
                b_score = yins + wins
                b_po = 0x0A

            # Closer between (1,0,0,1) and (0,1,1,0) will replace the further of a and b, if closer.
            if xins + wins > yins + zins:
                score = xins + wins
                if a_score >= b_score and score > b_score:
                    b_score = score
                    b_po = 0x09
                elif a_score < b_score and score > a_score:
                    a_score = score
                    a_po = 0x09

            else:
                score = yins + zins
                if a_score >= b_score and score > b_score:
                    b_score = score
                    b_po = 0x06
                elif a_score < b_score and score > a_score:
                    a_score = score
                    a_po = 0x06

            # Decide if (1,0,0,0) is closer.
            p1 = 2 - in_sum + xins
            if a_score >= b_score and p1 > b_score:
                b_score = p1
                b_po = 0x01
                b_is_bigger_side = False
            elif a_score < b_score and p1 > a_score:
                a_score = p1
                a_po = 0x01
                a_is_bigger_side = False

            # Decide if (0,1,0,0) is closer.
            p2 = 2 - in_sum + yins
            if a_score >= b_score and p2 > b_score:
                b_score = p2
                b_po = 0x02
                b_is_bigger_side = False
            elif a_score < b_score and p2 > a_score:
                a_score = p2
                a_po = 0x02
                a_is_bigger_side = False

            # Decide if (0,0,1,0) is closer.
            p3 = 2 - in_sum + zins
            if a_score >= b_score and p3 > b_score:
                b_score = p3
                b_po = 0x04
                b_is_bigger_side = False
            elif a_score < b_score and p3 > a_score:
                a_score = p3
                a_po = 0x04
                a_is_bigger_side = False

            # Decide if (0,0,0,1) is closer.
            p4 = 2 - in_sum + wins
            if a_score >= b_score and p4 > b_score:
                b_po = 0x08
                b_is_bigger_side = False
            elif a_score < b_score and p4 > a_score:
                a_po = 0x08
                a_is_bigger_side = False

            # Where each of the two closest pos are determines how the extra three vertices are calculated.
            if a_is_bigger_side == b_is_bigger_side:
                if a_is_bigger_side: # Both closest pos on the bigger side
                    c1 = (a_po | b_po)
                    c2 = (a_po & b_po)
                    if (c1 & 0x01) == 0:
                        xsv_ext0 = xsb
                        xsv_ext1 = xsb - 1
                        dx_ext0 = dx0 - 3 * SQUISH_CONSTANT_4D
                        dx_ext1 = dx0 + 1 - 2 * SQUISH_CONSTANT_4D
                    else:
                        xsv_ext0 = xsv_ext1 = xsb + 1
                        dx_ext0 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dx_ext1 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D

                    if (c1 & 0x02) == 0:
                        ysv_ext0 = ysb
                        ysv_ext1 = ysb - 1
                        dy_ext0 = dy0 - 3 * SQUISH_CONSTANT_4D
                        dy_ext1 = dy0 + 1 - 2 * SQUISH_CONSTANT_4D
                    else:
                        ysv_ext0 = ysv_ext1 = ysb + 1
                        dy_ext0 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dy_ext1 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D

                    if (c1 & 0x04) == 0:
                        zsv_ext0 = zsb
                        zsv_ext1 = zsb - 1
                        dz_ext0 = dz0 - 3 * SQUISH_CONSTANT_4D
                        dz_ext1 = dz0 + 1 - 2 * SQUISH_CONSTANT_4D
                    else:
                        zsv_ext0 = zsv_ext1 = zsb + 1
                        dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dz_ext1 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D

                    if (c1 & 0x08) == 0:
                        wsv_ext0 = wsb
                        wsv_ext1 = wsb - 1
                        dw_ext0 = dw0 - 3 * SQUISH_CONSTANT_4D
                        dw_ext1 = dw0 + 1 - 2 * SQUISH_CONSTANT_4D
                    else:
                        wsv_ext0 = wsv_ext1 = wsb + 1
                        dw_ext0 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dw_ext1 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D

                    # One combination is a _permutation of (0,0,0,2) based on c2
                    xsv_ext2 = xsb
                    ysv_ext2 = ysb
                    zsv_ext2 = zsb
                    wsv_ext2 = wsb
                    dx_ext2 = dx0 - 2 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - 2 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - 2 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 2 * SQUISH_CONSTANT_4D
                    if (c2 & 0x01) != 0:
                        xsv_ext2 += 2
                        dx_ext2 -= 2
                    elif (c2 & 0x02) != 0:
                        ysv_ext2 += 2
                        dy_ext2 -= 2
                    elif (c2 & 0x04) != 0:
                        zsv_ext2 += 2
                        dz_ext2 -= 2
                    else:
                        wsv_ext2 += 2
                        dw_ext2 -= 2

                else: # Both closest pos on the smaller side
                    # One of the two extra pos is (0,0,0,0)
                    xsv_ext2 = xsb
                    ysv_ext2 = ysb
                    zsv_ext2 = zsb
                    wsv_ext2 = wsb
                    dx_ext2 = dx0
                    dy_ext2 = dy0
                    dz_ext2 = dz0
                    dw_ext2 = dw0

                    # Other two pos are based on the omitted axes.
                    c = (a_po | b_po)

                    if (c & 0x01) == 0:
                        xsv_ext0 = xsb - 1
                        xsv_ext1 = xsb
                        dx_ext0 = dx0 + 1 - SQUISH_CONSTANT_4D
                        dx_ext1 = dx0 - SQUISH_CONSTANT_4D
                    else:
                        xsv_ext0 = xsv_ext1 = xsb + 1
                        dx_ext0 = dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_4D

                    if (c & 0x02) == 0:
                        ysv_ext0 = ysv_ext1 = ysb
                        dy_ext0 = dy_ext1 = dy0 - SQUISH_CONSTANT_4D
                        if (c & 0x01) == 0x01:
                            ysv_ext0 -= 1
                            dy_ext0 += 1
                        else:
                            ysv_ext1 -= 1
                            dy_ext1 += 1

                    else:
                        ysv_ext0 = ysv_ext1 = ysb + 1
                        dy_ext0 = dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_4D

                    if (c & 0x04) == 0:
                        zsv_ext0 = zsv_ext1 = zsb
                        dz_ext0 = dz_ext1 = dz0 - SQUISH_CONSTANT_4D
                        if (c & 0x03) == 0x03:
                            zsv_ext0 -= 1
                            dz_ext0 += 1
                        else:
                            zsv_ext1 -= 1
                            dz_ext1 += 1

                    else:
                        zsv_ext0 = zsv_ext1 = zsb + 1
                        dz_ext0 = dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_4D


                    if (c & 0x08) == 0:
                        wsv_ext0 = wsb
                        wsv_ext1 = wsb - 1
                        dw_ext0 = dw0 - SQUISH_CONSTANT_4D
                        dw_ext1 = dw0 + 1 - SQUISH_CONSTANT_4D
                    else:
                        wsv_ext0 = wsv_ext1 = wsb + 1
                        dw_ext0 = dw_ext1 = dw0 - 1 - SQUISH_CONSTANT_4D

            else: # One po on each "side"
                if a_is_bigger_side:
                    c1 = a_po
                    c2 = b_po
                else:
                    c1 = b_po
                    c2 = a_po

                # Two contributions are the bigger-sided po with each 0 replaced with -1.
                if (c1 & 0x01) == 0:
                    xsv_ext0 = xsb - 1
                    xsv_ext1 = xsb
                    dx_ext0 = dx0 + 1 - SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 - SQUISH_CONSTANT_4D
                else:
                    xsv_ext0 = xsv_ext1 = xsb + 1
                    dx_ext0 = dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_4D

                if (c1 & 0x02) == 0:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy_ext1 = dy0 - SQUISH_CONSTANT_4D
                    if (c1 & 0x01) == 0x01:
                        ysv_ext0 -= 1
                        dy_ext0 += 1
                    else:
                        ysv_ext1 -= 1
                        dy_ext1 += 1

                else:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_4D

                if (c1 & 0x04) == 0:
                    zsv_ext0 = zsv_ext1 = zsb
                    dz_ext0 = dz_ext1 = dz0 - SQUISH_CONSTANT_4D
                    if (c1 & 0x03) == 0x03:
                        zsv_ext0 -= 1
                        dz_ext0 += 1
                    else:
                        zsv_ext1 -= 1
                        dz_ext1 += 1

                else:
                    zsv_ext0 = zsv_ext1 = zsb + 1
                    dz_ext0 = dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_4D

                if (c1 & 0x08) == 0:
                    wsv_ext0 = wsb
                    wsv_ext1 = wsb - 1
                    dw_ext0 = dw0 - SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 + 1 - SQUISH_CONSTANT_4D
                else:
                    wsv_ext0 = wsv_ext1 = wsb + 1
                    dw_ext0 = dw_ext1 = dw0 - 1 - SQUISH_CONSTANT_4D

                # One contribution is a _permutation of (0,0,0,2) based on the smaller-sided po
                xsv_ext2 = xsb
                ysv_ext2 = ysb
                zsv_ext2 = zsb
                wsv_ext2 = wsb
                dx_ext2 = dx0 - 2 * SQUISH_CONSTANT_4D
                dy_ext2 = dy0 - 2 * SQUISH_CONSTANT_4D
                dz_ext2 = dz0 - 2 * SQUISH_CONSTANT_4D
                dw_ext2 = dw0 - 2 * SQUISH_CONSTANT_4D
                if (c2 & 0x01) != 0:
                    xsv_ext2 += 2
                    dx_ext2 -= 2
                elif (c2 & 0x02) != 0:
                    ysv_ext2 += 2
                    dy_ext2 -= 2
                elif (c2 & 0x04) != 0:
                    zsv_ext2 += 2
                    dz_ext2 -= 2
                else:
                    wsv_ext2 += 2
                    dw_ext2 -= 2

            # Contribution (1,0,0,0)
            dx1 = dx0 - 1 - SQUISH_CONSTANT_4D
            dy1 = dy0 - 0 - SQUISH_CONSTANT_4D
            dz1 = dz0 - 0 - SQUISH_CONSTANT_4D
            dw1 = dw0 - 0 - SQUISH_CONSTANT_4D
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb + 1, ysb + 0, zsb + 0, wsb + 0, dx1, dy1, dz1, dw1)

            # Contribution (0,1,0,0)
            dx2 = dx0 - 0 - SQUISH_CONSTANT_4D
            dy2 = dy0 - 1 - SQUISH_CONSTANT_4D
            dz2 = dz1
            dw2 = dw1
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb + 0, ysb + 1, zsb + 0, wsb + 0, dx2, dy2, dz2, dw2)

            # Contribution (0,0,1,0)
            dx3 = dx2
            dy3 = dy1
            dz3 = dz0 - 1 - SQUISH_CONSTANT_4D
            dw3 = dw1
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb + 0, ysb + 0, zsb + 1, wsb + 0, dx3, dy3, dz3, dw3)

            # Contribution (0,0,0,1)
            dx4 = dx2
            dy4 = dy1
            dz4 = dz1
            dw4 = dw0 - 1 - SQUISH_CONSTANT_4D
            attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
            if attn4 > 0:
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb + 0, ysb + 0, zsb + 0, wsb + 1, dx4, dy4, dz4, dw4)

            # Contribution (1,1,0,0)
            dx5 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            dy5 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            dz5 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            dw5 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5 - dw5 * dw5
            if attn5 > 0:
                attn5 *= attn5
                value += attn5 * attn5 * extrapolate(xsb + 1, ysb + 1, zsb + 0, wsb + 0, dx5, dy5, dz5, dw5)

            # Contribution (1,0,1,0)
            dx6 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            dy6 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            dz6 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            dw6 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6 - dw6 * dw6
            if attn6 > 0:
                attn6 *= attn6
                value += attn6 * attn6 * extrapolate(xsb + 1, ysb + 0, zsb + 1, wsb + 0, dx6, dy6, dz6, dw6)

            # Contribution (1,0,0,1)
            dx7 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            dy7 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            dz7 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            dw7 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            attn7 = 2 - dx7 * dx7 - dy7 * dy7 - dz7 * dz7 - dw7 * dw7
            if attn7 > 0:
                attn7 *= attn7
                value += attn7 * attn7 * extrapolate(xsb + 1, ysb + 0, zsb + 0, wsb + 1, dx7, dy7, dz7, dw7)

            # Contribution (0,1,1,0)
            dx8 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            dy8 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            dz8 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            dw8 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            attn8 = 2 - dx8 * dx8 - dy8 * dy8 - dz8 * dz8 - dw8 * dw8
            if attn8 > 0:
                attn8 *= attn8
                value += attn8 * attn8 * extrapolate(xsb + 0, ysb + 1, zsb + 1, wsb + 0, dx8, dy8, dz8, dw8)

            # Contribution (0,1,0,1)
            dx9 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            dy9 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            dz9 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            dw9 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            attn9 = 2 - dx9 * dx9 - dy9 * dy9 - dz9 * dz9 - dw9 * dw9
            if attn9 > 0:
                attn9 *= attn9
                value += attn9 * attn9 * extrapolate(xsb + 0, ysb + 1, zsb + 0, wsb + 1, dx9, dy9, dz9, dw9)

            # Contribution (0,0,1,1)
            dx10 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            dy10 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            dz10 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            dw10 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            attn10 = 2 - dx10 * dx10 - dy10 * dy10 - dz10 * dz10 - dw10 * dw10
            if attn10 > 0:
                attn10 *= attn10
                value += attn10 * attn10 * extrapolate(xsb + 0, ysb + 0, zsb + 1, wsb + 1, dx10, dy10, dz10, dw10)

        else: # We're inside the second dispentachoron (Rectified 4-Simplex)
            a_is_bigger_side = True
            b_is_bigger_side = True

            # Decide between (0,0,1,1) and (1,1,0,0)
            if xins + yins < zins + wins:
                a_score = xins + yins
                a_po = 0x0C
            else:
                a_score = zins + wins
                a_po = 0x03

            # Decide between (0,1,0,1) and (1,0,1,0)
            if xins + zins < yins + wins:
                b_score = xins + zins
                b_po = 0x0A
            else:
                b_score = yins + wins
                b_po = 0x05

            # Closer between (0,1,1,0) and (1,0,0,1) will replace the further of a and b, if closer.
            if xins + wins < yins + zins:
                score = xins + wins
                if a_score <= b_score and score < b_score:
                    b_score = score
                    b_po = 0x06
                elif a_score > b_score and score < a_score:
                    a_score = score
                    a_po = 0x06

            else:
                score = yins + zins
                if a_score <= b_score and score < b_score:
                    b_score = score
                    b_po = 0x09
                elif a_score > b_score and score < a_score:
                    a_score = score
                    a_po = 0x09

            # Decide if (0,1,1,1) is closer.
            p1 = 3 - in_sum + xins
            if a_score <= b_score and p1 < b_score:
                b_score = p1
                b_po = 0x0E
                b_is_bigger_side = False
            elif a_score > b_score and p1 < a_score:
                a_score = p1
                a_po = 0x0E
                a_is_bigger_side = False

            # Decide if (1,0,1,1) is closer.
            p2 = 3 - in_sum + yins
            if a_score <= b_score and p2 < b_score:
                b_score = p2
                b_po = 0x0D
                b_is_bigger_side = False
            elif a_score > b_score and p2 < a_score:
                a_score = p2
                a_po = 0x0D
                a_is_bigger_side = False

            # Decide if (1,1,0,1) is closer.
            p3 = 3 - in_sum + zins
            if a_score <= b_score and p3 < b_score:
                b_score = p3
                b_po = 0x0B
                b_is_bigger_side = False
            elif a_score > b_score and p3 < a_score:
                a_score = p3
                a_po = 0x0B
                a_is_bigger_side = False

            # Decide if (1,1,1,0) is closer.
            p4 = 3 - in_sum + wins
            if a_score <= b_score and p4 < b_score:
                b_po = 0x07
                b_is_bigger_side = False
            elif a_score > b_score and p4 < a_score:
                a_po = 0x07
                a_is_bigger_side = False

            # Where each of the two closest pos are determines how the extra three vertices are calculated.
            if a_is_bigger_side == b_is_bigger_side:
                if a_is_bigger_side: # Both closest pos on the bigger side
                    c1 = (a_po & b_po)
                    c2 = (a_po | b_po)

                    # Two contributions are _permutations of (0,0,0,1) and (0,0,0,2) based on c1
                    xsv_ext0 = xsv_ext1 = xsb
                    ysv_ext0 = ysv_ext1 = ysb
                    zsv_ext0 = zsv_ext1 = zsb
                    wsv_ext0 = wsv_ext1 = wsb
                    dx_ext0 = dx0 - SQUISH_CONSTANT_4D
                    dy_ext0 = dy0 - SQUISH_CONSTANT_4D
                    dz_ext0 = dz0 - SQUISH_CONSTANT_4D
                    dw_ext0 = dw0 - SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 - 2 * SQUISH_CONSTANT_4D
                    if (c1 & 0x01) != 0:
                        xsv_ext0 += 1
                        dx_ext0 -= 1
                        xsv_ext1 += 2
                        dx_ext1 -= 2
                    elif (c1 & 0x02) != 0:
                        ysv_ext0 += 1
                        dy_ext0 -= 1
                        ysv_ext1 += 2
                        dy_ext1 -= 2
                    elif (c1 & 0x04) != 0:
                        zsv_ext0 += 1
                        dz_ext0 -= 1
                        zsv_ext1 += 2
                        dz_ext1 -= 2
                    else:
                        wsv_ext0 += 1
                        dw_ext0 -= 1
                        wsv_ext1 += 2
                        dw_ext1 -= 2

                    # One contribution is a _permutation of (1,1,1,-1) based on c2
                    xsv_ext2 = xsb + 1
                    ysv_ext2 = ysb + 1
                    zsv_ext2 = zsb + 1
                    wsv_ext2 = wsb + 1
                    dx_ext2 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
                    if (c2 & 0x01) == 0:
                        xsv_ext2 -= 2
                        dx_ext2 += 2
                    elif (c2 & 0x02) == 0:
                        ysv_ext2 -= 2
                        dy_ext2 += 2
                    elif (c2 & 0x04) == 0:
                        zsv_ext2 -= 2
                        dz_ext2 += 2
                    else:
                        wsv_ext2 -= 2
                        dw_ext2 += 2

                else: # Both closest pos on the smaller side
                    # One of the two extra pos is (1,1,1,1)
                    xsv_ext2 = xsb + 1
                    ysv_ext2 = ysb + 1
                    zsv_ext2 = zsb + 1
                    wsv_ext2 = wsb + 1
                    dx_ext2 = dx0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 1 - 4 * SQUISH_CONSTANT_4D

                    # Other two pos are based on the shared axes.
                    c = (a_po & b_po)
                    if (c & 0x01) != 0:
                        xsv_ext0 = xsb + 2
                        xsv_ext1 = xsb + 1
                        dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT_4D
                        dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
                    else:
                        xsv_ext0 = xsv_ext1 = xsb
                        dx_ext0 = dx_ext1 = dx0 - 3 * SQUISH_CONSTANT_4D

                    if (c & 0x02) != 0:
                        ysv_ext0 = ysv_ext1 = ysb + 1
                        dy_ext0 = dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
                        if (c & 0x01) == 0:
                            ysv_ext0 += 1
                            dy_ext0 -= 1
                        else:
                            ysv_ext1 += 1
                            dy_ext1 -= 1

                    else:
                        ysv_ext0 = ysv_ext1 = ysb
                        dy_ext0 = dy_ext1 = dy0 - 3 * SQUISH_CONSTANT_4D

                    if (c & 0x04) != 0:
                        zsv_ext0 = zsv_ext1 = zsb + 1
                        dz_ext0 = dz_ext1 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
                        if (c & 0x03) == 0:
                            zsv_ext0 += 1
                            dz_ext0 -= 1
                        else:
                            zsv_ext1 += 1
                            dz_ext1 -= 1

                    else:
                        zsv_ext0 = zsv_ext1 = zsb
                        dz_ext0 = dz_ext1 = dz0 - 3 * SQUISH_CONSTANT_4D


                    if (c & 0x08) != 0:
                        wsv_ext0 = wsb + 1
                        wsv_ext1 = wsb + 2
                        dw_ext0 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dw_ext1 = dw0 - 2 - 3 * SQUISH_CONSTANT_4D
                    else:
                        wsv_ext0 = wsv_ext1 = wsb
                        dw_ext0 = dw_ext1 = dw0 - 3 * SQUISH_CONSTANT_4D

            else: # One po on each "side"
                if a_is_bigger_side:
                    c1 = a_po
                    c2 = b_po
                else:
                    c1 = b_po
                    c2 = a_po

                # Two contributions are the bigger-sided po with each 1 replaced with 2.
                if (c1 & 0x01) != 0:
                    xsv_ext0 = xsb + 2
                    xsv_ext1 = xsb + 1
                    dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
                else:
                    xsv_ext0 = xsv_ext1 = xsb
                    dx_ext0 = dx_ext1 = dx0 - 3 * SQUISH_CONSTANT_4D

                if (c1 & 0x02) != 0:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
                    if (c1 & 0x01) == 0:
                        ysv_ext0 += 1
                        dy_ext0 -= 1
                    else:
                        ysv_ext1 += 1
                        dy_ext1 -= 1

                else:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy_ext1 = dy0 - 3 * SQUISH_CONSTANT_4D

                if (c1 & 0x04) != 0:
                    zsv_ext0 = zsv_ext1 = zsb + 1
                    dz_ext0 = dz_ext1 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
                    if (c1 & 0x03) == 0:
                        zsv_ext0 += 1
                        dz_ext0 -= 1
                    else:
                        zsv_ext1 += 1
                        dz_ext1 -= 1

                else:
                    zsv_ext0 = zsv_ext1 = zsb
                    dz_ext0 = dz_ext1 = dz0 - 3 * SQUISH_CONSTANT_4D

                if (c1 & 0x08) != 0:
                    wsv_ext0 = wsb + 1
                    wsv_ext1 = wsb + 2
                    dw_ext0 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 - 2 - 3 * SQUISH_CONSTANT_4D
                else:
                    wsv_ext0 = wsv_ext1 = wsb
                    dw_ext0 = dw_ext1 = dw0 - 3 * SQUISH_CONSTANT_4D

                # One contribution is a _permutation of (1,1,1,-1) based on the smaller-sided po
                xsv_ext2 = xsb + 1
                ysv_ext2 = ysb + 1
                zsv_ext2 = zsb + 1
                wsv_ext2 = wsb + 1
                dx_ext2 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
                dy_ext2 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
                dz_ext2 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
                dw_ext2 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
                if (c2 & 0x01) == 0:
                    xsv_ext2 -= 2
                    dx_ext2 += 2
                elif (c2 & 0x02) == 0:
                    ysv_ext2 -= 2
                    dy_ext2 += 2
                elif (c2 & 0x04) == 0:
                    zsv_ext2 -= 2
                    dz_ext2 += 2
                else:
                    wsv_ext2 -= 2
                    dw_ext2 += 2

            # Contribution (1,1,1,0)
            dx4 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
            dy4 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
            dz4 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
            dw4 = dw0 - 3 * SQUISH_CONSTANT_4D
            attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
            if attn4 > 0:
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb + 1, ysb + 1, zsb + 1, wsb + 0, dx4, dy4, dz4, dw4)

            # Contribution (1,1,0,1)
            dx3 = dx4
            dy3 = dy4
            dz3 = dz0 - 3 * SQUISH_CONSTANT_4D
            dw3 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb + 1, ysb + 1, zsb + 0, wsb + 1, dx3, dy3, dz3, dw3)

            # Contribution (1,0,1,1)
            dx2 = dx4
            dy2 = dy0 - 3 * SQUISH_CONSTANT_4D
            dz2 = dz4
            dw2 = dw3
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb + 1, ysb + 0, zsb + 1, wsb + 1, dx2, dy2, dz2, dw2)

            # Contribution (0,1,1,1)
            dx1 = dx0 - 3 * SQUISH_CONSTANT_4D
            dz1 = dz4
            dy1 = dy4
            dw1 = dw3
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb + 0, ysb + 1, zsb + 1, wsb + 1, dx1, dy1, dz1, dw1)

            # Contribution (1,1,0,0)
            dx5 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            dy5 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            dz5 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            dw5 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5 - dw5 * dw5
            if attn5 > 0:
                attn5 *= attn5
                value += attn5 * attn5 * extrapolate(xsb + 1, ysb + 1, zsb + 0, wsb + 0, dx5, dy5, dz5, dw5)

            # Contribution (1,0,1,0)
            dx6 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            dy6 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            dz6 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            dw6 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6 - dw6 * dw6
            if attn6 > 0:
                attn6 *= attn6
                value += attn6 * attn6 * extrapolate(xsb + 1, ysb + 0, zsb + 1, wsb + 0, dx6, dy6, dz6, dw6)

            # Contribution (1,0,0,1)
            dx7 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            dy7 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            dz7 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            dw7 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            attn7 = 2 - dx7 * dx7 - dy7 * dy7 - dz7 * dz7 - dw7 * dw7
            if attn7 > 0:
                attn7 *= attn7
                value += attn7 * attn7 * extrapolate(xsb + 1, ysb + 0, zsb + 0, wsb + 1, dx7, dy7, dz7, dw7)

            # Contribution (0,1,1,0)
            dx8 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            dy8 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            dz8 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            dw8 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            attn8 = 2 - dx8 * dx8 - dy8 * dy8 - dz8 * dz8 - dw8 * dw8
            if attn8 > 0:
                attn8 *= attn8
                value += attn8 * attn8 * extrapolate(xsb + 0, ysb + 1, zsb + 1, wsb + 0, dx8, dy8, dz8, dw8)

            # Contribution (0,1,0,1)
            dx9 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            dy9 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            dz9 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            dw9 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            attn9 = 2 - dx9 * dx9 - dy9 * dy9 - dz9 * dz9 - dw9 * dw9
            if attn9 > 0:
                attn9 *= attn9
                value += attn9 * attn9 * extrapolate(xsb + 0, ysb + 1, zsb + 0, wsb + 1, dx9, dy9, dz9, dw9)

            # Contribution (0,0,1,1)
            dx10 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            dy10 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            dz10 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            dw10 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            attn10 = 2 - dx10 * dx10 - dy10 * dy10 - dz10 * dz10 - dw10 * dw10
            if attn10 > 0:
                attn10 *= attn10
                value += attn10 * attn10 * extrapolate(xsb + 0, ysb + 0, zsb + 1, wsb + 1, dx10, dy10, dz10, dw10)

        # First extra vertex
        attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0 - dw_ext0 * dw_ext0
        if attn_ext0 > 0:
            attn_ext0 *= attn_ext0
            value += attn_ext0 * attn_ext0 * extrapolate(xsv_ext0, ysv_ext0, zsv_ext0, wsv_ext0, dx_ext0, dy_ext0, dz_ext0, dw_ext0)

        # Second extra vertex
        attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1 - dw_ext1 * dw_ext1
        if attn_ext1 > 0:
            attn_ext1 *= attn_ext1
            value += attn_ext1 * attn_ext1 * extrapolate(xsv_ext1, ysv_ext1, zsv_ext1, wsv_ext1, dx_ext1, dy_ext1, dz_ext1, dw_ext1)

        # Third extra vertex
        attn_ext2 = 2 - dx_ext2 * dx_ext2 - dy_ext2 * dy_ext2 - dz_ext2 * dz_ext2 - dw_ext2 * dw_ext2
        if attn_ext2 > 0:
            attn_ext2 *= attn_ext2
            value += attn_ext2 * attn_ext2 * extrapolate(xsv_ext2, ysv_ext2, zsv_ext2, wsv_ext2, dx_ext2, dy_ext2, dz_ext2, dw_ext2)

        return value / NORM_CONSTANT_4D
