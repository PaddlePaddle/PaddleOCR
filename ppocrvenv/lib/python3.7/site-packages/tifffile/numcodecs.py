# tifffile/numcodecs.py

# Copyright (c) 2021, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""TIFF codec for numcodecs based on tifffile."""

__all__ = ('register_codec', 'Tiff')

from io import BytesIO

from numcodecs import registry
from numcodecs.abc import Codec

import tifffile


class Tiff(Codec):
    """TIFF codec for numcodecs."""

    codec_id = 'tifffile'

    def __init__(
        self,
        # TiffFile.asarray
        key=None,
        series=None,
        level=None,
        maxworkers=None,
        # TiffWriter
        bigtiff=None,
        byteorder=None,
        imagej=False,
        ome=None,
        # TiffWriter.write
        photometric=None,
        planarconfig=None,
        volumetric=None,
        tile=None,
        truncate=False,
        rowsperstrip=None,
        compression=None,
        predictor=None,
        subsampling=None,
        metadata={},
        extratags=(),
    ):
        self.key = key
        self.series = series
        self.level = level
        self.maxworkers = maxworkers

        self.bigtiff = bigtiff
        self.byteorder = byteorder
        self.imagej = imagej
        self.ome = ome

        self.photometric = photometric
        self.planarconfig = planarconfig
        self.volumetric = volumetric
        self.tile = tile
        self.truncate = truncate
        self.rowsperstrip = rowsperstrip
        self.compression = compression
        self.predictor = predictor
        self.subsampling = subsampling
        self.metadata = metadata
        self.extratags = extratags

    def encode(self, buf):
        """Return TIFF file as bytes."""
        with BytesIO() as fh:
            with tifffile.TiffWriter(
                fh,
                bigtiff=self.bigtiff,
                byteorder=self.byteorder,
                imagej=self.imagej,
                ome=self.ome,
            ) as tif:
                tif.write(
                    buf,
                    photometric=self.photometric,
                    planarconfig=self.planarconfig,
                    volumetric=self.volumetric,
                    tile=self.tile,
                    truncate=self.truncate,
                    rowsperstrip=self.rowsperstrip,
                    compression=self.compression,
                    predictor=self.predictor,
                    subsampling=self.subsampling,
                    metadata=self.metadata,
                    extratags=self.extratags,
                )
            result = fh.getvalue()
        return result

    def decode(self, buf, out=None):
        """Return decoded image as numpy array."""
        with BytesIO(buf) as fh:
            with tifffile.TiffFile(fh) as tif:
                result = tif.asarray(
                    key=self.key,
                    series=self.series,
                    level=self.level,
                    maxworkers=self.maxworkers,
                    out=out,
                )
        return result


def register_codec(cls=Tiff, codec_id=None):
    """Register Tiff codec with numcodecs."""
    registry.register_codec(cls, codec_id=codec_id)
