# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Read TIFF from FEI SEM microscopes.

Backend Library: internal

This format is based on :mod:`TIFF <imageio.plugins.tifffile>`, and supports the
same parameters. FEI microscopes append metadata as ASCII text at the end of the
file, which this reader correctly extracts.

Parameters
----------
discard_watermark : bool
    If True (default), discard the bottom rows of the image, which
    contain no image data, only a watermark with metadata.
watermark_height : int
    The height in pixels of the FEI watermark. The default is 70.

See Also
--------
    :mod:`imageio.plugins.tifffile`

"""


from .tifffile import TiffFormat


class FEISEMFormat(TiffFormat):
    """See :mod:`imageio.plugins.feisem`"""

    def _can_write(self, request):
        return False  # FEI-SEM only supports reading

    class Reader(TiffFormat.Reader):
        def _get_data(self, index=0, discard_watermark=True, watermark_height=70):
            """Get image and metadata from given index.

            FEI images usually (always?) contain a watermark at the
            bottom of the image, 70 pixels high. We discard this by
            default as it does not contain any information not present
            in the metadata.
            """
            im, meta = super(FEISEMFormat.Reader, self)._get_data(index)
            if discard_watermark:
                im = im[:-watermark_height]
            return im, meta

        def _get_meta_data(self, index=None):
            """Read the metadata from an FEI SEM TIFF.

            This metadata is included as ASCII text at the end of the file.

            The index, if provided, is ignored.

            Returns
            -------
            metadata : dict
                Dictionary of metadata.
            """
            if hasattr(self, "_fei_meta"):
                return self._fei_meta

            md = {"root": {}}
            current_tag = "root"
            reading_metadata = False
            filename = self.request.get_local_filename()
            with open(filename, encoding="utf8", errors="ignore") as fin:
                for line in fin:
                    if not reading_metadata:
                        if not line.startswith("Date="):
                            continue
                        else:
                            reading_metadata = True
                    line = line.rstrip()
                    if line.startswith("["):
                        current_tag = line.lstrip("[").rstrip("]")
                        md[current_tag] = {}
                    else:
                        if "=" in line:  # ignore empty and irrelevant lines
                            key, val = line.split("=", maxsplit=1)
                            for tag_type in (int, float):
                                try:
                                    val = tag_type(val)
                                except ValueError:
                                    continue
                                else:
                                    break
                            md[current_tag][key] = val
            if not md["root"] and len(md) == 1:
                raise ValueError("Input file %s contains no FEI metadata." % filename)

            self._fei_meta = md
            return md
