# tifffile.py

# Copyright (c) 2008-2021, Christoph Gohlke
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

r"""Read and write TIFF files.

Tifffile is a Python library to

(1) store numpy arrays in TIFF (Tagged Image File Format) files, and
(2) read image and metadata from TIFF-like files used in bioimaging.

Image and metadata can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, SGI,
NIHImage, ImageJ, MicroManager, FluoView, ScanImage, SEQ, GEL, SVS, SCN, SIS,
BIF, ZIF (Zoomable Image File Format), QPTIFF (QPI), NDPI, and GeoTIFF files.

Image data can be read as numpy arrays or zarr arrays/groups from strips,
tiles, pages (IFDs), SubIFDs, higher order series, and pyramidal levels.

Numpy arrays can be written to TIFF, BigTIFF, OME-TIFF, and ImageJ hyperstack
compatible files in multi-page, volumetric, pyramidal, memory-mappable, tiled,
predicted, or compressed form.

A subset of the TIFF specification is supported, mainly 8, 16, 32 and 64-bit
integer, 16, 32 and 64-bit float, grayscale and multi-sample images.
Specifically, CCITT and OJPEG compression, chroma subsampling without JPEG
compression, color space transformations, samples with differing types, or
IPTC, ICC, and XMP metadata are not implemented.

TIFF, the Tagged Image File Format, was created by the Aldus Corporation and
Adobe Systems Incorporated. BigTIFF allows for files larger than 4 GB.
STK, LSM, FluoView, SGI, SEQ, GEL, QPTIFF, NDPI, SCN, SVS, ZIF, BIF, and
OME-TIFF, are custom extensions defined by Molecular Devices (Universal Imaging
Corporation), Carl Zeiss MicroImaging, Olympus, Silicon Graphics International,
Media Cybernetics, Molecular Dynamics, PerkinElmer, Hamamatsu, Leica,
ObjectivePathology, Roche Digital Pathology, and the Open Microscopy
Environment consortium, respectively.

For command line usage run ``python -m tifffile --help``

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: BSD 3-Clause

:Version: 2021.11.2

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.7.9, 3.8.10, 3.9.7, 3.10.0, 64-bit <https://www.python.org>`_
* `Numpy 1.21.3 <https://pypi.org/project/numpy/>`_
* `Imagecodecs 2021.8.26  <https://pypi.org/project/imagecodecs/>`_
  (required only for encoding or decoding LZW, JPEG, etc.)
* `Matplotlib 3.4.3 <https://pypi.org/project/matplotlib/>`_
  (required only for plotting)
* `Lxml 4.6.3 <https://pypi.org/project/lxml/>`_
  (required only for validating and printing XML)
* `Zarr 2.10.3 <https://pypi.org/project/zarr/>`_
  (required only for opening zarr storage)

Revisions
---------
2021.11.2
    Pass 4731 tests.
    Lazy-load non-essential tag values (breaking).
    Warn when reading from closed file.
    Support ImageJ 'prop' metadata type (#103).
    Support writing indexed ImageJ format.
    Fix multi-threaded access of multi-page Zarr stores with chunkmode 2.
    Raise error if truncate is used with compression, packints, or tile.
    Read STK metadata without UIC2tag.
    Improve log and warning messages (WIP).
    Improve string representation of large tag values.
2021.10.12
    Revert renaming of 'file' parameter in FileSequence.asarray (breaking).
    Deprecate 'file' parameter in FileSequence.asarray.
2021.10.10
    Disallow letters as indices in FileSequence; use categories (breaking).
    Do not warn of missing files in FileSequence; use files_missing property.
    Support predictors in ZarrTiffStore.write_fsspec.
    Add option to specify zarr group name in write_fsspec.
    Add option to specify categories for FileSequence patterns (#76).
    Add option to specify chunk shape and dtype for ZarrFileSequenceStore.
    Add option to tile ZarrFileSequenceStore and FileSequence.asarray.
    Add option to pass additional zattrs to Zarr stores.
    Detect Roche BIF files.
2021.8.30
    Fix horizontal differencing with non-native byte order.
    Fix multi-threaded access of memory-mappable, multi-page Zarr stores (#67).
2021.8.8
    Fix tag offset and valueoffset for NDPI > 4 GB (#96).
2021.7.30
    Deprecate first parameter to TiffTag.overwrite (no longer required).
    TiffTag init API change (breaking).
    Detect Ventana BIF series and warn that tiles are not stitched.
    Enable reading PreviewImage from RAW formats (#93, #94).
    Work around numpy.ndarray.tofile is very slow for non-contiguous arrays.
    Fix issues with PackBits compression (requires imagecodecs 2021.7.30).
2021.7.2
    Decode complex integer images found in SAR GeoTIFF.
    Support reading NDPI with JPEG-XR compression.
    Deprecate TiffWriter RGB auto-detection, except for RGB24/48 and RGBA32/64.
2021.6.14
    Set stacklevel for deprecation warnings (#89).
    Fix svs_description_metadata for SVS with double header (#88, breaking).
    Fix reading JPEG compressed CMYK images.
    Support ALT_JPEG and JPEG_2000_LOSSY compression found in Bio-Formats.
    Log warning if TiffWriter auto-detects RGB mode (specify photometric).
2021.6.6
    Fix TIFF.COMPESSOR typo (#85).
    Round resolution numbers that do not fit in 64-bit rationals (#81).
    Add support for JPEG XL compression.
    Add numcodecs compatible TIFF codec.
    Rename ZarrFileStore to ZarrFileSequenceStore (breaking).
    Add method to export fsspec ReferenceFileSystem from ZarrFileStore.
    Fix fsspec ReferenceFileSystem v1 for multifile series.
    Fix creating OME-TIFF with micron character in OME-XML.
2021.4.8
    Fix reading OJPEG with wrong photometric or samplesperpixel tags (#75).
    Fix fsspec ReferenceFileSystem v1 and JPEG compression.
    Use TiffTagRegistry for NDPI_TAGS, EXIF_TAGS, GPS_TAGS, IOP_TAGS constants.
    Make TIFF.GEO_KEYS an Enum (breaking).
2021.3.31
    Use JPEG restart markers as tile offsets in NDPI.
    Support version 1 and more codecs in fsspec ReferenceFileSystem (untested).
2021.3.17
    Fix regression reading multi-file OME-TIFF with missing files (#72).
    Fix fsspec ReferenceFileSystem with non-native byte order (#56).
2021.3.16
    TIFF is no longer a defended trademark.
    Add method to export fsspec ReferenceFileSystem from ZarrTiffStore (#56).
2021.3.5
    Preliminary support for EER format (#68).
    Do not warn about unknown compression (#68).
2021.3.4
    Fix reading multi-file, multi-series OME-TIFF (#67).
    Detect ScanImage 2021 files (#46).
    Shape new version ScanImage series according to metadata (breaking).
    Remove Description key from TiffFile.scanimage_metadata dict (breaking).
    Also return ScanImage version from read_scanimage_metadata (breaking).
    Fix docstrings.
2021.2.26
    Squeeze axes of LSM series by default (breaking).
    Add option to preserve single dimensions when reading from series (WIP).
    Do not allow appending to OME-TIFF files.
    Fix reading STK files without name attribute in metadata.
    Make TIFF constants multi-thread safe and pickleable (#64).
    Add detection of NDTiffStorage MajorVersion to read_micromanager_metadata.
    Support ScanImage v4 files in read_scanimage_metadata.
2021.2.1
    Fix multi-threaded access of ZarrTiffStores using same TiffFile instance.
    Use fallback zlib and lzma codecs with imagecodecs lite builds.
    Open Olympus and Panasonic RAW files for parsing, albeit not supported.
    Support X2 and X4 differencing found in DNG.
    Support reading JPEG_LOSSY compression found in DNG.
2021.1.14
    Try ImageJ series if OME series fails (#54)
    Add option to use pages as chunks in ZarrFileStore (experimental).
    Fix reading from file objects with no readinto function.
2021.1.11
    Fix test errors on PyPy.
    Fix decoding bitorder with imagecodecs >= 2021.1.11.
2021.1.8
    Decode float24 using imagecodecs >= 2021.1.8.
    Consolidate reading of segments if possible.
2020.12.8
    Fix corrupted ImageDescription in multi shaped series if buffer too small.
    Fix libtiff warning that ImageDescription contains null byte in value.
    Fix reading invalid files using JPEG compression with palette colorspace.
2020.12.4
    Fix reading some JPEG compressed CFA images.
    Make index of SubIFDs a tuple.
    Pass through FileSequence.imread arguments in imread.
    Do not apply regex flags to FileSequence axes patterns (breaking).
2020.11.26
    Add option to pass axes metadata to ImageJ writer.
    Pad incomplete tiles passed to TiffWriter.write (#38).
    Split TiffTag constructor (breaking).
    Change TiffTag.dtype to TIFF.DATATYPES (breaking).
    Add TiffTag.overwrite method.
    Add script to change ImageDescription in files.
    Add TiffWriter.overwrite_description method (WIP).
2020.11.18
    Support writing SEPARATED color space (#37).
    Use imagecodecs.deflate codec if available.
    Fix SCN and NDPI series with Z dimensions.
    Add TiffReader alias for TiffFile.
    TiffPage.is_volumetric returns True if ImageDepth > 1.
    Zarr store getitem returns numpy arrays instead of bytes.
2020.10.1
    Formally deprecate unused TiffFile parameters (scikit-image #4996).
2020.9.30
    Allow to pass additional arguments to compression codecs.
    Deprecate TiffWriter.save method (use TiffWriter.write).
    Deprecate TiffWriter.save compress parameter (use compression).
    Remove multifile parameter from TiffFile (breaking).
    Pass all is_flag arguments from imread to TiffFile.
    Do not byte-swap JPEG2000, WEBP, PNG, JPEGXR segments in TiffPage.decode.
2020.9.29
    Fix reading files produced by ScanImage > 2015 (#29).
2020.9.28
    Derive ZarrStore from MutableMapping.
    Support zero shape ZarrTiffStore.
    Fix ZarrFileStore with non-TIFF files.
    Fix ZarrFileStore with missing files.
    Cache one chunk in ZarrFileStore.
    Keep track of already opened files in FileCache.
    Change parse_filenames function to return zero-based indices.
    Remove reopen parameter from asarray (breaking).
    Rename FileSequence.fromfile to imread (breaking).
2020.9.22
    Add experimental zarr storage interface (WIP).
    Remove unused first dimension from TiffPage.shaped (breaking).
    Move reading of STK planes to series interface (breaking).
    Always use virtual frames for ScanImage files.
    Use DimensionOrder to determine axes order in OmeXml.
    Enable writing striped volumetric images.
    Keep complete dataoffsets and databytecounts for TiffFrames.
    Return full size tiles from Tiffpage.segments.
    Rename TiffPage.is_sgi property to is_volumetric (breaking).
    Rename TiffPageSeries.is_pyramid to is_pyramidal (breaking).
    Fix TypeError when passing jpegtables to non-JPEG decode method (#25).
2020.9.3
    Do not write contiguous series by default (breaking).
    Allow to write to SubIFDs (WIP).
    Fix writing F-contiguous numpy arrays (#24).
2020.8.25
    Do not convert EPICS timeStamp to datetime object.
    Read incompletely written Micro-Manager image file stack header (#23).
    Remove tag 51123 values from TiffFile.micromanager_metadata (breaking).
2020.8.13
    Use tifffile metadata over OME and ImageJ for TiffFile.series (breaking).
    Fix writing iterable of pages with compression (#20).
    Expand error checking of TiffWriter data, dtype, shape, and tile arguments.
2020.7.24
    Parse nested OmeXml metadata argument (WIP).
    Do not lazy load TiffFrame JPEGTables.
    Fix conditionally skipping some tests.
2020.7.22
    Do not auto-enable OME-TIFF if description is passed to TiffWriter.save.
    Raise error writing empty bilevel or tiled images.
    Allow to write tiled bilevel images.
    Allow to write multi-page TIFF from iterable of single page images (WIP).
    Add function to validate OME-XML.
    Correct Philips slide width and length.
2020.7.17
    Initial support for writing OME-TIFF (WIP).
    Return samples as separate dimension in OME series (breaking).
    Fix modulo dimensions for multiple OME series.
    Fix some test errors on big endian systems (#18).
    Fix BytesWarning.
    Allow to pass TIFF.PREDICTOR values to TiffWriter.save.
2020.7.4
    Deprecate support for Python 3.6 (NEP 29).
    Move pyramidal subresolution series to TiffPageSeries.levels (breaking).
    Add parser for SVS, SCN, NDPI, and QPI pyramidal series.
    Read single-file OME-TIFF pyramids.
    Read NDPI files > 4 GB (#15).
    Include SubIFDs in generic series.
    Preliminary support for writing packed integer arrays (#11, WIP).
    Read more LSM info subrecords.
    Fix missing ReferenceBlackWhite tag for YCbCr photometrics.
    Fix reading lossless JPEG compressed DNG files.
2020.6.3
    ...

Refer to the CHANGES file for older revisions.

Notes
-----
The API is not stable yet and might change between revisions.

Tested on little-endian platforms only.

Python 32-bit versions are deprecated. Python <= 3.7 are no longer supported.

Tifffile relies on the `imagecodecs <https://pypi.org/project/imagecodecs/>`_
package for encoding and decoding LZW, JPEG, and other compressed image
segments.

Several TIFF-like formats do not strictly adhere to the TIFF6 specification,
some of which allow file or data sizes to exceed the 4 GB limit:

* *BigTIFF* is identified by version number 43 and uses different file
  header, IFD, and tag structures with 64-bit offsets. It adds more data types.
  Tifffile can read and write BigTIFF files.
* *ImageJ* hyperstacks store all image data, which may exceed 4 GB,
  contiguously after the first IFD. Files > 4 GB contain one IFD only.
  The size (shape and dtype) of the up to 6-dimensional image data can be
  determined from the ImageDescription tag of the first IFD, which is Latin-1
  encoded. Tifffile can read and write ImageJ hyperstacks.
* *OME-TIFF* stores up to 8-dimensional data in one or multiple TIFF of BigTIFF
  files. The 8-bit UTF-8 encoded OME-XML metadata found in the ImageDescription
  tag of the first IFD defines the position of TIFF IFDs in the high
  dimensional data. Tifffile can read OME-TIFF files, except when the OME-XML
  metadata are stored in a separate file. Tifffile can write numpy arrays
  to single-file OME-TIFF.
* *LSM* stores all IFDs below 4 GB but wraps around 32-bit StripOffsets.
  The StripOffsets of each series and position require separate unwrapping.
  The StripByteCounts tag contains the number of bytes for the uncompressed
  data. Tifffile can read large LSM files.
* *STK* (MetaMorph Stack) contains additional image planes stored contiguously
  after the image data of the first page. The total number of planes
  is equal to the counts of the UIC2tag. Tifffile can read STK files.
* *NDPI* uses some 64-bit offsets in the file header, IFD, and tag structures.
  Tag values/offsets can be corrected using high bits stored after IFD
  structures. Tifffile can read NDPI files > 4 GB.
  JPEG compressed segments with dimensions >65530 or missing restart markers
  are not decodable with libjpeg. Tifffile works around this limitation by
  separately decoding the MCUs between restart markers.
  BitsPerSample, SamplesPerPixel, and PhotometricInterpretation tags may
  contain wrong values, which can be corrected using the value of tag 65441.
* *Philips* TIFF slides store wrong ImageWidth and ImageLength tag values for
  tiled pages. The values can be corrected using the DICOM_PIXEL_SPACING
  attributes of the XML formatted description of the first page. Tifffile can
  read Philips slides.
* *Ventana/Roche BIF* slides store tiles and metadata in a BigTIFF container.
  Tiles may overlap and require stitching based on the TileJointInfo elements
  in the XMP tag. Volumetric scans are stored using the ImageDepth extension.
  Tifffile can read BIF and decode individual tiles, but does not perform
  stitching.
* *ScanImage* optionally allows corrupted non-BigTIFF files > 2 GB. The values
  of StripOffsets and StripByteCounts can be recovered using the constant
  differences of the offsets of IFD and tag values throughout the file.
  Tifffile can read such files if the image data are stored contiguously in
  each page.
* *GeoTIFF* sparse files allow strip or tile offsets and byte counts to be 0.
  Such segments are implicitly set to 0 or the NODATA value on reading.
  Tifffile can read GeoTIFF sparse files.

Other libraries for reading scientific TIFF files from Python:

* `Python-bioformats <https://github.com/CellProfiler/python-bioformats>`_
* `Imread <https://github.com/luispedro/imread>`_
* `GDAL <https://github.com/OSGeo/gdal/tree/master/gdal/swig/python>`_
* `OpenSlide-python <https://github.com/openslide/openslide-python>`_
* `Slideio <https://gitlab.com/bioslide/slideio>`_
* `PyLibTiff <https://github.com/pearu/pylibtiff>`_
* `SimpleITK <https://github.com/SimpleITK/SimpleITK>`_
* `PyLSM <https://launchpad.net/pylsm>`_
* `PyMca.TiffIO.py <https://github.com/vasole/pymca>`_ (same as fabio.TiffIO)
* `BioImageXD.Readers <http://www.bioimagexd.net/>`_
* `CellCognition <https://cellcognition-project.org/>`_
* `pymimage <https://github.com/ardoi/pymimage>`_
* `pytiff <https://github.com/FZJ-INM1-BDA/pytiff>`_
* `ScanImageTiffReaderPython
  <https://gitlab.com/vidriotech/scanimagetiffreader-python>`_
* `bigtiff <https://pypi.org/project/bigtiff>`_
* `Large Image <https://github.com/girder/large_image>`_

Some libraries are using tifffile to write OME-TIFF files:

* `Zeiss Apeer OME-TIFF library
  <https://github.com/apeer-micro/apeer-ometiff-library>`_
* `Allen Institute for Cell Science imageio
  <https://pypi.org/project/aicsimageio>`_
* `xtiff <https://github.com/BodenmillerGroup/xtiff>`_

Other tools for inspecting and manipulating TIFF files:

* `tifftools <https://github.com/DigitalSlideArchive/tifftools>`_
* `Tyf <https://github.com/Moustikitos/tyf>`_

References
----------
* TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
  https://www.adobe.io/open/standards/TIFF.html
* TIFF File Format FAQ. https://www.awaresystems.be/imaging/tiff/faq.html
* The BigTIFF File Format.
  https://www.awaresystems.be/imaging/tiff/bigtiff.html
* MetaMorph Stack (STK) Image File Format.
  http://mdc.custhelp.com/app/answers/detail/a_id/18862
* Image File Format Description LSM 5/7 Release 6.0 (ZEN 2010).
  Carl Zeiss MicroImaging GmbH. BioSciences. May 10, 2011
* The OME-TIFF format.
  https://docs.openmicroscopy.org/ome-model/latest/
* UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
  http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf
* Micro-Manager File Formats.
  https://micro-manager.org/wiki/Micro-Manager_File_Formats
* ScanImage BigTiff Specification - ScanImage 2019.
  http://scanimage.vidriotechnologies.com/display/SI2019/
  ScanImage+BigTiff+Specification
* ZIF, the Zoomable Image File format. http://zif.photo/
* GeoTIFF File Format https://gdal.org/drivers/raster/gtiff.html
* Cloud optimized GeoTIFF.
  https://github.com/cogeotiff/cog-spec/blob/master/spec.md
* Tags for TIFF and Related Specifications. Digital Preservation.
  https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
* CIPA DC-008-2016: Exchangeable image file format for digital still cameras:
  Exif Version 2.31.
  http://www.cipa.jp/std/documents/e/DC-008-Translation-2016-E.pdf
* The EER (Electron Event Representation) file format.
  https://github.com/fei-company/EerReaderLib
* Digital Negative (DNG) Specification. Version 1.5.0.0, June 2012.
  https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/
  dng_spec_1.5.0.0.pdf
* Roche Digital Pathology. BIF image file format for digital pathology.
  https://diagnostics.roche.com/content/dam/diagnostics/Blueprint/en/pdf/rmd/
  Roche-Digital-Pathology-BIF-Whitepaper.pdf

Examples
--------
Write a numpy array to a single-page RGB TIFF file:

>>> data = numpy.random.randint(0, 255, (256, 256, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb')

Read the image from the TIFF file as numpy array:

>>> image = imread('temp.tif')
>>> image.shape
(256, 256, 3)

Write a 3D numpy array to a multi-page, 16-bit grayscale TIFF file:

>>> data = numpy.random.randint(0, 2**12, (64, 301, 219), 'uint16')
>>> imwrite('temp.tif', data, photometric='minisblack')

Read the whole image stack from the TIFF file as numpy array:

>>> image_stack = imread('temp.tif')
>>> image_stack.shape
(64, 301, 219)
>>> image_stack.dtype
dtype('uint16')

Read the image from the first page in the TIFF file as numpy array:

>>> image = imread('temp.tif', key=0)
>>> image.shape
(301, 219)

Read images from a selected range of pages:

>>> images = imread('temp.tif', key=range(4, 40, 2))
>>> images.shape
(18, 301, 219)

Iterate over all pages in the TIFF file and successively read images:

>>> with TiffFile('temp.tif') as tif:
...     for page in tif.pages:
...         image = page.asarray()

Get information about the image stack in the TIFF file without reading
the image data:

>>> tif = TiffFile('temp.tif')
>>> len(tif.pages)  # number of pages in the file
64
>>> page = tif.pages[0]  # get shape and dtype of the image in the first page
>>> page.shape
(301, 219)
>>> page.dtype
dtype('uint16')
>>> page.axes
'YX'
>>> series = tif.series[0]  # get shape and dtype of the first image series
>>> series.shape
(64, 301, 219)
>>> series.dtype
dtype('uint16')
>>> series.axes
'QYX'
>>> tif.close()

Inspect the "XResolution" tag from the first page in the TIFF file:

>>> with TiffFile('temp.tif') as tif:
...     tag = tif.pages[0].tags['XResolution']
>>> tag.value
(1, 1)
>>> tag.name
'XResolution'
>>> tag.code
282
>>> tag.count
1
>>> tag.dtype
<DATATYPES.RATIONAL: 5>

Iterate over all tags in the TIFF file:

>>> with TiffFile('temp.tif') as tif:
...     for page in tif.pages:
...         for tag in page.tags:
...             tag_name, tag_value = tag.name, tag.value

Overwrite the value of an existing tag, e.g. XResolution:

>>> with TiffFile('temp.tif', mode='r+b') as tif:
...     _ = tif.pages[0].tags['XResolution'].overwrite((96000, 1000))

Write a floating-point ndarray and metadata using BigTIFF format, tiling,
compression, and planar storage:

>>> data = numpy.random.rand(2, 5, 3, 301, 219).astype('float32')
>>> imwrite('temp.tif', data, bigtiff=True, photometric='minisblack',
...         compression='zlib', planarconfig='separate', tile=(32, 32),
...         metadata={'axes': 'TZCYX'})

Write a 10 fps time series of volumes with xyz voxel size 2.6755x2.6755x3.9474
micron^3 to an ImageJ hyperstack formatted TIFF file:

>>> volume = numpy.random.randn(6, 57, 256, 256).astype('float32')
>>> imwrite('temp.tif', volume, imagej=True, resolution=(1./2.6755, 1./2.6755),
...         metadata={'spacing': 3.947368, 'unit': 'um', 'finterval': 1/10,
...                   'axes': 'TZYX'})

Read the volume and metadata from the ImageJ file:

>>> with TiffFile('temp.tif') as tif:
...     volume = tif.asarray()
...     axes = tif.series[0].axes
...     imagej_metadata = tif.imagej_metadata
>>> volume.shape
(6, 57, 256, 256)
>>> axes
'TZYX'
>>> imagej_metadata['slices']
57
>>> imagej_metadata['frames']
6

Create a TIFF file containing an empty image and write to the memory-mapped
numpy array:

>>> memmap_image = memmap(
...     'temp.tif', shape=(256, 256, 3), dtype='float32', photometric='rgb'
... )
>>> type(memmap_image)
<class 'numpy.memmap'>
>>> memmap_image[255, 255, 1] = 1.0
>>> memmap_image.flush()
>>> del memmap_image

Memory-map and read contiguous image data in the TIFF file:

>>> memmap_image = memmap('temp.tif')
>>> memmap_image.shape
(256, 256, 3)
>>> memmap_image[255, 255, 1]
1.0
>>> del memmap_image

Write two numpy arrays to a multi-series TIFF file:

>>> series0 = numpy.random.randint(0, 255, (32, 32, 3), 'uint8')
>>> series1 = numpy.random.randint(0, 1023, (4, 256, 256), 'uint16')
>>> with TiffWriter('temp.tif') as tif:
...     tif.write(series0, photometric='rgb')
...     tif.write(series1, photometric='minisblack')

Read the second image series from the TIFF file:

>>> series1 = imread('temp.tif', series=1)
>>> series1.shape
(4, 256, 256)

Successively write the frames of one contiguous series to a TIFF file:

>>> data = numpy.random.randint(0, 255, (30, 301, 219), 'uint8')
>>> with TiffWriter('temp.tif') as tif:
...     for frame in data:
...         tif.write(frame, contiguous=True)

Append an image series to the existing TIFF file:

>>> data = numpy.random.randint(0, 255, (301, 219, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb', append=True)

Create a TIFF file from a generator of tiles:

>>> data = numpy.random.randint(0, 2**12, (31, 33, 3), 'uint16')
>>> def tiles(data, tileshape):
...     for y in range(0, data.shape[0], tileshape[0]):
...         for x in range(0, data.shape[1], tileshape[1]):
...             yield data[y : y + tileshape[0], x : x + tileshape[1]]
>>> imwrite('temp.tif', tiles(data, (16, 16)), tile=(16, 16),
...         shape=data.shape, dtype=data.dtype, photometric='rgb')

Write two numpy arrays to a multi-series OME-TIFF file:

>>> series0 = numpy.random.randint(0, 255, (32, 32, 3), 'uint8')
>>> series1 = numpy.random.randint(0, 1023, (4, 256, 256), 'uint16')
>>> with TiffWriter('temp.ome.tif') as tif:
...     tif.write(series0, photometric='rgb')
...     tif.write(series1, photometric='minisblack',
...               metadata={'axes': 'ZYX', 'SignificantBits': 10,
...                         'Plane': {'PositionZ': [0.0, 1.0, 2.0, 3.0]}})

Write a tiled, multi-resolution, pyramidal, OME-TIFF file using
JPEG compression. Sub-resolution images are written to SubIFDs:

>>> data = numpy.arange(1024*1024*3, dtype='uint8').reshape((1024, 1024, 3))
>>> with TiffWriter('temp.ome.tif', bigtiff=True) as tif:
...     options = dict(tile=(256, 256), photometric='rgb', compression='jpeg')
...     tif.write(data, subifds=2, **options)
...     # save pyramid levels to the two subifds
...     # in production use resampling to generate sub-resolutions
...     tif.write(data[::2, ::2], subfiletype=1, **options)
...     tif.write(data[::4, ::4], subfiletype=1, **options)

Access the image levels in the pyramidal OME-TIFF file:

>>> baseimage = imread('temp.ome.tif')
>>> second_level = imread('temp.ome.tif', series=0, level=1)
>>> with TiffFile('temp.ome.tif') as tif:
...     baseimage = tif.series[0].asarray()
...     second_level = tif.series[0].levels[1].asarray()

Iterate over and decode single JPEG compressed tiles in the TIFF file:

>>> with TiffFile('temp.ome.tif') as tif:
...     fh = tif.filehandle
...     for page in tif.pages:
...         for index, (offset, bytecount) in enumerate(
...             zip(page.dataoffsets, page.databytecounts)
...         ):
...             _ = fh.seek(offset)
...             data = fh.read(bytecount)
...             tile, indices, shape = page.decode(
...                 data, index, jpegtables=page.jpegtables
...             )

Use zarr to read parts of the tiled, pyramidal images in the TIFF file:

>>> import zarr
>>> store = imread('temp.ome.tif', aszarr=True)
>>> z = zarr.open(store, mode='r')
>>> z
<zarr.hierarchy.Group '/' read-only>
>>> z[0]  # base layer
<zarr.core.Array '/0' (1024, 1024, 3) uint8 read-only>
>>> z[0][256:512, 512:768].shape  # read a tile from the base layer
(256, 256, 3)
>>> store.close()

Read images from a sequence of TIFF files as numpy array:

>>> imwrite('temp_C001T001.tif', numpy.random.rand(64, 64))
>>> imwrite('temp_C001T002.tif', numpy.random.rand(64, 64))
>>> image_sequence = imread(['temp_C001T001.tif', 'temp_C001T002.tif'])
>>> image_sequence.shape
(2, 64, 64)
>>> image_sequence.dtype
dtype('float64')

Read an image stack from a series of TIFF files with a file name pattern
as numpy or zarr arrays:

>>> image_sequence = TiffSequence('temp_C0*.tif', pattern=r'_(C)(\d+)(T)(\d+)')
>>> image_sequence.shape
(1, 2)
>>> image_sequence.axes
'CT'
>>> data = image_sequence.asarray()
>>> data.shape
(1, 2, 64, 64)
>>> with image_sequence.aszarr() as store:
...     zarr.open(store, mode='r')
<zarr.core.Array (1, 2, 64, 64) float64 read-only>
>>> image_sequence.close()

Write the zarr store to a fsspec ReferenceFileSystem in JSON format:

>>> with image_sequence.aszarr() as store:
...     store.write_fsspec('temp.json', url='file://')

Open the fsspec ReferenceFileSystem as a zarr array:

>>> import fsspec
>>> import tifffile.numcodecs
>>> tifffile.numcodecs.register_codec()
>>> mapper = fsspec.get_mapper(
...     'reference://', fo='temp.json', target_protocol='file')
>>> zarr.open(mapper, mode='r')
<zarr.core.Array (1, 2, 64, 64) float64 read-only>

"""

__version__ = '2021.11.2'

__all__ = (
    'OmeXml',
    'OmeXmlError',
    'TIFF',
    'TiffFile',
    'TiffFileError',
    'TiffFrame',
    'TiffPage',
    'TiffPageSeries',
    'TiffReader',
    'TiffSequence',
    'TiffTag',
    'TiffTags',
    'TiffTagRegistry',
    'TiffWriter',
    'ZarrFileSequenceStore',
    'ZarrStore',
    'ZarrTiffStore',
    'imread',
    'imshow',
    'imwrite',
    'lsm2bin',
    'memmap',
    'read_micromanager_metadata',
    'read_scanimage_metadata',
    'tiff2fsspec',
    'tiffcomment',
    # utility classes and functions used by oiffile, czifile, etc.
    'FileCache',
    'FileHandle',
    'FileSequence',
    'Timer',
    'askopenfilename',
    'astype',
    'create_output',
    'enumarg',
    'enumstr',
    'format_size',
    'lazyattr',
    'matlabstr2py',
    'natural_sorted',
    'nullfunc',
    'parse_kwargs',
    'pformat',
    'product',
    'repeat_nd',
    'reshape_axes',
    'reshape_nd',
    'squeeze_axes',
    'stripnull',
    'transpose_axes',
    'update_kwargs',
    'xml2dict',
    # deprecated
    'imsave',
    '_app_show',
)

import binascii
import collections
import datetime
import enum
import glob
import io
import json
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from collections.abc import Iterable, MutableMapping
from concurrent.futures import ThreadPoolExecutor

import numpy

try:
    import imagecodecs
except Exception:
    imagecodecs = None

# delay import of mmap, pprint, fractions, xml, lxml, matplotlib, tkinter,
#   logging, subprocess, multiprocessing, tempfile, zipfile, fnmatch


def imread(files=None, aszarr=False, **kwargs):
    """Return image data from TIFF file(s) as numpy array or zarr storage.

    Refer to the TiffFile and TiffSequence classes and their asarray
    functions for documentation.

    Parameters
    ----------
    files : str, path-like, binary stream, or sequence
        File name, seekable binary stream, glob pattern, or sequence of
        file names.
    aszarr : bool
        If True, return file sequences, series, or single pages as
        zarr storage instead of numpy array (experimental).
    kwargs : dict
        Parameters 'name', 'offset', 'size', and 'is_' flags are passed to
        TiffFile or TiffSequence.imread.
        Parameters 'imread', 'container', 'sort', 'pattern', 'axesorder',
        and 'categories' are passed to TiffSequence.
        Other parameters are passed to the asarray or aszarr functions.
        The first image series in the file is returned if no arguments are
        provided.

    Returns
    -------
    numpy.ndarray or zarr storage
        Image data from the specified pages.
        Zarr storage instances must be closed after use.
        See TiffPage.asarray for operations that are applied (or not)
        to the raw data stored in the file.

    """
    kwargs_file = parse_kwargs(
        kwargs,
        'name',
        'offset',
        'size',
        # private
        '_multifile',
        '_useframes',
        # deprecated, ignored
        # TODO: remove
        'fastij',
        'movie',
        'multifile',
        'multifile_close',
        # is_flags
        *(key for key in kwargs if key[:3] == 'is_'),
    )
    kwargs_seq = parse_kwargs(
        kwargs,
        'imread',
        'container',
        'sort',
        'pattern',
        'axesorder',
        'categories',
    )

    if kwargs.get('pages', None) is not None:
        # TODO: remove
        if kwargs.get('key', None) is not None:
            raise TypeError(
                "the 'pages' and 'key' parameters cannot be used together"
            )
        warnings.warn(
            '<tifffile.imread> '
            "the 'pages' parameter is deprecated since 2017.9.29. "
            "Use the 'key' parameter",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs['key'] = kwargs.pop('pages')

    if kwargs_seq.get('container', None) is None:
        if isinstance(files, str) and ('*' in files or '?' in files):
            files = glob.glob(files)
        if not files:
            raise ValueError('no files found')
        if (
            not hasattr(files, 'seek')
            and not isinstance(files, (str, os.PathLike))
            and len(files) == 1
        ):
            files = files[0]

        if isinstance(files, (str, os.PathLike)) or hasattr(files, 'seek'):
            with TiffFile(files, **kwargs_file) as tif:
                if aszarr:
                    return tif.aszarr(**kwargs)
                return tif.asarray(**kwargs)

    with TiffSequence(files, **kwargs_seq) as imseq:
        if aszarr:
            return imseq.aszarr(**kwargs, **kwargs_file)
        return imseq.asarray(**kwargs, **kwargs_file)


def imwrite(file, data=None, shape=None, dtype=None, **kwargs):
    """Write numpy array to TIFF file.

    Refer to the TiffWriter class and its write function for documentation.

    A BigTIFF file is created if the data's size is larger than 4 GB minus
    32 MB (for metadata), and 'bigtiff' is not specified, and 'imagej' or
    'truncate' are not enabled.

    Parameters
    ----------
    file : str, path-like, or binary stream
        File name or writable binary stream, such as an open file or BytesIO.
    data : array-like
        Input image. The last dimensions are assumed to be image depth,
        length, width, and samples.
        If None, an empty array of the specified shape and dtype is
        saved to file.
        Unless 'byteorder' is specified in 'kwargs', the TIFF file byte order
        is determined from the data's dtype or the dtype argument.
    shape : tuple
        If 'data' is None, shape of an empty array to save to the file.
    dtype : numpy.dtype
        If 'data' is None, datatype of an empty array to save to the file.
    kwargs : dict
        Parameters 'append', 'byteorder', 'bigtiff', 'imagej', and 'ome',
        are passed to TiffWriter().
        Other parameters are passed to TiffWriter.write().

    Returns
    -------
    offset, bytecount : tuple or None
        If the 'returnoffset' argument is True and the image data are written
        contiguously, return offset and bytecount of image data in the file.

    """
    tifargs = parse_kwargs(
        kwargs, 'append', 'bigtiff', 'byteorder', 'imagej', 'ome'
    )
    if data is None:
        dtype = numpy.dtype(dtype)
        datasize = product(shape) * dtype.itemsize
        byteorder = dtype.byteorder
    else:
        try:
            datasize = data.nbytes
            byteorder = data.dtype.byteorder
        except Exception:
            datasize = 0
            byteorder = None
    bigsize = kwargs.pop('bigsize', 2 ** 32 - 2 ** 25)
    if (
        'bigtiff' not in tifargs
        and datasize > bigsize
        and not tifargs.get('imagej', False)
        and not tifargs.get('truncate', False)
        and not kwargs.get('compression', False)
        and not kwargs.get('compress', False)  # TODO: remove deprecated
    ):
        tifargs['bigtiff'] = True
    if 'byteorder' not in tifargs:
        tifargs['byteorder'] = byteorder

    with TiffWriter(file, **tifargs) as tif:
        result = tif.write(data, shape, dtype, **kwargs)
    return result


def memmap(
    filename,
    shape=None,
    dtype=None,
    page=None,
    series=0,
    level=0,
    mode='r+',
    **kwargs,
):
    """Return memory-mapped numpy array stored in TIFF file.

    Memory-mapping requires data stored in native byte order, without tiling,
    compression, predictors, etc.
    If 'shape' and 'dtype' are provided, existing files are overwritten or
    appended to depending on the 'append' parameter.
    Otherwise the image data of a specified page or series in an existing
    file are memory-mapped. By default, the image data of the first
    series are memory-mapped.
    Call flush() to write any changes in the array to the file.
    Raise ValueError if the image data in the file are not memory-mappable.

    Parameters
    ----------
    filename : str or path-like
        Name of the TIFF file which stores the array.
    shape : tuple
        Shape of the empty array.
    dtype : numpy.dtype
        Datatype of the empty array.
    page : int
        Index of the page which image data to memory-map.
    series, level : int
        Index of the page series and pyramid level which image data to
        memory-map.
    mode : {'r+', 'r', 'c'}
        The file open mode. Default is to open existing file for reading and
        writing ('r+').
    kwargs : dict
        Additional parameters passed to imwrite() or TiffFile().

    Returns
    -------
    numpy.memmap
        Image data in TIFF file.

    """
    if shape is not None and dtype is not None:
        # create a new, empty array
        kwargs.update(
            data=None,
            shape=shape,
            dtype=dtype,
            align=TIFF.ALLOCATIONGRANULARITY,
            returnoffset=True,
        )
        result = imwrite(filename, **kwargs)
        if result is None:
            # TODO: fail before creating file or writing data
            raise ValueError('image data are not memory-mappable')
        offset = result[0]
    else:
        # use existing file
        with TiffFile(filename, **kwargs) as tif:
            if page is not None:
                page = tif.pages[page]
                if not page.is_memmappable:
                    raise ValueError('image data are not memory-mappable')
                offset, _ = page.is_contiguous
                shape = page.shape
                dtype = page.dtype
            else:
                series = tif.series[series]
                if series.offset is None:
                    raise ValueError('image data are not memory-mappable')
                shape = series.shape
                dtype = series.dtype
                offset = series.offset
            dtype = tif.byteorder + dtype.char
    return numpy.memmap(filename, dtype, mode, offset, shape, 'C')


class lazyattr:
    """Attribute whose value is computed on first access.

    Lazyattrs are not thread-safe.

    """

    # TODO: replace with functools.cached_property? requires Python >= 3.8
    __slots__ = ('func', '__dict__')

    def __init__(self, func):
        """Initialize instance from decorated function."""
        self.func = func
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__
        self.__name__ = func.__name__
        self.__qualname__ = func.__qualname__
        # self.lock = threading.RLock()

    def __get__(self, instance, owner):
        # with self.lock:
        if instance is None:
            return self
        try:
            value = self.func(instance)
        except AttributeError as exc:
            raise RuntimeError(exc)
        if value is NotImplemented:
            return getattr(super(owner, instance), self.func.__name__)
        setattr(instance, self.func.__name__, value)
        return value


class TiffFileError(Exception):
    """Exception to indicate invalid TIFF structure."""


class TiffWriter:
    """Write numpy arrays to TIFF file.

    TiffWriter's main purpose is saving nD numpy array's as TIFF, not to
    create any possible TIFF format. Specifically, ExifIFD and GPSIFD tags
    are not supported.

    TiffWriter instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    TiffWriter instances are not thread-safe.

    """

    def __init__(
        self,
        file,
        bigtiff=False,
        byteorder=None,
        append=False,
        imagej=False,
        ome=None,
    ):
        """Open TIFF file for writing.

        An empty TIFF file is created if the file does not exist, else the
        file is overwritten with an empty TIFF file unless 'append'
        is true. Use 'bigtiff=True' when creating files larger than 4 GB.

        Parameters
        ----------
        file : path-like, binary stream, or FileHandle
            File name or writable binary stream, such as an open file
            or BytesIO.
        bigtiff : bool (optional)
            If True, the BigTIFF format is used.
        byteorder : {'<', '>', '=', '|'} (optional)
            The endianness of the data in the file.
            By default, this is the system's native byte order.
        append : bool (optional)
            If True and 'file' is an existing standard TIFF file, image data
            and tags are appended to the file. This does not scale well with
            the number of pages already in the file.
            Appending data may corrupt specifically formatted TIFF files
            such as OME-TIFF, LSM, STK, ImageJ, or FluoView.
        imagej : bool (optional)
            If True and not 'ome', write an ImageJ hyperstack compatible file.
            This format can handle data types uint8, uint16, or float32 and
            data shapes up to 6 dimensions in TZCYXS order.
            RGB images (S=3 or S=4) must be uint8.
            ImageJ's default byte order is big-endian but this implementation
            uses the system's native byte order by default.
            ImageJ hyperstacks do not support BigTIFF or compression.
            The ImageJ file format is undocumented.
            When using compression, use ImageJ's Bio-Formats import function.
        ome : bool (optional)
            If True, write an OME-TIFF compatible file. If None (default),
            the value is determined from the file name extension, the value of
            the 'description' parameter in the first call of the write
            function, and the value of 'imagej'.
            Refer to the OME model for restrictions of this format.

        """
        if append:
            # determine if file is an existing TIFF file that can be extended
            try:
                with FileHandle(file, mode='rb', size=0) as fh:
                    pos = fh.tell()
                    try:
                        with TiffFile(fh) as tif:
                            if append != 'force' and not tif.is_appendable:
                                raise ValueError(
                                    'cannot append to file containing metadata'
                                )
                            byteorder = tif.byteorder
                            bigtiff = tif.is_bigtiff
                            self._ifdoffset = tif.pages.next_page_offset
                    finally:
                        fh.seek(pos)
            except (OSError, FileNotFoundError):
                append = False

        if byteorder in (None, '=', '|'):
            byteorder = '<' if sys.byteorder == 'little' else '>'
        elif byteorder not in ('<', '>'):
            raise ValueError(f'invalid byteorder {byteorder}')

        if byteorder == '<':
            self.tiff = TIFF.BIG_LE if bigtiff else TIFF.CLASSIC_LE
        else:
            self.tiff = TIFF.BIG_BE if bigtiff else TIFF.CLASSIC_BE

        self._truncate = False
        self._metadata = None
        self._colormap = None
        self._tags = None
        self._datashape = None  # shape of data in consecutive pages
        self._datadtype = None  # data type
        self._dataoffset = None  # offset to data
        self._databytecounts = None  # byte counts per plane
        self._dataoffsetstag = None  # strip or tile offset tag code
        self._descriptiontag = None  # TiffTag for updating comment
        self._subifds = 0  # number of subifds
        self._subifdslevel = -1  # index of current subifd level
        self._subifdsoffsets = []  # offsets to offsets to subifds
        self._nextifdoffsets = []  # offsets to offset to next ifd
        self._ifdindex = 0  # index of current ifd

        # normalized shape of data in consecutive pages
        # (pages, separate_samples, depth, length, width, contig_samples)
        self._storedshape = None

        if append:
            self._fh = FileHandle(file, mode='r+b', size=0)
            self._fh.seek(0, os.SEEK_END)
        else:
            self._fh = FileHandle(file, mode='wb', size=0)
            self._fh.write({'<': b'II', '>': b'MM'}[byteorder])
            if bigtiff:
                self._fh.write(struct.pack(byteorder + 'HHH', 43, 8, 0))
            else:
                self._fh.write(struct.pack(byteorder + 'H', 42))
            # first IFD
            self._ifdoffset = self._fh.tell()
            self._fh.write(struct.pack(self.tiff.offsetformat, 0))

        self._ome = None if ome is None else bool(ome)
        self._imagej = False if self._ome else bool(imagej)
        if self._imagej:
            self._ome = False

        if imagej and bigtiff:
            warnings.warn(
                f'{self!r} writing nonconformant BigTIFF ImageJ', UserWarning
            )

    @property
    def filehandle(self):
        """Return file handle."""
        return self._fh

    def write(
        self,
        data=None,
        shape=None,
        dtype=None,
        photometric=None,
        planarconfig=None,
        extrasamples=None,
        volumetric=False,
        tile=None,
        contiguous=False,
        truncate=False,
        align=None,
        rowsperstrip=None,
        bitspersample=None,
        compression=None,
        predictor=None,
        subsampling=None,
        jpegtables=None,
        colormap=None,
        description=None,
        datetime=None,
        resolution=None,
        subfiletype=0,
        software=None,
        subifds=None,
        metadata={},
        extratags=(),
        returnoffset=False,
        ijmetadata=None,  # deprecated: use metadata
        compress=None,  # deprecated: use compression
    ):
        """Write numpy ndarray to a series of TIFF pages.

        The ND image data are written to a series of TIFF pages/IFDs.
        By default, metadata in JSON, ImageJ, or OME-XML format are written
        to the ImageDescription tag of the first page to describe the series
        such that the image data can later be read back as a ndarray of same
        shape.

        The data shape's last dimensions are assumed to be image depth,
        length (height), width, and samples.
        If a colormap is provided, the data's dtype must be uint8 or uint16
        and the data values are indices into the last dimension of the
        colormap.
        If 'shape' and 'dtype' are specified instead of 'data', an empty array
        is written. This option cannot be used with compression, predictors,
        packed integers, bilevel images, or multiple tiles.
        If 'shape', 'dtype', and 'tile' are specified, 'data' must be an
        iterable of all tiles in the image.
        If 'shape', 'dtype', and 'data' are specified but not 'tile', 'data'
        must be an iterable of all single planes in the image.
        Image data are written uncompressed in one strip per plane by default.
        Dimensions larger than 2 to 4 (depending on photometric mode, planar
        configuration, and volumetric mode) are flattened and written as
        separate pages.
        If the data size is zero, a single page with shape (0, 0) is written.
        The SampleFormat tag is derived from the data type or dtype.

        A UserWarning is logged if RGB colorspace is auto-detected. Specify
        the 'photometric' parameter to avoid the warning.

        Parameters
        ----------
        data : numpy.ndarray, iterable of numpy.ndarray, or None
            Input image or iterable of tiles or images.
            A copy of the image data is made if 'data' is not a C-contiguous
            numpy array with the same byteorder as the TIFF file.
            Iterables must yield C-contiguous numpy array of TIFF byteorder.
            Iterable tiles must match 'dtype' and the shape specified in
            'tile'. Incomplete tiles are zero-padded.
            Iterable images must match 'dtype' and 'shape[1:]'.
        shape : tuple or None
            Shape of the empty or iterable data to write.
            Use only if 'data' is None or an iterable of tiles or images.
        dtype : numpy.dtype or None
            Datatype of the empty or iterable data to write.
            Use only if 'data' is None or an iterable of tiles or images.
        photometric : {MINISBLACK, MINISWHITE, RGB, PALETTE, SEPARATED, CFA}
            The color space of the image data according to TIFF.PHOTOMETRIC.
            By default, this setting is inferred from the data shape, dtype,
            and the value of colormap. Always specify this parameter to avoid
            ambiguities.
            For CFA images, the CFARepeatPatternDim, CFAPattern, and other
            DNG or TIFF/EP tags must be specified in 'extratags' to produce a
            valid file.
        planarconfig : {CONTIG, SEPARATE}
            Specifies if samples are stored interleaved or in separate planes.
            By default, this setting is inferred from the data shape.
            If this parameter is set, extra samples are used to store grayscale
            images.
            CONTIG: last dimension contains samples.
            SEPARATE: third (or fourth) last dimension contains samples.
        extrasamples : tuple of {UNSPECIFIED, ASSOCALPHA, UNASSALPHA}
            Defines the interpretation of extra components in pixels.
            UNSPECIFIED: no transparency information (default).
            ASSOCALPHA: single, true transparency with pre-multiplied color.
            UNASSALPHA: independent transparency masks.
        volumetric : bool
            If True, the SGI ImageDepth tag is used to write volumetric data
            in one page. The volumetric format is not officially specified,
            and few software can read it. OME and ImageJ formats are not
            compatible with volumetric storage.
        tile : tuple of int
            The shape ([depth,] length, width) of image tiles to write.
            If None (default), image data are written in strips.
            The tile length and width must be a multiple of 16.
            If a tile depth is provided, the SGI ImageDepth and TileDepth
            tags are used to write volumetric data.
            Tiles cannot be used to write contiguous series, except if tile
            matches the data shape.
        contiguous : bool
            If False (default), write data to a new series.
            If True and the data and parameters are compatible with previous
            written ones (same shape, no compression, etc.), the image data
            are stored contiguously after the previous one. In that case,
            'photometric', 'planarconfig', and 'rowsperstrip' are ignored.
            Metadata such as 'description', 'metadata', 'datetime', and
            'extratags' are written to the first page of a contiguous series
            only. Cannot be used with the OME or ImageJ formats.
        truncate : bool
            If True, only write the first page of a contiguous series if
            possible (uncompressed, contiguous, not tiled).
            Other TIFF readers will only be able to read part of the data.
            Cannot be used with the OME or ImageJ formats.
        align : int
            Byte boundary on which to align the image data in the file.
            Default 16. Use mmap.ALLOCATIONGRANULARITY for memory-mapped data.
            Following contiguous writes are not aligned.
        rowsperstrip : int
            The number of rows per strip. By default, strips are ~64 KB if
            compression is enabled, else rowsperstrip is set to the image
            length.
        bitspersample : int
            Number of bits per sample. By default, this is the number of
            bits of the data dtype. Different values for different samples
            are not supported. Unsigned integer data are packed into bytes
            as tightly as possible. Valid values are 1-8 for uint8, 9-16 for
            uint16 and 17-32 for uint32. Cannot be used with compression,
            contiguous series, or empty files.
        compression : str, (str, int), (str, int, dict)
            If None (default), data are written uncompressed.
            If a str, one of TIFF.COMPRESSION, e.g. 'JPEG' or 'ZSTD'.
            If a tuple, the first item is one of TIFF.COMPRESSION, the
            second item is the compression level, and the third item is a dict
            of arguments passed to the compression codec.
            Compression cannot be used to write contiguous series.
            Compressors may require certain data shapes, types or value ranges.
            For example, JPEG requires grayscale or RGB(A), uint8 or 12-bit
            uint16. JPEG compression is experimental. JPEG markers and TIFF
            tags may not match.
            Only a limited set of compression shemes are implemented.
        predictor : bool or TIFF.PREDICTOR
            If True, apply horizontal differencing or floating-point predictor
            before compression. Predictors are disabled for 64-bit integers.
        subsampling : {(1, 1), (2, 1), (2, 2), (4, 1)}
            The horizontal and vertical subsampling factors used for the
            chrominance components of images. The default is (2, 2).
            Currently applies to JPEG compression of RGB images only.
            Images are stored in YCbCr color space.
            Segment widths must be a multiple of 8 times the horizontal factor.
            Segment lengths and rowsperstrip must be a multiple of 8 times the
            vertical factor.
        jpegtables : bytes
            JPEG quantization and/or Huffman tables. Use for copying
            pre-compressed JPEG segments.
        colormap : numpy.ndarray
            RGB color values for the corresponding data value.
            Must be of shape (3, 2**(data.itemsize*8)) and dtype uint16.
        description : str or encoded bytes
            The subject of the image. Must be 7-bit ASCII. Cannot be used with
            the ImageJ or OME formats. Written with the first page of a series
            only.
        datetime : datetime, str, or bool
            Date and time of image creation in '%Y:%m:%d %H:%M:%S' format or
            datetime object. Else if True, the current date and time is used.
            Written with the first page of a series only.
        resolution : (float, float[, str]) or ((int, int), (int, int)[, str])
            X and Y resolutions in pixels per resolution unit as float or
            rational numbers. A third, optional parameter specifies the
            resolution unit, which must be None (default for ImageJ),
            'INCH' (default), or 'CENTIMETER'.
        subfiletype : int
            Bitfield to indicate the kind of data. Set bit 0 if the image
            is a reduced-resolution version of another image. Set bit 1 if
            the image is part of a multi-page image. Set bit 2 if the image
            is transparency mask for another image (photometric must be
            MASK, SamplesPerPixel and BitsPerSample must be 1).
        software : str or bool
            Name of the software used to create the file.
            If None (default), 'tifffile.py'. Must be 7-bit ASCII.
            Written with the first page of a series only.
        subifds : int
            Number of child IFDs. If greater than 0, the following 'subifds'
            number of series are written as child IFDs of the current
            series. The number of IFDs written for each SubIFD level must match
            the number of IFDs written for the current series. All pages
            written to a certain SubIFD level of the current series must have
            the same hash. SubIFDs cannot be used with truncated or ImageJ
            files. SubIFDs in OME-TIFF files must be sub-resolutions of the
            main IFDs.
        metadata : dict
            Additional metadata describing the image data, written along with
            shape information in JSON, OME-XML, or ImageJ formats in
            ImageDescription or IJMetadata tags.
            If None, do not write an ImageDescription tag with shape in JSON
            format.
            If ImageJ format, values for keys 'Info', 'Labels', 'Ranges',
            'LUTs', 'Plot', 'ROI', and 'Overlays' are written in IJMetadata and
            IJMetadataByteCounts tags. Refer to the imagej_metadata_tag
            function for valid values.
            Refer to the OmeXml class for supported keys when writing OME-TIFF.
            Strings must be 7-bit ASCII.
            Written with the first page of a series only.
        extratags : sequence of tuples
            Additional tags as [(code, dtype, count, value, writeonce)].

            code : int
                The TIFF tag Id.
            dtype : int or str
                Data type of items in 'value'. One of TIFF.DATATYPES.
            count : int
                Number of data values. Not used for string or bytes values.
            value : sequence
                'Count' values compatible with 'dtype'.
                Bytes must contain count values of dtype packed as binary data.
            writeonce : bool
                If True, the tag is written to the first page of a series only.

        returnoffset : bool
            If True and the image data in the file are memory-mappable, return
            the offset and number of bytes of the image data in the file.

        Returns
        -------
        offset, bytecount : tuple or None
            If 'returnoffset' is true and the image data in the file are
            memory-mappable, return the offset and number of bytes of the
            image data in the file.

        """
        # TODO: refactor this function
        fh = self._fh
        byteorder = self.tiff.byteorder

        if data is None:
            # empty
            dataiter = None
            datashape = tuple(shape)
            datadtype = numpy.dtype(dtype).newbyteorder(byteorder)
            datadtypechar = datadtype.char
        elif (
            shape is not None
            and dtype is not None
            and hasattr(data, '__iter__')
        ):
            # iterable pages or tiles
            if hasattr(data, '__next__'):
                dataiter = data
            else:
                dataiter = iter(data)
            datashape = tuple(shape)
            datadtype = numpy.dtype(dtype).newbyteorder(byteorder)
            datadtypechar = datadtype.char
        elif hasattr(data, '__next__'):
            # generator
            raise TypeError(
                "generators require 'shape' and 'dtype' parameters"
            )
        else:
            # whole image data
            # must be C-contiguous numpy array of TIFF byteorder
            if hasattr(data, 'dtype'):
                data = numpy.asarray(data, byteorder + data.dtype.char, 'C')
            else:
                datadtype = numpy.dtype(dtype).newbyteorder(byteorder)
                data = numpy.asarray(data, datadtype, 'C')

            if dtype is not None and dtype != data.dtype:
                warnings.warn(
                    f"{self!r} ignoring 'dtype' argument", UserWarning
                )
            if shape is not None and shape != data.shape:
                warnings.warn(
                    f"{self!r} ignoring 'shape' argument", UserWarning
                )
            dataiter = None
            datashape = data.shape
            datadtype = data.dtype
            datadtypechar = data.dtype.char

        returnoffset = returnoffset and datadtype.isnative

        if compression is None and compress is not None:
            # TODO: remove
            warnings.warn(
                "<tifffile.TiffWriter.write> the 'compress' parameter is "
                "deprecated since 2020.9.30. Use the 'compression' parameter",
                DeprecationWarning,
                stacklevel=2,
            )
            if isinstance(compress, (int, numpy.integer)) and compress > 0:
                # ADOBE_DEFLATE
                compression = 8, int(compress)
                if not 0 < compress <= 9:
                    raise ValueError(f'invalid compression level {compress}')
            else:
                compression = compress
            del compress

        bilevel = datadtypechar == '?'
        if bilevel:
            index = -1 if datashape[-1] > 1 else -2
            datasize = product(datashape[:index])
            if datashape[index] % 8:
                datasize *= datashape[index] // 8 + 1
            else:
                datasize *= datashape[index] // 8
        else:
            datasize = product(datashape) * datadtype.itemsize

        if datasize == 0:
            data = None
            compression = False
            bitspersample = None
            if metadata is not None:
                truncate = True
        elif compression in (None, 0, 1, 'NONE', 'none'):
            compression = False

        inputshape = datashape

        packints = (
            bitspersample is not None
            and bitspersample != datadtype.itemsize * 8
        )

        # just append contiguous data if possible
        if self._datashape is not None:
            if (
                not contiguous
                or self._datashape[1:] != datashape
                or self._datadtype != datadtype
                or not numpy.array_equal(colormap, self._colormap)
            ):
                # incompatible shape, dtype, or colormap
                self._write_remaining_pages()

                if self._imagej:
                    raise ValueError(
                        'the ImageJ format does not support '
                        'non-contiguous series'
                    )
                elif self._ome:
                    if self._subifdslevel < 0:
                        # add image to OME-XML
                        self._ome.addimage(
                            self._datadtype,
                            self._datashape[
                                0 if self._datashape[0] != 1 else 1 :
                            ],
                            self._storedshape,
                            **self._metadata,
                        )
                elif metadata is not None:
                    self._write_image_description()
                    # description might have been appended to file
                    fh.seek(0, os.SEEK_END)

                if self._subifds:
                    if self._truncate or truncate:
                        raise ValueError(
                            'SubIFDs cannot be used with truncated series'
                        )
                    self._subifdslevel += 1
                    if self._subifdslevel == self._subifds:
                        # done with writing SubIFDs
                        self._nextifdoffsets = []
                        self._subifdsoffsets = []
                        self._subifdslevel = -1
                        self._subifds = 0
                        self._ifdindex = 0
                    elif subifds:
                        raise ValueError(
                            'SubIFDs in SubIFDs are not supported'
                        )

                self._datashape = None
                self._colormap = None

            elif compression or packints or tile:
                raise ValueError(
                    'contiguous cannot be used with compression, tiles, etc.'
                )

            else:
                # consecutive mode
                # write contiguous data, write IFDs/tags later
                self._datashape = (self._datashape[0] + 1,) + datashape
                offset = fh.tell()
                if data is None:
                    fh.write_empty(datasize)
                else:
                    fh.write_array(data)
                if returnoffset:
                    return offset, datasize
                return None

        if self._ome is None:
            if description is None:
                self._ome = '.ome.tif' in fh.name
            else:
                self._ome = False
        self._truncate = False if self._ome else bool(truncate)

        if self._truncate and (compression or packints or tile):
            raise ValueError(
                'truncate cannot be used with compression, packints, or tiles'
            )

        if datasize == 0:
            # write single placeholder TiffPage for arrays with size=0
            datashape = (0, 0)
            warnings.warn(
                f'{self!r} writing zero size array to nonconformant TIFF',
                UserWarning,
            )
            # TODO: reconsider this
            # raise ValueError('cannot save zero size array')

        valueformat = f'{self.tiff.offsetsize}s'
        tagnoformat = self.tiff.tagnoformat
        offsetformat = self.tiff.offsetformat
        offsetsize = self.tiff.offsetsize
        tagsize = self.tiff.tagsize

        MINISBLACK = TIFF.PHOTOMETRIC.MINISBLACK
        MINISWHITE = TIFF.PHOTOMETRIC.MINISWHITE
        RGB = TIFF.PHOTOMETRIC.RGB
        YCBCR = TIFF.PHOTOMETRIC.YCBCR
        PALETTE = TIFF.PHOTOMETRIC.PALETTE
        CONTIG = TIFF.PLANARCONFIG.CONTIG
        SEPARATE = TIFF.PLANARCONFIG.SEPARATE

        # parse input
        if photometric is not None:
            photometric = enumarg(TIFF.PHOTOMETRIC, photometric)
        if planarconfig:
            planarconfig = enumarg(TIFF.PLANARCONFIG, planarconfig)
        if predictor:
            if not isinstance(predictor, bool):
                predictor = bool(enumarg(TIFF.PREDICTOR, predictor))
        if extrasamples is None:
            extrasamples_ = None
        else:
            extrasamples_ = tuple(
                enumarg(TIFF.EXTRASAMPLE, es) for es in sequence(extrasamples)
            )

        if compression:
            if isinstance(compression, (tuple, list)):
                if len(compression) == 2:
                    compressionargs = {'level': compression[1]}
                else:
                    compressionargs = dict(compression[2])
                    if compression[1] is not None:
                        compressionargs['level'] = compression[1]
                compression = compression[0]
            else:
                compressionargs = {}
            if isinstance(compression, str):
                compression = compression.upper()
                if compression == 'ZLIB':
                    compression = 8  # ADOBE_DEFLATE
            compressiontag = enumarg(TIFF.COMPRESSION, compression)
            compression = compressiontag > 1
        else:
            compression = False
            compressiontag = 1

        if not compression:
            compressionargs = {}
            predictor = False
            predictortag = 1

        if predictor:
            if compressiontag in (
                7,
                33003,
                33004,
                33005,
                33007,
                34712,
                34892,
                34933,
                34934,
                50001,
                50002,
            ):
                # disable predictor for JPEG, JPEG2000, WEBP, PNG, JPEGXR
                predictor = False
            elif datadtype.kind in 'iu':
                if datadtype.itemsize > 4:
                    predictor = False  # disable predictor for 64 bit
                else:
                    predictortag = 2
                    predictor = TIFF.PREDICTORS[2]
            elif datadtype.kind == 'f':
                predictortag = 3
                predictor = TIFF.PREDICTORS[3]
            else:
                raise ValueError(f'cannot apply predictor to {datadtype}')

        if self._ome:
            if description is not None:
                warnings.warn(
                    f'{self!r} not writing description to OME-TIFF',
                    UserWarning,
                )
                description = None
            if not isinstance(self._ome, OmeXml):
                self._ome = OmeXml(**metadata)
            if volumetric or (tile and len(tile) > 2):
                raise ValueError('OME-TIFF does not support ImageDepth')
            volumetric = False

        elif self._imagej:
            # if tile is not None or predictor or compression:
            #     warnings.warn(
            #         f'{self!r} the ImageJ format does not support '
            #         'tiles, predictors, compression'
            #     )
            if description is not None:
                warnings.warn(
                    f'{self!r} not writing description to ImageJ file',
                    UserWarning,
                )
                description = None
            if datadtypechar not in 'BHhf':
                raise ValueError(
                    'the ImageJ format does not support data type '
                    f'{datadtypechar!r}'
                )
            if volumetric or (tile and len(tile) > 2):
                raise ValueError(
                    'the ImageJ format does not support ImageDepth'
                )
            volumetric = False
            ijrgb = photometric == RGB if photometric else None
            if datadtypechar not in 'B':
                ijrgb = False
            ijshape = imagej_shape(
                datashape, ijrgb, metadata.get('axes', None)
            )
            if ijshape[-1] in (3, 4):
                photometric = RGB
                if datadtypechar not in 'B':
                    raise ValueError(
                        'the ImageJ format does not support '
                        f'data type {datadtypechar!r} for RGB'
                    )
            elif photometric is None:
                photometric = MINISBLACK
                planarconfig = None
            if planarconfig == SEPARATE:
                raise ValueError(
                    'the ImageJ format does not support planar images'
                )
            planarconfig = CONTIG if ijrgb else None

        # verify colormap and indices
        if colormap is not None:
            if datadtypechar not in 'BH':
                raise ValueError('invalid data dtype for palette mode')
            colormap = numpy.asarray(colormap, dtype=byteorder + 'H')
            if colormap.shape != (3, 2 ** (datadtype.itemsize * 8)):
                raise ValueError('invalid color map shape')
            self._colormap = colormap

        if tile:
            # verify tile shape
            tile = tuple(int(i) for i in tile[:3])
            if (
                len(tile) < 2
                or tile[-1] % 16
                or tile[-2] % 16
                or any(i < 1 for i in tile)
            ):
                raise ValueError('invalid tile shape')
            if volumetric and len(tile) == 2:
                tile = (1,) + tile
            volumetric = len(tile) == 3
        else:
            tile = ()
            volumetric = bool(volumetric)

        # normalize data shape to 5D or 6D, depending on volume:
        #   (pages, separate_samples, [depth,] length, width, contig_samples)
        storedshape = reshape_nd(
            datashape, TIFF.PHOTOMETRIC_SAMPLES.get(photometric, 2)
        )
        shape = storedshape
        ndim = len(storedshape)

        samplesperpixel = 1
        extrasamples = 0

        if volumetric and ndim < 3:
            volumetric = False

        if colormap is not None:
            photometric = PALETTE
            planarconfig = None

        if photometric is None:
            deprecate = False
            photometric = MINISBLACK
            if bilevel:
                photometric = MINISWHITE
            elif planarconfig == CONTIG:
                if ndim > 2 and shape[-1] in (3, 4):
                    photometric = RGB
                    deprecate = datadtypechar not in 'BH'
            elif planarconfig == SEPARATE:
                if volumetric and ndim > 3 and shape[-4] in (3, 4):
                    photometric = RGB
                    deprecate = True
                elif ndim > 2 and shape[-3] in (3, 4):
                    photometric = RGB
                    deprecate = True
            elif ndim > 2 and shape[-1] in (3, 4):
                photometric = RGB
                planarconfig = CONTIG
                deprecate = datadtypechar not in 'BH'
            elif self._imagej or self._ome:
                photometric = MINISBLACK
                planarconfig = None
            elif volumetric and ndim > 3 and shape[-4] in (3, 4):
                photometric = RGB
                planarconfig = SEPARATE
                deprecate = True
            elif ndim > 2 and shape[-3] in (3, 4):
                photometric = RGB
                planarconfig = SEPARATE
                deprecate = True

            if deprecate:
                if planarconfig == CONTIG:
                    msg = 'contiguous samples', "parameter is"
                else:
                    msg = (
                        'separate component planes',
                        "and 'planarconfig' parameters are",
                    )
                warnings.warn(
                    f"<tifffile.TiffWriter.write> data with shape {datashape} "
                    f"and dtype '{datadtype}' are stored as RGB with {msg[0]}."
                    " Future versions will store such data as MINISBLACK in "
                    "separate pages by default unless the 'photometric' "
                    f"{msg[1]} specified.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                del msg
            del deprecate

        del datashape
        photometricsamples = TIFF.PHOTOMETRIC_SAMPLES[photometric]

        if planarconfig and len(shape) <= (3 if volumetric else 2):
            # TODO: raise error?
            planarconfig = None
            if photometricsamples > 1:
                photometric = MINISBLACK

        if photometricsamples > 1:
            if len(shape) < 3:
                raise ValueError(f'not a {photometric!r} image')
            if len(shape) < 4:
                volumetric = False
            if planarconfig is None:
                if photometric == RGB:
                    samples_ = (photometricsamples, 4)  # allow common alpha
                else:
                    samples_ = (photometricsamples,)
                if shape[-1] in samples_:
                    planarconfig = CONTIG
                elif shape[-4 if volumetric else -3] in samples_:
                    planarconfig = SEPARATE
                elif shape[-1] > shape[-4 if volumetric else -3]:
                    # TODO: deprecated this?
                    planarconfig = SEPARATE
                else:
                    planarconfig = CONTIG
            if planarconfig == CONTIG:
                storedshape = (-1, 1) + shape[(-4 if volumetric else -3) :]
                samplesperpixel = storedshape[-1]
            else:
                storedshape = (
                    (-1,) + shape[(-4 if volumetric else -3) :] + (1,)
                )
                samplesperpixel = storedshape[1]
            if samplesperpixel > photometricsamples:
                extrasamples = samplesperpixel - photometricsamples

        elif photometric == TIFF.PHOTOMETRIC.CFA:
            if len(shape) != 2:
                raise ValueError('invalid CFA image')
            volumetric = False
            planarconfig = None
            storedshape = (-1, 1) + shape[-2:] + (1,)
            # if all(et[0] != 50706 for et in extratags):
            #     raise ValueError('must specify DNG tags for CFA image')

        elif planarconfig and len(shape) > (3 if volumetric else 2):
            if planarconfig == CONTIG:
                storedshape = (-1, 1) + shape[(-4 if volumetric else -3) :]
                samplesperpixel = storedshape[-1]
            else:
                storedshape = (
                    (-1,) + shape[(-4 if volumetric else -3) :] + (1,)
                )
                samplesperpixel = storedshape[1]
            extrasamples = samplesperpixel - 1

        else:
            planarconfig = None
            while len(shape) > 2 and shape[-1] == 1:
                shape = shape[:-1]  # remove trailing 1s
            if len(shape) < 3:
                volumetric = False
            if extrasamples_ is None:
                storedshape = (
                    (-1, 1) + shape[(-3 if volumetric else -2) :] + (1,)
                )
            else:
                storedshape = (-1, 1) + shape[(-4 if volumetric else -3) :]
                samplesperpixel = storedshape[-1]
                extrasamples = samplesperpixel - 1

        if subfiletype & 0b100:
            # FILETYPE_MASK
            if not (
                bilevel and samplesperpixel == 1 and photometric in (0, 1, 4)
            ):
                raise ValueError('invalid SubfileType MASK')
            photometric = TIFF.PHOTOMETRIC.MASK

        packints = False
        if bilevel:
            if bitspersample is not None and bitspersample != 1:
                raise ValueError(
                    f'bitspersample {bitspersample} must be 1 for bilevel'
                )
            bitspersample = 1
        elif compressiontag == 7 and datadtype == 'uint16':
            if bitspersample is not None and bitspersample != 12:
                raise ValueError(
                    f'bitspersample {bitspersample} must be 12 for JPEG '
                    'compressed uint16'
                )
            bitspersample = 12  # use 12-bit JPEG compression
        elif bitspersample is None:
            bitspersample = datadtype.itemsize * 8
        elif (
            datadtype.kind != 'u' or datadtype.itemsize > 4
        ) and bitspersample != datadtype.itemsize * 8:
            raise ValueError(
                f'bitspersample {bitspersample} does not match '
                f'dtype {datadtype}'
            )
        elif not (
            bitspersample > {1: 0, 2: 8, 4: 16}[datadtype.itemsize]
            and bitspersample <= datadtype.itemsize * 8
        ):
            raise ValueError(
                f'bitspersample {bitspersample} out of range of '
                f'dtype {datadtype}'
            )
        elif compression:
            if bitspersample != datadtype.itemsize * 8:
                raise ValueError(
                    f'bitspersample {bitspersample} cannot be used with '
                    'compression'
                )
        elif bitspersample != datadtype.itemsize * 8:
            packints = True

        # normalize storedshape to 6D
        if len(storedshape) not in (5, 6):
            raise RuntimeError(
                f'length of storedshape {len(storedshape)} not in (5, 6)'
            )
        if len(storedshape) == 5:
            storedshape = storedshape[:2] + (1,) + storedshape[2:]
        if storedshape[0] == -1:
            s0 = product(storedshape[1:])
            s0 = 1 if s0 == 0 else product(inputshape) // s0
            storedshape = (s0,) + storedshape[1:]
        try:
            data = data.reshape(storedshape)
        except AttributeError:
            pass  # data is None or iterator

        if photometric == PALETTE:
            if (
                samplesperpixel != 1
                or extrasamples
                or storedshape[1] != 1
                or storedshape[-1] != 1
            ):
                raise ValueError('invalid data shape for palette mode')

        if photometric == RGB and samplesperpixel == 2:
            raise ValueError('not a RGB image (samplesperpixel=2)')

        tags = []  # list of (code, ifdentry, ifdvalue, writeonce)

        if tile:
            tagbytecounts = 325  # TileByteCounts
            tagoffsets = 324  # TileOffsets
        else:
            tagbytecounts = 279  # StripByteCounts
            tagoffsets = 273  # StripOffsets
        self._dataoffsetstag = tagoffsets

        def pack(fmt, *val):
            if fmt[0] not in '<>':
                fmt = byteorder + fmt
            return struct.pack(fmt, *val)

        def addtag(code, dtype, count, value, writeonce=False):
            # compute ifdentry & ifdvalue bytes from code, dtype, count, value
            # append (code, ifdentry, ifdvalue, writeonce) to tags list
            if not isinstance(code, int):
                code = TIFF.TAGS[code]
            try:
                datatype = dtype
                dataformat = TIFF.DATA_FORMATS[datatype][-1]
            except KeyError as exc:
                try:
                    dataformat = dtype
                    if dataformat[0] in '<>':
                        dataformat = dataformat[1:]
                    datatype = TIFF.DATA_DTYPES[dataformat]
                except (KeyError, TypeError):
                    raise ValueError(f'unknown dtype {dtype}') from exc
            del dtype

            rawcount = count
            if datatype == 2:
                # string
                if isinstance(value, str):
                    # enforce 7-bit ASCII on Unicode strings
                    try:
                        value = value.encode('ascii')
                    except UnicodeEncodeError as exc:
                        raise ValueError(
                            'TIFF strings must be 7-bit ASCII'
                        ) from exc
                elif not isinstance(value, bytes):
                    raise ValueError('TIFF strings must be 7-bit ASCII')
                if len(value) == 0 or value[-1] != b'\x00':
                    value += b'\x00'
                count = len(value)
                if code == 270:
                    self._descriptiontag = TiffTag(
                        self, 0, 270, 2, count, None, 0
                    )
                    rawcount = value.find(b'\x00\x00')
                    if rawcount < 0:
                        rawcount = count
                    else:
                        # length of string without buffer
                        rawcount = max(offsetsize + 1, rawcount + 1)
                        rawcount = min(count, rawcount)
                else:
                    rawcount = count
                value = (value,)

            elif isinstance(value, bytes):
                # packed binary data
                itemsize = struct.calcsize(dataformat)
                if len(value) % itemsize:
                    raise ValueError('invalid packed binary data')
                count = len(value) // itemsize
                rawcount = count

            if datatype in (5, 10):  # rational
                count *= 2
                dataformat = dataformat[-1]

            ifdentry = [
                pack('HH', code, datatype),
                pack(offsetformat, rawcount),
            ]

            ifdvalue = None
            if struct.calcsize(dataformat) * count <= offsetsize:
                # value(s) can be written directly
                if isinstance(value, bytes):
                    ifdentry.append(pack(valueformat, value))
                elif count == 1:
                    if isinstance(value, (tuple, list, numpy.ndarray)):
                        value = value[0]
                    ifdentry.append(pack(valueformat, pack(dataformat, value)))
                else:
                    ifdentry.append(
                        pack(valueformat, pack(f'{count}{dataformat}', *value))
                    )
            else:
                # use offset to value(s)
                ifdentry.append(pack(offsetformat, 0))
                if isinstance(value, bytes):
                    ifdvalue = value
                elif isinstance(value, numpy.ndarray):
                    if value.size != count:
                        raise RuntimeError('value.size != count')
                    if value.dtype.char != dataformat:
                        raise RuntimeError('value.dtype.char != dtype')
                    ifdvalue = value.tobytes()
                elif isinstance(value, (tuple, list)):
                    ifdvalue = pack(f'{count}{dataformat}', *value)
                else:
                    ifdvalue = pack(dataformat, value)
            tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))

        def rational(arg):
            # return numerator and denominator from float or two integers
            from fractions import Fraction  # delayed import

            try:
                f = Fraction.from_float(arg)
            except TypeError:
                f = Fraction(arg[0], arg[1])
            try:
                numerator, denominator = f.as_integer_ratio()
            except AttributeError:
                # Python 3.7
                f = f.limit_denominator(4294967294)
                numerator, denominator = f.numerator, f.denominator
            if numerator > 4294967295 or denominator > 4294967295:
                s = 4294967295 / max(numerator, denominator)
                numerator = round(numerator * s)
                denominator = round(denominator * s)
            return numerator, denominator

        if description is not None:
            # ImageDescription: user provided description
            addtag(270, 2, 0, description, writeonce=True)

        # write shape and metadata to ImageDescription
        self._metadata = {} if not metadata else metadata.copy()
        if self._ome:
            if len(self._ome.images) == 0:
                description = ''  # rewritten later at end of file
            else:
                description = None
        elif self._imagej:
            if ijmetadata is None:
                ijmetadata = parse_kwargs(
                    self._metadata,
                    'Info',
                    'Labels',
                    'Ranges',
                    'LUTs',
                    'Plot',
                    'ROI',
                    'Overlays',
                    'Properties',
                    'info',
                    'labels',
                    'ranges',
                    'luts',
                    'plot',
                    'roi',
                    'overlays',
                    'prop',
                )
            else:
                # TODO: remove
                warnings.warn(
                    '<tifffile.TiffWriter.write> '
                    "the 'ijmetadata' parameter is deprecated since 2020.5.5. "
                    "Use the 'metadata' parameter",
                    DeprecationWarning,
                    stacklevel=2,
                )
            for t in imagej_metadata_tag(ijmetadata, byteorder):
                addtag(*t)
            description = imagej_description(
                inputshape,
                storedshape[-1] in (3, 4),
                self._colormap is not None,
                **self._metadata,
            )
            description += '\x00' * 64  # add buffer for in-place update
        elif metadata or metadata == {}:
            if self._truncate:
                self._metadata.update(truncated=True)
            description = json_description(inputshape, **self._metadata)
            description += '\x00' * 16  # add buffer for in-place update
        # elif metadata is None and self._truncate:
        #     raise ValueError('cannot truncate without writing metadata')
        else:
            description = None

        if description is not None:
            description = description.encode('ascii')
            addtag(270, 2, 0, description, writeonce=True)
        del description

        if software is None:
            software = 'tifffile.py'
        if software:
            addtag(305, 2, 0, software, writeonce=True)
        if datetime:
            if isinstance(datetime, str):
                if len(datetime) != 19 or datetime[16] != ':':
                    raise ValueError('invalid datetime string')
            else:
                try:
                    datetime = datetime.strftime('%Y:%m:%d %H:%M:%S')
                except AttributeError:
                    datetime = self._now().strftime('%Y:%m:%d %H:%M:%S')
            addtag(306, 2, 0, datetime, writeonce=True)
        addtag(259, 3, 1, compressiontag)  # Compression
        if compressiontag == 34887:
            # LERC without additional compression
            addtag(50674, 4, 2, (4, 0))
        if predictor:
            addtag(317, 3, 1, predictortag)
        addtag(256, 4, 1, storedshape[-2])  # ImageWidth
        addtag(257, 4, 1, storedshape[-3])  # ImageLength
        if tile:
            addtag(322, 4, 1, tile[-1])  # TileWidth
            addtag(323, 4, 1, tile[-2])  # TileLength
        if volumetric:
            addtag(32997, 4, 1, storedshape[-4])  # ImageDepth
            if tile:
                addtag(32998, 4, 1, tile[0])  # TileDepth
        if subfiletype:
            addtag(254, 4, 1, subfiletype)  # NewSubfileType
        if (subifds or self._subifds) and self._subifdslevel < 0:
            if self._subifds:
                subifds = self._subifds
            else:
                try:
                    self._subifds = subifds = int(subifds)
                except TypeError:
                    # allow TiffPage.subifds tuple
                    self._subifds = subifds = len(subifds)
            addtag(330, 18 if offsetsize > 4 else 13, subifds, [0] * subifds)
        if not bilevel and not datadtype.kind == 'u':
            sampleformat = {'u': 1, 'i': 2, 'f': 3, 'c': 6}[datadtype.kind]
            addtag(339, 3, samplesperpixel, (sampleformat,) * samplesperpixel)
        if colormap is not None:
            addtag(320, 3, colormap.size, colormap)
        addtag(277, 3, 1, samplesperpixel)
        if bilevel:
            pass
        elif planarconfig and samplesperpixel > 1:
            addtag(284, 3, 1, planarconfig.value)  # PlanarConfiguration
            addtag(  # BitsPerSample
                258, 3, samplesperpixel, (bitspersample,) * samplesperpixel
            )
        else:
            addtag(258, 3, 1, bitspersample)
        if extrasamples:
            if extrasamples_ is not None:
                if extrasamples != len(extrasamples_):
                    raise ValueError(
                        'wrong number of extrasamples '
                        f'{extrasamples} != {len(extrasamples_)}'
                    )
                addtag(338, 3, extrasamples, extrasamples_)
            elif photometric == RGB and extrasamples == 1:
                # Unassociated alpha channel
                addtag(338, 3, 1, 2)
            else:
                # Unspecified alpha channel
                addtag(338, 3, extrasamples, (0,) * extrasamples)

        if jpegtables is not None:
            addtag(347, 7, len(jpegtables), jpegtables)

        if (
            compressiontag == 7
            and planarconfig == 1
            and photometric in (RGB, YCBCR)
        ):
            # JPEG compression with subsampling
            # TODO: use JPEGTables for multiple tiles or strips
            if subsampling is None:
                subsampling = (2, 2)
            elif subsampling not in ((1, 1), (2, 1), (2, 2), (4, 1)):
                raise ValueError(
                    f'invalid subsampling factors {subsampling!r}'
                )
            maxsampling = max(subsampling) * 8
            if tile and (tile[-1] % maxsampling or tile[-2] % maxsampling):
                raise ValueError(f'tile shape not a multiple of {maxsampling}')
            if extrasamples > 1:
                raise ValueError('JPEG subsampling requires RGB(A) images')
            addtag(530, 3, 2, subsampling)  # YCbCrSubSampling
            # use PhotometricInterpretation YCBCR by default
            outcolorspace = enumarg(
                TIFF.PHOTOMETRIC, compressionargs.get('outcolorspace', 6)
            )
            compressionargs['subsampling'] = subsampling
            compressionargs['colorspace'] = photometric.name
            compressionargs['outcolorspace'] = outcolorspace.name
            addtag(262, 3, 1, outcolorspace)
            # ReferenceBlackWhite is required for YCBCR
            if all(et[0] != 532 for et in extratags):
                addtag(
                    532, 5, 6, (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1)
                )
        else:
            if subsampling not in (None, (1, 1)):
                log_warning(
                    f'{self!r} cannot apply subsampling {subsampling!r}'
                )
            subsampling = None
            maxsampling = 1
            addtag(262, 3, 1, photometric.value)  # PhotometricInterpretation
            if photometric == YCBCR:
                # YCbCrSubSampling and ReferenceBlackWhite
                addtag(530, 3, 2, (1, 1))
                if all(et[0] != 532 for et in extratags):
                    addtag(
                        532,
                        5,
                        6,
                        (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1),
                    )
        if resolution is not None:
            addtag(282, 5, 1, rational(resolution[0]))  # XResolution
            addtag(283, 5, 1, rational(resolution[1]))  # YResolution
            if len(resolution) > 2:
                unit = resolution[2]
                unit = 1 if unit is None else enumarg(TIFF.RESUNIT, unit)
            elif self._imagej:
                unit = 1
            else:
                unit = 2
            addtag(296, 3, 1, unit)  # ResolutionUnit
        elif not self._imagej:
            addtag(282, 5, 1, (1, 1))  # XResolution
            addtag(283, 5, 1, (1, 1))  # YResolution
            addtag(296, 3, 1, 1)  # ResolutionUnit

        def bytecount_format(
            bytecounts, compression=compression, size=offsetsize
        ):
            # return small bytecount format
            if len(bytecounts) == 1:
                return {4: 'I', 8: 'Q'}[size]
            bytecount = bytecounts[0]
            if compression:
                bytecount = bytecount * 10
            if bytecount < 2 ** 16:
                return 'H'
            if bytecount < 2 ** 32:
                return 'I'
            if size == 4:
                return 'I'
            return 'Q'

        # can save data array contiguous
        contiguous = not (compression or packints or bilevel)
        if tile:
            # one chunk per tile per plane
            if len(tile) == 2:
                tiles = (
                    (storedshape[3] + tile[0] - 1) // tile[0],
                    (storedshape[4] + tile[1] - 1) // tile[1],
                )
                contiguous = (
                    contiguous
                    and storedshape[3] == tile[0]
                    and storedshape[4] == tile[1]
                )
            else:
                tiles = (
                    (storedshape[2] + tile[0] - 1) // tile[0],
                    (storedshape[3] + tile[1] - 1) // tile[1],
                    (storedshape[4] + tile[2] - 1) // tile[2],
                )
                contiguous = (
                    contiguous
                    and storedshape[2] == tile[0]
                    and storedshape[3] == tile[1]
                    and storedshape[4] == tile[2]
                )
            numtiles = product(tiles) * storedshape[1]
            databytecounts = [
                product(tile) * storedshape[-1] * datadtype.itemsize
            ] * numtiles
            bytecountformat = bytecount_format(databytecounts)
            addtag(tagbytecounts, bytecountformat, numtiles, databytecounts)
            addtag(tagoffsets, offsetformat, numtiles, [0] * numtiles)
            bytecountformat = bytecountformat * numtiles
            if contiguous or dataiter is not None:
                pass
            else:
                dataiter = iter_tiles(data, tile, tiles)

        elif contiguous and rowsperstrip is None:
            count = storedshape[1] * storedshape[2]
            databytecounts = [
                product(storedshape[3:]) * datadtype.itemsize
            ] * count
            bytecountformat = bytecount_format(databytecounts)
            addtag(tagbytecounts, bytecountformat, count, databytecounts)
            addtag(tagoffsets, offsetformat, count, [0] * count)
            addtag(278, 4, 1, storedshape[-3])  # RowsPerStrip
            bytecountformat = bytecountformat * storedshape[1]
            if contiguous or dataiter is not None:
                pass
            else:
                dataiter = iter_images(data)

        else:
            # use rowsperstrip
            rowsize = product(storedshape[-2:]) * datadtype.itemsize
            if rowsperstrip is None:
                # compress ~64 KB chunks by default
                # TIFF-EP requires <= 64 KB
                if compression:
                    rowsperstrip = 65536 // rowsize
                else:
                    rowsperstrip = storedshape[-3]
            if rowsperstrip < 1:
                rowsperstrip = maxsampling
            elif rowsperstrip > storedshape[-3]:
                rowsperstrip = storedshape[-3]
            elif subsampling and rowsperstrip % maxsampling:
                rowsperstrip = (
                    math.ceil(rowsperstrip / maxsampling) * maxsampling
                )
            addtag(278, 4, 1, rowsperstrip)  # RowsPerStrip

            numstrips1 = (storedshape[-3] + rowsperstrip - 1) // rowsperstrip
            numstrips = numstrips1 * storedshape[1] * storedshape[2]
            # TODO: save bilevel data with rowsperstrip
            stripsize = rowsperstrip * rowsize
            databytecounts = [stripsize] * numstrips
            stripsize -= rowsize * (
                numstrips1 * rowsperstrip - storedshape[-3]
            )
            for i in range(numstrips1 - 1, numstrips, numstrips1):
                databytecounts[i] = stripsize
            bytecountformat = bytecount_format(databytecounts)
            addtag(tagbytecounts, bytecountformat, numstrips, databytecounts)
            addtag(tagoffsets, offsetformat, numstrips, [0] * numstrips)
            bytecountformat = bytecountformat * numstrips

            if contiguous or dataiter is not None:
                pass
            else:
                dataiter = iter_images(data)

        if data is None and not contiguous:
            raise ValueError('cannot write non-contiguous empty file')

        # add extra tags from user
        for t in extratags:
            addtag(*t)

        # TODO: check TIFFReadDirectoryCheckOrder warning in files containing
        #   multiple tags of same code
        # the entries in an IFD must be sorted in ascending order by tag code
        tags = sorted(tags, key=lambda x: x[0])

        # define compress function
        if bilevel:
            if compressiontag == 1:

                def compress(data, level=None):
                    return numpy.packbits(data, axis=-2).tobytes()

            elif compressiontag in (5, 32773):
                # LZW, PackBits
                def compress(
                    data,
                    compressor=TIFF.COMPRESSORS[compressiontag],
                    kwargs=compressionargs,
                ):
                    data = numpy.packbits(data, axis=-2).tobytes()
                    return compressor(data, **kwargs)

            else:
                raise NotImplementedError('cannot compress bilevel image')

        elif compression:
            compressor = TIFF.COMPRESSORS[compressiontag]

            if compressiontag == 32773:  # PackBits
                compressionargs['axis'] = -2

            if subsampling:
                # JPEG with subsampling
                def compress(
                    data, compressor=compressor, kwargs=compressionargs
                ):
                    return compressor(data, **kwargs)

            elif predictor:

                def compress(
                    data,
                    predictor=predictor,
                    compressor=compressor,
                    kwargs=compressionargs,
                ):
                    data = predictor(data, axis=-2)
                    return compressor(data, **kwargs)

            elif compressionargs:

                def compress(
                    data, compressor=compressor, kwargs=compressionargs
                ):
                    return compressor(data, **kwargs)

            else:
                compress = compressor

        elif packints:

            def compress(data, bps=bitspersample):
                return packints_encode(data, bps, axis=-2)

        else:
            compress = False

        del compression

        fhpos = fh.tell()
        if (
            not (offsetsize > 4 or self._imagej or compress)
            and fhpos + datasize > 2 ** 32 - 1
        ):
            raise ValueError('data too large for standard TIFF file')

        # if not compressed or multi-tiled, write the first IFD and then
        # all data contiguously; else, write all IFDs and data interleaved
        for pageindex in range(1 if contiguous else storedshape[0]):

            ifdpos = fhpos
            if ifdpos % 2:
                # location of IFD must begin on a word boundary
                fh.write(b'\x00')
                ifdpos += 1

            if self._subifdslevel < 0:
                # update pointer at ifdoffset
                fh.seek(self._ifdoffset)
                fh.write(pack(offsetformat, ifdpos))

            fh.seek(ifdpos)

            # create IFD in memory
            if pageindex < 2:
                subifdsoffsets = None
                ifd = io.BytesIO()
                ifd.write(pack(tagnoformat, len(tags)))
                tagoffset = ifd.tell()
                ifd.write(b''.join(t[1] for t in tags))
                ifdoffset = ifd.tell()
                ifd.write(pack(offsetformat, 0))  # offset to next IFD
                # write tag values and patch offsets in ifdentries
                for tagindex, tag in enumerate(tags):
                    offset = tagoffset + tagindex * tagsize + 4 + offsetsize
                    code = tag[0]
                    value = tag[2]
                    if value:
                        pos = ifd.tell()
                        if pos % 2:
                            # tag value is expected to begin on word boundary
                            ifd.write(b'\x00')
                            pos += 1
                        ifd.seek(offset)
                        ifd.write(pack(offsetformat, ifdpos + pos))
                        ifd.seek(pos)
                        ifd.write(value)
                        if code == tagoffsets:
                            dataoffsetsoffset = offset, pos
                        elif code == tagbytecounts:
                            databytecountsoffset = offset, pos
                        elif code == 270:
                            self._descriptiontag.offset = (
                                ifdpos + tagoffset + tagindex * tagsize
                            )
                            self._descriptiontag.valueoffset = ifdpos + pos
                        elif code == 330:
                            subifdsoffsets = offset, pos
                    elif code == tagoffsets:
                        dataoffsetsoffset = offset, None
                    elif code == tagbytecounts:
                        databytecountsoffset = offset, None
                    elif code == 270:
                        self._descriptiontag.offset = (
                            ifdpos + tagoffset + tagindex * tagsize
                        )
                        self._descriptiontag.valueoffset = (
                            self._descriptiontag.offset + offsetsize + 4
                        )
                    elif code == 330:
                        subifdsoffsets = offset, None
                ifdsize = ifd.tell()
                if ifdsize % 2:
                    ifd.write(b'\x00')
                    ifdsize += 1

            # write IFD later when strip/tile bytecounts and offsets are known
            fh.seek(ifdsize, os.SEEK_CUR)

            # write image data
            dataoffset = fh.tell()
            if align is None:
                align = 16
            skip = (align - (dataoffset % align)) % align
            fh.seek(skip, os.SEEK_CUR)
            dataoffset += skip
            if contiguous:
                if data is None:
                    fh.write_empty(datasize)
                elif dataiter is not None:
                    for pagedata in dataiter:
                        if pagedata.dtype != datadtype:
                            raise ValueError(
                                f'dtype of iterable {pagedata.dtype!r} '
                                f'does not match dtype {datadtype!r}'
                            )
                        fh.write_array(pagedata.reshape(storedshape[1:]))
                else:
                    fh.write_array(data)
            elif tile:
                if storedshape[-1] == 1:
                    tileshape = tile
                else:
                    tileshape = tile + (storedshape[-1],)
                tilesize = product(tileshape) * datadtype.itemsize
                if data is None:
                    fh.write_empty(numtiles * tilesize)
                elif compress:
                    isbytes = True
                    for tileindex in range(storedshape[1] * product(tiles)):
                        chunk = next(dataiter)
                        if chunk is None:
                            databytecounts[tileindex] = 0
                            continue
                        if isbytes and isinstance(chunk, bytes):
                            # pre-compressed
                            pass
                        else:
                            if chunk.nbytes != tilesize:
                                chunk = pad_tile(chunk, tileshape, datadtype)
                            isbytes = False
                            chunk = compress(chunk)
                        fh.write(chunk)
                        databytecounts[tileindex] = len(chunk)
                else:
                    for tileindex in range(storedshape[1] * product(tiles)):
                        chunk = next(dataiter)
                        if chunk is None:
                            # databytecounts[tileindex] = 0  # not contiguous
                            fh.write_empty(tilesize)
                            continue
                        if chunk.nbytes != tilesize:
                            chunk = pad_tile(chunk, tileshape, datadtype)
                        fh.write_array(chunk)
            elif compress:
                # write one strip per rowsperstrip
                numstrips = (
                    storedshape[-3] + rowsperstrip - 1
                ) // rowsperstrip
                stripindex = 0
                pagedata = next(dataiter).reshape(storedshape[1:])
                if pagedata.dtype != datadtype:
                    raise ValueError(
                        f'dtype of iterable {pagedata.dtype!r} '
                        f'does not match dtype {datadtype!r}'
                    )
                for plane in pagedata:
                    for depth in plane:
                        for i in range(numstrips):
                            strip = depth[
                                i * rowsperstrip : (i + 1) * rowsperstrip
                            ]
                            strip = compress(strip)
                            fh.write(strip)
                            databytecounts[stripindex] = len(strip)
                            stripindex += 1
            else:
                pagedata = next(dataiter).reshape(storedshape[1:])
                if pagedata.dtype != datadtype:
                    raise ValueError(
                        f'dtype of iterable {pagedata.dtype!r} '
                        f'does not match dtype {datadtype!r}'
                    )
                fh.write_array(pagedata)

            # update strip/tile offsets
            offset, pos = dataoffsetsoffset
            ifd.seek(offset)
            if pos:
                ifd.write(pack(offsetformat, ifdpos + pos))
                ifd.seek(pos)
                offset = dataoffset
                for size in databytecounts:
                    ifd.write(pack(offsetformat, offset))
                    offset += size
            else:
                ifd.write(pack(offsetformat, dataoffset))

            if compress:
                # update strip/tile bytecounts
                offset, pos = databytecountsoffset
                ifd.seek(offset)
                if pos:
                    ifd.write(pack(offsetformat, ifdpos + pos))
                    ifd.seek(pos)
                ifd.write(pack(bytecountformat, *databytecounts))

            if subifdsoffsets is not None:
                # update and save pointer to SubIFDs tag values if necessary
                offset, pos = subifdsoffsets
                if pos is not None:
                    ifd.seek(offset)
                    ifd.write(pack(offsetformat, ifdpos + pos))
                    self._subifdsoffsets.append(ifdpos + pos)
                else:
                    self._subifdsoffsets.append(ifdpos + offset)

            fhpos = fh.tell()
            fh.seek(ifdpos)
            fh.write(ifd.getbuffer())
            fh.flush()

            if self._subifdslevel < 0:
                self._ifdoffset = ifdpos + ifdoffset
            else:
                # update SubIFDs tag values
                fh.seek(
                    self._subifdsoffsets[self._ifdindex]
                    + self._subifdslevel * offsetsize
                )
                fh.write(pack(offsetformat, ifdpos))

                # update SubIFD chain offsets
                if self._subifdslevel == 0:
                    self._nextifdoffsets.append(ifdpos + ifdoffset)
                else:
                    fh.seek(self._nextifdoffsets[self._ifdindex])
                    fh.write(pack(offsetformat, ifdpos))
                    self._nextifdoffsets[self._ifdindex] = ifdpos + ifdoffset
                self._ifdindex += 1
                self._ifdindex %= len(self._subifdsoffsets)

            fh.seek(fhpos)

            # remove tags that should be written only once
            if pageindex == 0:
                tags = [tag for tag in tags if not tag[-1]]

        self._storedshape = storedshape
        self._datashape = (1,) + inputshape
        self._datadtype = datadtype
        self._dataoffset = dataoffset
        self._databytecounts = databytecounts

        if contiguous:
            # write remaining IFDs/tags later
            self._tags = tags
            # return offset and size of image data
            if returnoffset:
                return dataoffset, sum(databytecounts)
        return None

    def overwrite_description(self, description):
        """Overwrite the value of the last ImageDescription tag.

        Can be used to write OME-XML after writing the image data.
        Ends a contiguous series.

        """
        if self._descriptiontag is None:
            raise ValueError('no ImageDescription tag found')
        self._write_remaining_pages()
        self._descriptiontag.overwrite(description, erase=False)
        self._descriptiontag = None

    def _write_remaining_pages(self):
        """Write outstanding IFDs and tags to file."""
        if not self._tags or self._truncate or self._datashape is None:
            return

        pageno = self._storedshape[0] * self._datashape[0] - 1
        if pageno < 1:
            self._tags = None
            self._dataoffset = None
            self._databytecounts = None
            return

        fh = self._fh
        fhpos = fh.tell()
        if fhpos % 2:
            fh.write(b'\x00')
            fhpos += 1

        pack = struct.pack
        offsetformat = self.tiff.offsetformat
        offsetsize = self.tiff.offsetsize
        tagnoformat = self.tiff.tagnoformat
        tagsize = self.tiff.tagsize
        dataoffset = self._dataoffset
        pagedatasize = sum(self._databytecounts)
        subifdsoffsets = None

        # construct template IFD in memory
        # must patch offsets to next IFD and data before writing to file
        ifd = io.BytesIO()
        ifd.write(pack(tagnoformat, len(self._tags)))
        tagoffset = ifd.tell()
        ifd.write(b''.join(t[1] for t in self._tags))
        ifdoffset = ifd.tell()
        ifd.write(pack(offsetformat, 0))  # offset to next IFD
        # tag values
        for tagindex, tag in enumerate(self._tags):
            offset = tagoffset + tagindex * tagsize + offsetsize + 4
            code = tag[0]
            value = tag[2]
            if value:
                pos = ifd.tell()
                if pos % 2:
                    # tag value is expected to begin on word boundary
                    ifd.write(b'\x00')
                    pos += 1
                ifd.seek(offset)
                try:
                    ifd.write(pack(offsetformat, fhpos + pos))
                except Exception:  # struct.error
                    if self._imagej:
                        warnings.warn(
                            f'{self!r} truncating ImageJ file', UserWarning
                        )
                        self._truncate = True
                        return
                    raise ValueError('data too large for non-BigTIFF file')
                ifd.seek(pos)
                ifd.write(value)
                if code == self._dataoffsetstag:
                    # save strip/tile offsets for later updates
                    dataoffsetsoffset = offset, pos
                elif code == 330:
                    # save subifds offsets for later updates
                    subifdsoffsets = offset, pos
            elif code == self._dataoffsetstag:
                dataoffsetsoffset = offset, None
            elif code == 330:
                subifdsoffsets = offset, None

        ifdsize = ifd.tell()
        if ifdsize % 2:
            ifd.write(b'\x00')
            ifdsize += 1

        # check if all IFDs fit in file
        if offsetsize < 8 and fhpos + ifdsize * pageno > 2 ** 32 - 32:
            if self._imagej:
                warnings.warn(f'{self!r} truncating ImageJ file', UserWarning)
                self._truncate = True
                return
            raise ValueError('data too large for non-BigTIFF file')

        # assemble IFD chain in memory from IFD template
        ifds = io.BytesIO(bytes(ifdsize * pageno))
        ifdpos = fhpos
        for _ in range(pageno):
            # update strip/tile offsets in IFD
            dataoffset += pagedatasize  # offset to image data
            offset, pos = dataoffsetsoffset
            ifd.seek(offset)
            if pos is not None:
                ifd.write(pack(offsetformat, ifdpos + pos))
                ifd.seek(pos)
                offset = dataoffset
                for size in self._databytecounts:
                    ifd.write(pack(offsetformat, offset))
                    offset += size
            else:
                ifd.write(pack(offsetformat, dataoffset))

            if subifdsoffsets is not None:
                offset, pos = subifdsoffsets
                self._subifdsoffsets.append(
                    ifdpos + (pos if pos is not None else offset)
                )

            if self._subifdslevel < 0:
                if subifdsoffsets is not None:
                    # update pointer to SubIFDs tag values if necessary
                    offset, pos = subifdsoffsets
                    if pos is not None:
                        ifd.seek(offset)
                        ifd.write(pack(offsetformat, ifdpos + pos))

                # update pointer at ifdoffset to point to next IFD in file
                ifdpos += ifdsize
                ifd.seek(ifdoffset)
                ifd.write(pack(offsetformat, ifdpos))

            else:
                # update SubIFDs tag values in file
                fh.seek(
                    self._subifdsoffsets[self._ifdindex]
                    + self._subifdslevel * offsetsize
                )
                fh.write(pack(offsetformat, ifdpos))

                # update SubIFD chain
                if self._subifdslevel == 0:
                    self._nextifdoffsets.append(ifdpos + ifdoffset)
                else:
                    fh.seek(self._nextifdoffsets[self._ifdindex])
                    fh.write(pack(offsetformat, ifdpos))
                    self._nextifdoffsets[self._ifdindex] = ifdpos + ifdoffset
                self._ifdindex += 1
                self._ifdindex %= len(self._subifdsoffsets)
                ifdpos += ifdsize

            # write IFD entry
            ifds.write(ifd.getbuffer())

        # terminate IFD chain
        ifdoffset += ifdsize * (pageno - 1)
        ifds.seek(ifdoffset)
        ifds.write(pack(offsetformat, 0))
        # write IFD chain to file
        fh.seek(fhpos)
        fh.write(ifds.getbuffer())

        if self._subifdslevel < 0:
            # update file to point to new IFD chain
            pos = fh.tell()
            fh.seek(self._ifdoffset)
            fh.write(pack(offsetformat, fhpos))
            fh.flush()
            fh.seek(pos)
            self._ifdoffset = fhpos + ifdoffset

        self._tags = None
        self._dataoffset = None
        self._databytecounts = None
        # do not reset _storedshape, _datashape, _datadtype

    def _write_image_description(self):
        """Write metadata to ImageDescription tag."""
        if self._datashape is None or self._descriptiontag is None:
            self._descriptiontag = None
            return

        if self._ome:
            if self._subifdslevel < 0:
                self._ome.addimage(
                    self._datadtype,
                    self._datashape[0 if self._datashape[0] != 1 else 1 :],
                    self._storedshape,
                    **self._metadata,
                )
            description = self._ome.tostring(declaration=True).encode()
        elif self._datashape[0] == 1:
            # description already up-to-date
            self._descriptiontag = None
            return
        # elif self._subifdslevel >= 0:
        #     # don't write metadata to SubIFDs
        #     return
        elif self._imagej:
            colormapped = self._colormap is not None
            isrgb = self._storedshape[-1] in (3, 4)
            description = imagej_description(
                self._datashape, isrgb, colormapped, **self._metadata
            )
        else:
            description = json_description(self._datashape, **self._metadata)

        self._descriptiontag.overwrite(description, erase=False)
        self._descriptiontag = None

    def _now(self):
        """Return current date and time."""
        return datetime.datetime.now()

    def close(self):
        """Write remaining pages and close file handle."""
        if not self._truncate:
            self._write_remaining_pages()
        self._write_image_description()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return f'<tifffile.TiffWriter {snipstr(self.filehandle.name, 32)!r}>'


class TiffFile:
    """Read image and metadata from TIFF file.

    TiffFile instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    TiffFile instances are not thread-safe.

    Attributes
    ----------
    pages : TiffPages
        Sequence of TIFF pages in file.
    series : list of TiffPageSeries
        Sequences of closely related TIFF pages. These are computed
        from OME, LSM, ImageJ, etc. metadata or based on similarity
        of page properties such as shape, dtype, and compression.
    is_flag : bool
        If True, file is of a certain format.
        Flags are: bigtiff, uniform, shaped, ome, imagej, stk, lsm, fluoview,
        nih, vista, micromanager, metaseries, mdgel, mediacy, tvips, fei,
        sem, scn, svs, scanimage, andor, epics, ndpi, pilatus, qpi.

    All attributes are read-only.

    """

    def __init__(
        self,
        arg,
        mode=None,
        name=None,
        offset=None,
        size=None,
        _multifile=True,
        _useframes=None,
        _parent=None,
        **kwargs,
    ):
        """Initialize instance from file.

        Parameters
        ----------
        arg : path_like or open file
            Name of file or open file object.
            The file objects are closed in TiffFile.close().
        mode : str (optional)
            File open mode in case 'arg' is a file name. Must be 'rb' or 'r+b'.
            Default is 'rb'.
        name : str (optional)
            Optional name of file in case 'arg' is a file handle.
        offset : int (optional)
            Optional start position of embedded file. By default, this is
            the current file position.
        size : int (optional)
            Optional size of embedded file. By default, this is the number
            of bytes from the 'offset' to the end of the file.
        kwargs : bool (optional)
            'is_ome': If False, disable processing of OME-XML metadata.

        """
        if kwargs:
            # TODO: remove; formally deprecated in 2020.10.1
            for key in ('fastij', 'movie', 'multifile', 'multifile_close'):
                if key in kwargs:
                    del kwargs[key]
                    warnings.warn(
                        f'<tifffile.TiffFile> the {key!r} argument is ignored',
                        DeprecationWarning,
                        stacklevel=2,
                    )
            if 'pages' in kwargs:
                raise TypeError(
                    "the 'pages' parameter is no longer supported."
                    "\n\nUse TiffFile.asarray(key=[...]) to read image data "
                    "from specific pages.\n"
                )

            for key, value in kwargs.items():
                if key[:3] == 'is_' and key[3:] in TIFF.FILE_FLAGS:
                    if value is not None:
                        setattr(self, key, bool(value))
                else:
                    raise TypeError(f'unexpected keyword argument: {key}')

        if mode not in (None, 'rb', 'r+b'):
            raise ValueError(f'invalid mode {mode!r}')

        fh = FileHandle(arg, mode=mode, name=name, offset=offset, size=size)
        self._fh = fh
        self._multifile = bool(_multifile)
        self._files = {fh.name: self}  # cache of TiffFile instances
        self._decoders = {}  # cache of TiffPage.decode functions
        self._parent = self if _parent is None else _parent  # OME master file

        try:
            fh.seek(0)
            header = fh.read(4)
            try:
                byteorder = {b'II': '<', b'MM': '>', b'EP': '<'}[header[:2]]
            except KeyError:
                raise TiffFileError(f'not a TIFF file {header!r}')

            version = struct.unpack(byteorder + 'H', header[2:4])[0]
            if version == 43:
                # BigTiff
                offsetsize, zero = struct.unpack(byteorder + 'HH', fh.read(4))
                if zero != 0 or offsetsize != 8:
                    raise TiffFileError(
                        f'invalid BigTIFF offset size {(offsetsize, zero)}'
                    )
                if byteorder == '>':
                    self.tiff = TIFF.BIG_BE
                else:
                    self.tiff = TIFF.BIG_LE
            elif version == 42:
                # Classic TIFF
                if byteorder == '>':
                    self.tiff = TIFF.CLASSIC_BE
                elif kwargs.get('is_ndpi', False) or fh.name.endswith('ndpi'):
                    # NDPI uses 64 bit IFD offsets
                    self.tiff = TIFF.NDPI_LE
                else:
                    self.tiff = TIFF.CLASSIC_LE
            elif version == 0x4E31:
                # NIFF
                if byteorder == '>':
                    raise TiffFileError('invalid NIFF file')
                log_warning(f'{self!r} NIFF format not supported')
                self.tiff = TIFF.CLASSIC_LE
            elif version == 0x55 or version == 0x4F52 or version == 0x5352:
                # Panasonic or Olympus RAW
                log_warning(
                    f'{self!r} RAW format 0x{version:04X} not supported'
                )
                if byteorder == '>':
                    self.tiff = TIFF.CLASSIC_BE
                else:
                    self.tiff = TIFF.CLASSIC_LE
            else:
                raise TiffFileError(f'invalid TIFF version {version}')

            # file handle is at offset to offset to first page
            self.pages = TiffPages(self)

            if self.is_lsm and (
                self.filehandle.size >= 2 ** 32
                or self.pages[0].compression != 1
                or self.pages[1].compression != 1
            ):
                self._lsm_load_pages()

            elif self.is_scanimage and not self.is_bigtiff:
                # ScanImage <= 2015
                try:
                    self.pages._load_virtual_frames()
                except Exception as exc:
                    log_warning(
                        f'{self!r} _load_virtual_frames failed with '
                        f'{exc.__class__.__name__}: {exc}'
                    )

            elif self.is_philips:
                try:
                    self._philips_load_pages()
                except Exception as exc:
                    log_warning(
                        f'{self!r} _philips_load_pages failed with '
                        f'{exc.__class__.__name__}: {exc}'
                    )

            elif self.is_ndpi:
                try:
                    self._ndpi_load_pages()
                except Exception as exc:
                    log_warning(
                        f'{self!r} _ndpi_load_pages failed with '
                        f'{exc.__class__.__name__}: {exc}'
                    )

            elif _useframes:
                self.pages.useframes = True

        except Exception:
            fh.close()
            raise

    @property
    def byteorder(self):
        return self.tiff.byteorder

    @property
    def filehandle(self):
        """Return file handle."""
        return self._fh

    @property
    def filename(self):
        """Return name of file handle."""
        return self._fh.name

    @lazyattr
    def fstat(self):
        """Return status of file handle as stat_result object."""
        try:
            return os.fstat(self._fh.fileno())
        except Exception:  # io.UnsupportedOperation
            return None

    def close(self):
        """Close open file handle(s)."""
        for tif in self._files.values():
            tif.filehandle.close()

    def asarray(
        self,
        key=None,
        series=None,
        level=None,
        squeeze=None,
        out=None,
        maxworkers=None,
    ):
        """Return image data from selected TIFF page(s) as numpy array.

        By default, the data from the first series is returned.

        Parameters
        ----------
        key : int, slice, or sequence of indices
            Defines which pages to return as array.
            If None (default), data from a series (default 0) is returned.
            If not None, data from the specified pages in the whole file
            (if 'series' is None) or a specified series are returned as a
            stacked array.
            Requesting an array from multiple pages that are not compatible
            wrt. shape, dtype, compression etc. is undefined, i.e. may crash
            or return incorrect values.
        series : int or TiffPageSeries
            Defines which series of pages to return as array.
        level : int
            Defines which pyramid level of a series to return as array.
        squeeze : bool
            If True, all length-1 dimensions (except X and Y) are squeezed
            out from the array.
            If False, single pages are returned as 5D array (TiffPage.shaped).
            For series, the shape of the returned array also includes singlet
            dimensions specified in some file formats. E.g. ImageJ series, and
            most commonly also OME series, are returned in TZCYXS order.
            If None (default), all but "shaped" series are squeezed.
        out : numpy.ndarray, str, or file-like object
            Buffer where image data are saved.
            If None (default), a new array is created.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If 'memmap', directly memory-map the image data in the TIFF file
            if possible; else create a memory-mapped array in a temporary file.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        maxworkers : int or None
            Maximum number of threads to concurrently get data from multiple
            pages or compressed segments.
            If None (default), up to half the CPU cores are used.
            If 1, multi-threading is disabled.
            Reading data from file is limited to a single thread.
            Using multiple threads can significantly speed up this function
            if the bottleneck is decoding compressed data, e.g. in case of
            large LZW compressed LSM files or JPEG compressed tiled slides.
            If the bottleneck is I/O or pure Python code, using multiple
            threads might be detrimental.

        Returns
        -------
        numpy.ndarray
            Image data from the specified pages.
            See TiffPage.asarray for operations that are applied (or not)
            to the raw data stored in the file.

        """
        if not self.pages:
            return numpy.array([])
        if key is None and series is None:
            series = 0
        if series is None:
            pages = self.pages
        else:
            try:
                series = self.series[series]
            except (KeyError, TypeError):
                pass
            if level is not None:
                series = series.levels[level]
            pages = series.pages

        if key is None:
            pass
        elif series is None:
            pages = self.pages._getlist(key)
        elif isinstance(key, (int, numpy.integer)):
            pages = [pages[key]]
        elif isinstance(key, slice):
            pages = pages[key]
        elif isinstance(key, Iterable):
            pages = [pages[k] for k in key]
        else:
            raise TypeError('key must be an int, slice, or sequence')

        if not pages:
            raise ValueError('no pages selected')

        if key is None and series and series.offset:
            typecode = self.byteorder + series.dtype.char
            if (
                pages[0].is_memmappable
                and isinstance(out, str)
                and out == 'memmap'
            ):
                # direct mapping
                shape = series.get_shape(squeeze)
                result = self.filehandle.memmap_array(
                    typecode, shape, series.offset
                )
            else:
                # read into output
                shape = series.get_shape(squeeze)
                if out is not None:
                    out = create_output(out, shape, series.dtype)
                self.filehandle.seek(series.offset)
                result = self.filehandle.read_array(
                    typecode, product(shape), out=out
                )
        elif len(pages) == 1:
            result = pages[0].asarray(out=out, maxworkers=maxworkers)
        else:
            result = stack_pages(pages, out=out, maxworkers=maxworkers)

        if result is None:
            return None

        if key is None:
            shape = series.get_shape(squeeze)
            try:
                result.shape = shape
            except ValueError:
                try:
                    log_warning(
                        f'{self!r} '
                        f'asarray failed to reshape {result.shape} to {shape}'
                    )
                    # try series of expected shapes
                    result.shape = (-1,) + shape
                except ValueError:
                    # revert to generic shape
                    result.shape = (-1,) + pages[0].shape
        elif len(pages) == 1:
            if squeeze is None:
                squeeze = True
            result.shape = pages[0].shape if squeeze else pages[0].shaped
        else:
            if squeeze is None:
                squeeze = True
            result.shape = (-1,) + (
                pages[0].shape if squeeze else pages[0].shaped
            )
        return result

    def aszarr(self, key=None, series=None, level=None, **kwargs):
        """Return image data from selected TIFF page(s) as zarr storage."""
        if not self.pages:
            raise NotImplementedError('empty zarr arrays not supported')
        if key is None and series is None:
            return self.series[0].aszarr(level=level, **kwargs)
        if series is None:
            pages = self.pages
        else:
            try:
                series = self.series[series]
            except (KeyError, TypeError):
                pass
            if key is None:
                return series.aszarr(level=level, **kwargs)
            pages = series.pages
        if isinstance(key, (int, numpy.integer)):
            return pages[key].aszarr(**kwargs)
        raise TypeError('key must be an integer index')

    @lazyattr
    def series(self):
        """Return related pages as TiffPageSeries.

        Side effect: after calling this function, TiffFile.pages might contain
        TiffPage and TiffFrame instances.

        """
        if not self.pages:
            return []

        useframes = self.pages.useframes
        keyframe = self.pages.keyframe.index
        series = []
        for name in (
            'shaped',
            'lsm',
            'ome',
            'imagej',
            'fluoview',
            'stk',
            'sis',
            'svs',
            'scn',
            'qpi',
            'ndpi',
            'bif',
            'scanimage',
            'mdgel',  # adds second page to cache
            'uniform',
        ):
            if getattr(self, 'is_' + name, False):
                series = getattr(self, '_series_' + name)()
                if not series and name == 'ome' and self.is_imagej:
                    # try ImageJ series if OME series fails.
                    # clear pages cache since _series_ome() might leave some
                    # frames without keyframe
                    self.pages._clear()
                    continue
                break
        self.pages.useframes = useframes
        self.pages.keyframe = keyframe
        if not series:
            series = self._series_generic()

        # remove empty series, e.g. in MD Gel files
        # series = [s for s in series if product(s.shape) > 0]

        for i, s in enumerate(series):
            s.index = i
        return series

    def _series_uniform(self):
        """Return all images in file as single series."""
        page = self.pages[0]
        validate = not (page.is_scanimage or page.is_nih)
        pages = self.pages._getlist(validate=validate)
        shape = (len(pages),) + page.shape
        axes = 'I' + page.axes
        dtype = page.dtype
        if page.is_nih:
            kind = 'NIHImage'
        else:
            kind = 'Uniform'
        return [TiffPageSeries(pages, shape, dtype, axes, kind=kind)]

    def _series_generic(self):
        """Return image series in file.

        A series is a sequence of TiffPages with the same hash.

        """
        pages = self.pages
        pages._clear(False)
        pages.useframes = False
        if pages.cache:
            pages._load()

        series = []
        keys = []
        seriesdict = {}

        def addpage(page):
            # add page to seriesdict
            if not page.shape:  # or product(page.shape) == 0:
                return
            key = page.hash
            if key in seriesdict:
                for p in seriesdict[key]:
                    if p.offset == page.offset:
                        break  # remove duplicate page
                else:
                    seriesdict[key].append(page)
            else:
                keys.append(key)
                seriesdict[key] = [page]

        for page in pages:
            addpage(page)
            if page.subifds is not None:
                for i, offset in enumerate(page.subifds):
                    if offset < 8:
                        continue
                    try:
                        self._fh.seek(offset)
                        subifd = TiffPage(self, (page.index, i))
                    except Exception as exc:
                        log_warning(
                            f'{self!r} generic series failed with '
                            f'{exc.__class__.__name__}: {exc}'
                        )
                    else:
                        addpage(subifd)

        for key in keys:
            pages = seriesdict[key]
            page = pages[0]
            shape = (len(pages),) + page.shape
            axes = 'I' + page.axes
            if 'S' not in axes:
                shape += (1,)
                axes += 'S'
            series.append(
                TiffPageSeries(pages, shape, page.dtype, axes, kind='Generic')
            )

        self.is_uniform = len(series) == 1
        pyramidize_series(series)
        return series

    def _series_shaped(self):
        """Return image series in "shaped" file."""

        def append(series, pages, axes, shape, reshape, name, truncated):
            # append TiffPageSeries to series
            page = pages[0]
            if not axes:
                shape = page.shape
                axes = page.axes
                if len(pages) > 1:
                    shape = (len(pages),) + shape
                    axes = 'Q' + axes
            size = product(shape)
            resize = product(reshape)
            if page.is_contiguous and resize > size and resize % size == 0:
                if truncated is None:
                    truncated = True
                axes = 'Q' + axes
                shape = (resize // size,) + shape
            try:
                axes = reshape_axes(axes, shape, reshape)
                shape = reshape
            except ValueError as exc:
                log_warning(
                    f'{self!r} shaped series failed with '
                    f'{exc.__class__.__name__}: {exc}'
                )
            series.append(
                TiffPageSeries(
                    pages,
                    shape,
                    page.dtype,
                    axes,
                    name=name,
                    kind='Shaped',
                    truncated=truncated,
                    squeeze=False,
                )
            )

        def detect_series(pages, series, issubifds=False):
            lenpages = len(pages)
            keyframe = axes = shape = reshape = name = None
            index = 0
            while True:
                if index >= lenpages:
                    break
                if issubifds:
                    keyframe = pages[0]
                else:
                    # new keyframe; start of new series
                    pages.keyframe = index
                    keyframe = pages.keyframe
                if not keyframe.is_shaped:
                    log_warning(
                        f'{self!r} '
                        'invalid shaped series metadata or corrupted file'
                    )
                    return None
                # read metadata
                axes = None
                shape = None
                metadata = json_description_metadata(keyframe.is_shaped)
                name = metadata.get('name', '')
                reshape = metadata['shape']
                truncated = None if keyframe.subifds is None else False
                truncated = metadata.get('truncated', truncated)
                if 'axes' in metadata:
                    axes = metadata['axes']
                    if len(axes) == len(reshape):
                        shape = reshape
                    else:
                        axes = ''
                        log_warning(
                            f'{self!r} shaped series axes do not match shape'
                        )
                # skip pages if possible
                spages = [keyframe]
                size = product(reshape)
                if size > 0:
                    npages, mod = divmod(size, product(keyframe.shape))
                else:
                    npages = 1
                    mod = 0
                if mod:
                    log_warning(
                        f'{self!r} '
                        'shaped series shape does not match page shape'
                    )
                    return None
                if 1 < npages <= lenpages - index:
                    size *= keyframe._dtype.itemsize
                    if truncated:
                        npages = 1
                    elif (
                        keyframe.is_final
                        and keyframe.offset + size < pages[index + 1].offset
                        and keyframe.subifds is None
                    ):
                        truncated = False
                    else:
                        # must read all pages for series
                        truncated = False
                        for j in range(index + 1, index + npages):
                            page = pages[j]
                            page.keyframe = keyframe
                            spages.append(page)
                append(series, spages, axes, shape, reshape, name, truncated)
                index += npages

                # create series from SubIFDs
                if keyframe.subifds:
                    for i, offset in enumerate(keyframe.subifds):
                        if offset < 8:
                            continue
                        subifds = []
                        for j, page in enumerate(spages):
                            # if page.subifds is not None:
                            try:
                                self._fh.seek(page.subifds[i])
                                if j == 0:
                                    subifd = TiffPage(self, (page.index, i))
                                    keysubifd = subifd
                                else:
                                    subifd = TiffFrame(
                                        self,
                                        (page.index, i),
                                        keyframe=keysubifd,
                                    )
                            except Exception as exc:
                                log_warning(
                                    f'{self!r} shaped series failed with '
                                    f'{exc.__class__.__name__}: {exc}'
                                )
                                return None
                            subifds.append(subifd)
                        if subifds:
                            series = detect_series(subifds, series, True)
                            if series is None:
                                return None
            return series

        self.pages.useframes = True
        series = detect_series(self.pages, [])
        if series is None:
            return None
        self.is_uniform = len(series) == 1
        pyramidize_series(series, isreduced=True)
        return series

    def _series_imagej(self):
        """Return image series in ImageJ file."""
        # ImageJ's dimension order is TZCYXS
        # TODO: fix loading of color, composite, or palette images
        pages = self.pages
        pages.useframes = True
        pages.keyframe = 0
        page = pages[0]
        meta = self.imagej_metadata

        def is_virtual():
            # ImageJ virtual hyperstacks store all image metadata in the first
            # page and image data are stored contiguously before the second
            # page, if any
            if not page.is_final:
                return False
            images = meta.get('images', 0)
            if images <= 1:
                return False
            offset, count = page.is_contiguous
            if (
                count != product(page.shape) * page.bitspersample // 8
                or offset + count * images > self.filehandle.size
            ):
                raise ValueError
            # check that next page is stored after data
            if len(pages) > 1 and offset + count * images > pages[1].offset:
                return False
            return True

        try:
            isvirtual = is_virtual()
        except (ValueError, RuntimeError):
            log_warning(
                f'{self!r} ImageJ series metadata invalid or corrupted file'
            )
            return None
        if isvirtual:
            # no need to read other pages
            pages = [page]
        else:
            pages = pages[:]

        images = meta.get('images', len(pages))
        frames = meta.get('frames', 1)
        slices = meta.get('slices', 1)
        channels = meta.get('channels', 1)

        shape = (frames, slices, channels)
        axes = 'TZC'

        remain = images // product(shape)
        if remain > 1:
            log_warning(
                f'{self!r} ImageJ series contains unidentified dimension'
            )
            shape = (remain,) + shape
            axes = 'I' + axes

        if page.shaped[0] > 1:
            # planar storage, S == C, saved by Bio-Formats
            if page.shaped[0] != channels:
                log_warning(
                    f'{self!r} ImageJ series number of channels {channels} '
                    f'does not match separate samples {page.shaped[0]}'
                )
            shape = shape[:-1] + page.shape
            axes += page.axes[1:]
        elif page.shaped[-1] == channels and channels > 1:
            # keep contig storage, C = 1
            shape = (frames, slices, 1) + page.shape
            axes += page.axes
        else:
            shape += page.shape
            axes += page.axes

        if 'S' not in axes:
            shape += (1,)
            axes += 'S'
        # assert axes.endswith('TZCYXS'), axes

        truncated = (
            isvirtual
            and len(self.pages) == 1
            and page.is_contiguous[1]
            != (product(shape) * page.bitspersample // 8)
        )

        self.is_uniform = True

        return [
            TiffPageSeries(
                pages,
                shape,
                page.dtype,
                axes,
                kind='ImageJ',
                truncated=truncated,
            )
        ]

    def _series_scanimage(self):
        """Return image series in ScanImage file."""
        pages = self.pages._getlist(validate=False)
        page = pages[0]
        dtype = page.dtype
        shape = None

        framedata = self.scanimage_metadata.get('FrameData', {})
        if 'SI.hChannels.channelSave' in framedata:
            try:
                channels = framedata['SI.hChannels.channelSave']
                try:
                    # channelSave is a list
                    channels = len(channels)
                except TypeError:
                    # channelSave is an int
                    channels = int(channels)
                # slices = framedata.get(
                #    'SI.hStackManager.actualNumSlices',
                #     framedata.get('SI.hStackManager.numSlices', None),
                # )
                # if slices is None:
                #     raise ValueError('unable to determine numSlices')
                slices = None
                try:
                    frames = int(framedata['SI.hStackManager.framesPerSlice'])
                except Exception:
                    # framesPerSlice is inf
                    slices = 1
                    if len(pages) % channels:
                        raise ValueError('unable to determine framesPerSlice')
                    frames = len(pages) // channels
                if slices is None:
                    slices = max(len(pages) // (frames * channels), 1)
                shape = (slices, frames, channels) + page.shape
                axes = 'ZTC' + page.axes
            except Exception as exc:
                log_warning(
                    f'{self!r} ScanImage series failed with'
                    f'{exc.__class__.__name__}: {exc}'
                )

        # TODO: older versions of ScanImage store non-varying frame data in
        # the ImageDescription tag. Candidates are scanimage.SI5.channelsSave,
        # scanimage.SI5.stackNumSlices, scanimage.SI5.acqNumFrames
        # scanimage.SI4., state.acq.numberOfFrames, state.acq.numberOfFrames...

        if shape is None:
            shape = (len(pages),) + page.shape
            axes = 'I' + page.axes

        return [TiffPageSeries(pages, shape, dtype, axes, kind='ScanImage')]

    def _series_fluoview(self):
        """Return image series in FluoView file."""
        pages = self.pages._getlist(validate=False)

        mm = self.fluoview_metadata
        mmhd = list(reversed(mm['Dimensions']))
        axes = ''.join(TIFF.MM_DIMENSIONS.get(i[0].upper(), 'Q') for i in mmhd)
        shape = tuple(int(i[1]) for i in mmhd)
        self.is_uniform = True
        return [
            TiffPageSeries(
                pages,
                shape,
                pages[0].dtype,
                axes,
                name=mm['ImageName'],
                kind='FluoView',
            )
        ]

    def _series_mdgel(self):
        """Return image series in MD Gel file."""
        # only a single page, scaled according to metadata in second page
        self.pages.useframes = False
        self.pages.keyframe = 0
        md = self.mdgel_metadata
        if md['FileTag'] in (2, 128):
            dtype = numpy.dtype('float32')
            scale = md['ScalePixel']
            scale = scale[0] / scale[1]  # rational
            if md['FileTag'] == 2:
                # squary root data format
                def transform(a):
                    return a.astype('float32') ** 2 * scale

            else:

                def transform(a):
                    return a.astype('float32') * scale

        else:
            transform = None
        page = self.pages[0]
        self.is_uniform = False
        return [
            TiffPageSeries(
                [page],
                page.shape,
                dtype,
                page.axes,
                transform=transform,
                kind='MDGel',
            )
        ]

    def _series_ndpi(self):
        """Return pyramidal image series in NDPI file."""
        series = self._series_generic()
        for s in series:
            s.kind = 'NDPI'
            if s.axes[0] == 'I':
                s.axes = 'Z' + s.axes[1:]
            if s.is_pyramidal:
                name = s.pages[0].tags.valueof(65427)
                s.name = 'Baseline' if name is None else name
                continue
            mag = s.pages[0].tags.valueof(65421)
            if mag is not None:
                if mag == -1.0:
                    s.name = 'Macro'
                elif mag == -2.0:
                    s.name = 'Map'
        return series

    def _series_sis(self):
        """Return image series in Olympus SIS file."""
        pages = self.pages._getlist(validate=False)
        page = pages[0]
        lenpages = len(pages)
        md = self.sis_metadata
        if 'shape' in md and 'axes' in md:
            shape = md['shape'] + page.shape
            axes = md['axes'] + page.axes
        else:
            shape = (lenpages,) + page.shape
            axes = 'I' + page.axes
        self.is_uniform = True
        return [TiffPageSeries(pages, shape, page.dtype, axes, kind='SIS')]

    def _series_qpi(self):
        """Return image series in PerkinElmer QPI file."""
        series = []
        pages = self.pages
        pages.cache = True
        pages.useframes = False
        pages.keyframe = 0
        pages._load()

        # Baseline
        # TODO: get name from ImageDescription XML
        ifds = []
        index = 0
        axes = 'C' + pages[0].axes
        dtype = pages[0].dtype
        pshape = pages[0].shape
        while index < len(pages):
            page = pages[index]
            if page.shape != pshape:
                break
            ifds.append(page)
            index += 1
        shape = (len(ifds),) + pshape
        series.append(
            TiffPageSeries(
                ifds, shape, dtype, axes, name='Baseline', kind='QPI'
            )
        )

        if index < len(pages):
            # Thumbnail
            page = pages[index]
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Thumbnail',
                    kind='QPI',
                )
            )
            index += 1

        if pages[0].is_tiled:
            # Resolutions
            while index < len(pages):
                pshape = (pshape[0] // 2, pshape[1] // 2) + pshape[2:]
                ifds = []
                while index < len(pages):
                    page = pages[index]
                    if page.shape != pshape:
                        break
                    ifds.append(page)
                    index += 1
                if len(ifds) != len(series[0].pages):
                    break
                shape = (len(ifds),) + pshape
                series[0].levels.append(
                    TiffPageSeries(
                        ifds, shape, dtype, axes, name='Resolution', kind='QPI'
                    )
                )

        if series[0].is_pyramidal and index < len(pages):
            # Macro
            page = pages[index]
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Macro',
                    kind='QPI',
                )
            )
            index += 1
            # Label
            if index < len(pages):
                page = pages[index]
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Label',
                        kind='QPI',
                    )
                )

        self.is_uniform = False
        return series

    def _series_svs(self):
        """Return image series in Aperio SVS file."""
        if not self.pages[0].is_tiled:
            return None

        series = []
        self.is_uniform = False
        self.pages.cache = True
        self.pages.useframes = False
        self.pages.keyframe = 0
        self.pages._load()

        # Baseline
        index = 0
        page = self.pages[index]
        series.append(
            TiffPageSeries(
                [page],
                page.shape,
                page.dtype,
                page.axes,
                name='Baseline',
                kind='SVS',
            )
        )
        # Thumbnail
        index += 1
        if index == len(self.pages):
            return series
        page = self.pages[index]
        series.append(
            TiffPageSeries(
                [page],
                page.shape,
                page.dtype,
                page.axes,
                name='Thumbnail',
                kind='SVS',
            )
        )
        # Resolutions
        # TODO: resolutions not by two
        index += 1
        while index < len(self.pages):
            page = self.pages[index]
            if not page.is_tiled or page.is_reduced:
                break
            series[0].levels.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Resolution',
                    kind='SVS',
                )
            )
            index += 1
        # Label, Macro; subfiletype 1, 9
        for name in ('Label', 'Macro'):
            if index == len(self.pages):
                break
            page = self.pages[index]
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name=name,
                    kind='SVS',
                )
            )
            index += 1

        return series

    def _series_scn(self):
        """Return pyramidal image series in Leica SCN file."""
        # TODO: support collections
        from xml.etree import ElementTree as etree  # delayed import

        scnxml = self.pages[0].description
        root = etree.fromstring(scnxml)

        series = []
        self.is_uniform = False
        self.pages.cache = True
        self.pages.useframes = False
        self.pages.keyframe = 0
        self.pages._load()

        for collection in root:
            if not collection.tag.endswith('collection'):
                continue
            for image in collection:
                if not image.tag.endswith('image'):
                    continue
                name = image.attrib.get('name', 'Unknown')
                for pixels in image:
                    if not pixels.tag.endswith('pixels'):
                        continue
                    resolutions = {}
                    for dimension in pixels:
                        if not dimension.tag.endswith('dimension'):
                            continue
                        if int(image.attrib.get('sizeZ', 1)) > 1:
                            raise NotImplementedError(
                                'SCN series: Z-Stacks not supported. '
                                'Please submit a sample file.'
                            )
                        sizex = int(dimension.attrib['sizeX'])
                        sizey = int(dimension.attrib['sizeY'])
                        c = int(dimension.attrib.get('c', 0))
                        z = int(dimension.attrib.get('z', 0))
                        r = int(dimension.attrib.get('r', 0))
                        ifd = int(dimension.attrib['ifd'])
                        if r in resolutions:
                            level = resolutions[r]
                            if c > level['channels']:
                                level['channels'] = c
                            if z > level['sizez']:
                                level['sizez'] = z
                            level['ifds'][(c, z)] = ifd
                        else:
                            resolutions[r] = {
                                'size': [sizey, sizex],
                                'channels': c,
                                'sizez': z,
                                'ifds': {(c, z): ifd},
                            }
                    if not resolutions:
                        continue
                    levels = []
                    for r, level in sorted(resolutions.items()):
                        shape = (level['channels'] + 1, level['sizez'] + 1)
                        axes = 'CZ'

                        ifds = [None] * product(shape)
                        for (c, z), ifd in sorted(level['ifds'].items()):
                            ifds[c * shape[1] + z] = self.pages[ifd]

                        axes += ifds[0].axes
                        shape += ifds[0].shape
                        dtype = ifds[0].dtype

                        levels.append(
                            TiffPageSeries(
                                ifds,
                                shape,
                                dtype,
                                axes,
                                parent=self,
                                name=name,
                                kind='SCN',
                            )
                        )
                    levels[0].levels.extend(levels[1:])
                    series.append(levels[0])

        return series

    def _series_bif(self):
        """Return image series in Ventana/Roche BIF file."""
        series = []
        baseline = None
        self.is_uniform = False
        self.pages.cache = True
        self.pages.useframes = False
        self.pages.keyframe = 0
        self.pages._load()

        for page in self.pages:
            if page.description[:5] == 'Label':
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Label',
                        kind='BIF',
                    )
                )
            elif (
                page.description == 'Thumbnail'
                or page.description[:11] == 'Probability'
            ):
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Thumbnail',
                        kind='BIF',
                    )
                )
            elif 'level' not in page.description:
                # TODO: is this necessary?
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Unknown',
                        kind='BIF',
                    )
                )
            elif baseline is None:
                baseline = TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Baseline',
                    kind='BIF',
                )
                series.insert(0, baseline)
            else:
                baseline.levels.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Resolution',
                        kind='SVS',
                    )
                )

        log_warning(f'{self!r} BIF series tiles are not stiched')
        return series

    def _series_ome(self):
        """Return image series in OME-TIFF file(s)."""
        # xml.etree found to be faster than lxml
        from xml.etree import ElementTree as etree  # delayed import

        omexml = self.pages[0].description
        try:
            root = etree.fromstring(omexml)
        except etree.ParseError as exc:
            # TODO: test badly encoded OME-XML
            log_warning(
                f'{self!r} OME series failed with '
                f'{exc.__class__.__name__}: {exc}'
            )
            try:
                omexml = omexml.decode(errors='ignore').encode()
                root = etree.fromstring(omexml)
            except Exception:
                return None

        self.pages.cache = True
        self.pages.useframes = True
        self.pages.keyframe = 0
        self.pages._load(keyframe=None)

        root_uuid = root.attrib.get('UUID', None)
        self._files = {root_uuid: self}
        dirname = self._fh.dirname
        moduloref = []
        modulo = {}
        series = []
        for element in root:
            if element.tag.endswith('BinaryOnly'):
                # TODO: load OME-XML from master or companion file
                log_warning(
                    f'{self!r} OME series is BinaryOnly, '
                    'not an OME-TIFF master file '
                )
                break
            if element.tag.endswith('StructuredAnnotations'):
                for annot in element:
                    if not annot.attrib.get('Namespace', '').endswith(
                        'modulo'
                    ):
                        continue
                    modulo[annot.attrib['ID']] = mod = {}
                    for value in annot:
                        for modul in value:
                            for along in modul:
                                if not along.tag[:-1].endswith('Along'):
                                    continue
                                axis = along.tag[-1]
                                newaxis = along.attrib.get('Type', 'other')
                                newaxis = TIFF.AXES_LABELS[newaxis]
                                if 'Start' in along.attrib:
                                    step = float(along.attrib.get('Step', 1))
                                    start = float(along.attrib['Start'])
                                    stop = float(along.attrib['End']) + step
                                    labels = numpy.arange(start, stop, step)
                                else:
                                    labels = [
                                        label.text
                                        for label in along
                                        if label.tag.endswith('Label')
                                    ]
                                mod[axis] = (newaxis, labels)

            if not element.tag.endswith('Image'):
                continue

            for annot in element:
                if annot.tag.endswith('AnnotationRef'):
                    annotationref = annot.attrib['ID']
                    break
            else:
                annotationref = None

            attr = element.attrib
            name = attr.get('Name', None)

            for pixels in element:
                if not pixels.tag.endswith('Pixels'):
                    continue
                attr = pixels.attrib
                # dtype = attr.get('PixelType', None)
                axes = ''.join(reversed(attr['DimensionOrder']))
                shape = [int(attr['Size' + ax]) for ax in axes]
                ifds = []
                spp = 1  # samples per pixel
                first = True

                for data in pixels:
                    if data.tag.endswith('Channel'):
                        attr = data.attrib
                        if first:
                            first = False
                            spp = int(attr.get('SamplesPerPixel', spp))
                            if spp > 1:
                                # correct channel dimension for spp
                                shape = [
                                    shape[i] // spp if ax == 'C' else shape[i]
                                    for i, ax in enumerate(axes)
                                ]
                        elif int(attr.get('SamplesPerPixel', 1)) != spp:
                            raise ValueError(
                                'OME series cannot handle differing '
                                'SamplesPerPixel'
                            )
                        continue

                    if not data.tag.endswith('TiffData'):
                        continue

                    attr = data.attrib
                    ifd = int(attr.get('IFD', 0))
                    num = int(attr.get('NumPlanes', 1 if 'IFD' in attr else 0))
                    num = int(attr.get('PlaneCount', num))
                    idx = [int(attr.get('First' + ax, 0)) for ax in axes[:-2]]
                    try:
                        idx = int(numpy.ravel_multi_index(idx, shape[:-2]))
                    except ValueError:
                        # ImageJ produces invalid ome-xml when cropping
                        log_warning(
                            f'{self!r} '
                            'OME series contains invalid TiffData index'
                        )
                        continue
                    for uuid in data:
                        if not uuid.tag.endswith('UUID'):
                            continue
                        if root_uuid is None and uuid.text is not None:
                            # no global UUID, use this file
                            root_uuid = uuid.text
                            self._files[root_uuid] = self._files[None]
                        elif uuid.text not in self._files:
                            if not self._multifile:
                                # abort reading multifile OME series
                                # and fall back to generic series
                                return []
                            fname = uuid.attrib['FileName']
                            try:
                                tif = TiffFile(
                                    os.path.join(dirname, fname), _parent=self
                                )
                                tif.pages.cache = True
                                tif.pages.useframes = True
                                tif.pages.keyframe = 0
                                tif.pages._load(keyframe=None)
                            except (OSError, FileNotFoundError, ValueError):
                                log_warning(
                                    f'{self!r} OME series failed to read '
                                    f'{fname!r}'
                                )
                                # assume that size is same as in previous file
                                # if no NumPlanes or PlaneCount are given
                                size = num if num else size  # noqa: undefined
                                ifds.extend([None] * (size + idx - len(ifds)))
                                break
                            self._files[uuid.text] = tif
                            tif.close()
                        pages = self._files[uuid.text].pages
                        try:
                            size = num if num else len(pages)
                            ifds.extend([None] * (size + idx - len(ifds)))
                            for i in range(size):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            log_warning(
                                f'{self!r} '
                                'OME series contains index out of range'
                            )
                        # only process first UUID
                        break
                    else:
                        # no uuid found
                        pages = self.pages
                        try:
                            size = num if num else len(pages)
                            ifds.extend([None] * (size + idx - len(ifds)))
                            for i in range(size):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            log_warning(
                                f'{self!r} '
                                'OME series contains index out of range'
                            )

                if not ifds or all(i is None for i in ifds):
                    # skip images without data
                    continue

                # find a keyframe
                keyframe = None
                for ifd in ifds:
                    # try find a TiffPage
                    if ifd and ifd == ifd.keyframe:
                        keyframe = ifd
                        break
                if keyframe is None:
                    # reload a TiffPage from file
                    for i, keyframe in enumerate(ifds):
                        if keyframe:
                            isclosed = keyframe.parent.filehandle.closed
                            if isclosed:
                                keyframe.parent.filehandle.open()
                            keyframe.parent.pages.keyframe = keyframe.index
                            keyframe = keyframe.parent.pages[keyframe.index]
                            ifds[i] = keyframe
                            if isclosed:
                                keyframe.parent.filehandle.close()
                            break

                # does the series spawn multiple files
                multifile = False
                for ifd in ifds:
                    if ifd and ifd.parent != keyframe.parent:
                        multifile = True
                        break

                if spp > 1:
                    if keyframe.planarconfig == 1:
                        shape += [spp]
                        axes += 'S'
                    else:
                        shape = shape[:-2] + [spp] + shape[-2:]
                        axes = axes[:-2] + 'S' + axes[-2:]
                if 'S' not in shape:
                    shape += [1]
                    axes += 'S'

                # there might be more pages in the file than referenced in XML
                # e.g. Nikon-cell011.ome.tif
                size = max(product(shape) // keyframe.size, 1)
                if size != len(ifds):
                    log_warning(
                        f'{self!r} '
                        f'OME series expected {size} frames, got {len(ifds)}'
                    )
                    ifds = ifds[:size]

                # FIXME: this implementation assumes the last dimensions are
                # stored in TIFF pages. Apparently that is not always the case.
                # E.g. TCX (20000, 2, 500) is stored in 2 pages of (20000, 500)
                # in 'Image 7.ome_h00.tiff'.
                # For now, verify that shapes of keyframe and series match.
                # If not, skip series.
                squeezed = squeeze_axes(shape, axes)[0]
                if keyframe.shape != tuple(squeezed[-len(keyframe.shape) :]):
                    log_warning(
                        f'{self!r} OME series '
                        'cannot handle discontiguous storage (%s != %s)',
                        keyframe.shape,
                        tuple(squeezed[-len(keyframe.shape) :]),
                    )
                    del ifds
                    continue

                # set keyframe on all IFDs
                keyframes = {keyframe.parent.filehandle.name: keyframe}
                for i, page in enumerate(ifds):
                    if page is None:
                        continue
                    fh = page.parent.filehandle
                    if fh.name not in keyframes:
                        if page.keyframe != page:
                            # reload TiffPage from file
                            isclosed = fh.closed
                            if isclosed:
                                fh.open()
                            page.parent.pages.keyframe = page.index
                            page = page.parent.pages[page.index]
                            ifds[i] = page
                            if isclosed:
                                fh.close()
                        keyframes[fh.name] = page
                    if page.keyframe != page:
                        page.keyframe = keyframes[fh.name]

                moduloref.append(annotationref)
                series.append(
                    TiffPageSeries(
                        ifds,
                        shape,
                        keyframe.dtype,
                        axes,
                        parent=self,
                        name=name,
                        multifile=multifile,
                        kind='OME',
                    )
                )
                del ifds

        for serie, annotationref in zip(series, moduloref):
            if annotationref not in modulo:
                continue
            shape = list(serie.get_shape(False))
            axes = serie.get_axes(False)
            for axis, (newaxis, labels) in modulo[annotationref].items():
                i = axes.index(axis)
                size = len(labels)
                if shape[i] == size:
                    axes = axes.replace(axis, newaxis, 1)
                else:
                    shape[i] //= size
                    shape.insert(i + 1, size)
                    axes = axes.replace(axis, axis + newaxis, 1)
            serie.set_shape_axes(shape, axes)

        # pyramids
        for serie in series:
            keyframe = serie.keyframe
            if keyframe.subifds is None:
                continue
            if len(self._files) > 1:
                # TODO: support multi-file pyramids; must re-open/close
                log_warning(
                    f'{self!r} OME series cannot read multi-file pyramids'
                )
                break
            for level in range(len(keyframe.subifds)):
                keyframe = None
                ifds = []
                for page in serie.pages:
                    if page is None:
                        ifds.append(None)
                        continue
                    page.parent.filehandle.seek(page.subifds[level])
                    if page.keyframe == page:
                        ifd = keyframe = TiffPage(self, (page.index, level))
                    elif keyframe is None:
                        raise RuntimeError('no keyframe')
                    else:
                        ifd = TiffFrame(self, page.index, keyframe=keyframe)
                    ifds.append(ifd)
                # fix shape
                shape = []
                for i, ax in enumerate(serie.axes):
                    if ax == 'X':
                        shape.append(keyframe.imagewidth)
                    elif ax == 'Y':
                        shape.append(keyframe.imagelength)
                    else:
                        shape.append(serie.shape[i])
                # add series
                serie.levels.append(
                    TiffPageSeries(
                        ifds,
                        tuple(shape),
                        keyframe.dtype,
                        serie.axes,
                        parent=self,
                        name=f'level {level + 1}',
                        kind='OME',
                    )
                )

        self.is_uniform = len(series) == 1 and len(series[0].levels) == 1

        return series

    def _series_stk(self):
        """Return series in STK file."""
        page = self.pages[0]
        meta = self.stk_metadata
        planes = meta['NumberPlanes']
        name = meta.get('Name', '')
        if planes == 1:
            shape = (1,) + page.shape
            axes = 'I' + page.axes
        elif numpy.all(meta['ZDistance'] != 0):
            shape = (planes,) + page.shape
            axes = 'Z' + page.axes
        elif numpy.all(numpy.diff(meta['TimeCreated']) != 0):
            shape = (planes,) + page.shape
            axes = 'T' + page.axes
        else:
            # TODO: determine other/combinations of dimensions
            shape = (planes,) + page.shape
            axes = 'I' + page.axes
        self.is_uniform = True
        series = TiffPageSeries(
            [page],
            shape,
            page.dtype,
            axes,
            name=name,
            truncated=planes > 1,
            kind='STK',
        )
        return [series]

    def _series_lsm(self):
        """Return main and thumbnail series in LSM file."""
        lsmi = self.lsm_metadata
        axes = TIFF.CZ_LSMINFO_SCANTYPE[lsmi['ScanType']]
        if self.pages[0].photometric == 2:  # RGB; more than one channel
            axes = axes.replace('C', '').replace('XY', 'XYC')
        if lsmi.get('DimensionP', 0) > 0:
            axes += 'P'
        if lsmi.get('DimensionM', 0) > 0:
            axes += 'M'
        axes = axes[::-1]
        shape = tuple(int(lsmi[TIFF.CZ_LSMINFO_DIMENSIONS[i]]) for i in axes)

        name = lsmi.get('Name', '')
        pages = self.pages._getlist(slice(0, None, 2), validate=False)
        dtype = pages[0].dtype
        series = [
            TiffPageSeries(pages, shape, dtype, axes, name=name, kind='LSM')
        ]

        page = self.pages[1]
        if page.is_reduced:
            pages = self.pages._getlist(slice(1, None, 2), validate=False)
            dtype = page.dtype
            cp = 1
            i = 0
            while cp < len(pages) and i < len(shape) - 2:
                cp *= shape[i]
                i += 1
            shape = shape[:i] + page.shape
            axes = axes[:i] + 'SYX'
            series.append(
                TiffPageSeries(
                    pages, shape, dtype, axes, name=name, kind='LSMreduced'
                )
            )

        self.is_uniform = False
        return series

    def _lsm_load_pages(self):
        """Load and fix all pages from LSM file."""
        # cache all pages to preserve corrected values
        pages = self.pages
        pages.cache = True
        pages.useframes = True
        # use first and second page as keyframes
        pages.keyframe = 1
        pages.keyframe = 0
        # load remaining pages as frames
        pages._load(keyframe=None)
        # fix offsets and bytecounts first
        # TODO: fix multiple conversions between lists and tuples
        self._lsm_fix_strip_offsets()
        self._lsm_fix_strip_bytecounts()
        # assign keyframes for data and thumbnail series
        keyframe = pages[0]
        for page in pages[::2]:
            page.keyframe = keyframe
        keyframe = pages[1]
        for page in pages[1::2]:
            page.keyframe = keyframe

    def _lsm_fix_strip_offsets(self):
        """Unwrap strip offsets for LSM files greater than 4 GB.

        Each series and position require separate unwrapping (undocumented).

        """
        if self.filehandle.size < 2 ** 32:
            return

        pages = self.pages
        npages = len(pages)
        series = self.series[0]
        axes = series.axes

        # find positions
        positions = 1
        for i in 0, 1:
            if series.axes[i] in 'PM':
                positions *= series.shape[i]

        # make time axis first
        if positions > 1:
            ntimes = 0
            for i in 1, 2:
                if axes[i] == 'T':
                    ntimes = series.shape[i]
                    break
            if ntimes:
                div, mod = divmod(npages, 2 * positions * ntimes)
                if mod != 0:
                    raise RuntimeError('mod != 0')
                shape = (positions, ntimes, div, 2)
                indices = numpy.arange(product(shape)).reshape(shape)
                indices = numpy.moveaxis(indices, 1, 0)
        else:
            indices = numpy.arange(npages).reshape(-1, 2)

        # images of reduced page might be stored first
        if pages[0].dataoffsets[0] > pages[1].dataoffsets[0]:
            indices = indices[..., ::-1]

        # unwrap offsets
        wrap = 0
        previousoffset = 0
        for i in indices.flat:
            page = pages[int(i)]
            dataoffsets = []
            for currentoffset in page.dataoffsets:
                if currentoffset < previousoffset:
                    wrap += 2 ** 32
                dataoffsets.append(currentoffset + wrap)
                previousoffset = currentoffset
            page.dataoffsets = tuple(dataoffsets)

    def _lsm_fix_strip_bytecounts(self):
        """Set databytecounts to size of compressed data.

        The StripByteCounts tag in LSM files contains the number of bytes
        for the uncompressed data.

        """
        pages = self.pages
        if pages[0].compression == 1:
            return
        # sort pages by first strip offset
        pages = sorted(pages, key=lambda p: p.dataoffsets[0])
        npages = len(pages) - 1
        for i, page in enumerate(pages):
            if page.index % 2:
                continue
            offsets = page.dataoffsets
            bytecounts = page.databytecounts
            if i < npages:
                lastoffset = pages[i + 1].dataoffsets[0]
            else:
                # LZW compressed strips might be longer than uncompressed
                lastoffset = min(
                    offsets[-1] + 2 * bytecounts[-1], self._fh.size
                )
            bytecounts = list(bytecounts)
            for j in range(len(bytecounts) - 1):
                bytecounts[j] = offsets[j + 1] - offsets[j]
            bytecounts[-1] = lastoffset - offsets[-1]
            page.databytecounts = tuple(bytecounts)

    def _ndpi_load_pages(self):
        """Load and fix pages from NDPI slide file if CaptureMode > 6.

        If the value of the CaptureMode tag is greater than 6, change the
        attributes of the TiffPages that are part of the pyramid to match
        16-bit grayscale data. TiffTags are not corrected.

        """
        pages = self.pages
        capturemode = pages[0].tags.valueof(65441)
        if capturemode is None or capturemode < 6:
            return

        pages.cache = True
        pages.useframes = False
        pages._load()

        for page in pages:
            mag = page.tags.valueof(65421)
            if mag is None or mag > 0:
                page.photometric = TIFF.PHOTOMETRIC.MINISBLACK
                page.samplesperpixel = 1
                page.sampleformat = 1
                page.bitspersample = 16
                page.dtype = page._dtype = numpy.dtype('uint16')
                if page.shaped[-1] > 1:
                    page.axes = page.axes[:-1]
                    page.shape = page.shape[:-1]
                    page.shaped = page.shaped[:-1] + (1,)

    def _philips_load_pages(self):
        """Load and fix all pages from Philips slide file.

        The imagewidth and imagelength values of all tiled pages are corrected
        using the DICOM_PIXEL_SPACING attributes of the XML formatted
        description of the first page.

        """
        from xml.etree import ElementTree as etree  # delayed import

        pages = self.pages
        pages.cache = True
        pages.useframes = False
        pages._load()
        npages = len(pages)

        root = etree.fromstring(pages[0].description)

        imagewidth = pages[0].imagewidth
        imagelength = pages[0].imagelength
        sizes = None
        for elem in root.iter():
            if (
                elem.tag != 'Attribute'
                or elem.attrib.get('Name', '') != 'DICOM_PIXEL_SPACING'
            ):
                continue
            w, h = (float(v) for v in elem.text.replace('"', '').split())
            if sizes is None:
                imagelength *= h
                imagewidth *= w
                sizes = []
            else:
                sizes.append(
                    (
                        int(math.ceil(imagelength / h)),
                        int(math.ceil(imagewidth / w)),
                    )
                )

        i = 0
        for imagelength, imagewidth in sizes:
            while i < npages and pages[i].tilewidth == 0:
                # Label, Macro
                i += 1
                continue
            if i == npages:
                break
            page = pages[i]
            page.imagewidth = imagewidth
            page.imagelength = imagelength
            if page.shaped[-1] > 1:
                page.shape = (imagelength, imagewidth, page.shape[-1])
            elif page.shaped[0] > 1:
                page.shape = (page.shape[0], imagelength, imagewidth)
            else:
                page.shape = (imagelength, imagewidth)
            page.shaped = (
                page.shaped[:2] + (imagelength, imagewidth) + page.shaped[-1:]
            )
            i += 1

    def __getattr__(self, name):
        """Return 'is_flag' attributes from first page."""
        if name[3:] in TIFF.FILE_FLAGS:
            if not self.pages:
                return False
            value = bool(getattr(self.pages[0], name))
            setattr(self, name, value)
            return value
        raise AttributeError(
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return f'<tifffile.TiffFile {snipstr(self._fh.name, 32)!r}>'

    def __str__(self, detail=0, width=79):
        """Return string containing information about TiffFile.

        The detail parameter specifies the level of detail returned:

        0: file only.
        1: all series, first page of series and its tags.
        2: large tag values and file metadata.
        3: all pages.

        """
        info = [
            "TiffFile '{}'",
            format_size(self._fh.size),
            ''
            if byteorder_isnative(self.byteorder)
            else {'<': 'little-endian', '>': 'big-endian'}[self.byteorder],
        ]
        if self.is_bigtiff:
            info.append('BigTiff')
        info.append(' '.join(f.lower() for f in self.flags))
        if len(self.pages) > 1:
            info.append(f'{len(self.pages)} Pages')
        if len(self.series) > 1:
            info.append(f'{len(self.series)} Series')
        if len(self._files) > 1:
            info.append(f'{len(self._files)} Files')
        info = '  '.join(info)
        info = info.replace('    ', '  ').replace('   ', '  ')
        info = info.format(
            snipstr(self._fh.name, max(12, width + 2 - len(info)))
        )
        if detail <= 0:
            return info
        info = [info]
        info.append('\n'.join(str(s) for s in self.series))
        if detail >= 3:
            for p in self.pages:
                if p is None:
                    continue
                info.append(TiffPage.__str__(p, detail=detail, width=width))
                for s in p.pages:
                    info.append(
                        TiffPage.__str__(s, detail=detail, width=width)
                    )
        elif self.series:
            info.extend(
                TiffPage.__str__(s.keyframe, detail=detail, width=width)
                for s in self.series
                if not s.keyframe.parent.filehandle.closed  # avoid warning
            )
        elif self.pages and self.pages[0]:
            info.append(
                TiffPage.__str__(self.pages[0], detail=detail, width=width)
            )
        if detail >= 2:
            for name in sorted(self.flags):
                if hasattr(self, name + '_metadata'):
                    m = getattr(self, name + '_metadata')
                    if m:
                        info.append(
                            '{}_METADATA\n{}'.format(
                                name.upper(),
                                pformat(m, width=width, height=detail * 24),
                            )
                        )
        return '\n\n'.join(info).replace('\n\n\n', '\n\n')

    @lazyattr
    def flags(self):
        """Return set of file flags, a potentially expensive operation."""
        return {
            name.lower()
            for name in sorted(TIFF.FILE_FLAGS)
            if getattr(self, 'is_' + name)
        }

    @property
    def is_bigtiff(self):
        """Return if file has BigTIFF format."""
        return self.tiff.version == 43

    @lazyattr
    def is_mdgel(self):
        """Return if file has MD Gel format."""
        # side effect: add second page, if exists, to cache
        try:
            ismdgel = (
                self.pages[0].is_mdgel
                or self.pages.get(1, cache=True).is_mdgel
            )
            if ismdgel:
                self.is_uniform = False
            return ismdgel
        except IndexError:
            return False

    @lazyattr
    def is_uniform(self):
        """Return if file contains a uniform series of pages."""
        # the hashes of IFDs 0, 7, and -1 are the same
        pages = self.pages
        page = pages[0]
        if page.subifds:
            return False
        if page.is_scanimage or page.is_nih:
            return True
        try:
            useframes = pages.useframes
            pages.useframes = False
            h = page.hash
            for i in (1, 7, -1):
                if pages[i].aspage().hash != h:
                    return False
        except IndexError:
            return False
        finally:
            pages.useframes = useframes
        return True

    @property
    def is_appendable(self):
        """Return if pages can be appended to file without corrupting."""
        # TODO: check other formats
        return not (
            self.is_ome
            or self.is_lsm
            or self.is_stk
            or self.is_imagej
            or self.is_fluoview
            or self.is_micromanager
        )

    @lazyattr
    def shaped_metadata(self):
        """Return tifffile metadata from JSON descriptions as dicts."""
        if not self.is_shaped:
            return None
        return tuple(
            json_description_metadata(s.pages[0].is_shaped)
            for s in self.series
            if s.kind.lower() == 'shaped'
        )

    @property
    def ome_metadata(self):
        """Return OME XML."""
        if not self.is_ome:
            return None
        # return xml2dict(self.pages[0].description)['OME']
        return self.pages[0].description

    @property
    def scn_metadata(self):
        """Return Leica SCN XML."""
        if not self.is_scn:
            return None
        return self.pages[0].description

    @property
    def philips_metadata(self):
        """Return Philips DP XML."""
        if not self.is_philips:
            return None
        return self.pages[0].description

    @property
    def lsm_metadata(self):
        """Return LSM metadata from CZ_LSMINFO tag as dict."""
        if not self.is_lsm:
            return None
        return self.pages[0].tags.valueof(34412)  # CZ_LSMINFO

    @lazyattr
    def stk_metadata(self):
        """Return STK metadata from UIC tags as dict."""
        if not self.is_stk:
            return None
        page = self.pages[0]
        tags = page.tags
        result = {}
        if page.description:
            result['PlaneDescriptions'] = page.description.split('\x00')
            # result['plane_descriptions'] = stk_description_metadata(
            #    page.image_description)
        tag = tags.get(33629)  # UIC2tag
        result['NumberPlanes'] = 1 if tag is None else tag.count
        value = tags.valueof(33628)  # UIC1tag
        if value is not None:
            result.update(value)
        value = tags.valueof(33630)  # UIC3tag
        if value is not None:
            result.update(value)  # wavelengths
        value = tags.valueof(33631)  # UIC4tag
        if value is not None:
            result.update(value)  # override UIC1 tags
        uic2tag = tags.valueof(33629)
        if uic2tag is not None:
            result['ZDistance'] = uic2tag['ZDistance']
            result['TimeCreated'] = uic2tag['TimeCreated']
            result['TimeModified'] = uic2tag['TimeModified']
            try:
                result['DatetimeCreated'] = numpy.array(
                    [
                        julian_datetime(*dt)
                        for dt in zip(
                            uic2tag['DateCreated'], uic2tag['TimeCreated']
                        )
                    ],
                    dtype='datetime64[ns]',
                )
                result['DatetimeModified'] = numpy.array(
                    [
                        julian_datetime(*dt)
                        for dt in zip(
                            uic2tag['DateModified'], uic2tag['TimeModified']
                        )
                    ],
                    dtype='datetime64[ns]',
                )
            except ValueError as exc:
                log_warning(
                    f'{self!r} STK metadata failed with '
                    f'{exc.__class__.__name__}: {exc}'
                )
        return result

    @lazyattr
    def imagej_metadata(self):
        """Return consolidated ImageJ metadata as dict."""
        if not self.is_imagej:
            return None
        page = self.pages[0]
        result = imagej_description_metadata(page.is_imagej)
        value = page.tags.valueof(50839)  # IJMetadata
        if value is not None:
            try:
                result.update(value)
            except Exception:
                pass
        return result

    @lazyattr
    def fluoview_metadata(self):
        """Return consolidated FluoView metadata as dict."""
        if not self.is_fluoview:
            return None
        result = {}
        page = self.pages[0]
        value = page.tags.valueof(34361)  # MM_Header
        if value is not None:
            result.update(value)
        # TODO: read stamps from all pages
        value = page.tags.valueof(34362)  # MM_Stamp
        if value is not None:
            result['Stamp'] = value
        # skip parsing image description; not reliable
        # try:
        #     t = fluoview_description_metadata(page.image_description)
        #     if t is not None:
        #         result['ImageDescription'] = t
        # except Exception as exc:
        #     log_warning(
        #         f'{self!r} fluoview_description_metadata failed with'
        #         f'{exc.__class__.__name__}: {exc}'
        #     )
        return result

    @property
    def nih_metadata(self):
        """Return NIH Image metadata from NIHImageHeader tag as dict."""
        if not self.is_nih:
            return None
        return self.pages[0].tags.valueof(43314)  # NIHImageHeader

    @property
    def fei_metadata(self):
        """Return FEI metadata from SFEG or HELIOS tags as dict."""
        if not self.is_fei:
            return None
        tags = self.pages[0].tags
        return tags.valueof(34680, tags.valueof(34682))  # FEI_SFEG, FEI_HELIOS

    @property
    def sem_metadata(self):
        """Return SEM metadata from CZ_SEM tag as dict."""
        if not self.is_sem:
            return None
        return self.pages[0].tags.valueof(34118)

    @property
    def sis_metadata(self):
        """Return Olympus SIS metadata from SIS and INI tags as dict."""
        if not self.is_sis:
            return None
        tags = self.pages[0].tags
        result = {}
        try:
            result.update(tags.valueof(33471))  # OlympusINI
        except Exception:
            pass
        try:
            result.update(tags.valueof(33560))  # OlympusSIS
        except Exception:
            pass
        return result

    @lazyattr
    def mdgel_metadata(self):
        """Return consolidated metadata from MD GEL tags as dict."""
        for page in self.pages[:2]:
            if 33445 in page.tags:  # MDFileTag
                tags = page.tags
                break
        else:
            return None
        result = {}
        for code in range(33445, 33453):
            if code not in tags:
                continue
            name = TIFF.TAGS[code]
            result[name[2:]] = tags.valueof(code)
        return result

    @property
    def andor_metadata(self):
        """Return Andor tags as dict."""
        return self.pages[0].andor_tags

    @property
    def epics_metadata(self):
        """Return EPICS areaDetector tags as dict."""
        return self.pages[0].epics_tags

    @property
    def tvips_metadata(self):
        """Return TVIPS tag as dict."""
        if not self.is_tvips:
            return None
        return self.pages[0].tags.valueof(37706)

    @lazyattr
    def metaseries_metadata(self):
        """Return MetaSeries metadata from image description as dict."""
        if not self.is_metaseries:
            return None
        return metaseries_description_metadata(self.pages[0].description)

    @lazyattr
    def pilatus_metadata(self):
        """Return Pilatus metadata from image description as dict."""
        if not self.is_pilatus:
            return None
        return pilatus_description_metadata(self.pages[0].description)

    @lazyattr
    def micromanager_metadata(self):
        """Return MicroManager non-TIFF settings from file as dict."""
        if not self.is_micromanager:
            return None
        # from file header
        return read_micromanager_metadata(self._fh)

    @lazyattr
    def scanimage_metadata(self):
        """Return ScanImage non-varying frame and ROI metadata as dict.

        The returned dict may be empty or contain 'FrameData', 'RoiGroups',
        and 'version' keys.

        The varying frame data can be found in the ImageDescription tags.

        """
        if not self.is_scanimage:
            return None
        result = {}
        try:
            framedata, roidata, version = read_scanimage_metadata(self._fh)
            result['version'] = version
            result['FrameData'] = framedata
            result.update(roidata)
        except ValueError:
            pass
        return result

    @property
    def geotiff_metadata(self):
        """Return GeoTIFF metadata from first page as dict."""
        if not self.is_geotiff:
            return None
        return self.pages[0].geotiff_tags

    @property
    def eer_metadata(self):
        """Return EER metadata from first page as XML."""
        if not self.is_eer:
            return None
        value = self.pages[0].tags.valueof(65001)
        return None if value is None else value.decode()


class TiffPages:
    """Sequence of TIFF image file directories (IFD chain).

    Instances of TiffPages have a state (cache, keyframe, etc.) and are not
    thread-safe.

    """

    def __init__(self, arg, index=None):
        """Initialize instance and read first TiffPage from file.

        If arg is a TiffFile, the file position must be at an offset to an
        offset to a TiffPage. If arg is a TiffPage, page offsets are read
        from the SubIFDs tag.

        """
        self.parent = None
        self.pages = []  # cache of TiffPages, TiffFrames, or their offsets
        self._indexed = False  # True if offsets to all pages were read
        self._cached = False  # True if all pages were read into cache
        self._tiffpage = TiffPage  # class used for reading pages
        self._keyframe = None  # page that is currently used as keyframe
        self._cache = False  # do not cache frames or pages (if not keyframe)
        self._offset = 0
        self._nextpageoffset = None
        if isinstance(index, (int, numpy.integer)):
            self._index = (int(index),)
        elif index is None:
            self._index = None
        else:
            self._index = tuple(index)

        if isinstance(arg, TiffFile):
            # read offset to first page from current file position
            self.parent = arg
            fh = self.parent.filehandle
            self._nextpageoffset = fh.tell()
            offset = struct.unpack(
                self.parent.tiff.offsetformat,
                fh.read(self.parent.tiff.offsetsize),
            )[0]
            if offset == 0:
                log_warning(f'{arg!r} contains no pages')
                self._indexed = True
                return
        elif 330 in arg.tags:
            # use offsets from SubIFDs tag
            self.parent = arg.parent
            fh = self.parent.filehandle
            offsets = arg.tags[330].value
            offset = offsets[0]
            if offset == 0:
                log_warning(f'{arg!r} contains invalid SubIFDs')
                self._indexed = True
                return
        else:
            self._indexed = True
            return

        self._offset = offset
        if offset >= fh.size:
            log_warning(f'{self!r} invalid offset to first page {offset!r}')
            self._indexed = True
            return

        pageindex = 0 if self._index is None else self._index + (0,)

        # read and cache first page
        fh.seek(offset)
        page = TiffPage(self.parent, index=pageindex)
        self.pages.append(page)
        self._keyframe = page
        if self._nextpageoffset is None:
            # offsets from SubIFDs tag
            self.pages.extend(offsets[1:])
            self._indexed = True
            self._cached = True

    @property
    def cache(self):
        """Return if pages/frames are currently being cached."""
        return self._cache

    @cache.setter
    def cache(self, value):
        """Enable or disable caching of pages/frames. Clear cache if False."""
        value = bool(value)
        if self._cache and not value:
            self._clear()
        self._cache = value

    @property
    def useframes(self):
        """Return if currently using TiffFrame (True) or TiffPage (False)."""
        return self._tiffpage == TiffFrame and TiffFrame is not TiffPage

    @useframes.setter
    def useframes(self, value):
        """Set to use TiffFrame (True) or TiffPage (False)."""
        self._tiffpage = TiffFrame if value else TiffPage

    @property
    def keyframe(self):
        """Return current keyframe."""
        return self._keyframe

    @keyframe.setter
    def keyframe(self, index):
        """Set current keyframe. Load TiffPage from file if necessary."""
        index = int(index)
        if index < 0:
            index %= len(self)
        if self._keyframe.index == index:
            return
        if index == 0:
            self._keyframe = self.pages[0]
            return
        if self._indexed or index < len(self.pages):
            page = self.pages[index]
            if isinstance(page, TiffPage):
                self._keyframe = page
                return
            if isinstance(page, TiffFrame):
                # remove existing TiffFrame
                self.pages[index] = page.offset
        # load TiffPage from file
        tiffpage = self._tiffpage
        self._tiffpage = TiffPage
        try:
            self._keyframe = self._getitem(index)
        finally:
            self._tiffpage = tiffpage
        # always cache keyframes
        self.pages[index] = self._keyframe

    @property
    def next_page_offset(self):
        """Return offset where offset to a new page can be stored."""
        if not self._indexed:
            self._seek(-1)
        return self._nextpageoffset

    def get(self, key, default=None, validate=False, cache=None, aspage=True):
        """Return specified page from cache or file."""
        try:
            return self._getitem(
                key, validate=validate, cache=cache, aspage=aspage
            )
        except IndexError:
            if default is None:
                raise
        return default

    def _load(self, keyframe=True):
        """Read all remaining pages from file."""
        if self._cached:
            return
        pages = self.pages
        if not pages:
            return
        if not self._indexed:
            self._seek(-1)
        if not self._cache:
            return
        fh = self.parent.filehandle
        if keyframe is not None:
            keyframe = self._keyframe
        for i, page in enumerate(pages):
            if isinstance(page, (int, numpy.integer)):
                pageindex = i if self._index is None else self._index + (i,)
                fh.seek(page)
                page = self._tiffpage(
                    self.parent, index=pageindex, keyframe=keyframe
                )
                pages[i] = page
        self._cached = True

    def _load_virtual_frames(self):
        """Calculate virtual TiffFrames."""
        pages = self.pages
        try:
            if len(pages) > 1:
                raise ValueError('pages already loaded')
            page = pages[0]
            if not page.is_contiguous:
                raise ValueError('data not contiguous')
            self._seek(4)
            delta = pages[2] - pages[1]
            if pages[3] - pages[2] != delta or pages[4] - pages[3] != delta:
                raise ValueError('page offsets not equidistant')
            page1 = self._getitem(1, validate=page.hash)
            offsetoffset = page1.dataoffsets[0] - page1.offset
            if offsetoffset < 0 or offsetoffset > delta:
                raise ValueError('page offsets not equidistant')
            pages = [page, page1]
            filesize = self.parent.filehandle.size - delta

            for index, offset in enumerate(
                range(page1.offset + delta, filesize, delta)
            ):
                pageindex = index + 2
                d = pageindex * delta
                offsets = tuple(i + d for i in page.dataoffsets)
                offset = offset if offset < 2 ** 31 - 1 else None
                if self._index is not None:
                    pageindex = self._index + (pageindex,)
                pages.append(
                    TiffFrame(
                        parent=page.parent,
                        index=pageindex,
                        offset=offset,
                        offsets=offsets,
                        bytecounts=page.databytecounts,
                        keyframe=page,
                    )
                )
            self.pages = pages
            self._cache = True
            self._cached = True
            self._indexed = True
        except Exception as exc:
            if self.parent.filehandle.size >= 2147483648:
                log_warning(
                    f'{self!r} _load_virtual_frames failed with '
                    f'({exc.__class__.__name__}: {exc})'
                )

    def _clear(self, fully=True):
        """Delete all but first page from cache. Set keyframe to first page."""
        pages = self.pages
        if not pages:
            return
        self._keyframe = pages[0]
        if fully:
            # delete all but first TiffPage/TiffFrame
            for i, page in enumerate(pages[1:]):
                if not isinstance(page, int) and page.offset is not None:
                    pages[i + 1] = page.offset
        elif TiffFrame is not TiffPage:
            # delete only TiffFrames
            for i, page in enumerate(pages):
                if isinstance(page, TiffFrame) and page.offset is not None:
                    pages[i] = page.offset
        self._cached = False

    def _seek(self, index, maxpages=None):
        """Seek file to offset of page specified by index."""
        pages = self.pages
        lenpages = len(pages)
        if lenpages == 0:
            raise IndexError('index out of range')

        fh = self.parent.filehandle
        if fh.closed:
            raise ValueError('seek of closed file')

        if self._indexed or 0 <= index < lenpages:
            page = pages[index]
            offset = page if isinstance(page, int) else page.offset
            fh.seek(offset)
            return

        tiff = self.parent.tiff
        offsetformat = tiff.offsetformat
        offsetsize = tiff.offsetsize
        tagnoformat = tiff.tagnoformat
        tagnosize = tiff.tagnosize
        tagsize = tiff.tagsize
        unpack = struct.unpack

        page = pages[-1]
        offset = page if isinstance(page, int) else page.offset

        if maxpages is None:
            maxpages = 2 ** 22
        while lenpages < maxpages:
            # read offsets to pages from file until index is reached
            fh.seek(offset)
            # skip tags
            try:
                tagno = unpack(tagnoformat, fh.read(tagnosize))[0]
                if tagno > 4096:
                    raise TiffFileError(f'suspicious number of tags {tagno!r}')
            except Exception:
                log_warning(
                    f'{self!r} corrupted tag list of page '
                    f'{lenpages} @{offset}'
                )
                del pages[-1]
                lenpages -= 1
                self._indexed = True
                break
            self._nextpageoffset = offset + tagnosize + tagno * tagsize
            fh.seek(self._nextpageoffset)

            # read offset to next page
            try:
                offset = unpack(offsetformat, fh.read(offsetsize))[0]
            except Exception:
                log_warning(
                    f'{self!r} invalid offset to page '
                    f'{lenpages + 1} @{self._nextpageoffset}'
                )
                self._indexed = True
                break
            if offset == 0:
                self._indexed = True
                break
            if offset >= fh.size:
                log_warning(f'{self!r} invalid page offset {offset!r}')
                self._indexed = True
                break

            pages.append(offset)
            lenpages += 1
            if 0 <= index < lenpages:
                break

            # detect some circular references
            if lenpages == 100:
                for p in pages[:-1]:
                    if offset == (p if isinstance(p, int) else p.offset):
                        raise TiffFileError('invalid circular IFD reference')

        if index >= lenpages:
            raise IndexError('index out of range')

        page = pages[index]
        fh.seek(page if isinstance(page, int) else page.offset)

    def _getlist(self, key=None, useframes=True, validate=True):
        """Return specified pages as list of TiffPages or TiffFrames.

        The first item is a TiffPage, and is used as a keyframe for
        following TiffFrames.

        """
        getitem = self._getitem
        _useframes = self.useframes

        if key is None:
            key = iter(range(len(self)))
        elif isinstance(key, Iterable):
            key = iter(key)
        elif isinstance(key, slice):
            start, stop, _ = key.indices(2 ** 31 - 1)
            if not self._indexed and max(stop, start) > len(self.pages):
                self._seek(-1)
            key = iter(range(*key.indices(len(self.pages))))
        elif isinstance(key, (int, numpy.integer)):
            # return single TiffPage
            self.useframes = False
            if key == 0:
                return [self.pages[key]]
            try:
                return [getitem(key)]
            finally:
                self.useframes = _useframes
        else:
            raise TypeError('key must be an integer, slice, or iterable')

        # use first page as keyframe
        keyframe = self._keyframe
        self.keyframe = next(key)
        if validate:
            validate = self._keyframe.hash
        if useframes:
            self.useframes = True
        try:
            pages = [getitem(i, validate) for i in key]
            pages.insert(0, self._keyframe)
        finally:
            # restore state
            self._keyframe = keyframe
            if useframes:
                self.useframes = _useframes

        return pages

    def _getitem(self, key, validate=False, cache=None, aspage=None):
        """Return specified page from cache or file."""
        key = int(key)
        pages = self.pages

        if key < 0:
            key %= len(self)
        elif self._indexed and key >= len(pages):
            raise IndexError(f'index {key} out of range({len(pages)})')

        tiffpage = TiffPage if aspage else self._tiffpage

        if key < len(pages):
            page = pages[key]
            if self._cache and not aspage:
                if not isinstance(page, (int, numpy.integer)):
                    if validate and validate != page.hash:
                        raise RuntimeError('page hash mismatch')
                    return page
            elif isinstance(page, (TiffPage, tiffpage)):
                if validate and validate != page.hash:
                    raise RuntimeError('page hash mismatch')
                return page

        pageindex = key if self._index is None else self._index + (key,)
        self._seek(key)
        page = tiffpage(self.parent, index=pageindex, keyframe=self._keyframe)
        if validate and validate != page.hash:
            raise RuntimeError('page hash mismatch')
        if self._cache or cache:
            pages[key] = page
        return page

    def __getitem__(self, key):
        """Return specified page(s)."""
        pages = self.pages
        getitem = self._getitem

        if isinstance(key, (int, numpy.integer)):
            if key == 0:
                return pages[key]
            return getitem(key)

        if isinstance(key, slice):
            start, stop, _ = key.indices(2 ** 31 - 1)
            if not self._indexed and max(stop, start) > len(pages):
                self._seek(-1)
            return [getitem(i) for i in range(*key.indices(len(pages)))]

        if isinstance(key, Iterable):
            return [getitem(k) for k in key]

        raise TypeError('key must be an integer, slice, or iterable')

    def __iter__(self):
        """Return iterator over all pages."""
        i = 0
        while True:
            try:
                yield self._getitem(i)
                i += 1
            except IndexError:
                break
        if self._cache:
            self._cached = True

    def __bool__(self):
        """Return True if file contains any pages."""
        return len(self.pages) > 0

    def __len__(self):
        """Return number of pages in file."""
        if not self._indexed:
            self._seek(-1)
        return len(self.pages)

    def __repr__(self):
        return f'<tifffile.TiffPages @{self._offset}>'


class TiffPage:
    """TIFF image file directory (IFD).

    Attributes
    ----------
    index : int or tuple of int
        Index of the page in file.
    dtype : numpy.dtype or None
        Data type (native byte order) of the image in IFD.
    shape : tuple of int
        Dimensions of the image in IFD, as returned by asarray.
    axes : str
        Axes label codes for each dimension in shape:
        'S' sample,
        'X' width,
        'Y' length,
        'Z' depth,
    tags : TiffTags
        Multidict like interface to tags in IFD.
    colormap : numpy.ndarray
        Color look up table, if exists.
    shaped : tuple of int
        Normalized 5-dimensional shape of the image in IFD:
        0 : separate samplesperpixel or 1.
        1 : imagedepth Z or 1.
        2 : imagelength Y.
        3 : imagewidth X.
        4 : contig samplesperpixel or 1.

    All attributes are read-only.

    """

    # default properties; will be updated from tags
    subfiletype = 0
    imagewidth = 0
    imagelength = 0
    imagedepth = 1
    tilewidth = 0
    tilelength = 0
    tiledepth = 1
    bitspersample = 1
    samplesperpixel = 1
    sampleformat = 1
    rowsperstrip = 2 ** 32 - 1
    compression = 1
    planarconfig = 1
    fillorder = 1
    photometric = 0
    predictor = 1
    extrasamples = ()
    subsampling = None
    subifds = None
    jpegtables = None
    jpegheader = None  # NDPI only
    software = ''
    description = ''
    description1 = ''
    nodata = 0

    def __init__(self, parent, index, keyframe=None):
        """Initialize instance from file.

        The file handle position must be at offset to a valid IFD.

        """
        self.parent = parent
        self.index = index
        self.shape = ()
        self.shaped = ()
        self.dtype = None
        self._dtype = None
        self.axes = ''
        self.tags = tags = TiffTags()
        self.dataoffsets = ()
        self.databytecounts = ()

        tiff = parent.tiff

        # read TIFF IFD structure and its tags from file
        fh = parent.filehandle
        self.offset = fh.tell()  # offset to this IFD
        try:
            tagno = struct.unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
            if tagno > 4096:
                raise ValueError(f'suspicious number of tags {tagno}')
        except Exception as exc:
            raise TiffFileError(
                f'corrup tag list at offset {self.offset}'
            ) from exc

        tagoffset = self.offset + tiff.tagnosize  # fh.tell()
        tagsize = tagsize_ = tiff.tagsize

        data = fh.read(tagsize * tagno)
        if len(data) != tagsize * tagno:
            raise TiffFileError('corrupted IFD structure')
        if tiff.version == 42 and tiff.offsetsize == 8:
            # patch offsets/values for 64-bit NDPI file
            tagsize = 16
            fh.seek(8, os.SEEK_CUR)
            ext = fh.read(4 * tagno)  # high bits
            data = b''.join(
                data[i * 12 : i * 12 + 12] + ext[i * 4 : i * 4 + 4]
                for i in range(tagno)
            )

        tagindex = -tagsize
        for i in range(tagno):
            tagindex += tagsize
            tagdata = data[tagindex : tagindex + tagsize]
            try:
                tag = TiffTag.fromfile(
                    parent, tagoffset + i * tagsize_, tagdata
                )
            except TiffFileError as exc:
                log_warning(f'{self!r} {exc}')
                continue
            tags.add(tag)

        if not tags:
            return  # found in FIBICS

        for code, name in TIFF.TAG_ATTRIBUTES.items():
            value = tags.valueof(code)
            if value is None:
                continue
            if (code == 270 or code == 305) and not isinstance(value, str):
                # wrong string type for software or description
                continue
            setattr(self, name, value)

        value = tags.valueof(270, index=1)
        if value:
            self.description1 = value

        if self.subfiletype == 0:
            value = tags.valueof(255)  # SubfileType
            if value == 2:
                self.subfiletype = 0b1  # reduced image
            elif value == 3:
                self.subfiletype = 0b10  # multi-page

        # consolidate private tags; remove them from self.tags
        # if self.is_andor:
        #     self.andor_tags
        # elif self.is_epics:
        #     self.epics_tags
        # elif self.is_ndpi:
        #     self.ndpi_tags
        # if self.is_sis and 34853 in tags:
        #     # TODO: can't change tag.name
        #     tags[34853].name = 'OlympusSIS2'

        # dataoffsets and databytecounts
        # TileOffsets
        self.dataoffsets = tags.valueof(324)
        if self.dataoffsets is None:
            # StripOffsets
            self.dataoffsets = tags.valueof(273)
            if self.dataoffsets is None:
                # JPEGInterchangeFormat et al.
                self.dataoffsets = tags.valueof(513)
                if self.dataoffsets is None:
                    self.dataoffsets = ()
                    log_warning(f'{self!r} missing data offset tag')
        # TileByteCounts
        self.databytecounts = tags.valueof(325)
        if self.databytecounts is None:
            # StripByteCounts
            self.databytecounts = tags.valueof(279)
            if self.databytecounts is None:
                # JPEGInterchangeFormatLength et al.
                self.databytecounts = tags.valueof(514)

        if (
            self.imagewidth == 0
            and self.imagelength == 0
            and self.dataoffsets
            and self.databytecounts
        ):
            # dimensions may be missing in some RAW formats
            # read dimensions from assumed JPEG encoded segment
            try:
                fh.seek(self.dataoffsets[0])
                (
                    precision,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                ) = jpeg_shape(fh.read(min(self.databytecounts[0], 4096)))
            except Exception:
                pass
            else:
                self.imagelength = imagelength
                self.imagewidth = imagewidth
                self.samplesperpixel = samplesperpixel
                if 258 not in tags:
                    self.bitspersample = 8 if precision <= 8 else 16
                if 262 not in tags and samplesperpixel == 3:
                    self.photometric = 6  # YCbCr
                if 259 not in tags:
                    self.compression = 6  # OJPEG
                if 278 not in tags:
                    self.rowsperstrip = imagelength

        elif self.compression == 6:
            # OJPEG hack. See libtiff v4.2.0 tif_dirread.c#L4082
            if 262 not in tags:
                # PhotometricInterpretation missing
                self.photometric = 6  # YCbCr
            elif self.photometric == 2:
                # RGB -> YCbCr
                self.photometric = 6
            if 258 not in tags:
                # BitsPerSample missing
                self.bitspersample = 8
            if 277 not in tags:
                # SamplesPerPixel missing
                if self.photometric in (2, 6):
                    self.samplesperpixel = 3
                elif self.photometric in (0, 1):
                    self.samplesperpixel = 3

        elif self.is_lsm or (self.index != 0 and self.parent.is_lsm):
            # correct non standard LSM bitspersample tags
            tags[258]._fix_lsm_bitspersample()
            if self.compression == 1 and self.predictor != 1:
                # work around bug in LSM510 software
                self.predictor = 1

        elif self.is_vista or (self.index != 0 and self.parent.is_vista):
            # ISS Vista writes wrong ImageDepth tag
            self.imagedepth = 1

        elif self.is_stk:
            if tags.get(33629) is not None:  # UIC2tag
                # read UIC1tag again now that plane count is known
                tag = tags.get(33628)  # UIC1tag
                fh.seek(tag.valueoffset)
                try:
                    tag.value = read_uic1tag(
                        fh,
                        tiff.byteorder,
                        tag.dtype,
                        tag.count,
                        None,
                        tags[33629].count,  # UIC2tag
                    )
                except Exception as exc:
                    log_warning(
                        f'{self!r} read_uic1tag failed with '
                        f'{exc.__class__.__name__}: {exc}'
                    )

        tag = tags.get(50839)
        if tag is not None:
            # decode IJMetadata tag
            try:
                tag.value = imagej_metadata(
                    tag.value,
                    tags[50838].value,  # IJMetadataByteCounts
                    tiff.byteorder,
                )
            except Exception as exc:
                log_warning(
                    f'{self!r} imagej_metadata failed with '
                    f'{exc.__class__.__name__}: {exc}'
                )

        # BitsPerSample
        value = tags.valueof(258)
        if value is not None:
            if self.bitspersample != 1:
                pass  # bitspersample was set by ojpeg hack
            elif tags[258].count == 1:
                self.bitspersample = value
            else:
                # LSM might list more items than samplesperpixel
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    self.bitspersample = value
                else:
                    self.bitspersample = value[0]

        # SampleFormat
        value = tags.valueof(339)
        if value is not None:
            if tags[339].count == 1:
                self.sampleformat = value
            else:
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    self.sampleformat = value
                else:
                    self.sampleformat = value[0]

        if 322 in tags:  # TileWidth
            self.rowsperstrip = None
        elif 257 in tags:  # ImageLength
            if 278 not in tags or tags[278].count > 1:  # RowsPerStrip
                self.rowsperstrip = self.imagelength
            self.rowsperstrip = min(self.rowsperstrip, self.imagelength)
            # self.stripsperimage = int(math.floor(
            #    float(self.imagelength + self.rowsperstrip - 1) /
            #    self.rowsperstrip))

        # determine dtype
        dtype = TIFF.SAMPLE_DTYPES.get(
            (self.sampleformat, self.bitspersample), None
        )
        if dtype is not None:
            dtype = numpy.dtype(dtype)
        self.dtype = self._dtype = dtype

        # determine shape of data
        imagelength = self.imagelength
        imagewidth = self.imagewidth
        imagedepth = self.imagedepth
        samplesperpixel = self.samplesperpixel

        if self.photometric == 2 or samplesperpixel > 1:  # PHOTOMETRIC.RGB
            if self.planarconfig == 1:
                self.shaped = (
                    1,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                )
                if imagedepth == 1:
                    self.shape = (imagelength, imagewidth, samplesperpixel)
                    self.axes = 'YXS'
                else:
                    self.shape = (
                        imagedepth,
                        imagelength,
                        imagewidth,
                        samplesperpixel,
                    )
                    self.axes = 'ZYXS'
            else:
                self.shaped = (
                    samplesperpixel,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    1,
                )
                if imagedepth == 1:
                    self.shape = (samplesperpixel, imagelength, imagewidth)
                    self.axes = 'SYX'
                else:
                    self.shape = (
                        samplesperpixel,
                        imagedepth,
                        imagelength,
                        imagewidth,
                    )
                    self.axes = 'SZYX'
        else:
            self.shaped = (1, imagedepth, imagelength, imagewidth, 1)
            if imagedepth == 1:
                self.shape = (imagelength, imagewidth)
                self.axes = 'YX'
            else:
                self.shape = (imagedepth, imagelength, imagewidth)
                self.axes = 'ZYX'

        if not self.databytecounts:
            self.databytecounts = (
                product(self.shape) * (self.bitspersample // 8),
            )
            if self.compression != 1:
                log_warning(f'{self!r} missing ByteCounts tag')

        if imagelength and self.rowsperstrip and not self.is_lsm:
            # fix incorrect number of strip bytecounts and offsets
            maxstrips = (
                int(
                    math.floor(imagelength + self.rowsperstrip - 1)
                    / self.rowsperstrip
                )
                * self.imagedepth
            )
            if self.planarconfig == 2:
                maxstrips *= self.samplesperpixel
            if maxstrips != len(self.databytecounts):
                log_warning(
                    f'{self!r} incorrect StripByteCounts count '
                    f'({len(self.databytecounts)} != {maxstrips})'
                )
                self.databytecounts = self.databytecounts[:maxstrips]
            if maxstrips != len(self.dataoffsets):
                log_warning(
                    f'{self!r} incorrect StripOffsets count '
                    f'({len(self.dataoffsets)} != {maxstrips})'
                )
                self.dataoffsets = self.dataoffsets[:maxstrips]

        value = tags.valueof(42113)  # GDAL_NODATA
        if value is not None:
            try:
                pytype = type(dtype.type(0).item())
                self.nodata = pytype(value)
            except Exception:
                pass

        mcustarts = tags.valueof(65426)
        if mcustarts is not None and self.is_ndpi:
            # use NDPI JPEG McuStarts as tile offsets
            high = tags.valueof(65432)
            if high is not None:
                # McuStartsHighBytes
                high = high.astype('uint64')
                high <<= 32
                mcustarts = mcustarts.astype('uint64')
                mcustarts += high
            fh.seek(self.dataoffsets[0])
            jpegheader = fh.read(mcustarts[0])
            try:
                (
                    self.tilelength,
                    self.tilewidth,
                    self.jpegheader,
                ) = ndpi_jpeg_tile(jpegheader)
            except ValueError as exc:
                log_warning(
                    f'{self!r} ndpi_jpeg_tile failed with '
                    f'{exc.__class__.__name__}: {exc}'
                )
            else:
                self.databytecounts = (
                    mcustarts[1:] - mcustarts[:-1]
                ).tolist() + [self.databytecounts[0] - int(mcustarts[-1])]
                self.dataoffsets = (mcustarts + self.dataoffsets[0]).tolist()

    @lazyattr
    def decode(self):
        """Return decoded segment, its shape, and indices in image.

        The decode function is implemeted as a closure.

        Parameters
        ----------
        data : bytes
            Encoded bytes of a segment (aka strile, strip or tile)
            or None for empty segments.
        index : int
            The index of the segment in the Offsets and Bytecount tag values.
        jpegtables : bytes or None
            For JPEG compressed segments only, the value of the JPEGTables tag
            if any.

        Returns
        -------
        segment : numpy.ndarray
            Decoded segment or None for empty segments.
        indices : tuple of int
            The position of the segment in the image array of normalized shape:
            (separate sample, depth, length, width, contig sample).
        shape : tuple of int
            The shape of the segment: (depth, length, width, contig samples).
            The shape of strips depends on their linear index.

        Raises ValueError or NotImplementedError if decoding is not supported.

        """
        if self.hash in self.parent._parent._decoders:
            return self.parent._parent._decoders[self.hash]

        def cache(decode):
            self.parent._parent._decoders[self.hash] = decode
            return decode

        if self.dtype is None:

            def decode(*args, **kwargs):
                raise ValueError(
                    'data type not supported '
                    f'(SampleFormat {self.sampleformat}, '
                    f'{self.bitspersample}-bit)'
                )

            return cache(decode)

        if 0 in self.shaped:

            def decode(*args, **kwargs):
                raise ValueError('empty image')

            return cache(decode)

        try:
            if self.compression == 1:
                decompress = None
            else:
                decompress = TIFF.DECOMPRESSORS[self.compression]
        except KeyError as exc:

            def decode(*args, exc=str(exc)[1:-1], **kwargs):
                raise ValueError(f'{exc}')

            return cache(decode)

        try:
            if self.predictor == 1:
                unpredict = None
            else:
                unpredict = TIFF.UNPREDICTORS[self.predictor]
        except KeyError as exc:

            def decode(*args, exc=str(exc)[1:-1], **kwargs):
                raise ValueError(f'{exc}')

            return cache(decode)

        if self.tags.get(339) is not None:
            tag = self.tags[339]  # SampleFormat
            if tag.count != 1 and any(i - tag.value[0] for i in tag.value):

                def decode(*args, **kwargs):
                    raise ValueError(
                        f'sample formats do not match {tag.value}'
                    )

                return cache(decode)

        if self.is_subsampled and (
            self.compression not in (6, 7) or self.planarconfig == 2
        ):

            def decode(*args, **kwargs):
                raise NotImplementedError('chroma subsampling not supported')

            return cache(decode)

        # normalize segments shape to [depth, length, length, contig]
        if self.is_tiled:
            stshape = [self.tiledepth, self.tilelength, self.tilewidth, 1]
        else:
            stshape = [1, self.rowsperstrip, self.imagewidth, 1]
        if self.planarconfig == 1:
            stshape[-1] = self.samplesperpixel
        stshape = tuple(stshape)

        stdepth, stlength, stwidth, samples = stshape
        imdepth, imlength, imwidth, samples = self.shaped[1:]

        if self.is_tiled:

            width = (imwidth + stwidth - 1) // stwidth
            length = (imlength + stlength - 1) // stlength
            depth = (imdepth + stdepth - 1) // stdepth

            def indices(tileindex):
                # return indices and shape of tile in image array
                return (
                    (
                        tileindex // (width * length * depth),
                        (tileindex // (width * length)) % depth * stdepth,
                        (tileindex // width) % length * stlength,
                        tileindex % width * stwidth,
                        0,
                    ),
                    stshape,
                )

            def reshape(data, indices, shape):
                # return reshaped tile
                if data is None:
                    return data
                size = shape[0] * shape[1] * shape[2] * shape[3]
                if data.size > size:
                    # decompression / unpacking might return too many bytes
                    data.shape = -1
                    data = data[:size]
                if data.size == size:
                    # complete tile
                    # data might be non-contiguous; cannot reshape inplace
                    data = data.reshape(shape)
                else:
                    # data fills remaining space
                    # found in some JPEG/PNG compressed tiles
                    try:
                        data = data.reshape(
                            (
                                min(imdepth - indices[1], shape[0]),
                                min(imlength - indices[2], shape[1]),
                                min(imwidth - indices[3], shape[2]),
                                samples,
                            )
                        )
                    except ValueError:
                        # incomplete tile; see gdal issue #1179
                        log_warning(
                            '<tifffile.TiffPage.decode> '
                            f'incomplete tile {data.shape} {shape}'
                        )
                        t = numpy.zeros(size, data.dtype)
                        size = min(data.size, size)
                        t[:size] = data[:size]
                        data = t.reshape(shape)
                return data

            def pad(data, shape, nodata=self.nodata):
                # pad tile to shape
                if data is None or data.shape == shape:
                    return data, shape
                padwidth = [(0, i - j) for i, j in zip(shape, data.shape)]
                data = numpy.pad(data, padwidth, constant_values=nodata)
                return data, data.shape

        else:
            # strips
            length = (imlength + stlength - 1) // stlength

            def indices(stripindex):
                # return indices and shape of strip in image array
                indices = (
                    stripindex // (length * imdepth),
                    (stripindex // length) % imdepth * stdepth,
                    stripindex % length * stlength,
                    0,
                    0,
                )
                shape = (
                    stdepth,
                    min(stlength, imlength - indices[2]),
                    stwidth,
                    samples,
                )
                return indices, shape

            def reshape(data, indices, shape):
                # return reshaped strip
                if data is None:
                    return data
                size = shape[0] * shape[1] * shape[2] * shape[3]
                if data.size > size:
                    # decompression / unpacking might return too many bytes
                    data.shape = -1
                    data = data[:size]
                if data.size == size:
                    # expected size
                    data.shape = shape
                else:
                    # should not happen, but try different length
                    data.shape = shape[0], -1, shape[2], shape[3]
                    # raise RuntimeError(
                    #     f'invalid strip shape {data.shape} or size {size}'
                    # )
                return data

            def pad(data, shape, nodata=self.nodata):
                # pad strip length to rowsperstrip
                shape = (shape[0], stlength, shape[2], shape[3])
                if data is None or data.shape == shape:
                    return data, shape
                padwidth = [
                    (0, 0),
                    (0, stlength - data.shape[1]),
                    (0, 0),
                    (0, 0),
                ]
                data = numpy.pad(data, padwidth, constant_values=nodata)
                return data, data.shape

        if self.compression in (6, 7, 34892, 33007):
            # JPEG needs special handling
            if self.fillorder == 2:
                log_warning(f'{self!r} disabling LSB2MSB for JPEG')
            if unpredict:
                log_warning(f'{self!r} disabling predictor for JPEG')
            if 28672 in self.tags:  # SonyRawFileType
                log_warning(
                    f'{self!r} SonyRawFileType might need additional '
                    'unpacking (see issue #95)'
                )

            colorspace, outcolorspace = jpeg_decode_colorspace(
                self.photometric, self.planarconfig, self.extrasamples
            )

            def decode(
                data,
                segmentindex,
                jpegtables=None,
                jpegheader=None,
                _fullsize=False,
                bitspersample=self.bitspersample,
                colorspace=colorspace,
                outcolorspace=outcolorspace,
            ):
                # return decoded segment, its shape, and indices in image
                index, shape = indices(segmentindex)
                if data is None:
                    if _fullsize:
                        data, shape = pad(data, shape)
                    return data, index, shape
                data = imagecodecs.jpeg_decode(
                    data,
                    bitspersample=bitspersample,
                    tables=jpegtables,
                    header=jpegheader,
                    colorspace=colorspace,
                    outcolorspace=outcolorspace,
                    shape=shape[1:3],
                )
                data = reshape(data, index, shape)
                if _fullsize:
                    data, shape = pad(data, shape)
                return data, index, shape

            return cache(decode)

        if self.compression in (
            33003,
            33004,
            33005,
            34712,
            34933,
            34934,
            22610,
            50001,
            50002,
        ):
            # JPEG2000, WEBP, PNG, JPEGXR
            # presume codecs always return correct dtype, native byte order...
            if self.fillorder == 2:
                log_warning(
                    f'{self!r} '
                    f'disabling LSB2MSB for compression {self.compression}'
                )
            if unpredict:
                log_warning(
                    f'{self!r} '
                    f'disabling predictor for compression {self.compression}'
                )

            def decode(data, segmentindex, jpegtables=None, _fullsize=False):
                # return decoded segment, its shape, and indices in image
                index, shape = indices(segmentindex)
                if data is None:
                    if _fullsize:
                        data, shape = pad(data, shape)
                    return data, index, shape
                data = decompress(data)
                data = reshape(data, index, shape)
                if _fullsize:
                    data, shape = pad(data, shape)
                return data, index, shape

            return cache(decode)

        dtype = numpy.dtype(self.parent.byteorder + self._dtype.char)

        if self.sampleformat == 5:
            # complex integer
            if unpredict is not None:
                raise NotImplementedError(
                    'unpredicting complex integers not supported'
                )

            itype = numpy.dtype(
                f'{self.parent.byteorder}i{self.bitspersample // 16}'
            )
            ftype = numpy.dtype(
                f'{self.parent.byteorder}f{dtype.itemsize // 2}'
            )

            def unpack(data):
                # return complex integer as numpy.complex
                data = numpy.frombuffer(data, itype)
                return data.astype(ftype).view(dtype)

        elif self.bitspersample in (8, 16, 32, 64, 128):
            # regular data types

            if (self.bitspersample * stwidth * samples) % 8:
                raise ValueError('data and sample size mismatch')
            if self.predictor == 3:  # PREDICTOR.FLOATINGPOINT
                # floating-point horizontal differencing decoder needs
                # raw byte order
                dtype = numpy.dtype(self._dtype.char)

            def unpack(data):
                # return numpy array from buffer
                try:
                    # read only numpy array
                    return numpy.frombuffer(data, dtype)
                except ValueError:
                    # e.g. LZW strips may be missing EOI
                    bps = self.bitspersample // 8
                    size = (len(data) // bps) * bps
                    return numpy.frombuffer(data[:size], dtype)

        elif isinstance(self.bitspersample, tuple):
            # e.g. RGB 565
            def unpack(data):
                # return numpy array from packed integers
                return unpack_rgb(data, dtype, self.bitspersample)

        elif self.bitspersample == 24 and dtype.char == 'f':
            # float24
            if unpredict is not None:
                # floatpred_decode requires numpy.float24, which does not exist
                raise NotImplementedError('unpredicting float24 not supported')

            def unpack(data, byteorder=self.parent.byteorder):
                # return numpy.float32 array from float24
                return float24_decode(data, byteorder)

        else:
            # bilevel and packed integers
            def unpack(data):
                # return numpy array from packed integers
                return packints_decode(
                    data, dtype, self.bitspersample, stwidth * samples
                )

        def decode(data, segmentindex, jpegtables=None, _fullsize=False):
            # return decoded segment, its shape, and indices in image
            index, shape = indices(segmentindex)
            if data is None:
                if _fullsize:
                    data, shape = pad(data, shape)
                return data, index, shape
            if self.fillorder == 2:
                data = bitorder_decode(data)
            if decompress is not None:
                # TODO: calculate correct size for packed integers
                size = shape[0] * shape[1] * shape[2] * shape[3]
                data = decompress(data, out=size * dtype.itemsize)
            data = unpack(data)
            data = reshape(data, index, shape)
            data = data.astype('=' + dtype.char)
            if unpredict is not None:
                # unpredict is faster with native byte order
                data = unpredict(data, axis=-2, out=data)
            if _fullsize:
                data, shape = pad(data, shape)
            return data, index, shape

        return cache(decode)

    def segments(
        self, lock=None, maxworkers=None, func=None, sort=False, _fullsize=None
    ):
        """Return iterator over decoded segments in TiffPage.

        See the decode function for return values.

        """
        keyframe = self.keyframe  # self or keyframe
        fh = self.parent.filehandle
        if lock is None:
            lock = fh.lock
        if _fullsize is None:
            _fullsize = keyframe.is_tiled

        decodeargs = {'_fullsize': bool(_fullsize)}
        if keyframe.compression in (6, 7, 34892, 33007):  # JPEG
            decodeargs['jpegtables'] = self.jpegtables
            decodeargs['jpegheader'] = keyframe.jpegheader

        if func is None:

            def decode(args, decodeargs=decodeargs, keyframe=keyframe):
                return keyframe.decode(*args, **decodeargs)

        else:

            def decode(
                args, decodeargs=decodeargs, keyframe=keyframe, func=func
            ):
                return func(keyframe.decode(*args, **decodeargs))

        if maxworkers is None or maxworkers < 1:
            maxworkers = keyframe.maxworkers
        if maxworkers < 2:
            for segment in fh.read_segments(
                self.dataoffsets,
                self.databytecounts,
                lock=lock,
                sort=sort,
                flat=True,
            ):
                yield decode(segment)
        else:
            # reduce memory overhead by processing chunks of up to
            # ~64 MB of segments because ThreadPoolExecutor.map is not
            # collecting iterables lazily
            with ThreadPoolExecutor(maxworkers) as executor:
                for segments in fh.read_segments(
                    self.dataoffsets,
                    self.databytecounts,
                    lock=lock,
                    sort=sort,
                    flat=False,
                ):
                    yield from executor.map(decode, segments)

    def asarray(self, out=None, squeeze=True, lock=None, maxworkers=None):
        """Read image data from file and return as numpy array.

        Raise ValueError if format is not supported.

        Parameters
        ----------
        out : numpy.ndarray, str, or file-like object
            Buffer where image data are saved.
            If None (default), a new array is created.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If 'memmap', directly memory-map the image data in the TIFF file
            if possible; else create a memory-mapped array in a temporary file.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        squeeze : bool
            If True (default), all length-1 dimensions (except X and Y) are
            squeezed out from the array.
            If False, the shape of the returned array is the normalized
            5-dimensional shape (TiffPage.shaped).
        lock : {RLock, NullContext}
            A reentrant lock used to synchronize seeks and reads from file.
            If None (default), the lock of the parent's filehandle is used.
        maxworkers : int or None
            Maximum number of threads to concurrently decode strips or tiles.
            If None (default), up to half the CPU cores are used.
            See remarks in TiffFile.asarray.

        Returns
        -------
        numpy.ndarray
            Numpy array of decompressed, unpredicted, and unpacked image data
            read from Strip/Tile Offsets/ByteCounts, formatted according to
            shape and dtype metadata found in tags and parameters.
            Photometric conversion, pre-multiplied alpha, orientation, and
            colorimetry corrections are not applied. Specifically, CMYK images
            are not converted to RGB, MinIsWhite images are not inverted,
            and color palettes are not applied. Exception are YCbCr JPEG
            compressed images, which are converted to RGB.

        """
        keyframe = self.keyframe  # self or keyframe

        if not keyframe.shaped or product(keyframe.shaped) == 0:
            return None

        if len(self.dataoffsets) == 0:
            raise TiffFileError('missing data offset')

        fh = self.parent.filehandle
        if lock is None:
            lock = fh.lock
        with lock:
            closed = fh.closed
            if closed:
                # this is an inefficient resort in case a user calls
                # asarray of a TiffPage or TiffFrame with a closed FileHandle.
                warnings.warn(
                    f'{self!r} reading array from closed file', UserWarning
                )
                fh.open()

        if (
            isinstance(out, str)
            and out == 'memmap'
            and keyframe.is_memmappable
        ):
            # direct memory map array in file
            with lock:
                result = fh.memmap_array(
                    keyframe.parent.byteorder + keyframe._dtype.char,
                    keyframe.shaped,
                    offset=self.dataoffsets[0],
                )

        elif keyframe.is_contiguous:
            # read contiguous bytes to array
            if keyframe.is_subsampled:
                raise NotImplementedError('chroma subsampling not supported')
            if out is not None:
                out = create_output(out, keyframe.shaped, keyframe._dtype)
            with lock:
                fh.seek(self.dataoffsets[0])
                result = fh.read_array(
                    keyframe.parent.byteorder + keyframe._dtype.char,
                    product(keyframe.shaped),
                    out=out,
                )
            if keyframe.fillorder == 2:
                bitorder_decode(result, out=result)
            if keyframe.predictor != 1:
                # predictors without compression
                unpredict = TIFF.UNPREDICTORS[keyframe.predictor]
                if keyframe.predictor == 1:
                    unpredict(result, axis=-2, out=result)
                else:
                    # floatpred cannot decode in-place
                    out = unpredict(result, axis=-2, out=result)
                    result[:] = out

        elif (
            keyframe.jpegheader is not None
            and keyframe is self
            and 273 in self.tags  # striped ...
            and self.is_tiled  # but reported as tiled
            and self.imagewidth <= 65500
            and self.imagelength <= 65500
        ):
            # decode the whole NDPI JPEG strip
            with lock:
                fh.seek(self.tags[273].value[0])  # StripOffsets
                data = fh.read(self.tags[279].value[0])  # StripByteCounts
            decompress = TIFF.DECOMPRESSORS[self.compression]
            result = decompress(
                data, bitspersample=self.bitspersample, out=out
            )
            del data

        else:
            # decode individual strips or tiles
            result = create_output(out, keyframe.shaped, keyframe._dtype)
            keyframe.decode  # init TiffPage.decode function

            def func(decoderesult, keyframe=keyframe, out=out):
                # copy decoded segments to output array
                segment, (s, d, l, w, _), shape = decoderesult
                if segment is None:
                    segment = keyframe.nodata
                else:
                    segment = segment[
                        : keyframe.imagedepth - d,
                        : keyframe.imagelength - l,
                        : keyframe.imagewidth - w,
                    ]
                result[
                    s, d : d + shape[0], l : l + shape[1], w : w + shape[2]
                ] = segment
                # except IndexError:
                #     pass  # corrupted file e.g. with too many strips

            for _ in self.segments(
                func=func,
                lock=lock,
                maxworkers=maxworkers,
                sort=True,
                _fullsize=False,
            ):
                pass

        result.shape = keyframe.shaped
        if squeeze:
            try:
                result.shape = keyframe.shape
            except ValueError:
                log_warning(
                    f'{self!r} '
                    f'failed to reshape {result.shape} to {keyframe.shape}'
                )

        if closed:
            # TODO: close file if an exception occurred above
            fh.close()
        return result

    def aszarr(self, **kwargs):
        """Return image data as zarr storage."""
        return ZarrTiffStore(self, **kwargs)

    def asrgb(
        self,
        uint8=False,
        alpha=None,
        colormap=None,
        dmin=None,
        dmax=None,
        **kwargs,
    ):
        """Return image data as RGB(A).

        Work in progress.

        """
        data = self.asarray(**kwargs)
        keyframe = self.keyframe  # self or keyframe

        if keyframe.photometric == TIFF.PHOTOMETRIC.PALETTE:
            colormap = keyframe.colormap
            if (
                colormap.shape[1] < 2 ** keyframe.bitspersample
                or keyframe.dtype.char not in 'BH'
            ):
                raise ValueError('cannot apply colormap')
            if uint8:
                if colormap.max() > 255:
                    colormap >>= 8
                colormap = colormap.astype('uint8')
            if 'S' in keyframe.axes:
                data = data[..., 0] if keyframe.planarconfig == 1 else data[0]
            data = apply_colormap(data, colormap)

        elif keyframe.photometric == TIFF.PHOTOMETRIC.RGB:
            if keyframe.extrasamples:
                if alpha is None:
                    alpha = TIFF.EXTRASAMPLE
                for i, exs in enumerate(keyframe.extrasamples):
                    if exs in alpha:
                        if keyframe.planarconfig == 1:
                            data = data[..., [0, 1, 2, 3 + i]]
                        else:
                            data = data[:, [0, 1, 2, 3 + i]]
                        break
            else:
                if keyframe.planarconfig == 1:
                    data = data[..., :3]
                else:
                    data = data[:, :3]
            # TODO: convert to uint8?

        elif keyframe.photometric == TIFF.PHOTOMETRIC.MINISBLACK:
            raise NotImplementedError
        elif keyframe.photometric == TIFF.PHOTOMETRIC.MINISWHITE:
            raise NotImplementedError
        elif keyframe.photometric == TIFF.PHOTOMETRIC.SEPARATED:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return data

    def _gettags(self, codes=None, lock=None):
        """Return list of (code, TiffTag)."""
        return [
            (tag.code, tag)
            for tag in self.tags
            if codes is None or tag.code in codes
        ]

    def _nextifd(self):
        """Return offset to next IFD from file."""
        fh = self.parent.filehandle
        tiff = self.parent.tiff
        fh.seek(self.offset)
        tagno = struct.unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
        fh.seek(self.offset + tiff.tagnosize + tagno * tiff.tagsize)
        return struct.unpack(tiff.offsetformat, fh.read(tiff.offsetsize))[0]

    def aspage(self):
        """Return self."""
        return self

    @property
    def keyframe(self):
        """Return keyframe, self."""
        return self

    @keyframe.setter
    def keyframe(self, index):
        """Set keyframe, NOP."""
        return

    @property
    def ndim(self):
        """Return number of array dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Return number of elements in array."""
        return product(self.shape)

    @property
    def nbytes(self):
        """Return number of bytes in array."""
        return product(self.shape) * self.dtype.itemsize

    @property
    def colormap(self):
        """Return colormap as numpy array."""
        return self.tags.valueof(320)

    @property
    def transferfunction(self):
        """Return transferfunction as numpy array."""
        return self.tags.valueof(301)

    @lazyattr
    def chunks(self):
        """Return shape of tiles or stripes."""
        shape = []
        if self.tiledepth > 1:
            shape.append(self.tiledepth)
        if self.is_tiled:
            shape.extend((self.tilelength, self.tilewidth))
        else:
            shape.extend((self.rowsperstrip, self.imagewidth))
        if self.planarconfig == 1 and self.samplesperpixel > 1:
            shape.append(self.samplesperpixel)
        return tuple(shape)

    @lazyattr
    def chunked(self):
        """Return shape of chunked image."""
        shape = []
        if self.planarconfig == 2 and self.samplesperpixel > 1:
            shape.append(self.samplesperpixel)
        if self.is_tiled:
            if self.imagedepth > 1:
                shape.append(
                    (self.imagedepth + self.tiledepth - 1) // self.tiledepth
                )
            shape.append(
                (self.imagelength + self.tilelength - 1) // self.tilelength
            )
            shape.append(
                (self.imagewidth + self.tilewidth - 1) // self.tilewidth
            )
        else:
            if self.imagedepth > 1:
                shape.append(self.imagedepth)
            shape.append(
                (self.imagelength + self.rowsperstrip - 1) // self.rowsperstrip
            )
            shape.append(1)
        if self.planarconfig == 1 and self.samplesperpixel > 1:
            shape.append(1)
        return tuple(shape)

    @lazyattr
    def hash(self):
        """Return checksum to identify pages in same series.

        Pages with the same hash can use the same decode function.

        """
        return hash(
            self.shaped
            + (
                self.parent.byteorder,
                self.tilewidth,
                self.tilelength,
                self.tiledepth,
                self.sampleformat,
                self.bitspersample,
                self.rowsperstrip,
                self.fillorder,
                self.predictor,
                self.extrasamples,
                self.photometric,
                self.planarconfig,
                self.compression,
            )
        )

    @lazyattr
    def pages(self):
        """Return sequence of sub-pages, SubIFDs."""
        if 330 not in self.tags:
            return ()
        return TiffPages(self, index=self.index)

    @lazyattr
    def maxworkers(self):
        """Return maximum number of threads for decoding segments.

        Return 0 to disable multi-threading also for stacking pages.

        """
        if self.is_contiguous or self.dtype is None:
            return 0
        if self.compression in (
            6,
            7,
            33003,
            33004,
            33005,
            33007,
            34712,
            34892,
            34933,
            34934,
            22610,
            50001,
            50002,
        ):
            # image codecs
            return min(TIFF.MAXWORKERS, len(self.dataoffsets))
        bytecount = product(self.chunks) * self.dtype.itemsize
        if bytecount < 2048:
            # disable multi-threading for small segments
            return 0
        if self.compression != 1 or self.fillorder != 1 or self.predictor != 1:
            if self.compression == 5 and bytecount < 16384:
                # disable multi-threading for small LZW compressed segments
                return 0
        if len(self.dataoffsets) < 4:
            return 1
        if self.compression != 1 or self.fillorder != 1 or self.predictor != 1:
            if imagecodecs is not None:
                return min(TIFF.MAXWORKERS, len(self.dataoffsets))
        return 2  # optimum for large number of uncompressed tiles

    @lazyattr
    def is_contiguous(self):
        """Return offset and size of contiguous data, else None.

        Excludes prediction and fill_order.

        """
        if self.sampleformat == 5:
            return None
        if self.compression != 1 or self.bitspersample not in (8, 16, 32, 64):
            return None
        if 322 in self.tags:  # TileWidth
            if (
                self.imagewidth != self.tilewidth
                or self.imagelength % self.tilelength
                or self.tilewidth % 16
                or self.tilelength % 16
            ):
                return None
            if (
                32997 in self.tags  # ImageDepth
                and 32998 in self.tags  # TileDepth
                and (
                    self.imagelength != self.tilelength
                    or self.imagedepth % self.tiledepth
                )
            ):
                return None
        offsets = self.dataoffsets
        bytecounts = self.databytecounts
        if len(offsets) == 0:
            return None
        if len(offsets) == 1:
            return offsets[0], bytecounts[0]
        if self.is_stk or self.is_lsm:
            return offsets[0], sum(bytecounts)
        if all(
            bytecounts[i] != 0 and offsets[i] + bytecounts[i] == offsets[i + 1]
            for i in range(len(offsets) - 1)
        ):
            return offsets[0], sum(bytecounts)
        return None

    @lazyattr
    def is_final(self):
        """Return if page's image data are stored in final form.

        Excludes byte-swapping.

        """
        return (
            self.is_contiguous
            and self.fillorder == 1
            and self.predictor == 1
            and not self.is_subsampled
        )

    @lazyattr
    def is_memmappable(self):
        """Return if page's image data in file can be memory-mapped."""
        return (
            self.parent.filehandle.is_file
            and self.is_final
            # and (self.bitspersample == 8 or self.parent.isnative)
            # aligned?
            and self.is_contiguous[0] % self.dtype.itemsize == 0
        )

    def __repr__(self):
        return f'<tifffile.TiffPage {self.index} @{self.offset}>'

    def __str__(self, detail=0, width=79):
        """Return string containing information about TiffPage."""
        if self.keyframe != self:
            return TiffFrame.__str__(self, detail, width)
        attr = ''
        for name in ('memmappable', 'final', 'contiguous'):
            attr = getattr(self, 'is_' + name)
            if attr:
                attr = name.upper()
                break

        def tostr(name, skip=1):
            obj = getattr(self, name)
            try:
                value = getattr(obj, 'name')
            except AttributeError:
                return ''
            if obj != skip:
                return value
            return ''

        info = '  '.join(
            s.lower()
            for s in (
                'x'.join(str(i) for i in self.shape),
                '{}{}'.format(
                    TIFF.SAMPLEFORMAT(self.sampleformat).name,
                    self.bitspersample,
                ),
                ' '.join(
                    i
                    for i in (
                        TIFF.PHOTOMETRIC(self.photometric).name,
                        'REDUCED' if self.is_reduced else '',
                        'MASK' if self.is_mask else '',
                        'TILED' if self.is_tiled else '',
                        tostr('compression'),
                        tostr('planarconfig'),
                        tostr('predictor'),
                        tostr('fillorder'),
                    )
                    + tuple(f.upper() for f in self.flags)
                    + (attr,)
                    if i
                ),
            )
            if s
        )
        info = f'TiffPage {self.index} @{self.offset}  {info}'
        if detail <= 0:
            return info
        info = [info, self.tags.__str__(detail + 1, width=width)]
        if detail > 1:
            for name in ('ndpi',):
                name = name + '_tags'
                attr = getattr(self, name, False)
                if attr:
                    info.append(
                        '{}\n{}'.format(
                            name.upper(),
                            pformat(attr, width=width, height=detail * 8),
                        )
                    )
        if detail > 3:
            try:
                info.append(
                    'DATA\n{}'.format(
                        pformat(self.asarray(), width=width, height=detail * 8)
                    )
                )
            except Exception:
                pass
        return '\n\n'.join(info)

    @lazyattr
    def flags(self):
        """Return set of flags."""
        return {
            name.lower()
            for name in sorted(TIFF.FILE_FLAGS)
            if getattr(self, 'is_' + name)
        }

    @lazyattr
    def andor_tags(self):
        """Return consolidated metadata from Andor tags as dict."""
        if not self.is_andor:
            return None
        result = {'Id': self.tags[4864].value}  # AndorId
        for tag in self.tags:  # list(self.tags.values()):
            code = tag.code
            if not 4864 < code < 5031:
                continue
            name = tag.name
            name = name[5:] if len(name) > 5 else name
            result[name] = tag.value
            # del self.tags[code]
        return result

    @lazyattr
    def epics_tags(self):
        """Return consolidated metadata from EPICS areaDetector tags as dict.

        Use epics_datetime() to get a datetime object from the epicsTSSec and
        epicsTSNsec tags.

        """
        if not self.is_epics:
            return None
        result = {}
        for tag in self.tags:  # list(self.tags.values()):
            code = tag.code
            if not 65000 <= code < 65500:
                continue
            value = tag.value
            if code == 65000:
                # not a POSIX timestamp
                # https://github.com/bluesky/area-detector-handlers/issues/20
                result['timeStamp'] = float(value)
            elif code == 65001:
                result['uniqueID'] = int(value)
            elif code == 65002:
                result['epicsTSSec'] = int(value)
            elif code == 65003:
                result['epicsTSNsec'] = int(value)
            else:
                key, value = value.split(':', 1)
                result[key] = astype(value)
            # del self.tags[code]
        return result

    @lazyattr
    def ndpi_tags(self):
        """Return consolidated metadata from Hamamatsu NDPI as dict."""
        # TODO: parse 65449 ini style comments
        if not self.is_ndpi:
            return None
        tags = self.tags
        result = {}
        for name in ('Make', 'Model', 'Software'):
            result[name] = tags[name].value
        for code, name in TIFF.NDPI_TAGS.items():
            if code in tags:
                result[name] = tags[code].value
                # del tags[code]
        if 'McuStarts' in result:
            mcustarts = result['McuStarts']
            if 'McuStartsHighBytes' in result:
                high = result['McuStartsHighBytes'].astype('uint64')
                high <<= 32
                mcustarts = mcustarts.astype('uint64')
                mcustarts += high
                del result['McuStartsHighBytes']
            result['McuStarts'] = mcustarts
        return result

    @lazyattr
    def geotiff_tags(self):
        """Return consolidated metadata from GeoTIFF tags as dict."""
        if not self.is_geotiff:
            return None
        tags = self.tags

        gkd = tags.valueof(34735)  # GeoKeyDirectoryTag
        if gkd is None or len(gkd) < 2 or gkd[0] != 1:
            log_warning(f'{self!r} invalid GeoKeyDirectoryTag')
            return {}

        result = {
            'KeyDirectoryVersion': gkd[0],
            'KeyRevision': gkd[1],
            'KeyRevisionMinor': gkd[2],
            # 'NumberOfKeys': gkd[3],
        }
        # deltags = ['GeoKeyDirectoryTag']
        geokeys = TIFF.GEO_KEYS
        geocodes = TIFF.GEO_CODES
        for index in range(gkd[3]):
            try:
                keyid, tagid, count, offset = gkd[
                    4 + index * 4 : index * 4 + 8
                ]
            except Exception as exc:
                log_warning(
                    f'{self!r} corrupted GeoKeyDirectoryTag '
                    f'({exc.__class__.__name__}: {exc})'
                )
                continue
            if tagid == 0:
                value = offset
            else:
                try:
                    value = tags[tagid].value[offset : offset + count]
                except TiffFileError:
                    log_warning(
                        f'{self!r} corrupted GeoKeyDirectoryTag {tagid}'
                    )
                    continue
                except KeyError:
                    log_warning(
                        f'{self!r} GeoKeyDirectoryTag {tagid} not found'
                    )
                    continue
                if tagid == 34737 and count > 1 and value[-1] == '|':
                    value = value[:-1]
                value = value if count > 1 else value[0]
            if keyid in geocodes:
                try:
                    value = geocodes[keyid](value)
                except Exception:
                    pass
            try:
                key = geokeys(keyid).name
            except ValueError:
                key = keyid
            result[key] = value

        value = tags.valueof(33920)  # IntergraphMatrixTag
        if value is not None:
            value = numpy.array(value)
            if len(value) == 16:
                value = value.reshape((4, 4)).tolist()
            result['IntergraphMatrix'] = value

        value = tags.valueof(33550)  # ModelPixelScaleTag
        if value is not None:
            result['ModelPixelScale'] = numpy.array(value).tolist()

        value = tags.valueof(33922)  # ModelTiepointTag
        if value is not None:
            value = numpy.array(value).reshape((-1, 6)).squeeze().tolist()
            result['ModelTiepoint'] = value

        value = tags.valueof(34264)  # ModelTransformationTag
        if value is not None:
            value = numpy.array(value).reshape((4, 4)).tolist()
            result['ModelTransformation'] = value

        # if 33550 in tags and 33922 in tags:
        #     sx, sy, sz = tags[33550].value  # ModelPixelScaleTag
        #     tiepoints = tags[33922].value  # ModelTiepointTag
        #     transforms = []
        #     for tp in range(0, len(tiepoints), 6):
        #         i, j, k, x, y, z = tiepoints[tp : tp + 6]
        #         transforms.append(
        #             [
        #                 [sx, 0.0, 0.0, x - i * sx],
        #                 [0.0, -sy, 0.0, y + j * sy],
        #                 [0.0, 0.0, sz, z - k * sz],
        #                 [0.0, 0.0, 0.0, 1.0],
        #             ]
        #         )
        #     if len(tiepoints) == 6:
        #         transforms = transforms[0]
        #     result['ModelTransformation'] = transforms

        rpcc = tags.valueof(50844)  # RPCCoefficientTag
        if rpcc is not None:
            result['RPCCoefficient'] = {
                'ERR_BIAS': rpcc[0],
                'ERR_RAND': rpcc[1],
                'LINE_OFF': rpcc[2],
                'SAMP_OFF': rpcc[3],
                'LAT_OFF': rpcc[4],
                'LONG_OFF': rpcc[5],
                'HEIGHT_OFF': rpcc[6],
                'LINE_SCALE': rpcc[7],
                'SAMP_SCALE': rpcc[8],
                'LAT_SCALE': rpcc[9],
                'LONG_SCALE': rpcc[10],
                'HEIGHT_SCALE': rpcc[11],
                'LINE_NUM_COEFF': rpcc[12:33],
                'LINE_DEN_COEFF ': rpcc[33:53],
                'SAMP_NUM_COEFF': rpcc[53:73],
                'SAMP_DEN_COEFF': rpcc[73:],
            }
        return result

    @property
    def is_reduced(self):
        """Page is reduced image of another image."""
        return self.subfiletype & 0b1

    @property
    def is_multipage(self):
        """Page is part of multi-page image."""
        return self.subfiletype & 0b10

    @property
    def is_mask(self):
        """Page is transparency mask for another image."""
        return self.subfiletype & 0b100

    @property
    def is_mrc(self):
        """Page is part of Mixed Raster Content."""
        return self.subfiletype & 0b1000

    @property
    def is_tiled(self):
        """Page contains tiled image."""
        return self.tilewidth > 0  # return 322 in self.tags  # TileWidth

    @property
    def is_subsampled(self):
        """Page contains chroma subsampled image."""
        if self.subsampling is not None:
            return self.subsampling != (1, 1)
        return (
            self.compression == 7
            and self.planarconfig == 1
            and self.photometric in (2, 6)
        )

    @lazyattr
    def is_imagej(self):
        """Return ImageJ description if exists, else None."""
        for description in (self.description, self.description1):
            if not description:
                return None
            if description[:7] == 'ImageJ=':
                return description
        return None

    @lazyattr
    def is_shaped(self):
        """Return description containing array shape if exists, else None."""
        for description in (self.description, self.description1):
            if not description or '"mibi.' in description:
                return None
            if description[:1] == '{' and '"shape":' in description:
                return description
            if description[:6] == 'shape=':
                return description
        return None

    @property
    def is_mdgel(self):
        """Page contains MDFileTag tag."""
        return 33445 in self.tags  # MDFileTag

    @property
    def is_mediacy(self):
        """Page contains Media Cybernetics Id tag."""
        tag = self.tags.get(50288)  # MC_Id
        try:
            return tag is not None and tag.value[:7] == b'MC TIFF'
        except Exception:
            return False

    @property
    def is_stk(self):
        """Page contains UIC1Tag tag."""
        return 33628 in self.tags

    @property
    def is_lsm(self):
        """Page contains CZ_LSMINFO tag."""
        return 34412 in self.tags

    @property
    def is_fluoview(self):
        """Page contains FluoView MM_STAMP tag."""
        return 34362 in self.tags

    @property
    def is_nih(self):
        """Page contains NIHImageHeader tag."""
        return 43314 in self.tags

    @property
    def is_volumetric(self):
        """Page contains SGI ImageDepth tag with value > 1."""
        return self.imagedepth > 1

    @property
    def is_vista(self):
        """Software tag is 'ISS Vista'."""
        return self.software == 'ISS Vista'

    @property
    def is_metaseries(self):
        """Page contains MDS MetaSeries metadata in ImageDescription tag."""
        if self.index != 0 or self.software != 'MetaSeries':
            return False
        d = self.description
        return d.startswith('<MetaData>') and d.endswith('</MetaData>')

    @property
    def is_ome(self):
        """Page contains OME-XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-4:] == 'OME>'  # and [:13] == '<?xml version'

    @property
    def is_scn(self):
        """Page contains Leica SCN XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-6:] == '</scn>'

    @property
    def is_micromanager(self):
        """Page contains MicroManagerMetadata tag."""
        return 51123 in self.tags

    @property
    def is_andor(self):
        """Page contains Andor Technology tags 4864-5030."""
        return 4864 in self.tags

    @property
    def is_pilatus(self):
        """Page contains Pilatus tags."""
        return self.software[:8] == 'TVX TIFF' and self.description[:2] == '# '

    @property
    def is_epics(self):
        """Page contains EPICS areaDetector tags."""
        return (
            self.description == 'EPICS areaDetector'
            or self.software == 'EPICS areaDetector'
        )

    @property
    def is_tvips(self):
        """Page contains TVIPS metadata."""
        return 37706 in self.tags

    @property
    def is_fei(self):
        """Page contains FEI_SFEG or FEI_HELIOS tags."""
        return 34680 in self.tags or 34682 in self.tags

    @property
    def is_sem(self):
        """Page contains CZ_SEM tag."""
        return 34118 in self.tags

    @property
    def is_svs(self):
        """Page contains Aperio metadata."""
        return self.description[:7] == 'Aperio '

    @property
    def is_bif(self):
        """Page contains Ventana metadata."""
        try:
            return 700 in self.tags and (
                # avoid reading XMP tag from file at this point
                # b'<iScan' in self.tags[700].value[:4096]
                'Ventana' in self.software
                or self.software[:17] == 'ScanOutputManager'
                or self.description == 'Label Image'
                or self.description == 'Label_Image'
                or self.description == 'Probability_Image'
            )
        except Exception:
            return False

    @property
    def is_scanimage(self):
        """Page contains ScanImage metadata."""
        return (
            self.software[:3] == 'SI.'
            or self.description[:6] == 'state.'
            or 'scanimage.SI' in self.description[-256:]
        )

    @property
    def is_qpi(self):
        """Page contains PerkinElmer tissue images metadata."""
        # The ImageDescription tag contains XML with a top-level
        # <PerkinElmer-QPI-ImageDescription> element
        return self.software[:15] == 'PerkinElmer-QPI'

    @property
    def is_geotiff(self):
        """Page contains GeoTIFF metadata."""
        return 34735 in self.tags  # GeoKeyDirectoryTag

    @property
    def is_tiffep(self):
        """Page contains TIFF/EP metadata."""
        return 37398 in self.tags  # TIFF/EPStandardID

    @property
    def is_sis(self):
        """Page contains Olympus SIS metadata."""
        return 33560 in self.tags or 33471 in self.tags

    @property
    def is_ndpi(self):
        """Page contains NDPI metadata."""
        return 65420 in self.tags and 271 in self.tags

    @property
    def is_philips(self):
        """Page contains Philips DP metadata."""
        return (
            self.software[:10] == 'Philips DP'
            and self.description[-13:] == '</DataObject>'
        )

    @property
    def is_eer(self):
        """Page contains EER metadata."""
        return (
            self.parent.is_bigtiff
            and self.compression in (65000, 65001)
            and 65001 in self.tags
        )


class TiffFrame:
    """Lightweight TIFF image file directory (IFD).

    Only a limited number of tag values are read from file.
    Other tag values are assumed to be identical with a specified TiffPage
    instance, the keyframe.

    TiffFrame is intended to reduce resource usage and speed up reading image
    data from file, not for introspection of metadata.

    """

    __slots__ = (
        'index',
        'parent',
        'offset',
        'dataoffsets',
        'databytecounts',
        'subifds',
        'jpegtables',
        '_keyframe',
    )

    is_mdgel = False
    pages = ()
    # tags = {}

    def __init__(
        self,
        parent,
        index,
        offset=None,
        keyframe=None,
        offsets=None,
        bytecounts=None,
    ):
        """Initialize TiffFrame from file or values.

        The file handle position must be at the offset to a valid IFD.

        """
        self._keyframe = None
        self.parent = parent
        self.index = index
        self.offset = offset
        self.subifds = None
        self.jpegtables = None
        self.dataoffsets = ()
        self.databytecounts = ()

        if offsets is not None:
            # initialize "virtual frame" from offsets and bytecounts
            self.dataoffsets = offsets
            self.databytecounts = bytecounts
            self._keyframe = keyframe
            return

        if offset is None:
            self.offset = self.parent.filehandle.tell()
        else:
            self.parent.filehandle.seek(offset)

        if keyframe is None:
            tags = {273, 279, 324, 325, 330, 347}
        elif keyframe.is_contiguous:
            # use databytecounts from keyframe
            tags = {256, 273, 324, 330}
            self.databytecounts = keyframe.databytecounts
        else:
            tags = {256, 273, 279, 324, 325, 330, 347}

        for code, tag in self._gettags(tags):
            if code == 273 or code == 324:
                self.dataoffsets = tag.value
            elif code == 279 or code == 325:
                self.databytecounts = tag.value
            elif code == 330:
                self.subifds = tag.value
            elif code == 347:
                self.jpegtables = tag.value
            elif code == 256 and keyframe.imagewidth != tag.value:
                raise RuntimeError('incompatible keyframe')

        if not self.dataoffsets:
            log_warning(f'{self!r} is missing required tags')

        self.keyframe = keyframe

    def _gettags(self, codes=None, lock=None):
        """Return list of (code, TiffTag) from file."""
        fh = self.parent.filehandle
        tiff = self.parent.tiff
        unpack = struct.unpack
        lock = NullContext() if lock is None else lock
        tags = []

        with lock:
            fh.seek(self.offset)
            try:
                tagno = unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
                if tagno > 4096:
                    raise TiffFileError(f'suspicious number of tags {tagno}')
            except Exception as exc:
                raise TiffFileError('corrupted tag list') from exc

            tagoffset = self.offset + tiff.tagnosize  # fh.tell()
            tagsize = tiff.tagsize
            tagindex = -tagsize
            codeformat = tiff.tagformat1[:2]
            tagbytes = fh.read(tagsize * tagno)

            for _ in range(tagno):
                tagindex += tagsize
                code = unpack(codeformat, tagbytes[tagindex : tagindex + 2])[0]
                if codes and code not in codes:
                    continue
                try:
                    tag = TiffTag.fromfile(
                        self.parent,
                        tagoffset + tagindex,
                        tagbytes[tagindex : tagindex + tagsize],
                    )
                except TiffFileError as exc:
                    log_warning(f'{self!r} {exc}')
                    continue
                tags.append((code, tag))

        return tags

    def _nextifd(self):
        """Return offset to next IFD from file."""
        return TiffPage._nextifd(self)

    def aspage(self):
        """Return TiffPage from file."""
        if self.offset is None:
            raise ValueError('cannot return virtual frame as page')
        fh = self.parent.filehandle
        closed = fh.closed
        if closed:
            # this is an inefficient resort in case a user calls aspage
            # of a TiffFrame with a closed FileHandle.
            warnings.warn(
                f'{self!r} reading TiffPage from closed file', UserWarning
            )
            fh.open()
        try:
            fh.seek(self.offset)
            page = TiffPage(self.parent, index=self.index)
        finally:
            if closed:
                fh.close()
        return page

    def asarray(self, *args, **kwargs):
        """Read image data from file and return as numpy array."""
        return TiffPage.asarray(self, *args, **kwargs)

    def aszarr(self, **kwargs):
        """Return image data as zarr storage."""
        return TiffPage.aszarr(self, **kwargs)

    def asrgb(self, *args, **kwargs):
        """Read image data from file and return RGB image as numpy array."""
        return TiffPage.asrgb(self, *args, **kwargs)

    def segments(self, *args, **kwargs):
        """Return iterator over decoded segments in TiffFrame."""
        return TiffPage.segments(self, *args, **kwargs)

    @property
    def keyframe(self):
        """Return keyframe."""
        return self._keyframe

    @keyframe.setter
    def keyframe(self, keyframe):
        """Set keyframe."""
        if self._keyframe == keyframe:
            return
        if self._keyframe is not None:
            raise RuntimeError('cannot reset keyframe')
        if len(self.dataoffsets) != len(keyframe.dataoffsets):
            raise RuntimeError('incompatible keyframe')
        if keyframe.is_contiguous:
            self.databytecounts = keyframe.databytecounts
        self._keyframe = keyframe

    @property
    def is_contiguous(self):
        """Return offset and size of contiguous data, else None."""
        # if self._keyframe is None:
        #     raise RuntimeError('keyframe not set')
        if self._keyframe.is_contiguous:
            return self.dataoffsets[0], self._keyframe.is_contiguous[1]
        return None

    @property
    def is_memmappable(self):
        """Return if page's image data in file can be memory-mapped."""
        # if self._keyframe is None:
        #     raise RuntimeError('keyframe not set')
        return self._keyframe.is_memmappable

    @property
    def hash(self):
        """Return checksum to identify pages in same series."""
        # if self._keyframe is None:
        #     raise RuntimeError('keyframe not set')
        return self._keyframe.hash

    def __getattr__(self, name):
        """Return attribute from keyframe."""
        if name in TIFF.FRAME_ATTRS:
            return getattr(self._keyframe, name)
        # this error could be raised because an AttributeError was
        # raised inside a @property function
        raise AttributeError(
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
        )

    def __repr__(self):
        return f'<tifffile.TiffFrame {self.index} @{self.offset}>'

    def __str__(self, detail=0, width=79):
        """Return string containing information about TiffFrame."""
        if self._keyframe is None:
            info = ''
            kf = None
        else:
            info = '  '.join(
                s
                for s in (
                    'x'.join(str(i) for i in self.shape),
                    str(self.dtype),
                )
            )
            kf = TiffPage.__str__(self._keyframe, width=width - 11)
        if detail > 3:
            of = pformat(self.dataoffsets, width=width - 9, height=detail - 3)
            bc = pformat(
                self.databytecounts, width=width - 13, height=detail - 3
            )
            info = f'\n Keyframe {kf}\n Offsets {of}\n Bytecounts {bc}'
        return f'TiffFrame {self.index} @{self.offset}  {info}'


class TiffTag:
    """TIFF tag structure.

    Attributes
    ----------
    name : string
        Name of tag, TIFF.TAGS[code].
    code : int
        Decimal code of tag.
    dtype : int
        Datatype of tag data. One of TIFF.DATATYPES.
    count : int
        Number of values.
    value : various types
        Tag data as Python object.
    valueoffset : int
        Location of value in file.
    offset : int
        Location of tag structure in file.
    parent : TiffFile or TiffWriter
        Reference to parent TIFF file.

    All attributes are read-only.

    """

    __slots__ = (
        'parent',
        'offset',
        'code',
        'dtype',
        'count',
        '_value',
        'valueoffset',
    )

    def __init__(self, parent, offset, code, dtype, count, value, valueoffset):
        """Initialize TiffTag instance from values."""
        self.parent = parent
        self.offset = int(offset)
        self.code = int(code)
        self.count = int(count)
        self._value = value
        self.valueoffset = valueoffset
        try:
            self.dtype = TIFF.DATATYPES(dtype)
        except ValueError:
            self.dtype = int(dtype)

    @classmethod
    def fromfile(cls, parent, offset=None, header=None, validate=True):
        """Return TiffTag instance read from file."""
        tiff = parent.tiff

        if header is None:
            if offset is None:
                offset = parent.filehandle.tell()
            else:
                parent.filehandle.seek(offset)
            header = parent.filehandle.read(tiff.tagsize)
        elif offset is None:
            offset = parent.filehandle.tell()

        valueoffset = offset + tiff.tagsize - tiff.tagoffsetthreshold
        code, dtype = struct.unpack(tiff.tagformat1, header[:4])
        count, value = struct.unpack(tiff.tagformat2, header[4:])

        try:
            valueformat = TIFF.DATA_FORMATS[dtype]
        except KeyError:
            msg = (
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid data type {dtype!r}'
            )
            if validate:
                raise TiffFileError(msg)
            log_warning(msg)
            return cls(parent, offset, code, dtype, count, None, None)

        valuesize = count * struct.calcsize(valueformat)
        if (
            valuesize > tiff.tagoffsetthreshold
            or code in TIFF.TAG_READERS  # TODO: only works with offsets?
        ):
            valueoffset = struct.unpack(tiff.offsetformat, value)[0]
            if validate and code in TIFF.TAG_LOAD:
                value = TiffTag._read_value(
                    parent, offset, code, dtype, count, valueoffset
                )
            elif (
                valueoffset < 8
                or valueoffset + valuesize > parent.filehandle.size
            ):
                msg = (
                    f'<tifffile.TiffTag {code} @{offset}> '
                    f'invalid value offset {valueoffset}'
                )
                if validate:
                    raise TiffFileError(msg)
                log_warning(msg)
                value = None
            elif code in TIFF.TAG_LOAD:
                value = TiffTag._read_value(
                    parent, offset, code, dtype, count, valueoffset
                )
            else:
                value = None
        elif dtype == 1 or dtype == 2 or dtype == 7:
            # BYTES, ASCII, UNDEFINED
            value = value[:valuesize]
        elif (
            tiff.version == 42
            and tiff.offsetsize == 8
            and count == 1
            and (dtype == 4 or dtype == 13)
            and value[4:] != b'\x00\x00\x00\x00'
        ):
            # NDPI LONG or IFD
            value = struct.unpack('<Q', value)
        else:
            fmt = '{}{}{}'.format(
                tiff.byteorder, count * int(valueformat[0]), valueformat[1]
            )
            value = struct.unpack(fmt, value[:valuesize])

        value = TiffTag._process_value(value, code, dtype, offset)

        return cls(parent, offset, code, dtype, count, value, valueoffset)

    @staticmethod
    def _read_value(parent, offset, code, dtype, count, valueoffset):
        """Read tag value from file."""
        try:
            valueformat = TIFF.DATA_FORMATS[dtype]
        except KeyError:
            raise TiffFileError(
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid data type {dtype!r}'
            )

        fh = parent.filehandle
        tiff = parent.tiff

        valuesize = count * struct.calcsize(valueformat)
        if (
            valueoffset is None
            or valueoffset < 8
            or valueoffset + valuesize > fh.size
        ):
            raise TiffFileError(
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid value offset {valueoffset}'
            )
        # if valueoffset % 2:
        #     log_warning(
        #         f'<tifffile.TiffTag {code} @{offset}> '
        #         'value does not begin on word boundary'
        #     )

        fh.seek(valueoffset)
        if code in TIFF.TAG_READERS:
            readfunc = TIFF.TAG_READERS[code]
            value = readfunc(fh, tiff.byteorder, dtype, count, tiff.offsetsize)
        elif dtype == 1 or dtype == 2 or dtype == 7:
            # BYTES, ASCII, UNDEFINED
            value = fh.read(valuesize)
            if len(value) != valuesize:
                log_warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    'could not read all values'
                )
        elif code not in TIFF.TAG_TUPLE and count > 1024:
            value = read_numpy(
                fh, tiff.byteorder, dtype, count, tiff.offsetsize
            )
        else:
            fmt = '{}{}{}'.format(
                tiff.byteorder, count * int(valueformat[0]), valueformat[1]
            )
            value = struct.unpack(fmt, fh.read(valuesize))
        return value

    @staticmethod
    def _process_value(value, code, dtype, offset):
        """Process tag value. ."""
        if (
            value is None
            or dtype == 1  # BYTE
            or dtype == 7  # UNDEFINED
            or code in TIFF.TAG_READERS
            or not isinstance(value, (bytes, str, tuple))
        ):
            return value

        if dtype == 2:
            # TIFF ASCII fields can contain multiple strings,
            #   each terminated with a NUL
            try:
                value = bytes2str(stripnull(value, first=False).strip())
            except UnicodeDecodeError:
                log_warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    'coercing invalid ASCII to bytes'
                )
            return value

        if code in TIFF.TAG_ENUM:
            t = TIFF.TAG_ENUM[code]
            try:
                value = tuple(t(v) for v in value)
            except ValueError as exc:
                if code not in (259, 317):  # ignore compression/predictor
                    log_warning(f'<tifffile.TiffTag {code} @{offset}> {exc}')

        if len(value) == 1 and code not in TIFF.TAG_TUPLE:
            value = value[0]

        return value

    @property
    def value(self):
        """Return value of tag. Load from file if necessary."""
        if self._value is None:
            # print(
            #     f'_read_value {self.code} {TIFF.TAGS.get(self.code)} '
            #     f'{self.dtype}[{self.count}] @{self.valueoffset} '
            # )
            fh = self.parent.filehandle
            with fh.lock:
                closed = fh.closed
                if closed:
                    # this is an inefficient resort in case a user delay loads
                    # tag values from a TiffPage with a closed FileHandle.
                    warnings.warn(
                        f'{self!r} reading value from closed file', UserWarning
                    )
                    fh.open()
                try:
                    value = TiffTag._read_value(
                        self.parent,
                        self.offset,
                        self.code,
                        self.dtype,
                        self.count,
                        self.valueoffset,
                    )
                finally:
                    if closed:
                        fh.close()
            self._value = TiffTag._process_value(
                value,
                self.code,
                self.dtype,
                self.offset,
            )
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def dtype_name(self):
        try:
            return self.dtype.name
        except AttributeError:
            return f'TYPE{self.dtype}'

    @property
    def name(self):
        """Return name of tag from TIFF.TAGS registry."""
        return TIFF.TAGS.get(self.code, str(self.code))

    @property
    def dataformat(self):
        """Return data type as Python struct format."""
        return TIFF.DATA_FORMATS[self.dtype]

    @property
    def valuebytecount(self):
        """Return size of value in file."""
        return self.count * struct.calcsize(TIFF.DATA_FORMATS[self.dtype])

    def _astuple(self):
        """Return tag code, dtype, count, and encoded value.

        The encoded value is read from file if necessary.

        """
        # TODO: make this method public
        if isinstance(self.value, bytes):
            value = self.value
        else:
            dataformat = TIFF.DATA_FORMATS[self.dtype]
            count = self.count * int(dataformat[0])
            fmt = '{}{}{}'.format(
                self.parent.tiff.byteorder, count, dataformat[1]
            )
            try:
                if count == 1:
                    value = struct.pack(fmt, self.value)
                else:
                    value = struct.pack(fmt, *self.value)
            except Exception:
                tiff = self.parent.tiff
                if tiff.version == 42 and tiff.offsetsize == 8:
                    raise NotImplementedError(
                        'cannot read from NDPI > 4 GB files'
                    )
                fh = self.parent.filehandle
                pos = fh.tell()
                fh.seek(self.valueoffset)
                value = fh.read(struct.calcsize(fmt))
                fh.seek(pos)
        return self.code, self.dtype.value, self.count, value

    def overwrite(self, value, _arg=None, dtype=None, erase=True):
        """Write new tag value to file and return new TiffTag instance.

        The value must be compatible with the struct.pack formats in
        TIFF.DATA_FORMATS.

        The new packed value is appended to the file if it is longer than the
        old value. The old value is zeroed. The file position is left where it
        was.

        """
        if self.offset < 8 or self.valueoffset < 8:
            raise ValueError(f'cannot rewrite tag at offset {self.offset} < 8')

        if hasattr(value, 'filehandle'):
            value = _arg
            warnings.warn(
                '<tifffile.TiffTag.overwrite> passing a TiffFile instance is '
                'deprecated and no longer required since 2021.7.30.',
                DeprecationWarning,
                stacklevel=2,
            )

        fh = self.parent.filehandle
        tiff = self.parent.tiff

        if tiff.version == 42 and tiff.offsetsize == 8:
            # TODO: support patching NDPI > 4 GB files
            raise NotImplementedError('cannot patch NDPI > 4 GB files')

        if value is None:
            value = b''
        if dtype is None:
            dtype = self.dtype

        try:
            dataformat = TIFF.DATA_FORMATS[dtype]
        except KeyError as exc:
            try:
                dataformat = dtype[-1:]
                if dataformat[0] in '<>':
                    dataformat = dataformat[1:]
                dtype = TIFF.DATA_DTYPES[dataformat]
            except (KeyError, TypeError):
                raise ValueError(f'unknown dtype {dtype}') from exc

        if dtype == 2:
            # strings
            if isinstance(value, str):
                # enforce 7-bit ASCII on Unicode strings
                try:
                    value = value.encode('ascii')
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        'TIFF strings must be 7-bit ASCII'
                    ) from exc
            elif not isinstance(value, bytes):
                raise ValueError('TIFF strings must be 7-bit ASCII')
            if len(value) == 0 or value[-1] != b'\x00':
                value += b'\x00'
            count = len(value)
            value = (value,)
        elif isinstance(value, bytes):
            # pre-packed binary data
            dtsize = struct.calcsize(dataformat)
            if len(value) % dtsize:
                raise ValueError('invalid packed binary data')
            count = len(value) // dtsize
            value = (value,)
        else:
            try:
                count = len(value)
            except TypeError:
                value = (value,)
                count = 1

        if dtype in (5, 10):
            if count < 2 or count % 2:
                raise ValueError('invalid RATIONAL value')
            count //= 2  # rational

        packedvalue = struct.pack(
            '{}{}{}'.format(
                tiff.byteorder, count * int(dataformat[0]), dataformat[1]
            ),
            *value,
        )
        newsize = len(packedvalue)
        oldsize = self.count * struct.calcsize(TIFF.DATA_FORMATS[self.dtype])
        valueoffset = self.valueoffset

        pos = fh.tell()
        try:
            if dtype != self.dtype:
                # rewrite data type
                fh.seek(self.offset + 2)
                fh.write(struct.pack(tiff.byteorder + 'H', dtype))

            if oldsize <= tiff.tagoffsetthreshold:
                if newsize <= tiff.tagoffsetthreshold:
                    # inline -> inline: overwrite
                    fh.seek(self.offset + 4)
                    fh.write(struct.pack(tiff.tagformat2, count, packedvalue))
                else:
                    # inline -> separate: append to file
                    fh.seek(0, os.SEEK_END)
                    valueoffset = fh.tell()
                    if valueoffset % 2:
                        # value offset must begin on a word boundary
                        fh.write(b'\x00')
                        valueoffset += 1
                    fh.write(packedvalue)
                    fh.seek(self.offset + 4)
                    fh.write(
                        struct.pack(
                            tiff.tagformat2,
                            count,
                            struct.pack(tiff.offsetformat, valueoffset),
                        )
                    )
            elif newsize <= tiff.tagoffsetthreshold:
                # separate -> inline: erase old value
                valueoffset = self.offset + 4 + tiff.offsetsize
                fh.seek(self.offset + 4)
                fh.write(struct.pack(tiff.tagformat2, count, packedvalue))
                if erase:
                    fh.seek(self.valueoffset)
                    fh.write(b'\x00' * oldsize)
            elif newsize <= oldsize or self.valueoffset + oldsize == fh.size:
                # separate -> separate smaller: overwrite, erase remaining
                fh.seek(self.offset + 4)
                fh.write(struct.pack(tiff.offsetformat, count))
                fh.seek(self.valueoffset)
                fh.write(packedvalue)
                if erase and oldsize - newsize > 0:
                    fh.write(b'\x00' * (oldsize - newsize))
            else:
                # separate -> separate larger: erase old value, append to file
                if erase:
                    fh.seek(self.valueoffset)
                    fh.write(b'\x00' * oldsize)
                fh.seek(0, os.SEEK_END)
                valueoffset = fh.tell()
                if valueoffset % 2:
                    # value offset must begin on a word boundary
                    fh.write(b'\x00')
                    valueoffset += 1
                fh.write(packedvalue)
                fh.seek(self.offset + 4)
                fh.write(
                    struct.pack(
                        tiff.tagformat2,
                        count,
                        struct.pack(tiff.offsetformat, valueoffset),
                    )
                )
        finally:
            fh.seek(pos)  # must restore file position

        return TiffTag(
            self.parent,
            self.offset,
            self.code,
            dtype,
            count,
            value,
            valueoffset,
        )

    def _fix_lsm_bitspersample(self):
        """Correct LSM bitspersample tag.

        Old LSM writers may use a separate region for two 16-bit values,
        although they fit into the tag value element of the tag.

        """
        if self.code != 258 or self.count != 2:
            return
        # TODO: test this case; need example file
        log_warning(f'{self!r} correcting LSM bitspersample tag')
        value = struct.pack('<HH', *self.value)
        self.valueoffset = struct.unpack('<I', value)[0]
        self.parent.filehandle.seek(self.valueoffset)
        self.value = struct.unpack('<HH', self.parent.filehandle.read(4))

    def __repr__(self):
        return f'<tifffile.TiffTag {self.code} @{self.offset}>'

    def __str__(self, detail=0, width=79):
        """Return string containing information about TiffTag."""
        height = 1 if detail <= 0 else 8 * detail
        dtype = self.dtype_name
        if self.count > 1:
            dtype += f'[{self.count}]'
        name = '|'.join(TIFF.TAGS.getall(self.code, ()))
        if name:
            name = f'{self.code} {name} @{self.offset}'
        else:
            name = f'{self.code} @{self.offset}'
        line = f'TiffTag {name} {dtype} @{self.valueoffset} '
        line = line[:width]
        try:
            value = self.value
        except TiffFileError:
            value = 'CORRUPT'
        else:
            try:
                if self.count == 1:
                    value = enumstr(value)
                else:
                    value = pformat(tuple(enumstr(v) for v in value))
            except Exception:
                value = pformat(value, width=width, height=height)
        if detail <= 0:
            line += value[:width]
            line = line[:width]
        else:
            line += '\n' + value
        return line


class TiffTags:
    """Multidict like interface to TiffTag instances in TiffPage.

    Differences to a regular dict:

    * values are instances of TiffTag.
    * keys are TiffTag.code (int).
    * multiple values can be stored per key.
    * can be indexed with TiffTag.name (str), although slower than by key.
    * iter() returns values instead of keys.
    * values() and items() contain all values sorted by offset stored in file.
    * len() returns the number of all values.
    * get() takes an optional index argument.
    * some functions are not implemented, e.g. update, setdefault, pop.

    """

    __slots__ = ('_dict', '_list')

    def __init__(self):
        """Initialize empty instance."""
        self._dict = {}
        self._list = [self._dict]

    def add(self, tag):
        """Add a tag."""
        code = tag.code
        for d in self._list:
            if code not in d:
                d[code] = tag
                break
        else:
            self._list.append({code: tag})

    def keys(self):
        """Return new view of all codes."""
        return self._dict.keys()

    def values(self):
        """Return all tags in order they are stored in file."""
        tags = (t for d in self._list for t in d.values())
        return sorted(tags, key=lambda t: t.offset)

    def items(self):
        """Return all (code, tag) pairs in order tags are stored in file."""
        items = (i for d in self._list for i in d.items())
        return sorted(items, key=lambda i: i[1].offset)

    def valueof(self, key, default=None, index=None):
        """Return value of tag if exists, else default."""
        tag = self.get(key, default=None, index=index)
        if tag is None:
            return default
        try:
            return tag.value
        except TiffFileError:
            return default  # corrupted tag

    def get(self, key, default=None, index=None):
        """Return tag of code or name if exists, else default."""
        if index is None:
            if key in self._dict:
                return self._dict[key]
            if not isinstance(key, str):
                return default
            index = 0
        try:
            tags = self._list[index]
        except IndexError:
            return default
        if key in tags:
            return tags[key]
        if not isinstance(key, str):
            return default
        for tag in tags.values():
            if tag.name == key:
                return tag
        return default

    def getall(self, key, default=None):
        """Return list of all tags of code or name if exists, else default."""
        result = []
        for tags in self._list:
            if key in tags:
                result.append(tags[key])
            else:
                break
        if result:
            return result
        if not isinstance(key, str):
            return default
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    result.append(tag)
                    break
            if not result:
                break
        return result if result else default

    def __getitem__(self, key):
        """Return first tag of code or name. Raise KeyError if not found."""
        if key in self._dict:
            return self._dict[key]
        if not isinstance(key, str):
            raise KeyError(key)
        for tag in self._dict.values():
            if tag.name == key:
                return tag
        raise KeyError(key)

    def __setitem__(self, code, tag):
        """Add a tag."""
        self.add(tag)

    def __delitem__(self, key):
        """Delete all tags of code or name."""
        found = False
        for tags in self._list:
            if key in tags:
                found = True
                del tags[key]
            else:
                break
        if found:
            return None
        if not isinstance(key, str):
            raise KeyError(key)
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    del tags[tag.code]
                    found = True
                    break
            else:
                break
        if not found:
            raise KeyError(key)
        return None

    def __contains__(self, item):
        """Return if tag is in map."""
        if item in self._dict:
            return True
        if not isinstance(item, str):
            return False
        for tag in self._dict.values():
            if tag.name == item:
                return True
        return False

    def __iter__(self):
        """Return iterator over all tags."""
        return iter(self.values())

    def __len__(self):
        """Return number of tags."""
        size = 0
        for d in self._list:
            size += len(d)
        return size

    def __repr__(self):
        return f'<tifffile.TiffTags @0x{id(self):016X}>'

    def __str__(self, detail=0, width=79):
        """Return string with information about TiffTags."""
        info = []
        tlines = []
        vlines = []
        for tag in self:
            value = tag.__str__(width=width + 1)
            tlines.append(value[:width].strip())
            if detail > 0 and len(value) > width:
                try:
                    value = tag.value
                except Exception:
                    # delay load failed or closed file
                    continue
                if detail < 2 and tag.code in (273, 279, 324, 325):
                    value = pformat(value, width=width, height=detail * 4)
                else:
                    value = pformat(value, width=width, height=detail * 12)
                if tag.count > 1:
                    vlines.append(
                        f'{tag.name} {tag.dtype_name}[{tag.count}]\n{value}'
                    )
                else:
                    vlines.append(f'{tag.name}\n{value}')
        info.append('\n'.join(tlines))
        if detail > 0 and vlines:
            info.append('\n')
            info.append('\n\n'.join(vlines))
        return '\n'.join(info)


class TiffTagRegistry:
    """Registry of TIFF tag codes and names.

    The registry allows to look up tag codes and names by indexing with names
    and codes respectively.
    One tag code may be registered with several names, e.g. 34853 is used for
    GPSTag or OlympusSIS2.
    Different tag codes may be registered with the same name, e.g. 37387 and
    41483 are both named FlashEnergy.

    """

    __slots__ = ('_dict', '_list')

    def __init__(self, arg):
        self._dict = {}
        self._list = [self._dict]
        self.update(arg)

    def update(self, arg):
        """Add codes and names from sequence or dict to registry."""
        if isinstance(arg, TiffTagRegistry):
            self._list.extend(arg._list)
            return
        if isinstance(arg, dict):
            arg = arg.items()
        for code, name in arg:
            self.add(code, name)

    def add(self, code, name):
        """Add code and name to registry."""
        for d in self._list:
            if code in d and d[code] == name:
                break
            if code not in d and name not in d:
                d[code] = name
                d[name] = code
                break
        else:
            self._list.append({code: name, name: code})

    def items(self):
        """Return all registry items as (code, name)."""
        items = (
            i for d in self._list for i in d.items() if isinstance(i[0], int)
        )
        return sorted(items, key=lambda i: i[0])

    def get(self, key, default=None):
        """Return first code/name if exists, else default."""
        for d in self._list:
            if key in d:
                return d[key]
        return default

    def getall(self, key, default=None):
        """Return list of all codes/names if exists, else default."""
        result = [d[key] for d in self._list if key in d]
        return result if result else default

    def __getitem__(self, key):
        """Return first code/name. Raise KeyError if not found."""
        for d in self._list:
            if key in d:
                return d[key]
        raise KeyError(key)

    def __delitem__(self, key):
        """Delete all tags of code or name."""
        found = False
        for d in self._list:
            if key in d:
                found = True
                value = d[key]
                del d[key]
                del d[value]
        if not found:
            raise KeyError(key)

    def __contains__(self, item):
        """Return if code or name is in registry."""
        for d in self._list:
            if item in d:
                return True
        return False

    def __iter__(self):
        """Return iterator over all items in registry."""
        return iter(self.items())

    def __len__(self):
        """Return number of registered tags."""
        size = 0
        for d in self._list:
            size += len(d)
        return size // 2

    def __repr__(self):
        return f'<tifffile.TiffTagRegistry @0x{id(self):016X}>'

    def __str__(self):
        """Return string with information about TiffTags."""
        return 'TiffTagRegistry(((\n  {}\n))'.format(
            ',\n  '.join(f'({code}, {name!r})' for code, name in self.items())
        )


class TiffPageSeries:
    """Series of TIFF pages with compatible shape and data type (same hash).

    Attributes
    ----------
    pages : list of TiffPage, TiffFrame, or None
        Sequence of TiffPages or TiffFrame in series.
        May be None if pages or files of pages are missing in the series.
        The file handles of TiffPages or TiffFrames may not be open.
    keyframe : TiffPage
        A key frame of the series.
    dtype : numpy.dtype
        Data type (native byte order) of the image array in series.
    shape : tuple
        Dimensions of the image array in series.
    axes : str
        Labels of axes in shape. See TIFF.AXES_LABELS.
    offset : int or None
        Position of image data in file if memory-mappable, else None.
    levels : list of TiffPageSeries
        Pyramid levels. levels[0] is 'self'.

    """

    def __init__(
        self,
        pages,
        shape=None,
        dtype=None,
        axes=None,
        parent=None,
        name=None,
        transform=None,
        kind=None,
        truncated=False,
        multifile=False,
        squeeze=True,
    ):
        """Initialize instance."""
        self.index = 0
        self._pages = pages  # might contain only first of contiguous pages
        self.levels = [self]
        if shape is None:
            shape = pages[0].shape
        if axes is None:
            axes = pages[0].axes
        if dtype is None:
            dtype = pages[0].dtype

        self.set_shape_axes(shape, axes, squeeze)

        self.dtype = numpy.dtype(dtype)
        self.kind = kind if kind else ''
        self.name = name if name else ''
        self.transform = transform
        self.keyframe = next(p.keyframe for p in pages if p is not None)
        self.is_multifile = bool(multifile)

        if parent:
            self.parent = parent
        elif pages:
            self.parent = self.keyframe.parent
        else:
            self.parent = None
        if not truncated and len(pages) == 1:
            s = product(pages[0].shape)
            if s > 0:
                self._len = int(product(self.shape) // s)
            else:
                self._len = len(pages)
        else:
            self._len = len(pages)

    def set_shape_axes(self, shape, axes, squeeze=True):
        """Set shape and axes."""
        shape = tuple(shape)
        axes = ''.join(axes)
        # expanded shape according to metadata
        self._shape_expanded = shape
        self._axes_expanded = axes
        # squeezed shape and axes
        self._shape_squeezed, self._axes_squeezed = squeeze_axes(shape, axes)
        # default shape and axes returned by asarray
        self.shape = self._shape_squeezed if squeeze else self._shape_expanded
        self.axes = self._axes_squeezed if squeeze else self._axes_expanded

    def get_shape(self, squeeze=None):
        """Return default, squeezed, or expanded shape."""
        if squeeze is None:
            return self.shape
        return self._shape_squeezed if squeeze else self._shape_expanded

    def get_axes(self, squeeze=None):
        """Return default, squeezed, or expanded axes."""
        if squeeze is None:
            return self.axes
        return self._axes_squeezed if squeeze else self._axes_expanded

    def asarray(self, level=None, **kwargs):
        """Return image data from series of TIFF pages as numpy array."""
        if level is not None:
            return self.levels[level].asarray(**kwargs)
        if self.parent:
            result = self.parent.asarray(series=self, **kwargs)
            if self.transform is not None:
                result = self.transform(result)
            return result
        return None

    def aszarr(self, level=None, **kwargs):
        """Return image data from series of TIFF pages as zarr storage."""
        if self.parent:
            return ZarrTiffStore(self, level=level, **kwargs)
        return None

    @lazyattr
    def offset(self):
        """Return offset to series data in file, if any."""
        if not self._pages:
            return None

        pos = 0
        for page in self._pages:
            if page is None:
                return None
            if not page.is_final:
                return None
            if not pos:
                pos = page.is_contiguous[0] + page.is_contiguous[1]
                continue
            if pos != page.is_contiguous[0]:
                return None
            pos += page.is_contiguous[1]

        page = self._pages[0]
        offset = page.is_contiguous[0]
        if (page.is_imagej or page.is_shaped or page.is_stk) and len(
            self._pages
        ) == 1:
            # truncated files
            return offset
        if pos == offset + product(self.shape) * self.dtype.itemsize:
            return offset
        return None

    @property
    def is_pyramidal(self):
        """Return if series contains several levels."""
        return len(self.levels) > 1

    @property
    def ndim(self):
        """Return number of array dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Return number of elements in array."""
        return int(product(self.shape))

    @property
    def pages(self):
        """Return sequence of all pages in series."""
        # a workaround to keep the old interface working
        return self

    def _getitem(self, key):
        """Return specified page of series from cache or file."""
        key = int(key)
        if key < 0:
            key %= self._len
        if len(self._pages) == 1 and 0 < key < self._len:
            index = self._pages[0].index
            return self.parent.pages._getitem(index + key)
        return self._pages[key]

    def __getitem__(self, key):
        """Return specified page(s)."""
        if isinstance(key, (int, numpy.integer)):
            return self._getitem(key)
        if isinstance(key, slice):
            return [self._getitem(i) for i in range(*key.indices(self._len))]
        if isinstance(key, Iterable):
            return [self._getitem(k) for k in key]
        raise TypeError('key must be an integer, slice, or iterable')

    def __iter__(self):
        """Return iterator over pages in series."""
        if len(self._pages) == self._len:
            yield from self._pages
        else:
            pages = self.parent.pages
            index = self._pages[0].index
            for i in range(self._len):
                yield pages[index + i]

    def __len__(self):
        """Return number of pages in series."""
        return self._len

    def __repr__(self):
        return f'<tifffile.TiffPageSeries {self.index}>'

    def __str__(self):
        """Return string with information about TiffPageSeries."""
        s = '  '.join(
            s
            for s in (
                snipstr(f'{self.name!r}', 20) if self.name else '',
                'x'.join(str(i) for i in self.shape),
                str(self.dtype),
                self.axes,
                self.kind,
                (f'{len(self.levels)} Levels') if self.is_pyramidal else '',
                f'{len(self)} Pages',
                (f'@{self.offset}') if self.offset else '',
            )
            if s
        )
        return f'TiffPageSeries {self.index}  {s}'


class ZarrStore(MutableMapping):
    """Zarr storage base class.

    ZarrStore instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    https://zarr.readthedocs.io/en/stable/spec/v2.html
    https://forum.image.sc/t/multiscale-arrays-v0-1/37930

    """

    def __init__(self, fillvalue=None, chunkmode=None):
        """Initialize ZarrStore."""
        self._store = {}
        self._fillvalue = 0 if fillvalue is None else fillvalue
        if chunkmode is None:
            self._chunkmode = TIFF.CHUNKMODE(0)
        else:
            self._chunkmode = enumarg(TIFF.CHUNKMODE, chunkmode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Close ZarrStore."""

    def flush(self):
        """Flush ZarrStore."""
        raise PermissionError('ZarrStore is read-only')

    def clear(self):
        """Clear ZarrStore."""
        raise PermissionError('ZarrStore is read-only')

    def keys(self):
        """Return keys in ZarrStore."""
        return self._store.keys()

    def items(self):
        """Return items in ZarrStore."""
        return self._store.items()

    def values(self):
        """Return values in ZarrStore."""
        return self._store.values()

    def __iter__(self):
        return iter(self._store.keys())

    def __len__(self):
        return len(self._store)

    def __delitem__(self, key):
        raise PermissionError('ZarrStore is read-only')

    def __contains__(self, key):
        return key in self._store

    def __setitem__(self, key, value):
        raise PermissionError('ZarrStore is read-only')

    def __getitem__(self, key):
        if key in self._store:
            return self._store[key]
        return self._getitem(key)

    def _getitem(self, key):
        """Return chunk from file."""
        raise NotImplementedError

    @property
    def is_multiscales(self):
        """Return if ZarrStore is multiscales."""
        return b'multiscales' in self._store['.zattrs']

    @staticmethod
    def _empty_chunk(shape, dtype, fillvalue):
        """Return empty chunk."""
        if fillvalue is None or fillvalue == 0:
            return bytes(product(shape) * dtype.itemsize)
        chunk = numpy.empty(shape, dtype)
        chunk[:] = fillvalue
        return chunk  # .tobytes()

    @staticmethod
    def _dtype(dtype):
        """Return dtype as string with native byte order."""
        if dtype.itemsize == 1:
            byteorder = '|'
        else:
            byteorder = {'big': '>', 'little': '<'}[sys.byteorder]
        return byteorder + dtype.str[1:]

    @staticmethod
    def _json(obj):
        """Serialize obj to a JSON formatted string."""
        return json.dumps(
            obj,
            indent=1,
            sort_keys=True,
            ensure_ascii=True,
            separators=(',', ': '),
        ).encode('ascii')

    @staticmethod
    def _value(value, dtype):
        """Return value which is serializable to JSON."""
        if value is None:
            return value
        if dtype.kind == 'b':
            return bool(value)
        if dtype.kind in 'ui':
            return int(value)
        if dtype.kind == 'f':
            if numpy.isnan(value):
                return 'NaN'
            if numpy.isposinf(value):
                return 'Infinity'
            if numpy.isneginf(value):
                return '-Infinity'
            return float(value)
        if dtype.kind in 'c':
            value = numpy.array(value, dtype)
            return (
                ZarrStore._value(value.real, dtype.type().real.dtype),
                ZarrStore._value(value.imag, dtype.type().imag.dtype),
            )
        return value

    @staticmethod
    def _ndindex(shape, chunks):
        """Return iterator over all chunk index strings."""
        assert len(shape) == len(chunks)
        chunked = tuple(
            i // j + (1 if i % j else 0) for i, j in zip(shape, chunks)
        )
        for indices in numpy.ndindex(chunked):
            yield '.'.join(str(index) for index in indices)


class ZarrTiffStore(ZarrStore):
    """Zarr storage interface to image data in TiffPage or TiffPageSeries.

    ZarrTiffStore instances are using a TiffFile instance for reading and
    decoding chunks. Therefore ZarrTiffStore instances cannot be pickled.

    """

    def __init__(
        self,
        arg,
        level=None,
        chunkmode=None,
        fillvalue=None,
        zattrs=None,
        lock=None,
        squeeze=None,
        maxworkers=None,
        _openfiles=None,
    ):
        """Initialize Zarr storage.

        Parameters
        ----------
        arg : TiffPage or TiffPageSeries
            The TiffPage or TiffPageSeries instance to wrap as a zarr store.
        level : int (optional)
            Specifies a pyramidal level to wrap.
        chunkmode : {0, 2} (optional)
            Specifies to use strips/tiles (0, the default) or whole page data
            (2) as chunks.
        fillvalue : number (optional)
            Value to use for missing chunks of the Zarr store. Default: 0.
        zattrs : dict (optional)
            Additional attributes to store in .zattrs.
        lock : {RLock, NullContext} (optional)
            A reentrant lock used to synchronize seeks and reads from file.
            If None (default), the lock of the parent's filehandle is used.
        squeeze : bool (optional)
            Squeeze shape of TiffPageSeries.
        maxworkers : int or None
            Maximum number of threads to concurrently decode strips or tiles
            if chunkmode=2.  If None (default), up to half the CPU cores are
            used. See remarks in TiffFile.asarray.

        """
        super().__init__(fillvalue=fillvalue, chunkmode=chunkmode)

        if self._chunkmode not in (0, 2):
            raise NotImplementedError(f'{self._chunkmode!r} not implemented')

        self._maxworkers = maxworkers
        self._squeeze = None if squeeze is None else bool(squeeze)
        self._transform = getattr(arg, 'transform', None)
        self._data = getattr(arg, 'levels', [TiffPageSeries([arg])])
        if level is not None:
            self._data = [self._data[level]]

        if lock is None:
            fh = self._data[0].keyframe.parent._parent.filehandle
            fh.lock = True
            lock = fh.lock
        self._filecache = FileCache(size=_openfiles, lock=lock)

        zattrs = {} if zattrs is None else dict(zattrs)

        if len(self._data) > 1:
            # multiscales
            self._store['.zgroup'] = ZarrStore._json({'zarr_format': 2})
            self._store['.zattrs'] = ZarrStore._json(
                {
                    'multiscales': [
                        {
                            'datasets': [
                                {'path': str(i)}
                                for i in range(len(self._data))
                            ],
                            'metadata': {},
                            'name': arg.name,
                            # 'type': 'gaussian',
                            'version': '0.1',
                        }
                    ],
                    **zattrs,
                }
            )
            for level, series in enumerate(self._data):
                series.keyframe.decode  # cache decode function
                shape = series.get_shape(squeeze)
                dtype = series.dtype
                if fillvalue is None:
                    self._fillvalue = fillvalue = series.keyframe.nodata
                if self._chunkmode:
                    chunks = series.keyframe.shape
                else:
                    chunks = series.keyframe.chunks
                self._store[f'{level}/.zarray'] = ZarrStore._json(
                    {
                        'zarr_format': 2,
                        'shape': shape,
                        'chunks': ZarrTiffStore._chunks(chunks, shape),
                        'dtype': ZarrStore._dtype(dtype),
                        'compressor': None,
                        'fill_value': ZarrStore._value(fillvalue, dtype),
                        'order': 'C',
                        'filters': None,
                    }
                )
        else:
            series = self._data[0]
            series.keyframe.decode  # cache decode function
            shape = series.get_shape(squeeze)
            dtype = series.dtype
            if fillvalue is None:
                self._fillvalue = fillvalue = series.keyframe.nodata
            if self._chunkmode:
                chunks = series.keyframe.shape
            else:
                chunks = series.keyframe.chunks
            self._store['.zattrs'] = ZarrStore._json(zattrs)
            self._store['.zarray'] = ZarrStore._json(
                {
                    'zarr_format': 2,
                    'shape': shape,
                    'chunks': ZarrTiffStore._chunks(chunks, shape),
                    'dtype': ZarrStore._dtype(dtype),
                    'compressor': None,
                    'fill_value': ZarrStore._value(fillvalue, dtype),
                    'order': 'C',
                    'filters': None,
                }
            )

    def close(self):
        """Close ZarrTiffStore."""
        if hasattr(self, '_filecache'):
            self._filecache.clear()

    def write_fsspec(
        self,
        arg,
        url,
        groupname=None,
        compressors=None,
        version=None,
        _append=False,
    ):
        """Write fsspec ReferenceFileSystem as JSON to file.

        Url is the remote location of the TIFF file without the file name(s).

        Raise ValueError if TIFF store can not be represented as
        ReferenceFileSystem due to features that are not supported by zarr,
        numcodecs, or imagecodecs:

        * compressors, e.g. CCITT
        * filters, e.g. bitorder reversal, packed integers
        * dtypes, e.g. float24
        * JPEGTables in multi-page files
        * incomplete chunks, e.g. if imagelength % rowsperstrip != 0

        Files containing incomplete tiles may fail at runtime.

        https://github.com/intake/fsspec-reference-maker

        """
        compressors = {
            1: None,
            8: 'zlib',
            32946: 'zlib',
            34925: 'lzma',
            50000: 'zstd',
            5: 'imagecodecs_lzw',
            7: 'imagecodecs_jpeg',
            22610: 'imagecodecs_jpegxr',  # NDPI
            32773: 'imagecodecs_packbits',
            33003: 'imagecodecs_jpeg2k',
            33004: 'imagecodecs_jpeg2k',
            33005: 'imagecodecs_jpeg2k',
            33007: 'imagecodecs_jpeg',  # ALT_JPG
            34712: 'imagecodecs_jpeg2k',
            34887: 'imagecodecs_lerc',
            34892: 'imagecodecs_jpeg',  # DNG lossy
            34933: 'imagecodecs_png',
            34934: 'imagecodecs_jpegxr',  # ZIF
            50001: 'imagecodecs_webp',
            50002: 'imagecodecs_jpegxl',
            **({} if compressors is None else compressors),
        }

        for series in self._data:
            errormsg = ' not supported by the fsspec ReferenceFileSystem'
            keyframe = series.keyframe
            if keyframe.compression not in compressors:
                raise ValueError(f'{keyframe.compression!r} is' + errormsg)
            if keyframe.fillorder != 1:
                raise ValueError(f'{keyframe.fillorder!r} is' + errormsg)
            if keyframe.sampleformat not in (1, 2, 3, 6):
                # TODO: support float24 and cint via filters?
                raise ValueError(f'{keyframe.sampleformat!r} is' + errormsg)
            if keyframe.bitspersample not in (
                8,
                16,
                32,
                64,
                128,
            ) and keyframe.compression not in (
                7,
                33007,
                34892,
            ):  # JPEG
                raise ValueError(
                    f'BitsPerSample {keyframe.bitspersample} is' + errormsg
                )
            if (
                not self._chunkmode
                and not keyframe.is_tiled
                and keyframe.imagelength % keyframe.rowsperstrip
            ):
                raise ValueError('incomplete chunks are' + errormsg)
            if self._chunkmode and not keyframe.is_final:
                raise ValueError(f'{self._chunkmode!r} is' + errormsg)
            if keyframe.jpegtables is not None and len(series.pages) > 1:
                raise ValueError(
                    'JPEGTables in multi-page files are' + errormsg
                )

        if url is None:
            url = ''
        elif url and url[-1] != '/':
            url += '/'

        if groupname is None:
            groupname = ''
        elif groupname and groupname[-1] != '/':
            groupname += '/'

        byteorder = '<' if sys.byteorder == 'big' else '>'
        if (
            self._data[0].keyframe.parent.byteorder != byteorder
            or self._data[0].keyframe.dtype.itemsize == 1
        ):
            byteorder = None

        refs = dict()
        if version == 1:
            if _append:
                raise ValueError('cannot append to version 1')
            refs['version'] = 1
            refs['templates'] = {}
            refs['gen'] = []
            templates = {}
            if self._data[0].is_multifile:
                i = 0
                for page in self._data[0].pages:
                    if page is None:
                        continue
                    fname = page.keyframe.parent.filehandle.name
                    if fname in templates:
                        continue
                    key = f'u{i}'
                    templates[fname] = '{{%s}}' % key
                    refs['templates'][key] = url + fname
                    i += 1
            else:
                fname = self._data[0].keyframe.parent.filehandle.name
                key = 'u'
                templates[fname] = '{{%s}}' % key
                refs['templates'][key] = url + fname

            refs['refs'] = refzarr = dict()
        else:
            refzarr = refs

        if groupname and not _append:
            # TODO: support nested groups
            refzarr['.zgroup'] = ZarrStore._json({'zarr_format': 2}).decode()

        for key, value in self._store.items():
            if '.zarray' in key:
                level = int(key.split('/')[0]) if '/' in key else 0
                keyframe = self._data[level].keyframe
                value = json.loads(value)
                codec_id = compressors[keyframe.compression]
                if codec_id == 'imagecodecs_jpeg':
                    # TODO: handle JPEG colorspaces
                    tables = keyframe.jpegtables
                    if tables is not None:
                        import base64

                        tables = base64.b64encode(tables).decode()
                    header = keyframe.jpegheader
                    if header is not None:
                        import base64

                        header = base64.b64encode(header).decode()
                    colorspace_jpeg, colorspace_data = jpeg_decode_colorspace(
                        keyframe.photometric,
                        keyframe.planarconfig,
                        keyframe.extrasamples,
                    )
                    value['compressor'] = {
                        'id': codec_id,
                        'tables': tables,
                        'header': header,
                        'bitspersample': keyframe.bitspersample,
                        'colorspace_jpeg': colorspace_jpeg,
                        'colorspace_data': colorspace_data,
                    }
                elif codec_id is not None:
                    value['compressor'] = {'id': codec_id}
                if keyframe.predictor > 1:
                    # predictors need access to chunk shape and dtype
                    # requires imagecodecs > 2021.8.26 to read
                    if keyframe.predictor in (2, 34892, 34893):
                        filter_id = 'imagecodecs_delta'
                    else:
                        filter_id = 'imagecodecs_floatpred'
                    if keyframe.predictor <= 3:
                        dist = 1
                    elif keyframe.predictor in (34892, 34894):
                        dist = 2
                    else:
                        dist = 4
                    if (
                        keyframe.planarconfig == 1
                        and keyframe.samplesperpixel > 1
                    ):
                        axis = -2
                    else:
                        axis = -1
                    value['filters'] = [
                        {
                            'id': filter_id,
                            'axis': axis,
                            'dist': dist,
                            'shape': value['chunks'],
                            'dtype': value['dtype'],
                        }
                    ]
                if byteorder is not None:
                    value['dtype'] = byteorder + value['dtype'][1:]
                value = ZarrStore._json(value)

            refzarr[groupname + key] = value.decode()

        if hasattr(arg, 'write'):
            fh = arg
        else:
            fh = open(arg, 'w')

        if version == 1:
            fh.write(json.dumps(refs, indent=1).rsplit('}"', 1)[0] + '}"')
            indent = '  '
        elif _append:
            fh.write(',\n')
            fh.write(json.dumps(refs, indent=1)[2:-2])
            indent = ' '
        else:
            fh.write(json.dumps(refs, indent=1)[:-2])
            indent = ' '

        for key, value in self._store.items():
            if '.zarray' in key:
                value = json.loads(value)
                shape = value['shape']
                chunks = value['chunks']
                level = (key.split('/')[0] + '/') if '/' in key else ''
                for chunkindex in ZarrStore._ndindex(shape, chunks):
                    key = level + chunkindex
                    keyframe, page, _, offset, bytecount = self._parse_key(key)
                    if page and self._chunkmode and offset is None:
                        offset = page.dataoffsets[0]
                        bytecount = keyframe.nbytes
                    if offset and bytecount:
                        fname = keyframe.parent.filehandle.name
                        if version == 1:
                            fname = templates[fname]
                        else:
                            fname = f'{url}{fname}'
                        fh.write(
                            f',\n{indent}"{groupname}{key}": '
                            f'["{fname}", {offset}, {bytecount}]'
                        )

        # TODO: support nested groups
        if version == 1:
            fh.write('\n }\n}')
        elif not _append:
            fh.write('\n}')

        if not hasattr(arg, 'write'):
            fh.close()

    def _getitem(self, key):
        """Return chunk from file."""
        keyframe, page, chunkindex, offset, bytecount = self._parse_key(key)

        if self._chunkmode:
            chunks = keyframe.shape
        else:
            chunks = keyframe.chunks

        if page is None or offset == 0 or bytecount == 0:
            chunk = ZarrStore._empty_chunk(
                chunks, keyframe.dtype, self._fillvalue
            )
            if self._transform is not None:
                chunk = self._transform(chunk)
            return chunk

        fh = page.parent.filehandle

        if self._chunkmode and offset is None:
            self._filecache.open(fh)
            chunk = page.asarray(
                lock=self._filecache.lock, maxworkers=self._maxworkers
            )
            self._filecache.close(fh)
            if self._transform is not None:
                chunk = self._transform(chunk)
            return chunk

        chunk = self._filecache.read(fh, offset, bytecount)

        decodeargs = {'_fullsize': True}
        if page.jpegtables is not None:
            decodeargs['jpegtables'] = page.jpegtables
        if keyframe.jpegheader is not None:
            decodeargs['jpegheader'] = keyframe.jpegheader

        chunk = keyframe.decode(chunk, chunkindex, **decodeargs)[0]
        if self._transform is not None:
            chunk = self._transform(chunk)

        if chunk.size != product(chunks):
            raise RuntimeError(f'{chunk.size} != {product(chunks)}')
        return chunk  # .tobytes()

    def _parse_key(self, key):
        """Return keyframe, page, index, offset, and bytecount from key."""
        if len(self._data) > 1:
            # multiscales
            level, key = key.split('/')
            series = self._data[int(level)]
        else:
            series = self._data[0]
        keyframe = series.keyframe
        pageindex, chunkindex = self._indices(key, series)
        if pageindex > 0 and len(series) == 1:
            # truncated ImageJ, STK, or shaped
            if series.offset is None:
                raise RuntimeError('truncated series is not contiguous')
            page = series[0]
            if page is None:
                return keyframe, None, chunkindex, 0, 0
            offset = pageindex * page.size * page.dtype.itemsize
            offset += page.dataoffsets[chunkindex]
            if self._chunkmode:
                bytecount = page.size * page.dtype.itemsize
                return page.keyframe, page, chunkindex, offset, bytecount
        elif self._chunkmode:
            with self._filecache.lock:
                page = series[pageindex]
            if page is None:
                return keyframe, None, None, 0, 0
            return page.keyframe, page, None, None, None
        else:
            with self._filecache.lock:
                page = series[pageindex]
            if page is None:
                return keyframe, None, chunkindex, 0, 0
            offset = page.dataoffsets[chunkindex]
        bytecount = page.databytecounts[chunkindex]
        return page.keyframe, page, chunkindex, offset, bytecount

    def _indices(self, key, series):
        """Return page and strile indices from zarr chunk index."""
        keyframe = series.keyframe
        shape = series.get_shape(self._squeeze)
        indices = [int(i) for i in key.split('.')]
        assert len(indices) == len(shape)
        if self._chunkmode:
            chunked = (1,) * len(keyframe.shape)
        else:
            chunked = keyframe.chunked
        p = 1
        for i, s in enumerate(shape[::-1]):
            p *= s
            if p == keyframe.size:
                i = len(indices) - i - 1
                frames_indices = indices[:i]
                strile_indices = indices[i:]
                frames_chunked = shape[:i]
                strile_chunked = list(shape[i:])  # updated later
                break
        else:
            raise RuntimeError
        if len(strile_chunked) == len(keyframe.shape):
            strile_chunked = chunked
        else:
            # get strile_chunked including singleton dimensions
            i = len(strile_indices) - 1
            j = len(keyframe.shape) - 1
            while True:
                if strile_chunked[i] == keyframe.shape[j]:
                    strile_chunked[i] = chunked[j]
                    i -= 1
                    j -= 1
                elif strile_chunked[i] == 1:
                    i -= 1
                else:
                    raise RuntimeError('shape does not match page shape')
                if i < 0 or j < 0:
                    break
            assert product(strile_chunked) == product(chunked)
        if len(frames_indices) > 0:
            frameindex = int(
                numpy.ravel_multi_index(frames_indices, frames_chunked)
            )
        else:
            frameindex = 0
        if len(strile_indices) > 0:
            strileindex = int(
                numpy.ravel_multi_index(strile_indices, strile_chunked)
            )
        else:
            strileindex = 0
        return frameindex, strileindex

    @staticmethod
    def _chunks(chunks, shape):
        """Return chunks with same length as shape."""
        ndim = len(shape)
        if ndim == 0:
            return ()  # empty array
        if 0 in shape:
            return (1,) * ndim
        newchunks = []
        i = ndim - 1
        j = len(chunks) - 1
        while True:
            if j < 0:
                newchunks.append(1)
                i -= 1
            elif shape[i] > 1 and chunks[j] > 1:
                newchunks.append(chunks[j])
                i -= 1
                j -= 1
            elif shape[i] == chunks[j]:  # both 1
                newchunks.append(1)
                i -= 1
                j -= 1
            elif shape[i] == 1:
                newchunks.append(1)
                i -= 1
            elif chunks[j] == 1:
                newchunks.append(1)
                j -= 1
            else:
                raise RuntimeError
            if i < 0 or ndim == len(newchunks):
                break
        # assert ndim == len(newchunks)
        return tuple(newchunks[::-1])

    def __repr__(self):
        return f'<tifffile.ZarrTiffStore @0x{id(self):016X}>'


class ZarrFileSequenceStore(ZarrStore):
    """Zarr storage interface to image data in FileSequence."""

    def __init__(
        self,
        arg,
        fillvalue=None,
        chunkmode=None,
        chunkshape=None,
        dtype=None,
        axestiled=None,
        zattrs=None,
        **kwargs,
    ):
        """Initialize Zarr storage from FileSequence.

        Parameters
        ----------
        arg: FileSequence
            FileSequence instance to wrap as zarr store. Files in containers
            are not supported.
        fillvalue : number (optional)
            Default value to use for missing chunks of the Zarr store.
            Default: 0.
        chunkmode: {0, 3} (optional)
            Currently only one chunk per file is supported.
        chunkshape : tuple of int (optional)
            Shape of the chunk in each file.
            Must match `arg.imread(file, **kwargs).shape`.
        dtype : numpy.dtype (optional)
            Data type of the chunk in each file.
            Must match `arg.imread(file, **kwargs).dtype`.
        axestiled: dict (optional)
           Defines the axes to be tiled. Map stacked sequence axis to
           chunk axis.
        zattrs : dict
            Additional attributes to store in .zattrs.
        kwargs: dict
            Additional parameters passed to the FileSequence.imread function.

        Notes
        -----
        If chunkshape or dtype are None (default), their values are determined
        by reading the first file using `arg.imread(arg.files[0], **kwargs)`.

        """
        super().__init__(fillvalue=fillvalue, chunkmode=chunkmode)

        if self._chunkmode not in (0, 3):
            raise ValueError(f'invalid chunkmode {self._chunkmode!r}')

        if not isinstance(arg, FileSequence):
            raise TypeError('not a FileSequence')

        if arg._container:
            raise NotImplementedError('cannot open container as zarr storage')

        self._kwargs = kwargs
        self._imread = arg.imread
        self._cache = {}  # TODO: cache MRU number chunks
        self._commonpath = arg.commonpath()

        if chunkshape is None or dtype is None:
            chunk = arg.imread(arg.files[0], **kwargs)
            self._chunks = chunk.shape
            self._dtype = chunk.dtype
        else:
            self._chunks = tuple(chunkshape)
            self._dtype = numpy.dtype(dtype)
            chunk = None

        self._tiled = TiledSequence(arg.shape, self._chunks, axestiled)
        self._lookup = dict(zip(self._tiled.indices(arg.indices), arg.files))
        if chunk is not None:
            self._cache[next(self._tiled.indices(arg.indices[0:1]))] = chunk

        zattrs = {} if zattrs is None else dict(zattrs)

        self._store['.zattrs'] = ZarrStore._json(zattrs)
        self._store['.zarray'] = ZarrStore._json(
            {
                'zarr_format': 2,
                'shape': self._tiled.shape,
                'chunks': self._tiled.chunks,
                'dtype': ZarrStore._dtype(self._dtype),
                'compressor': None,
                'fill_value': ZarrStore._value(fillvalue, self._dtype),
                'order': 'C',
                'filters': None,
            }
        )

    def _getitem(self, key):
        """Return chunk from file."""
        indices = tuple(int(i) for i in key.split('.'))
        if indices in self._cache:
            return self._cache[indices]
        self._cache.clear()
        filename = self._lookup.get(indices, None)
        if filename is None:
            chunk = ZarrStore._empty_chunk(
                self._chunks, self._dtype, self._fillvalue
            )
        else:
            chunk = self._imread(filename, **self._kwargs)  # .tobytes()
        self._cache[indices] = chunk
        return chunk

    def close(self):
        """Clear chunk cache."""
        self._cache.clear()

    def write_fsspec(
        self,
        arg,
        url,
        groupname=None,
        codec_id=None,
        version=None,
        _append=False,
    ):
        """Write fsspec ReferenceFileSystem as JSON to file.

        Url is the remote location of the files without the file names.

        """
        from urllib.parse import quote

        kwargs = self._kwargs.copy()

        if codec_id is not None:
            pass
        elif self._imread == imread:
            codec_id = 'tifffile'
        elif 'imagecodecs.' in self._imread.__module__:
            if (
                self._imread.__name__ != 'imread'
                or 'codec' not in self._kwargs
            ):
                raise ValueError('can not determine codec_id')
            codec = kwargs.pop('codec')
            if isinstance(codec, (list, tuple)):
                codec = codec[0]
            if callable(codec):
                codec = codec.__name__.split('_')[0]
            codec_id = {
                'avif': 'imagecodecs_avif',
                'gif': 'imagecodecs_gif',
                'jpeg': 'imagecodecs_jpeg',
                'jpeg8': 'imagecodecs_jpeg',
                'jpeg12': 'imagecodecs_jpeg',
                'jpeg2k': 'imagecodecs_jpeg2k',
                'jpegls': 'imagecodecs_jpegls',
                'jpegxl': 'imagecodecs_jpegxl',
                'jpegxr': 'imagecodecs_jpegxr',
                'ljpeg': 'imagecodecs_ljpeg',
                # 'npy': 'imagecodecs_npy',
                'png': 'imagecodecs_png',
                'tiff': 'imagecodecs_tiff',
                'webp': 'imagecodecs_webp',
                'zfp': 'imagecodecs_zfp',
            }[codec]
        else:
            raise ValueError('can not determine codec_id')

        if url is None:
            url = ''
        elif url and url[-1] != '/':
            url += '/'

        if groupname is None:
            groupname = ''
        elif groupname and groupname[-1] != '/':
            groupname += '/'

        refs = dict()
        if version == 1:
            if _append:
                raise ValueError('cannot append when using version 1')
            refs['version'] = 1
            refs['templates'] = {'u': url}
            refs['gen'] = []
            refs['refs'] = refzarr = dict()
            url = '{{u}}'
        else:
            refzarr = refs

        if groupname and not _append:
            refzarr['.zgroup'] = ZarrStore._json({'zarr_format': 2}).decode()

        for key, value in self._store.items():
            if '.zarray' in key:
                value = json.loads(value)
                # TODO: make kwargs serializable
                value['compressor'] = {'id': codec_id, **kwargs}
                value = ZarrStore._json(value)
            refzarr[groupname + key] = value.decode()

        if hasattr(arg, 'write'):
            fh = arg
        else:
            fh = open(arg, 'w')

        if version == 1:
            fh.write(json.dumps(refs, indent=1).rsplit('}"', 1)[0] + '}"')
            indent = '  '
        elif _append:
            fh.write(',\n')
            fh.write(json.dumps(refs, indent=1)[2:-2])
            indent = ' '
        else:
            fh.write(json.dumps(refs, indent=1)[:-2])
            indent = ' '

        prefix = len(self._commonpath)

        for key, value in self._store.items():
            if '.zarray' in key:
                value = json.loads(value)
                for index, filename in sorted(
                    self._lookup.items(), key=lambda x: x[0]
                ):
                    filename = quote(filename[prefix:].replace('\\', '/'))
                    if filename[0] == '/':
                        filename = filename[1:]
                    index = '.'.join(str(i) for i in index)
                    fh.write(
                        f',\n{indent}"{groupname}{index}": ["{url}{filename}"]'
                    )

        if version == 1:
            fh.write('\n }\n}')
        elif not _append:
            fh.write('\n}')

        if not hasattr(arg, 'write'):
            fh.close()

    def __repr__(self):
        return f'<tifffile.ZarrFileSequenceStore @0x{id(self):016X}>'

    def __str__(self):
        """Return information about instance."""
        return '\n '.join(
            (
                self.__class__.__name__,
                'shape: {}'.format(
                    ', '.join(str(i) for i in self._tiled.shape)
                ),
                'chunks: {}'.format(
                    ', '.join(str(i) for i in self._tiled.chunks)
                ),
                f'dtype: {self._dtype}',
                f'fillvalue: {self._fillvalue}',
            )
        )


class FileSequence:
    """Series of files containing array data of compatible shape and type.

    Attributes
    ----------
    files : list
        List of file names.
    shape : tuple
        Shape of file series. Excludes shape of chunks in files.
    axes : str
        One letter labels of axes in shape.
    labels : tuple of str
        Labels of axes in shape.
    indices : tuple of tuples
        ND indices of files in shape.

    """

    def __init__(
        self,
        imread,
        files,
        container=None,
        sort=None,
        parse=None,
        **kwargs,
    ):
        r"""Initialize instance from multiple files.

        Parameters
        ----------
        imread : function or class
            Array read function or class with asarray function returning numpy
            array from single file.
        files : str, path-like, or sequence thereof
            Glob filename pattern or sequence of file names. Default: \*.
            Binary streams are not supported.
        container : str or container instance (optional)
            Name or open instance of ZIP file in which files are stored.
        sort : function (optional)
            Sort function used to sort file names when 'files' is a pattern.
            The default (None) is the natural_sorted function.
            If False, disable sorting.
        parse : func (optional)
            Parse function used to parse the sequence of sorted file names to
            axes labels, shape, chunk indices, and filtered file names.
            The default (None) is the parse_filenames function if kwargs
            contains 'pattern'.
        kwargs : dict
            Additional arguments passed to the parse function.

        """
        if files is None:
            files = '*'
        if sort is None:
            sort = natural_sorted
        self._container = container
        if container:
            import fnmatch

            if isinstance(container, (str, os.PathLike)):
                import zipfile

                self._container = zipfile.ZipFile(container)
            elif not hasattr(self._container, 'open'):
                raise ValueError('invalid container')
            if isinstance(files, str):
                files = fnmatch.filter(self._container.namelist(), files)
                if sort:
                    files = sort(files)
        elif isinstance(files, os.PathLike):
            files = [os.fspath(files)]
        elif isinstance(files, str):
            files = glob.glob(files)
            if sort:
                files = sort(files)

        files = [os.fspath(f) for f in files]
        if not files:
            raise ValueError('no files found')

        if hasattr(imread, 'asarray'):
            # redefine imread to use asarray from imread class
            if not callable(imread.asarray):
                raise ValueError('invalid imread function')
            _imread_ = imread

            def imread(fname, **kwargs):
                with _imread_(fname) as handle:
                    return handle.asarray(**kwargs)

        elif not callable(imread):
            raise ValueError('invalid imread function')

        if container:
            # redefine imread to read from container
            _imread_ = imread

            def imread(fname, **kwargs):
                with self._container.open(fname) as handle1:
                    with io.BytesIO(handle1.read()) as handle2:
                        return _imread_(handle2, **kwargs)

        if parse is None and kwargs.get('pattern', None):
            parse = parse_filenames

        if parse:
            try:
                labels, shape, indices, files = parse(files, **kwargs)
            except ValueError as exc:
                raise ValueError('failed to parse file names') from exc
        else:
            labels = ('I',)
            shape = (len(files),)
            indices = tuple((i,) for i in range(len(files)))

        self.imread = imread
        self.files = files
        self.axes = ''.join(label[0] for label in labels).upper()
        self.labels = tuple(labels)
        self.shape = tuple(shape)
        self.indices = indices

    @property
    def files_missing(self):
        """Return number of empty chunks."""
        return product(self.shape) - len(self.files)

    def __str__(self):
        """Return string with information about file FileSequence."""
        file = str(self._container) if self._container else self.files[0]
        file = os.path.split(file)[-1]
        return '\n '.join(
            (
                self.__class__.__name__,
                file,
                f'files: {len(self.files)} ({self.files_missing} missing)',
                'shape: {}'.format(', '.join(str(i) for i in self.shape)),
                'labels: {}'.format(', '.join(s for s in self.labels)),
                # f'axes: {self.axes}',
            )
        )

    def __repr__(self):
        return f'<tifffile.FileSequence @0x{id(self):016X}>'

    def __len__(self):
        return len(self.files)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._container:
            self._container.close()
        self._container = None

    def asarray(
        self,
        file=None,  # deprecated
        axestiled=None,
        ioworkers=1,
        out=None,
        **kwargs,
    ):
        """Read image data from files and return as numpy array.

        Raise IndexError or ValueError if array shapes do not match.

        Parameters
        ----------
        file : int or str (optional, deprecated)
            Index or name of single file to read.
        axestiled: dict (optional)
           Defines the axes to be tiled. Map stacked sequence axis to
           chunk axis.
        ioworkers : int  (optional)
            Maximum number of threads to execute the array read function
            asynchronously. Default: 1.
            If None, default to the number of processors multiplied by 5.
            Using threads can significantly improve runtime when
            reading many small files from a network share.
        out : numpy.ndarray, str, or file-like object (optional)
            Buffer where image data are saved.
            If None (default), a new array is created.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If 'memmap', create a memory-mapped array in a temporary file.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        kwargs : dict
            Additional parameters passed to the array read function.

        """
        if file is not None:
            warnings.warn(
                "<tifffile.FileSequence.asarray> "
                "the 'file' parameter is deprecated since 2021.10.12.",
                DeprecationWarning,
                stacklevel=2,
            )
            if isinstance(file, (int, numpy.integer)):
                return self.imread(self.files[file], **kwargs)
            return self.imread(file, **kwargs)

        if len(self.files) < 2:
            ioworkers = 1
        elif ioworkers is None or ioworkers < 1:
            import multiprocessing

            ioworkers = max(multiprocessing.cpu_count() * 5, 1)

        im = self.imread(self.files[0], **kwargs)

        if axestiled:
            tiled = TiledSequence(self.shape, im.shape, axestiled)
            result = create_output(out, tiled.shape, dtype=im.dtype)

            def func(index, fname):
                # read single image from file into result
                # if index is None:
                #     return
                result[index] = self.imread(fname, **kwargs)

            if ioworkers < 2:
                for index, fname in zip(
                    tiled.slices(self.indices), self.files
                ):
                    func(index, fname)
            else:
                with ThreadPoolExecutor(ioworkers) as executor:
                    for _ in executor.map(
                        func, tiled.slices(self.indices), self.files
                    ):
                        pass
        else:
            shape = self.shape + im.shape
            result = create_output(out, shape, dtype=im.dtype)
            result = result.reshape(-1, *im.shape)

            def func(index, fname):
                # read single image from file into result
                if index is None:
                    return
                index = int(numpy.ravel_multi_index(index, self.shape))
                im = self.imread(fname, **kwargs)
                result[index] = im

            if ioworkers < 2:
                for index, fname in zip(self.indices, self.files):
                    func(index, fname)
            else:
                with ThreadPoolExecutor(ioworkers) as executor:
                    for _ in executor.map(func, self.indices, self.files):
                        pass

            result.shape = shape

        return result

    def aszarr(self, **kwargs):
        """Return image data from files as zarr storage."""
        return ZarrFileSequenceStore(self, **kwargs)

    def commonpath(self):
        """Return longest common sub-path of each file in sequence."""
        if len(self.files) == 1:
            commonpath = os.path.dirname(self.files[0])
        else:
            commonpath = os.path.commonpath(self.files)
        return commonpath


class TiffSequence(FileSequence):
    """Series of TIFF files."""

    def __init__(self, files=None, imread=imread, **kwargs):
        """Initialize instance from multiple TIFF files."""
        super().__init__(imread, '*.tif' if files is None else files, **kwargs)

    def __repr__(self):
        return f'<tifffile.TiffSequence @0x{id(self):016X}>'


class TiledSequence:
    """Tiled Sequence.

    Transform a sequence of stacked chunks to tiled chunks.

    Attributes
    ----------
    shape : tuple of int
        Shape of the tiled sequence.
    chunks : tuple of int
        Shape of the chunks in the tiled sequence.

    Examples
    --------
    >>> ts = TiledSequence((1, 2), (3, 4), {1: 0})
    >>> ts.shape
    (1, 6, 4)
    >>> ts.chunks
    (1, 3, 4)

    """

    def __init__(self, stackshape, chunkshape, axestiled=None):
        """Initialize from shape of stacked sequence and axes to be tiled.

        Parameters
        ----------
        stackshape : tuple of int
            Shape of the stacked sequence excluding chunks.
        chunkshape : tuple of int
            Shape of the chunks excluding stack axes.
        axestiled: dict (optional)
           Defines the axes to be tiled. Map stacked sequence axis to
           chunk axis.

        """
        self._stackdims = len(stackshape)
        self._chunkdims = len(chunkshape)
        self._stackshape = tuple(stackshape) + tuple(chunkshape)

        if axestiled:
            axestiled = dict(axestiled)
            for ax0, ax1 in axestiled.items():
                axestiled[ax0] = ax1 + self._stackdims
            self._axestiled = tuple(reversed(sorted(axestiled.items())))

            shape = list(self._stackshape)
            chunks = [1] * self._stackdims + list(chunkshape)
            used = set()
            for ax0, ax1 in self._axestiled:
                if ax0 in used or ax1 in used:
                    raise ValueError('duplicate axis')
                used.add(ax0)
                used.add(ax1)
                shape[ax1] *= stackshape[ax0]
            for ax0, ax1 in self._axestiled:
                del shape[ax0]
                del chunks[ax0]
            self.shape = tuple(shape)
            self.chunks = tuple(chunks)
        else:
            self._axestiled = ()
            self.shape = self._stackshape
            self.chunks = (1,) * self._stackdims + tuple(chunkshape)

    def indices(self, indices):
        """Return iterator over chunk indices of tiled sequence.

        Parameters
        ----------
        indices : sequence of tuple of int
            Indices of chunks in the stacked sequence.

        """
        chunkindex = [0] * self._chunkdims
        for index in indices:
            if index is None:
                yield None
            else:
                if len(index) != self._stackdims:
                    raise ValueError(f'{len(index)} != {self._stackdims}')
                index = list(index) + chunkindex
                for ax0, ax1 in self._axestiled:
                    index[ax1] = index[ax0]
                for ax0, ax1 in self._axestiled:
                    del index[ax0]
                yield tuple(index)

    def slices(self, indices):
        """Return iterator over slices of chunks in tiled sequence.

        Parameters
        ----------
        indices : sequence of tuple of int
            Indices of chunks in the stacked sequence.

        """
        chunkslice = [slice(None)] * self._chunkdims
        for index in indices:
            if index is None:
                yield None
            else:
                assert len(index) == self._stackdims
                index = list(index) + chunkslice
                for ax0, ax1 in self._axestiled:
                    j = self._stackshape[ax1]
                    i = index[ax0] * j
                    index[ax1] = slice(i, i + j)
                for ax0, ax1 in self._axestiled:
                    del index[ax0]
                yield tuple(index)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_tiled(self):
        return bool(self._axestiled)


class FileHandle:
    """Binary file handle.

    A limited, special purpose file handle that can:

    * handle embedded files (e.g. for LSM within LSM files)
    * re-open closed files (for multi-file formats, such as OME-TIFF)
    * read and write numpy arrays and records from file like objects

    Only 'rb', 'r+b', and 'wb' modes are supported. Concurrently reading and
    writing of the same stream is untested.

    When initialized from another file handle, do not use it unless this
    FileHandle is closed.

    Attributes
    ----------
    name : str
        Name of the file.
    path : str
        Absolute path to file.
    size : int
        Size of file in bytes.
    is_file : bool
        If True, file has a fileno and can be memory-mapped.

    All attributes are read-only.

    """

    __slots__ = (
        '_fh',
        '_file',
        '_mode',
        '_name',
        '_dir',
        '_lock',
        '_offset',
        '_size',
        '_close',
        'is_file',
    )

    def __init__(self, file, mode=None, name=None, offset=None, size=None):
        """Initialize file handle from file name or another file handle.

        Parameters
        ----------
        file : str, path-like, binary stream, or FileHandle
            File name or seekable binary stream, such as an open file
            or BytesIO.
        mode : str
            File open mode in case 'file' is a file name. Must be 'rb', 'r+b',
            or 'wb'. Default is 'rb'.
        name : str
            Optional name of file in case 'file' is a binary stream.
        offset : int
            Optional start position of embedded file. By default, this is
            the current file position.
        size : int
            Optional size of embedded file. By default, this is the number
            of bytes from the 'offset' to the end of the file.

        """
        self._fh = None
        self._file = file
        self._mode = 'rb' if mode is None else mode
        self._name = name
        self._dir = ''
        self._offset = offset
        self._size = size
        self._close = True
        self.is_file = None
        self._lock = NullContext()
        self.open()

    def open(self):
        """Open or re-open file."""
        if self._fh is not None:
            return  # file is open

        if isinstance(self._file, os.PathLike):
            self._file = os.fspath(self._file)
        if isinstance(self._file, str):
            # file name
            self._file = os.path.realpath(self._file)
            self._dir, self._name = os.path.split(self._file)
            self._fh = open(self._file, self._mode)
            self._close = True
            if self._offset is None:
                self._offset = 0
        elif isinstance(self._file, FileHandle):
            # FileHandle
            self._fh = self._file._fh
            if self._offset is None:
                self._offset = 0
            self._offset += self._file._offset
            self._close = False
            if not self._name:
                if self._offset:
                    name, ext = os.path.splitext(self._file._name)
                    self._name = f'{name}@{self._offset}{ext}'
                else:
                    self._name = self._file._name
            if self._mode and self._mode != self._file._mode:
                raise ValueError('FileHandle has wrong mode')
            self._mode = self._file._mode
            self._dir = self._file._dir
        elif hasattr(self._file, 'seek'):
            # binary stream: open file, BytesIO
            try:
                self._file.tell()
            except Exception:
                raise ValueError('binary stream is not seekable')
            self._fh = self._file
            if self._offset is None:
                self._offset = self._file.tell()
            self._close = False
            if not self._name:
                try:
                    self._dir, self._name = os.path.split(self._fh.name)
                except AttributeError:
                    self._name = 'Unnamed binary stream'
            try:
                self._mode = self._fh.mode
            except AttributeError:
                pass
        else:
            raise ValueError(
                'the first parameter must be a file name, '
                'seekable binary stream, or FileHandle'
            )

        if self._offset:
            self._fh.seek(self._offset)

        if self._size is None:
            pos = self._fh.tell()
            self._fh.seek(self._offset, os.SEEK_END)
            self._size = self._fh.tell()
            self._fh.seek(pos)

        if self.is_file is None:
            try:
                self._fh.fileno()
                self.is_file = True
            except Exception:
                self.is_file = False

    def close(self):
        """Close file."""
        if self._close and self._fh is not None:
            self._fh.close()
            self._fh = None

    def tell(self):
        """Return file's current position."""
        return self._fh.tell() - self._offset

    def seek(self, offset, whence=0):
        """Set file's current position."""
        if self._offset:
            if whence == 0:
                return (
                    self._fh.seek(self._offset + offset, whence) - self._offset
                )
            if whence == 2 and self._size > 0:
                return (
                    self._fh.seek(self._offset + self._size + offset, 0)
                    - self._offset
                )
        return self._fh.seek(offset, whence)

    def read(self, size=-1):
        """Read 'size' bytes from file, or until EOF is reached."""
        if size < 0 and self._offset:
            size = self._size
        return self._fh.read(size)

    def readinto(self, b):
        """Read up to len(b) bytes into b and return number of bytes read."""
        return self._fh.readinto(b)

    def write(self, bytestring):
        """Write bytes to file."""
        return self._fh.write(bytestring)

    def flush(self):
        """Flush write buffers if applicable."""
        return self._fh.flush()

    def memmap_array(self, dtype, shape, offset=0, mode='r', order='C'):
        """Return numpy.memmap of data stored in file."""
        if not self.is_file:
            raise ValueError('cannot memory-map file without fileno')
        return numpy.memmap(
            self._fh,
            dtype=dtype,
            mode=mode,
            offset=self._offset + offset,
            shape=shape,
            order=order,
        )

    def read_array(self, dtype, count=-1, out=None):
        """Return numpy array from file in native byte order."""
        dtype = numpy.dtype(dtype)

        if count < 0:
            nbytes = self._size if out is None else out.nbytes
            count = nbytes // dtype.itemsize
        else:
            nbytes = count * dtype.itemsize

        result = numpy.empty(count, dtype) if out is None else out

        if result.nbytes != nbytes:
            raise ValueError('size mismatch')

        try:
            n = self._fh.readinto(result)
        except AttributeError:
            result[:] = numpy.frombuffer(self._fh.read(nbytes), dtype).reshape(
                result.shape
            )
            n = nbytes

        if n != nbytes:
            raise ValueError(f'failed to read {nbytes} bytes, got {n}')

        if not result.dtype.isnative:
            if not dtype.isnative:
                result.byteswap(True)
            result = result.newbyteorder()
        elif result.dtype.isnative != dtype.isnative:
            result.byteswap(True)

        if out is not None:
            if hasattr(out, 'flush'):
                out.flush()

        return result

    def read_record(self, dtype, shape=1, byteorder=None):
        """Return numpy record from file."""
        rec = numpy.rec
        try:
            record = rec.fromfile(self._fh, dtype, shape, byteorder=byteorder)
        except Exception:
            dtype = numpy.dtype(dtype)
            if shape is None:
                shape = self._size // dtype.itemsize
            size = product(sequence(shape)) * dtype.itemsize
            # data = bytearray(size)
            # n = self._fh.readinto(data)
            # data = data[:n]
            # TODO: record is not writable
            data = self._fh.read(size)
            record = rec.fromstring(data, dtype, shape, byteorder=byteorder)
        return record[0] if shape == 1 else record

    def write_empty(self, size):
        """Append size bytes to file. Position must be at end of file."""
        if size < 1:
            return
        self._fh.seek(size - 1, os.SEEK_CUR)
        self._fh.write(b'\x00')

    def write_array(self, data):
        """Write numpy array to binary file."""
        try:
            # writing non-contiguous arrays is very slow
            numpy.ascontiguousarray(data).tofile(self._fh)
        except Exception:
            # numpy can't write to BytesIO
            self._fh.write(data.tobytes())

    def read_segments(
        self,
        offsets,
        bytecounts,
        indices=None,
        sort=True,
        lock=None,
        buffersize=None,
        flat=True,
    ):
        """Return iterator over segments read from file and their indices.

        The purpose of this function is to

        * reduce small or random reads
        * reduce acquiring reentrant locks
        * synchronize seeks and reads
        * limit the size of segments read into memory at once
          (ThreadPoolExecutor.map is not collecting iterables lazily).

        Parameters
        ----------
        offsets, bytecounts : sequence of int
            offsets and bytecounts of the segments to read from file.
        indices : sequence of int
            Indices of the segments in the image. Default: range(len(offsets)).
        sort : bool
            If True (default), segments are read from file in the order of
            their offsets.
        lock:
            A reentrant lock used to synchronize seeks and reads.
        buffersize : int
            Approximate number of bytes to read from file in one pass.
            Default: 64 MB.
        flat : bool
            If True (default), return an iterator over individual
            (segment, index) tuples. Else return an iterator over a list
            of (segment, index) tuples that were acquired in one pass.

        Returns
        -------
        items : (bytes, int) or [(bytes, int)]
            Iterator over individual or lists of (segment, index) tuples.

        """
        # TODO: Cythonize this?
        length = len(offsets)
        if length < 1:
            return
        if length == 1:
            index = 0 if indices is None else indices[0]
            if bytecounts[index] > 0 and offsets[index] > 0:
                if lock is None:
                    lock = self._lock
                with lock:
                    self.seek(offsets[index])
                    data = self._fh.read(bytecounts[index])
            else:
                data = None
            yield (data, index) if flat else [(data, index)]
            return

        if lock is None:
            lock = self._lock
        if buffersize is None:
            buffersize = 67108864  # 2 ** 26, 64 MB

        if indices is None:
            segments = [(i, offsets[i], bytecounts[i]) for i in range(length)]
        else:
            segments = [
                (indices[i], offsets[i], bytecounts[i]) for i in range(length)
            ]
        if sort:
            segments = sorted(segments, key=lambda x: x[1])

        iscontig = True
        for i in range(length - 1):
            _, offset, bytecount = segments[i]
            nextoffset = segments[i + 1][1]
            if offset == 0 or bytecount == 0 or nextoffset == 0:
                continue
            if offset + bytecount != nextoffset:
                iscontig = False
                break

        seek = self.seek
        read = self._fh.read

        if iscontig:
            # consolidate reads
            i = 0
            while i < length:
                j = i
                offset = None
                bytecount = 0
                while bytecount < buffersize and i < length:
                    _, o, b = segments[i]
                    if o > 0 and b > 0:
                        if offset is None:
                            offset = o
                        bytecount += b
                    i += 1

                if offset is None:
                    data = None
                else:
                    with lock:
                        seek(offset)
                        data = read(bytecount)
                start = 0
                stop = 0
                result = []
                while j < i:
                    index, offset, bytecount = segments[j]
                    if offset > 0 and bytecount > 0:
                        stop += bytecount
                        result.append((data[start:stop], index))
                        start = stop
                    else:
                        result.append((None, index))
                    j += 1
                if flat:
                    yield from result
                else:
                    yield result
            return

        i = 0
        while i < length:
            result = []
            size = 0
            with lock:
                while size < buffersize and i < length:
                    index, offset, bytecount = segments[i]
                    if offset > 0 and bytecount > 0:
                        seek(offset)
                        result.append((read(bytecount), index))
                        # buffer = bytearray(bytecount)
                        # n = fh.readinto(buffer)
                        # data.append(buffer[:n])
                        size += bytecount
                    else:
                        result.append((None, index))
                    i += 1
            if flat:
                yield from result
            else:
                yield result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getattr__(self, name):
        """Return attribute from underlying file object."""
        if self._offset:
            warnings.warn(
                '<tifffile.FileHandle> '
                f'{name} not implemented for embedded files',
                UserWarning,
            )
        return getattr(self._fh, name)

    def __repr__(self):
        return f'<tifffile.FileHandle {snipstr(self.name, 32)!r}>'

    def __str__(self):
        """Return string with information about FileHandle."""
        return '\n '.join(
            (
                'FileHandle',
                self.name,
                self.dirname,
                f'{self.size} bytes',
                'closed' if self.closed else 'open',
            )
        )

    @property
    def name(self):
        return self._name

    @property
    def dirname(self):
        return self._dir

    @property
    def path(self):
        return os.path.join(self._dir, self._name)

    @property
    def size(self):
        return self._size

    @property
    def closed(self):
        return self._fh is None

    @property
    def lock(self):
        """Return current lock instance."""
        return self._lock

    @lock.setter
    def lock(self, value):
        if bool(value) == isinstance(self._lock, NullContext):
            self._lock = threading.RLock() if value else NullContext()

    @property
    def has_lock(self):
        """Return if a RLock is used."""
        return not isinstance(self._lock, NullContext)


class FileCache:
    """Keep FileHandles open."""

    __slots__ = ('files', 'keep', 'past', 'lock', 'size')

    def __init__(self, size=None, lock=None):
        """Initialize open file cache."""
        self.past = []  # FIFO of opened files
        self.files = {}  # refcounts of opened file handles
        self.keep = set()  # files to keep open
        self.lock = NullContext() if lock is None else lock
        self.size = 8 if size is None else int(size)

    def __len__(self):
        """Return number of open files."""
        return len(self.files)

    def open(self, filehandle):
        """Open file, re-open if necessary."""
        with self.lock:
            if filehandle in self.files:
                self.files[filehandle] += 1
            elif filehandle.closed:
                filehandle.open()
                self.files[filehandle] = 1
                self.past.append(filehandle)
            else:
                self.files[filehandle] = 2
                self.keep.add(filehandle)
                self.past.append(filehandle)

    def close(self, filehandle):
        """Close least recently used open files."""
        with self.lock:
            if filehandle in self.files:
                self.files[filehandle] -= 1
            self._trim()

    def clear(self):
        """Close all opened files if not in use when opened first."""
        with self.lock:
            for filehandle, refcount in list(self.files.items()):
                if filehandle not in self.keep:
                    filehandle.close()
                    del self.files[filehandle]
                    del self.past[self.past.index(filehandle)]

    def read(self, filehandle, offset, bytecount, whence=0):
        """Return bytes read from binary file."""
        # this function is more efficient than
        # filecache.open(filehandle)
        # with lock:
        #     filehandle.seek()
        #     data = filehandle.read()
        # filecache.close(filehandle)
        with self.lock:
            b = filehandle not in self.files
            if b:
                if filehandle.closed:
                    filehandle.open()
                    self.files[filehandle] = 0
                else:
                    self.files[filehandle] = 1
                    self.keep.add(filehandle)
                self.past.append(filehandle)
            filehandle.seek(offset, whence)
            data = filehandle.read(bytecount)
            if b:
                self._trim()
        return data

    def _trim(self):
        """Trim file cache."""
        index = 0
        size = len(self.past)
        while index < size > self.size:
            filehandle = self.past[index]
            if filehandle not in self.keep and self.files[filehandle] <= 0:
                filehandle.close()
                del self.files[filehandle]
                del self.past[index]
                size -= 1
            else:
                index += 1

    def __repr__(self):
        return f'<tifffile.FileCache @0x{id(self):016X}>'


class NullContext:
    """Null context manager.

    >>> with NullContext():
    ...     pass

    """

    __slots = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __repr__(self):
        return 'NullContext()'


class Timer:
    """Stopwatch for timing execution speed."""

    __slots__ = ('started', 'stopped', 'duration')

    clock = time.perf_counter

    def __init__(self, message=None, end=' ', started=None):
        """Initialize timer and print message."""
        if message is not None:
            print(message, end=end, flush=True)
        self.duration = 0
        if started is None:
            started = Timer.clock()
        self.started = self.stopped = started

    def start(self, message=None, end=' '):
        """Start timer and return current time."""
        if message is not None:
            print(message, end=end, flush=True)
        self.duration = 0
        self.started = self.stopped = Timer.clock()
        return self.started

    def stop(self, message=None, end=' '):
        """Return duration of timer till start."""
        self.stopped = Timer.clock()
        if message is not None:
            print(message, end=end, flush=True)
        self.duration = self.stopped - self.started
        return self.duration

    def print(self, message=None, end=None):
        """Print duration from timer start till last stop or now."""
        msg = str(self)
        if message is not None:
            print(message, end=' ')
        print(msg, end=end, flush=True)

    def __str__(self):
        """Return duration from timer start till last stop or now as string."""
        if self.duration <= 0:
            # not stopped
            duration = Timer.clock() - self.started
        else:
            duration = self.duration
        s = str(datetime.timedelta(seconds=duration))
        i = 0
        while i < len(s) and s[i : i + 2] in '0:0010203040506070809':
            i += 1
        if s[i : i + 1] == ':':
            i += 1
        return f'{s[i:]} s'

    def __repr__(self):
        return f'Timer(started={self.started})'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.print()


class OmeXmlError(Exception):
    """Exception to indicate invalid OME-XML or unsupported cases."""


class OmeXml:
    """OME-TIFF XML."""

    def __init__(self, **metadata):
        """Create a new instance.

        Creator : str (optional)
            Name of the creating application. Default 'tifffile.py'.
        UUID : str (optional)
            Unique identifier.

        """
        if 'OME' in metadata:
            metadata = metadata['OME']

        self.ifd = 0
        self.images = []
        self.annotations = []
        self.elements = []
        # TODO: parse other OME elements from metadata
        #   Project
        #   Dataset
        #   Folder
        #   Experiment
        #   Plate
        #   Screen
        #   Experimenter
        #   ExperimenterGroup
        #   Instrument
        #   StructuredAnnotations
        #   ROI
        if 'UUID' in metadata:
            self.uuid = metadata['UUID'].split(':')[-1]
        else:
            from uuid import uuid1  # noqa: delayed import

            self.uuid = str(uuid1())
        creator = OmeXml._attribute(
            metadata, 'Creator', default=f'tifffile.py {__version__}'
        )
        schema = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
        self.xml = (
            '{declaration}'
            f'<OME xmlns="{schema}" '
            f'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            f'xsi:schemaLocation="{schema} {schema}/ome.xsd" '
            f'UUID="urn:uuid:{self.uuid}" {creator}>'
            '{images}'
            '{annotations}'
            '{elements}'
            f'</OME>'
        )

    def addimage(self, dtype, shape, storedshape, axes=None, **metadata):
        """Add image to OME-XML.

        The OME model can handle up to 9 dimensional images for selected
        axes orders. Refer to the OME-XML specification for details.
        Non-TZCYXS (modulo) dimensions must be after a TZC dimension or
        require an unused TZC dimension.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type of image array.
        shape : tuple
            Shape of image array.
        storedshape: tuple
            Normalized shape describing how the image array is stored in TIFF:
            (pages, separate_samples, depth, length, width, contig_samples).
        axes : str (optional)
            Axes labels for each dimension in shape.
            By default, axes are matched to the shape in reverse order of
            TZC(S)YX(S) based on storedshape.
            The following axes codes are supported: 'S' sample, 'X' width,
            'Y' length, 'Z' depth, 'C' channel, 'T' time, 'A' angle, 'P' phase,
            'R' tile, 'H' lifetime, 'E' lambda, 'Q' other.
        metadata : miscellaneous (optional)
            Additional OME-XML attributes or elements to be stored.
            Image/Pixels: Name, AcquisitionDate, Description,
            PhysicalSizeX, PhysicalSizeXUnit, PhysicalSizeY, PhysicalSizeYUnit,
            PhysicalSizeZ, PhysicalSizeZUnit, TimeIncrement, TimeIncrementUnit.
            Per Plane: DeltaTUnit, ExposureTime, ExposureTimeUnit,
            PositionX, PositionXUnit, PositionY, PositionYUnit, PositionZ,
            PositionZUnit.
            Per Channel: Name, AcquisitionMode, Color, ContrastMethod,
            EmissionWavelength, EmissionWavelengthUnit, ExcitationWavelength,
            ExcitationWavelengthUnit, Fluor, IlluminationType, NDFilter,
            PinholeSize, PinholeSizeUnit, PockelCellSetting.

        """
        index = len(self.images)

        # get Image and Pixels metadata
        metadata = metadata.get('OME', metadata)
        metadata = metadata.get('Image', metadata)
        if isinstance(metadata, (list, tuple)):
            # multiple images
            metadata = metadata[index]
        if 'Pixels' in metadata:
            # merge with Image
            if 'ID' in metadata['Pixels']:
                del metadata['Pixels']['ID']
            metadata.update(metadata['Pixels'])
            del metadata['Pixels']

        try:
            dtype = numpy.dtype(dtype).name
            dtype = {
                'int8': 'int8',
                'int16': 'int16',
                'int32': 'int32',
                'uint8': 'uint8',
                'uint16': 'uint16',
                'uint32': 'uint32',
                'float32': 'float',
                'float64': 'double',
                'complex64': 'complex',
                'complex128': 'double-complex',
                'bool': 'bit',
            }[dtype]
        except KeyError:
            raise OmeXmlError(f'data type {dtype!r} not supported')

        if metadata.get('Type', dtype) != dtype:
            raise OmeXmlError(
                f'metadata Pixels Type {metadata["Type"]!r} '
                f'does not match array dtype {dtype!r}'
            )

        samples = 1
        planecount, separate, depth, length, width, contig = storedshape
        if depth != 1:
            raise OmeXmlError('ImageDepth not supported')
        if not (separate == 1 or contig == 1):
            raise ValueError('invalid stored shape')

        shape = tuple(int(i) for i in shape)
        ndim = len(shape)
        if ndim < 1 or product(shape) <= 0:
            raise OmeXmlError('empty arrays not supported')

        if axes is None:
            # get axes from shape, stored shape, and DimensionOrder
            if contig != 1 or shape[-3:] == (length, width, 1):
                axes = 'YXS'
                samples = contig
            elif separate != 1 or (
                ndim == 6 and shape[-3:] == (1, length, width)
            ):
                axes = 'SYX'
                samples = separate
            else:
                axes = 'YX'
            if not len(axes) <= ndim <= (6 if 'S' in axes else 5):
                raise OmeXmlError(f'{ndim} dimensions not supported')
            hiaxes = metadata.get('DimensionOrder', 'XYCZT')[:1:-1]
            axes = hiaxes[(6 if 'S' in axes else 5) - ndim :] + axes
            assert len(axes) == len(shape)

        else:
            # validate axes against shape and stored shape
            axes = axes.upper()
            if len(axes) != len(shape):
                raise ValueError('axes do not match shape')
            if not (axes.endswith('YX') or axes.endswith('YXS')):
                raise OmeXmlError('dimensions must end with YX or YXS')
            unique = []
            for ax in axes:
                if ax not in 'TZCYXSAPRHEQ':
                    raise OmeXmlError(f'dimension {ax!r} not supported')
                if ax in unique:
                    raise OmeXmlError(f'multiple {ax!r} dimensions')
                unique.append(ax)
            if ndim > (9 if 'S' in axes else 8):
                raise OmeXmlError('more than 8 dimensions not supported')
            if contig != 1:
                samples = contig
                if ndim < 3:
                    raise ValueError('dimensions do not match stored shape')
                if axes[-1] != 'S':
                    raise ValueError('axes do not match stored shape')
                if shape[-1] != contig or shape[-2] != width:
                    raise ValueError('shape does not match stored shape')
            elif separate != 1:
                samples = separate
                if ndim < 3:
                    raise ValueError('dimensions do not match stored shape')
                if axes[-3] != 'S':
                    raise ValueError('axes do not match stored shape')
                if shape[-3] != separate or shape[-1] != length:
                    raise ValueError('shape does not match stored shape')

        if shape[axes.index('X')] != width or shape[axes.index('Y')] != length:
            raise ValueError('shape does not match stored shape')

        if 'S' in axes:
            hiaxes = axes[: min(axes.index('S'), axes.index('Y'))]
        else:
            hiaxes = axes[: axes.index('Y')]

        if any(ax in 'APRHEQ' for ax in hiaxes):
            # modulo axes
            modulo = {}
            dimorder = []
            axestype = {
                'A': 'angle',
                'P': 'phase',
                'R': 'tile',
                'H': 'lifetime',
                'E': 'lambda',
                'Q': 'other',
            }
            for i, ax in enumerate(hiaxes):
                if ax in 'APRHEQ':
                    x = hiaxes[i - 1 : i]
                    if x and x in 'TZC':
                        # use previous axis
                        modulo[x] = axestype[ax], shape[i]
                    else:
                        # use next unused axis
                        for x in 'TZC':
                            if x not in dimorder and x not in modulo:
                                modulo[x] = axestype[ax], shape[i]
                                dimorder.append(x)
                                break
                        else:
                            # TODO: support any order of axes, e.g. APRTZC
                            raise OmeXmlError('more than 3 modulo dimensions')
                else:
                    dimorder.append(ax)
            hiaxes = ''.join(dimorder)

            # TODO: use user-specified start, stop, step, or labels
            moduloalong = ''.join(
                f'<ModuloAlong{ax} Type="{axtype}" Start="0" End="{size-1}"/>'
                for ax, (axtype, size) in modulo.items()
            )
            annotationref = f'<AnnotationRef ID="Annotation:{index}"/>'
            annotations = (
                f'<XMLAnnotation ID="Annotation:{index}" '
                'Namespace="openmicroscopy.org/omero/dimension/modulo">'
                '<Value>'
                '<Modulo namespace='
                '"http://www.openmicroscopy.org/Schemas/Additions/2011-09">'
                f'{moduloalong}'
                '</Modulo>'
                '</Value>'
                '</XMLAnnotation>'
            )
            self.annotations.append(annotations)
        else:
            modulo = {}
            annotationref = ''

        hiaxes = hiaxes[::-1]
        for dimorder in (
            metadata.get('DimensionOrder', 'XYCZT'),
            'XYCZT',
            'XYZCT',
            'XYZTC',
            'XYCTZ',
            'XYTCZ',
            'XYTZC',
        ):
            if hiaxes in dimorder:
                break
        else:
            raise OmeXmlError(f'dimension order {axes!r} not supported')

        dimsizes = []
        for ax in dimorder:
            if ax == 'S':
                continue
            if ax in axes:
                size = shape[axes.index(ax)]
            else:
                size = 1
            if ax == 'C':
                sizec = size
                size *= samples
            if ax in modulo:
                size *= modulo[ax][1]
            dimsizes.append(size)
        sizes = ''.join(
            f' Size{ax}="{size}"' for ax, size in zip(dimorder, dimsizes)
        )

        # verify DimensionOrder in metadata is compatible
        if 'DimensionOrder' in metadata:
            omedimorder = metadata['DimensionOrder']
            omedimorder = ''.join(
                ax for ax in omedimorder if dimsizes[dimorder.index(ax)] > 1
            )
            if hiaxes not in omedimorder:
                raise OmeXmlError(
                    f'metadata DimensionOrder does not match {axes!r}'
                )

        # verify metadata Size values match shape
        for ax, size in zip(dimorder, dimsizes):
            if metadata.get(f'Size{ax}', size) != size:
                raise OmeXmlError(
                    f'metadata Size{ax} does not match {shape!r}'
                )

        dimsizes[dimorder.index('C')] //= samples
        if planecount != product(dimsizes[2:]):
            raise ValueError('shape does not match stored shape')

        planes = []
        planeattributes = metadata.get('Plane', '')
        if planeattributes:
            cztorder = tuple(dimorder[2:].index(ax) for ax in 'CZT')
            for p in range(planecount):
                attributes = OmeXml._attributes(
                    planeattributes,
                    p,
                    'DeltaTUnit',
                    'ExposureTime',
                    'ExposureTimeUnit',
                    'PositionX',
                    'PositionXUnit',
                    'PositionY',
                    'PositionYUnit',
                    'PositionZ',
                    'PositionZUnit',
                )
                unraveled = numpy.unravel_index(p, dimsizes[2:], order='F')
                c, z, t = (int(unraveled[i]) for i in cztorder)
                planes.append(
                    f'<Plane TheC="{c}" TheZ="{z}" TheT="{t}"{attributes}/>'
                )
                # TODO: if possible, verify c, z, t match planeattributes
        planes = ''.join(planes)

        channels = []
        for c in range(sizec):
            lightpath = '<LightPath/>'
            # TODO: use LightPath elements from metadata
            #    'AnnotationRef',
            #    'DichroicRef',
            #    'EmissionFilterRef',
            #    'ExcitationFilterRef'
            attributes = OmeXml._attributes(
                metadata.get('Channel', ''),
                c,
                'Name',
                'AcquisitionMode',
                'Color',
                'ContrastMethod',
                'EmissionWavelength',
                'EmissionWavelengthUnit',
                'ExcitationWavelength',
                'ExcitationWavelengthUnit',
                'Fluor',
                'IlluminationType',
                'NDFilter',
                'PinholeSize',
                'PinholeSizeUnit',
                'PockelCellSetting',
            )
            channels.append(
                f'<Channel ID="Channel:{index}:{c}" '
                f'SamplesPerPixel="{samples}"'
                f'{attributes}>'
                f'{lightpath}'
                '</Channel>'
            )
        channels = ''.join(channels)

        # TODO: support more Image elements
        elements = OmeXml._elements(metadata, 'AcquisitionDate', 'Description')

        name = OmeXml._attribute(metadata, 'Name', default=f'Image{index}')
        attributes = OmeXml._attributes(
            metadata,
            None,
            'SignificantBits',
            'PhysicalSizeX',
            'PhysicalSizeXUnit',
            'PhysicalSizeY',
            'PhysicalSizeYUnit',
            'PhysicalSizeZ',
            'PhysicalSizeZUnit',
            'TimeIncrement',
            'TimeIncrementUnit',
        )
        if separate > 1 or contig > 1:
            interleaved = 'false' if separate > 1 else 'true'
            interleaved = f' Interleaved="{interleaved}"'
        else:
            interleaved = ''

        self.images.append(
            f'<Image ID="Image:{index}"{name}>'
            f'{elements}'
            f'<Pixels ID="Pixels:{index}" '
            f'DimensionOrder="{dimorder}" '
            f'Type="{dtype}"'
            f'{sizes}'
            f'{interleaved}'
            f'{attributes}>'
            f'{channels}'
            f'<TiffData IFD="{self.ifd}" PlaneCount="{planecount}"/>'
            f'{planes}'
            f'</Pixels>'
            f'{annotationref}'
            f'</Image>'
        )
        self.ifd += planecount

    def tostring(self, declaration=False):
        """Return OME-XML string."""
        # TODO: support other top-level elements
        elements = ''.join(self.elements)
        images = ''.join(self.images)
        annotations = ''.join(self.annotations)
        if annotations:
            annotations = (
                f'<StructuredAnnotations>{annotations}</StructuredAnnotations>'
            )
        if declaration:
            declaration = '<?xml version="1.0" encoding="UTF-8"?>'
        else:
            declaration = ''
        xml = self.xml.format(
            declaration=declaration,
            images=images,
            annotations=annotations,
            elements=elements,
        )
        return xml

    def __repr__(self):
        return f'<tifffile.OmeXml @0x{id(self):016X}>'

    def __str__(self):
        """Return OME-XML string."""
        xml = self.tostring()
        try:
            from lxml import etree  # noqa: delayed import

            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.fromstring(xml, parser)
            xml = etree.tostring(
                tree, encoding='utf-8', pretty_print=True, xml_declaration=True
            ).decode()
        except Exception as exc:
            warnings.warn(
                f'<tiffile.OmeXml.__str__> {exc.__class__.__name__}: {exc}',
                UserWarning,
            )
        except ImportError:
            pass
        return xml

    @staticmethod
    def _escape(value):
        """Return escaped string of value."""
        if not isinstance(value, str):
            value = str(value)
        elif '&amp;' in value or '&gt;' in value or '&lt;' in value:
            return value
        value = value.replace('&', '&amp;')
        value = value.replace('>', '&gt;')
        value = value.replace('<', '&lt;')
        return value

    @staticmethod
    def _element(metadata, name, default=None):
        """Return XML formatted element if name in metadata."""
        value = metadata.get(name, default)
        if value is None:
            return None
        return f'<{name}>{OmeXml._escape(value)}</{name}>'

    @staticmethod
    def _elements(metadata, *names):
        """Return XML formatted elements."""
        if not metadata:
            return ''
        elements = (OmeXml._element(metadata, name) for name in names)
        return ''.join(e for e in elements if e)

    @staticmethod
    def _attribute(metadata, name, index=None, default=None):
        """Return XML formatted attribute if name in metadata."""
        value = metadata.get(name, default)
        if value is None:
            return None
        if index is not None:
            if isinstance(value, (list, tuple)):
                value = value[index]
            elif index > 0:
                raise TypeError(
                    f'{type(value).__name__!r} is not a list or tuple'
                )
        return f' {name}="{OmeXml._escape(value)}"'

    @staticmethod
    def _attributes(metadata, index_, *names):
        """Return XML formatted attributes."""
        if not metadata:
            return ''
        if index_ is None:
            attributes = (OmeXml._attribute(metadata, name) for name in names)
        elif isinstance(metadata, (list, tuple)):
            metadata = metadata[index_]
            attributes = (OmeXml._attribute(metadata, name) for name in names)
        elif isinstance(metadata, dict):
            attributes = (
                OmeXml._attribute(metadata, name, index_) for name in names
            )
        return ''.join(a for a in attributes if a)

    @staticmethod
    def _reference(metadata, name):
        """Return XML formatted reference element."""
        value = metadata.get(name, None)
        if value is None:
            return ''
        try:
            value = value['ID']
        except KeyError:
            pass
        return f'<{name} ID="{OmeXml._escape(value)}"/>'

    @staticmethod
    def validate(omexml, omexsd=None, assert_=True, _schema=[]):
        """Return if OME-XML is valid according to XMLSchema.

        If 'assert_' is True, raise an AssertionError if validation fails.

        On first run, this function takes several seconds to download and
        parse the 2016-06 OME XMLSchema.

        """
        from lxml import etree  # noqa: delay import

        if not _schema:
            if omexsd is None:
                omexsd = os.path.join(os.path.dirname(__file__), 'ome.xsd')
                if os.path.exists(omexsd):
                    with open(omexsd, 'rb') as fh:
                        omexsd = fh.read()
                else:
                    import urllib.request  # noqa: delay import

                    with urllib.request.urlopen(
                        'https://www.openmicroscopy.org/'
                        'Schemas/OME/2016-06/ome.xsd'
                    ) as fh:
                        omexsd = fh.read()
            if omexsd.startswith(b'<?xml'):
                omexsd = omexsd.split(b'>', 1)[-1]
            try:
                _schema.append(
                    etree.XMLSchema(etree.fromstring(omexsd.decode()))
                )
            except Exception:
                # raise
                _schema.append(None)
        if _schema and _schema[0] is not None:
            if omexml.startswith('<?xml'):
                omexml = omexml.split('>', 1)[-1]
            tree = etree.fromstring(omexml)
            if assert_:
                _schema[0].assert_(tree)
                return True
            return _schema[0].validate(tree)
        return None


class LazyConst:
    """Class whose attributes are computed on first access from its methods."""

    def __init__(self, cls):
        self._cls = cls
        self.__doc__ = cls.__doc__
        self.__module__ = cls.__module__
        self.__name__ = cls.__name__
        self.__qualname__ = cls.__qualname__
        self.lock = threading.RLock()

    def __reduce__(self):
        # decorated class will be pickled by name
        return self._cls.__qualname__

    def __getattr__(self, name):
        with self.lock:
            if name in self.__dict__:
                # another thread set attribute while awaiting lock
                value = self.__dict__[name]
            else:
                func = getattr(self._cls, name)
                value = func() if callable(func) else func
                try:
                    setattr(value, '__module__', getattr(func, '__module__'))
                except AttributeError:
                    pass
                try:
                    setattr(value, '__name__', getattr(func, '__name__'))
                except AttributeError:
                    pass
                try:
                    setattr(
                        value, '__qualname__', getattr(func, '__qualname__')
                    )
                except AttributeError:
                    pass
                # __doc__
                # __annotations__
                setattr(self, name, value)
        return value


@LazyConst
class TIFF:
    """Namespace for module constants."""

    def CLASSIC_LE():
        class ClassicTiffLe:
            __slots__ = ()
            version = 42
            byteorder = '<'
            offsetsize = 4
            offsetformat = '<I'
            tagnosize = 2
            tagnoformat = '<H'
            tagsize = 12
            tagformat1 = '<HH'
            tagformat2 = '<I4s'
            tagoffsetthreshold = 4

        return ClassicTiffLe

    def CLASSIC_BE():
        class ClassicTiffBe:
            __slots__ = ()
            version = 42
            byteorder = '>'
            offsetsize = 4
            offsetformat = '>I'
            tagnosize = 2
            tagnoformat = '>H'
            tagsize = 12
            tagformat1 = '>HH'
            tagformat2 = '>I4s'
            tagoffsetthreshold = 4

        return ClassicTiffBe

    def BIG_LE():
        class BigTiffLe:
            __slots__ = ()
            version = 43
            byteorder = '<'
            offsetsize = 8
            offsetformat = '<Q'
            tagnosize = 8
            tagnoformat = '<Q'
            tagsize = 20
            tagformat1 = '<HH'
            tagformat2 = '<Q8s'
            tagoffsetthreshold = 8

        return BigTiffLe

    def BIG_BE():
        class BigTiffBe:
            __slots__ = ()
            version = 43
            byteorder = '>'
            offsetsize = 8
            offsetformat = '>Q'
            tagnosize = 8
            tagnoformat = '>Q'
            tagsize = 20
            tagformat1 = '>HH'
            tagformat2 = '>Q8s'
            tagoffsetthreshold = 8

        return BigTiffBe

    def NDPI_LE():
        class NdpiTiffLe:
            __slots__ = ()
            version = 42
            byteorder = '<'
            offsetsize = 8  # NDPI uses 8 bytes IFD and tag offsets
            offsetformat = '<Q'
            tagnosize = 2
            tagnoformat = '<H'
            tagsize = 12  # 16 after patching
            tagformat1 = '<HH'
            tagformat2 = '<I8s'  # after patching
            tagoffsetthreshold = 4

        return NdpiTiffLe

    def TAGS():
        # TIFF tag codes and names from TIFF6, TIFF/EP, EXIF, and other specs
        # TODO: divide into baseline, exif, private, ... tags
        return TiffTagRegistry(
            (
                (11, 'ProcessingSoftware'),
                (254, 'NewSubfileType'),
                (255, 'SubfileType'),
                (256, 'ImageWidth'),
                (257, 'ImageLength'),
                (258, 'BitsPerSample'),
                (259, 'Compression'),
                (262, 'PhotometricInterpretation'),
                (263, 'Thresholding'),
                (264, 'CellWidth'),
                (265, 'CellLength'),
                (266, 'FillOrder'),
                (269, 'DocumentName'),
                (270, 'ImageDescription'),
                (271, 'Make'),
                (272, 'Model'),
                (273, 'StripOffsets'),
                (274, 'Orientation'),
                (277, 'SamplesPerPixel'),
                (278, 'RowsPerStrip'),
                (279, 'StripByteCounts'),
                (280, 'MinSampleValue'),
                (281, 'MaxSampleValue'),
                (282, 'XResolution'),
                (283, 'YResolution'),
                (284, 'PlanarConfiguration'),
                (285, 'PageName'),
                (286, 'XPosition'),
                (287, 'YPosition'),
                (288, 'FreeOffsets'),
                (289, 'FreeByteCounts'),
                (290, 'GrayResponseUnit'),
                (291, 'GrayResponseCurve'),
                (292, 'T4Options'),
                (293, 'T6Options'),
                (296, 'ResolutionUnit'),
                (297, 'PageNumber'),
                (300, 'ColorResponseUnit'),
                (301, 'TransferFunction'),
                (305, 'Software'),
                (306, 'DateTime'),
                (315, 'Artist'),
                (316, 'HostComputer'),
                (317, 'Predictor'),
                (318, 'WhitePoint'),
                (319, 'PrimaryChromaticities'),
                (320, 'ColorMap'),
                (321, 'HalftoneHints'),
                (322, 'TileWidth'),
                (323, 'TileLength'),
                (324, 'TileOffsets'),
                (325, 'TileByteCounts'),
                (326, 'BadFaxLines'),
                (327, 'CleanFaxData'),
                (328, 'ConsecutiveBadFaxLines'),
                (330, 'SubIFDs'),
                (332, 'InkSet'),
                (333, 'InkNames'),
                (334, 'NumberOfInks'),
                (336, 'DotRange'),
                (337, 'TargetPrinter'),
                (338, 'ExtraSamples'),
                (339, 'SampleFormat'),
                (340, 'SMinSampleValue'),
                (341, 'SMaxSampleValue'),
                (342, 'TransferRange'),
                (343, 'ClipPath'),
                (344, 'XClipPathUnits'),
                (345, 'YClipPathUnits'),
                (346, 'Indexed'),
                (347, 'JPEGTables'),
                (351, 'OPIProxy'),
                (400, 'GlobalParametersIFD'),
                (401, 'ProfileType'),
                (402, 'FaxProfile'),
                (403, 'CodingMethods'),
                (404, 'VersionYear'),
                (405, 'ModeNumber'),
                (433, 'Decode'),
                (434, 'DefaultImageColor'),
                (435, 'T82Options'),
                (437, 'JPEGTables'),  # 347
                (512, 'JPEGProc'),
                (513, 'JPEGInterchangeFormat'),
                (514, 'JPEGInterchangeFormatLength'),
                (515, 'JPEGRestartInterval'),
                (517, 'JPEGLosslessPredictors'),
                (518, 'JPEGPointTransforms'),
                (519, 'JPEGQTables'),
                (520, 'JPEGDCTables'),
                (521, 'JPEGACTables'),
                (529, 'YCbCrCoefficients'),
                (530, 'YCbCrSubSampling'),
                (531, 'YCbCrPositioning'),
                (532, 'ReferenceBlackWhite'),
                (559, 'StripRowCounts'),
                (700, 'XMP'),  # XMLPacket
                (769, 'GDIGamma'),  # GDI+
                (770, 'ICCProfileDescriptor'),  # GDI+
                (771, 'SRGBRenderingIntent'),  # GDI+
                (800, 'ImageTitle'),  # GDI+
                (999, 'USPTO_Miscellaneous'),
                (4864, 'AndorId'),  # TODO, Andor Technology 4864 - 5030
                (4869, 'AndorTemperature'),
                (4876, 'AndorExposureTime'),
                (4878, 'AndorKineticCycleTime'),
                (4879, 'AndorAccumulations'),
                (4881, 'AndorAcquisitionCycleTime'),
                (4882, 'AndorReadoutTime'),
                (4884, 'AndorPhotonCounting'),
                (4885, 'AndorEmDacLevel'),
                (4890, 'AndorFrames'),
                (4896, 'AndorHorizontalFlip'),
                (4897, 'AndorVerticalFlip'),
                (4898, 'AndorClockwise'),
                (4899, 'AndorCounterClockwise'),
                (4904, 'AndorVerticalClockVoltage'),
                (4905, 'AndorVerticalShiftSpeed'),
                (4907, 'AndorPreAmpSetting'),
                (4908, 'AndorCameraSerial'),
                (4911, 'AndorActualTemperature'),
                (4912, 'AndorBaselineClamp'),
                (4913, 'AndorPrescans'),
                (4914, 'AndorModel'),
                (4915, 'AndorChipSizeX'),
                (4916, 'AndorChipSizeY'),
                (4944, 'AndorBaselineOffset'),
                (4966, 'AndorSoftwareVersion'),
                (18246, 'Rating'),
                (18247, 'XP_DIP_XML'),
                (18248, 'StitchInfo'),
                (18249, 'RatingPercent'),
                (20481, 'ResolutionXUnit'),  # GDI+
                (20482, 'ResolutionYUnit'),  # GDI+
                (20483, 'ResolutionXLengthUnit'),  # GDI+
                (20484, 'ResolutionYLengthUnit'),  # GDI+
                (20485, 'PrintFlags'),  # GDI+
                (20486, 'PrintFlagsVersion'),  # GDI+
                (20487, 'PrintFlagsCrop'),  # GDI+
                (20488, 'PrintFlagsBleedWidth'),  # GDI+
                (20489, 'PrintFlagsBleedWidthScale'),  # GDI+
                (20490, 'HalftoneLPI'),  # GDI+
                (20491, 'HalftoneLPIUnit'),  # GDI+
                (20492, 'HalftoneDegree'),  # GDI+
                (20493, 'HalftoneShape'),  # GDI+
                (20494, 'HalftoneMisc'),  # GDI+
                (20495, 'HalftoneScreen'),  # GDI+
                (20496, 'JPEGQuality'),  # GDI+
                (20497, 'GridSize'),  # GDI+
                (20498, 'ThumbnailFormat'),  # GDI+
                (20499, 'ThumbnailWidth'),  # GDI+
                (20500, 'ThumbnailHeight'),  # GDI+
                (20501, 'ThumbnailColorDepth'),  # GDI+
                (20502, 'ThumbnailPlanes'),  # GDI+
                (20503, 'ThumbnailRawBytes'),  # GDI+
                (20504, 'ThumbnailSize'),  # GDI+
                (20505, 'ThumbnailCompressedSize'),  # GDI+
                (20506, 'ColorTransferFunction'),  # GDI+
                (20507, 'ThumbnailData'),
                (20512, 'ThumbnailImageWidth'),  # GDI+
                (20513, 'ThumbnailImageHeight'),  # GDI+
                (20514, 'ThumbnailBitsPerSample'),  # GDI+
                (20515, 'ThumbnailCompression'),
                (20516, 'ThumbnailPhotometricInterp'),  # GDI+
                (20517, 'ThumbnailImageDescription'),  # GDI+
                (20518, 'ThumbnailEquipMake'),  # GDI+
                (20519, 'ThumbnailEquipModel'),  # GDI+
                (20520, 'ThumbnailStripOffsets'),  # GDI+
                (20521, 'ThumbnailOrientation'),  # GDI+
                (20522, 'ThumbnailSamplesPerPixel'),  # GDI+
                (20523, 'ThumbnailRowsPerStrip'),  # GDI+
                (20524, 'ThumbnailStripBytesCount'),  # GDI+
                (20525, 'ThumbnailResolutionX'),
                (20526, 'ThumbnailResolutionY'),
                (20527, 'ThumbnailPlanarConfig'),  # GDI+
                (20528, 'ThumbnailResolutionUnit'),
                (20529, 'ThumbnailTransferFunction'),
                (20530, 'ThumbnailSoftwareUsed'),  # GDI+
                (20531, 'ThumbnailDateTime'),  # GDI+
                (20532, 'ThumbnailArtist'),  # GDI+
                (20533, 'ThumbnailWhitePoint'),  # GDI+
                (20534, 'ThumbnailPrimaryChromaticities'),  # GDI+
                (20535, 'ThumbnailYCbCrCoefficients'),  # GDI+
                (20536, 'ThumbnailYCbCrSubsampling'),  # GDI+
                (20537, 'ThumbnailYCbCrPositioning'),
                (20538, 'ThumbnailRefBlackWhite'),  # GDI+
                (20539, 'ThumbnailCopyRight'),  # GDI+
                (20545, 'InteroperabilityIndex'),
                (20546, 'InteroperabilityVersion'),
                (20624, 'LuminanceTable'),
                (20625, 'ChrominanceTable'),
                (20736, 'FrameDelay'),  # GDI+
                (20737, 'LoopCount'),  # GDI+
                (20738, 'GlobalPalette'),  # GDI+
                (20739, 'IndexBackground'),  # GDI+
                (20740, 'IndexTransparent'),  # GDI+
                (20752, 'PixelUnit'),  # GDI+
                (20753, 'PixelPerUnitX'),  # GDI+
                (20754, 'PixelPerUnitY'),  # GDI+
                (20755, 'PaletteHistogram'),  # GDI+
                (28672, 'SonyRawFileType'),  # Sony ARW
                (28722, 'VignettingCorrParams'),  # Sony ARW
                (28725, 'ChromaticAberrationCorrParams'),  # Sony ARW
                (28727, 'DistortionCorrParams'),  # Sony ARW
                # Private tags >= 32768
                (32781, 'ImageID'),
                (32931, 'WangTag1'),
                (32932, 'WangAnnotation'),
                (32933, 'WangTag3'),
                (32934, 'WangTag4'),
                (32953, 'ImageReferencePoints'),
                (32954, 'RegionXformTackPoint'),
                (32955, 'WarpQuadrilateral'),
                (32956, 'AffineTransformMat'),
                (32995, 'Matteing'),
                (32996, 'DataType'),  # use SampleFormat
                (32997, 'ImageDepth'),
                (32998, 'TileDepth'),
                (33300, 'ImageFullWidth'),
                (33301, 'ImageFullLength'),
                (33302, 'TextureFormat'),
                (33303, 'TextureWrapModes'),
                (33304, 'FieldOfViewCotangent'),
                (33305, 'MatrixWorldToScreen'),
                (33306, 'MatrixWorldToCamera'),
                (33405, 'Model2'),
                (33421, 'CFARepeatPatternDim'),
                (33422, 'CFAPattern'),
                (33423, 'BatteryLevel'),
                (33424, 'KodakIFD'),
                (33434, 'ExposureTime'),
                (33437, 'FNumber'),
                (33432, 'Copyright'),
                (33445, 'MDFileTag'),
                (33446, 'MDScalePixel'),
                (33447, 'MDColorTable'),
                (33448, 'MDLabName'),
                (33449, 'MDSampleInfo'),
                (33450, 'MDPrepDate'),
                (33451, 'MDPrepTime'),
                (33452, 'MDFileUnits'),
                (33465, 'NiffRotation'),  # NIFF
                (33466, 'NiffNavyCompression'),  # NIFF
                (33467, 'NiffTileIndex'),  # NIFF
                (33471, 'OlympusINI'),
                (33550, 'ModelPixelScaleTag'),
                (33560, 'OlympusSIS'),  # see also 33471 and 34853
                (33589, 'AdventScale'),
                (33590, 'AdventRevision'),
                (33628, 'UIC1tag'),  # Metamorph  Universal Imaging Corp STK
                (33629, 'UIC2tag'),
                (33630, 'UIC3tag'),
                (33631, 'UIC4tag'),
                (33723, 'IPTCNAA'),
                (33858, 'ExtendedTagsOffset'),  # DEFF points IFD with tags
                (33918, 'IntergraphPacketData'),  # INGRPacketDataTag
                (33919, 'IntergraphFlagRegisters'),  # INGRFlagRegisters
                (33920, 'IntergraphMatrixTag'),  # IrasBTransformationMatrix
                (33921, 'INGRReserved'),
                (33922, 'ModelTiepointTag'),
                (33923, 'LeicaMagic'),
                (34016, 'Site'),  # 34016..34032 ANSI IT8 TIFF/IT
                (34017, 'ColorSequence'),
                (34018, 'IT8Header'),
                (34019, 'RasterPadding'),
                (34020, 'BitsPerRunLength'),
                (34021, 'BitsPerExtendedRunLength'),
                (34022, 'ColorTable'),
                (34023, 'ImageColorIndicator'),
                (34024, 'BackgroundColorIndicator'),
                (34025, 'ImageColorValue'),
                (34026, 'BackgroundColorValue'),
                (34027, 'PixelIntensityRange'),
                (34028, 'TransparencyIndicator'),
                (34029, 'ColorCharacterization'),
                (34030, 'HCUsage'),
                (34031, 'TrapIndicator'),
                (34032, 'CMYKEquivalent'),
                (34118, 'CZ_SEM'),  # Zeiss SEM
                (34152, 'AFCP_IPTC'),
                (34232, 'PixelMagicJBIGOptions'),  # EXIF, also TI FrameCount
                (34263, 'JPLCartoIFD'),
                (34122, 'IPLAB'),  # number of images
                (34264, 'ModelTransformationTag'),
                (34306, 'WB_GRGBLevels'),  # Leaf MOS
                (34310, 'LeafData'),
                (34361, 'MM_Header'),
                (34362, 'MM_Stamp'),
                (34363, 'MM_Unknown'),
                (34377, 'ImageResources'),  # Photoshop
                (34386, 'MM_UserBlock'),
                (34412, 'CZ_LSMINFO'),
                (34665, 'ExifTag'),
                (34675, 'InterColorProfile'),  # ICCProfile
                (34680, 'FEI_SFEG'),  #
                (34682, 'FEI_HELIOS'),  #
                (34683, 'FEI_TITAN'),  #
                (34687, 'FXExtensions'),
                (34688, 'MultiProfiles'),
                (34689, 'SharedData'),
                (34690, 'T88Options'),
                (34710, 'MarCCD'),  # offset to MarCCD header
                (34732, 'ImageLayer'),
                (34735, 'GeoKeyDirectoryTag'),
                (34736, 'GeoDoubleParamsTag'),
                (34737, 'GeoAsciiParamsTag'),
                (34750, 'JBIGOptions'),
                (34821, 'PIXTIFF'),  # ? Pixel Translations Inc
                (34850, 'ExposureProgram'),
                (34852, 'SpectralSensitivity'),
                (34853, 'GPSTag'),  # GPSIFD  also OlympusSIS2
                (34853, 'OlympusSIS2'),
                (34855, 'ISOSpeedRatings'),
                (34855, 'PhotographicSensitivity'),
                (34856, 'OECF'),  # optoelectric conversion factor
                (34857, 'Interlace'),
                (34858, 'TimeZoneOffset'),
                (34859, 'SelfTimerMode'),
                (34864, 'SensitivityType'),
                (34865, 'StandardOutputSensitivity'),
                (34866, 'RecommendedExposureIndex'),
                (34867, 'ISOSpeed'),
                (34868, 'ISOSpeedLatitudeyyy'),
                (34869, 'ISOSpeedLatitudezzz'),
                (34908, 'HylaFAXFaxRecvParams'),
                (34909, 'HylaFAXFaxSubAddress'),
                (34910, 'HylaFAXFaxRecvTime'),
                (34911, 'FaxDcs'),
                (34929, 'FedexEDR'),
                (34954, 'LeafSubIFD'),
                (34959, 'Aphelion1'),
                (34960, 'Aphelion2'),
                (34961, 'AphelionInternal'),  # ADCIS
                (36864, 'ExifVersion'),
                (36867, 'DateTimeOriginal'),
                (36868, 'DateTimeDigitized'),
                (36873, 'GooglePlusUploadCode'),
                (36880, 'OffsetTime'),
                (36881, 'OffsetTimeOriginal'),
                (36882, 'OffsetTimeDigitized'),
                # TODO, Pilatus/CHESS/TV6 36864..37120 conflicting with Exif
                (36864, 'TVX_Unknown'),
                (36865, 'TVX_NumExposure'),
                (36866, 'TVX_NumBackground'),
                (36867, 'TVX_ExposureTime'),
                (36868, 'TVX_BackgroundTime'),
                (36870, 'TVX_Unknown'),
                (36873, 'TVX_SubBpp'),
                (36874, 'TVX_SubWide'),
                (36875, 'TVX_SubHigh'),
                (36876, 'TVX_BlackLevel'),
                (36877, 'TVX_DarkCurrent'),
                (36878, 'TVX_ReadNoise'),
                (36879, 'TVX_DarkCurrentNoise'),
                (36880, 'TVX_BeamMonitor'),
                (37120, 'TVX_UserVariables'),  # A/D values
                (37121, 'ComponentsConfiguration'),
                (37122, 'CompressedBitsPerPixel'),
                (37377, 'ShutterSpeedValue'),
                (37378, 'ApertureValue'),
                (37379, 'BrightnessValue'),
                (37380, 'ExposureBiasValue'),
                (37381, 'MaxApertureValue'),
                (37382, 'SubjectDistance'),
                (37383, 'MeteringMode'),
                (37384, 'LightSource'),
                (37385, 'Flash'),
                (37386, 'FocalLength'),
                (37387, 'FlashEnergy'),  # 37387
                (37388, 'SpatialFrequencyResponse'),  # 37388
                (37389, 'Noise'),
                (37390, 'FocalPlaneXResolution'),
                (37391, 'FocalPlaneYResolution'),
                (37392, 'FocalPlaneResolutionUnit'),
                (37393, 'ImageNumber'),
                (37394, 'SecurityClassification'),
                (37395, 'ImageHistory'),
                (37396, 'SubjectLocation'),
                (37397, 'ExposureIndex'),
                (37398, 'TIFFEPStandardID'),
                (37399, 'SensingMethod'),
                (37434, 'CIP3DataFile'),
                (37435, 'CIP3Sheet'),
                (37436, 'CIP3Side'),
                (37439, 'StoNits'),
                (37500, 'MakerNote'),
                (37510, 'UserComment'),
                (37520, 'SubsecTime'),
                (37521, 'SubsecTimeOriginal'),
                (37522, 'SubsecTimeDigitized'),
                (37679, 'MODIText'),  # Microsoft Office Document Imaging
                (37680, 'MODIOLEPropertySetStorage'),
                (37681, 'MODIPositioning'),
                (37706, 'TVIPS'),  # offset to TemData structure
                (37707, 'TVIPS1'),
                (37708, 'TVIPS2'),  # same TemData structure as undefined
                (37724, 'ImageSourceData'),  # Photoshop
                (37888, 'Temperature'),
                (37889, 'Humidity'),
                (37890, 'Pressure'),
                (37891, 'WaterDepth'),
                (37892, 'Acceleration'),
                (37893, 'CameraElevationAngle'),
                (40000, 'XPos'),  # Janelia
                (40001, 'YPos'),
                (40002, 'ZPos'),
                (40001, 'MC_IpWinScal'),  # Media Cybernetics
                (40001, 'RecipName'),  # MS FAX
                (40002, 'RecipNumber'),
                (40003, 'SenderName'),
                (40004, 'Routing'),
                (40005, 'CallerId'),
                (40006, 'TSID'),
                (40007, 'CSID'),
                (40008, 'FaxTime'),
                (40100, 'MC_IdOld'),
                (40106, 'MC_Unknown'),
                (40965, 'InteroperabilityTag'),  # InteropOffset
                (40091, 'XPTitle'),
                (40092, 'XPComment'),
                (40093, 'XPAuthor'),
                (40094, 'XPKeywords'),
                (40095, 'XPSubject'),
                (40960, 'FlashpixVersion'),
                (40961, 'ColorSpace'),
                (40962, 'PixelXDimension'),
                (40963, 'PixelYDimension'),
                (40964, 'RelatedSoundFile'),
                (40976, 'SamsungRawPointersOffset'),
                (40977, 'SamsungRawPointersLength'),
                (41217, 'SamsungRawByteOrder'),
                (41218, 'SamsungRawUnknown'),
                (41483, 'FlashEnergy'),
                (41484, 'SpatialFrequencyResponse'),
                (41485, 'Noise'),  # 37389
                (41486, 'FocalPlaneXResolution'),  # 37390
                (41487, 'FocalPlaneYResolution'),  # 37391
                (41488, 'FocalPlaneResolutionUnit'),  # 37392
                (41489, 'ImageNumber'),  # 37393
                (41490, 'SecurityClassification'),  # 37394
                (41491, 'ImageHistory'),  # 37395
                (41492, 'SubjectLocation'),  # 37395
                (41493, 'ExposureIndex '),  # 37397
                (41494, 'TIFF-EPStandardID'),
                (41495, 'SensingMethod'),  # 37399
                (41728, 'FileSource'),
                (41729, 'SceneType'),
                (41730, 'CFAPattern'),  # 33422
                (41985, 'CustomRendered'),
                (41986, 'ExposureMode'),
                (41987, 'WhiteBalance'),
                (41988, 'DigitalZoomRatio'),
                (41989, 'FocalLengthIn35mmFilm'),
                (41990, 'SceneCaptureType'),
                (41991, 'GainControl'),
                (41992, 'Contrast'),
                (41993, 'Saturation'),
                (41994, 'Sharpness'),
                (41995, 'DeviceSettingDescription'),
                (41996, 'SubjectDistanceRange'),
                (42016, 'ImageUniqueID'),
                (42032, 'CameraOwnerName'),
                (42033, 'BodySerialNumber'),
                (42034, 'LensSpecification'),
                (42035, 'LensMake'),
                (42036, 'LensModel'),
                (42037, 'LensSerialNumber'),
                (42080, 'CompositeImage'),
                (42081, 'SourceImageNumberCompositeImage'),
                (42082, 'SourceExposureTimesCompositeImage'),
                (42112, 'GDAL_METADATA'),
                (42113, 'GDAL_NODATA'),
                (42240, 'Gamma'),
                (43314, 'NIHImageHeader'),
                (44992, 'ExpandSoftware'),
                (44993, 'ExpandLens'),
                (44994, 'ExpandFilm'),
                (44995, 'ExpandFilterLens'),
                (44996, 'ExpandScanner'),
                (44997, 'ExpandFlashLamp'),
                (48129, 'PixelFormat'),  # HDP and WDP
                (48130, 'Transformation'),
                (48131, 'Uncompressed'),
                (48132, 'ImageType'),
                (48256, 'ImageWidth'),  # 256
                (48257, 'ImageHeight'),
                (48258, 'WidthResolution'),
                (48259, 'HeightResolution'),
                (48320, 'ImageOffset'),
                (48321, 'ImageByteCount'),
                (48322, 'AlphaOffset'),
                (48323, 'AlphaByteCount'),
                (48324, 'ImageDataDiscard'),
                (48325, 'AlphaDataDiscard'),
                (50003, 'KodakAPP3'),
                (50215, 'OceScanjobDescription'),
                (50216, 'OceApplicationSelector'),
                (50217, 'OceIdentificationNumber'),
                (50218, 'OceImageLogicCharacteristics'),
                (50255, 'Annotations'),
                (50288, 'MC_Id'),  # Media Cybernetics
                (50289, 'MC_XYPosition'),
                (50290, 'MC_ZPosition'),
                (50291, 'MC_XYCalibration'),
                (50292, 'MC_LensCharacteristics'),
                (50293, 'MC_ChannelName'),
                (50294, 'MC_ExcitationWavelength'),
                (50295, 'MC_TimeStamp'),
                (50296, 'MC_FrameProperties'),
                (50341, 'PrintImageMatching'),
                (50495, 'PCO_RAW'),  # TODO, PCO CamWare
                (50547, 'OriginalFileName'),
                (50560, 'USPTO_OriginalContentType'),  # US Patent Office
                (50561, 'USPTO_RotationCode'),
                (50648, 'CR2Unknown1'),
                (50649, 'CR2Unknown2'),
                (50656, 'CR2CFAPattern'),
                (50674, 'LercParameters'),  # ESGI 50674 .. 50677
                (50706, 'DNGVersion'),  # DNG 50706 .. 51114
                (50707, 'DNGBackwardVersion'),
                (50708, 'UniqueCameraModel'),
                (50709, 'LocalizedCameraModel'),
                (50710, 'CFAPlaneColor'),
                (50711, 'CFALayout'),
                (50712, 'LinearizationTable'),
                (50713, 'BlackLevelRepeatDim'),
                (50714, 'BlackLevel'),
                (50715, 'BlackLevelDeltaH'),
                (50716, 'BlackLevelDeltaV'),
                (50717, 'WhiteLevel'),
                (50718, 'DefaultScale'),
                (50719, 'DefaultCropOrigin'),
                (50720, 'DefaultCropSize'),
                (50721, 'ColorMatrix1'),
                (50722, 'ColorMatrix2'),
                (50723, 'CameraCalibration1'),
                (50724, 'CameraCalibration2'),
                (50725, 'ReductionMatrix1'),
                (50726, 'ReductionMatrix2'),
                (50727, 'AnalogBalance'),
                (50728, 'AsShotNeutral'),
                (50729, 'AsShotWhiteXY'),
                (50730, 'BaselineExposure'),
                (50731, 'BaselineNoise'),
                (50732, 'BaselineSharpness'),
                (50733, 'BayerGreenSplit'),
                (50734, 'LinearResponseLimit'),
                (50735, 'CameraSerialNumber'),
                (50736, 'LensInfo'),
                (50737, 'ChromaBlurRadius'),
                (50738, 'AntiAliasStrength'),
                (50739, 'ShadowScale'),
                (50740, 'DNGPrivateData'),
                (50741, 'MakerNoteSafety'),
                (50752, 'RawImageSegmentation'),
                (50778, 'CalibrationIlluminant1'),
                (50779, 'CalibrationIlluminant2'),
                (50780, 'BestQualityScale'),
                (50781, 'RawDataUniqueID'),
                (50784, 'AliasLayerMetadata'),
                (50827, 'OriginalRawFileName'),
                (50828, 'OriginalRawFileData'),
                (50829, 'ActiveArea'),
                (50830, 'MaskedAreas'),
                (50831, 'AsShotICCProfile'),
                (50832, 'AsShotPreProfileMatrix'),
                (50833, 'CurrentICCProfile'),
                (50834, 'CurrentPreProfileMatrix'),
                (50838, 'IJMetadataByteCounts'),
                (50839, 'IJMetadata'),
                (50844, 'RPCCoefficientTag'),
                (50879, 'ColorimetricReference'),
                (50885, 'SRawType'),
                (50898, 'PanasonicTitle'),
                (50899, 'PanasonicTitle2'),
                (50908, 'RSID'),  # DGIWG
                (50909, 'GEO_METADATA'),  # DGIWG XML
                (50931, 'CameraCalibrationSignature'),
                (50932, 'ProfileCalibrationSignature'),
                (50933, 'ProfileIFD'),  # EXTRACAMERAPROFILES
                (50934, 'AsShotProfileName'),
                (50935, 'NoiseReductionApplied'),
                (50936, 'ProfileName'),
                (50937, 'ProfileHueSatMapDims'),
                (50938, 'ProfileHueSatMapData1'),
                (50939, 'ProfileHueSatMapData2'),
                (50940, 'ProfileToneCurve'),
                (50941, 'ProfileEmbedPolicy'),
                (50942, 'ProfileCopyright'),
                (50964, 'ForwardMatrix1'),
                (50965, 'ForwardMatrix2'),
                (50966, 'PreviewApplicationName'),
                (50967, 'PreviewApplicationVersion'),
                (50968, 'PreviewSettingsName'),
                (50969, 'PreviewSettingsDigest'),
                (50970, 'PreviewColorSpace'),
                (50971, 'PreviewDateTime'),
                (50972, 'RawImageDigest'),
                (50973, 'OriginalRawFileDigest'),
                (50974, 'SubTileBlockSize'),
                (50975, 'RowInterleaveFactor'),
                (50981, 'ProfileLookTableDims'),
                (50982, 'ProfileLookTableData'),
                (51008, 'OpcodeList1'),
                (51009, 'OpcodeList2'),
                (51022, 'OpcodeList3'),
                (51023, 'FibicsXML'),  #
                (51041, 'NoiseProfile'),
                (51043, 'TimeCodes'),
                (51044, 'FrameRate'),
                (51058, 'TStop'),
                (51081, 'ReelName'),
                (51089, 'OriginalDefaultFinalSize'),
                (51090, 'OriginalBestQualitySize'),
                (51091, 'OriginalDefaultCropSize'),
                (51105, 'CameraLabel'),
                (51107, 'ProfileHueSatMapEncoding'),
                (51108, 'ProfileLookTableEncoding'),
                (51109, 'BaselineExposureOffset'),
                (51110, 'DefaultBlackRender'),
                (51111, 'NewRawImageDigest'),
                (51112, 'RawToPreviewGain'),
                (51113, 'CacheBlob'),
                (51114, 'CacheVersion'),
                (51123, 'MicroManagerMetadata'),
                (51125, 'DefaultUserCrop'),
                (51159, 'ZIFmetadata'),  # Objective Pathology Services
                (51160, 'ZIFannotations'),  # Objective Pathology Services
                (51177, 'DepthFormat'),
                (51178, 'DepthNear'),
                (51179, 'DepthFar'),
                (51180, 'DepthUnits'),
                (51181, 'DepthMeasureType'),
                (51182, 'EnhanceParams'),
                (59932, 'Padding'),
                (59933, 'OffsetSchema'),
                # Reusable Tags 65000-65535
                # (65000,  DimapDocumentXML'),
                # (65001, 'EER_XML'),
                # 65000-65112,  Photoshop Camera RAW EXIF tags
                # (65000, 'OwnerName'),
                # (65001, 'SerialNumber'),
                # (65002, 'Lens'),
                # (65024, 'KodakKDCPrivateIFD'),
                # (65100, 'RawFile'),
                # (65101, 'Converter'),
                # (65102, 'WhiteBalance'),
                # (65105, 'Exposure'),
                # (65106, 'Shadows'),
                # (65107, 'Brightness'),
                # (65108, 'Contrast'),
                # (65109, 'Saturation'),
                # (65110, 'Sharpness'),
                # (65111, 'Smoothness'),
                # (65112, 'MoireFilter'),
                (65200, 'FlexXML'),
            )
        )

    def TAG_READERS():
        # map tag codes to import functions
        return {
            301: read_colormap,
            320: read_colormap,
            # 700: read_bytes,  # read_utf8,
            # 34377: read_bytes,
            33723: read_bytes,
            # 34675: read_bytes,
            33628: read_uic1tag,  # Universal Imaging Corp STK
            33629: read_uic2tag,
            33630: read_uic3tag,
            33631: read_uic4tag,
            34118: read_cz_sem,  # Carl Zeiss SEM
            34361: read_mm_header,  # Olympus FluoView
            34362: read_mm_stamp,
            34363: read_numpy,  # MM_Unknown
            34386: read_numpy,  # MM_UserBlock
            34412: read_cz_lsminfo,  # Carl Zeiss LSM
            34680: read_fei_metadata,  # S-FEG
            34682: read_fei_metadata,  # Helios NanoLab
            37706: read_tvips_header,  # TVIPS EMMENU
            37724: read_bytes,  # ImageSourceData
            33923: read_bytes,  # read_leica_magic
            43314: read_nih_image_header,
            # 40001: read_bytes,
            40100: read_bytes,
            50288: read_bytes,
            50296: read_bytes,
            50839: read_bytes,
            51123: read_json,
            33471: read_sis_ini,
            33560: read_sis,
            34665: read_exif_ifd,
            34853: read_gps_ifd,  # conflicts with OlympusSIS
            40965: read_interoperability_ifd,
            65426: read_numpy,  # NDPI McuStarts
            65432: read_numpy,  # NDPI McuStartsHighBytes
            65439: read_numpy,  # NDPI unknown
            65459: read_bytes,  # NDPI bytes, not string
        }

    def TAG_LOAD():
        # tags whose values are not delay loaded
        return frozenset(
            (
                258,  # BitsPerSample
                270,  # ImageDescription
                273,  # StripOffsets
                277,  # SamplesPerPixel
                279,  # StripByteCounts
                282,  # XResolution
                283,  # YResolution
                # 301,  # TransferFunction
                305,  # Software
                # 306,  # DateTime
                # 320,  # ColorMap
                324,  # TileOffsets
                325,  # TileByteCounts
                330,  # SubIFDs
                338,  # ExtraSamples
                339,  # SampleFormat
                347,  # JPEGTables
                513,  # JPEGInterchangeFormat
                514,  # JPEGInterchangeFormatLength
                530,  # YCbCrSubSampling
                33628,  # UIC1tag
                42113,  # GDAL_NODATA
                50838,  # IJMetadataByteCounts
                50839,  # IJMetadata
            )
        )

    def TAG_TUPLE():
        # tags whose values must be stored as tuples
        return frozenset(
            (273, 279, 324, 325, 330, 338, 513, 514, 530, 531, 34736)
        )

    def TAG_ATTRIBUTES():
        # map tag codes to TiffPage attribute names
        return {
            254: 'subfiletype',
            256: 'imagewidth',
            257: 'imagelength',
            # 258: 'bitspersample',  # set manually
            259: 'compression',
            262: 'photometric',
            266: 'fillorder',
            270: 'description',
            277: 'samplesperpixel',
            278: 'rowsperstrip',
            284: 'planarconfig',
            # 301: 'transferfunction',  # delay load
            305: 'software',
            # 320: 'colormap',  # delay load
            317: 'predictor',
            322: 'tilewidth',
            323: 'tilelength',
            330: 'subifds',
            338: 'extrasamples',
            # 339: 'sampleformat',  # set manually
            347: 'jpegtables',
            530: 'subsampling',
            32997: 'imagedepth',
            32998: 'tiledepth',
        }

    def TAG_ENUM():
        # map tag codes to Enums
        class TAG_ENUM:

            __slots__ = ('_codes',)

            def __init__(self):
                self._codes = {
                    254: None,
                    255: None,
                    259: TIFF.COMPRESSION,
                    262: TIFF.PHOTOMETRIC,
                    263: None,
                    266: None,
                    274: None,
                    284: TIFF.PLANARCONFIG,
                    290: None,
                    296: None,
                    300: None,
                    317: None,
                    338: None,
                    339: TIFF.SAMPLEFORMAT,
                }

            def __contains__(self, key):
                return key in self._codes

            def __getitem__(self, key):
                value = self._codes[key]
                if value is not None:
                    return value
                if key == 254:
                    value = TIFF.FILETYPE
                elif key == 255:
                    value = TIFF.OFILETYPE
                elif key == 263:
                    value = TIFF.THRESHHOLD
                elif key == 266:
                    value = TIFF.FILLORDER
                elif key == 274:
                    value = TIFF.ORIENTATION
                elif key == 290:
                    value = TIFF.GRAYRESPONSEUNIT
                # elif key == 292:
                #     value = TIFF.GROUP3OPT
                # elif key == 293:
                #     value = TIFF.GROUP4OPT
                elif key == 296:
                    value = TIFF.RESUNIT
                elif key == 300:
                    value = TIFF.COLORRESPONSEUNIT
                elif key == 317:
                    value = TIFF.PREDICTOR
                elif key == 338:
                    value = TIFF.EXTRASAMPLE
                # elif key == 512:
                #     TIFF.JPEGPROC
                # elif key == 531:
                #     TIFF.YCBCRPOSITION
                else:
                    raise KeyError(key)
                self._codes[key] = value
                return value

        return TAG_ENUM()

    def FILETYPE():
        class FILETYPE(enum.IntFlag):
            UNDEFINED = 0
            REDUCEDIMAGE = 1
            PAGE = 2
            MASK = 4
            MACRO = 8  # Aperio SVS, or DNG Depth map
            ENHANCED = 16  # DNG
            DNG = 65536  # 65537: Alternative, 65540: Semantic mask

        return FILETYPE

    def OFILETYPE():
        class OFILETYPE(enum.IntEnum):
            UNDEFINED = 0
            IMAGE = 1
            REDUCEDIMAGE = 2
            PAGE = 3

        return OFILETYPE

    def COMPRESSION():
        class COMPRESSION(enum.IntEnum):
            NONE = 1  # Uncompressed
            CCITTRLE = 2  # CCITT 1D
            CCITT_T4 = 3  # T4/Group 3 Fax
            CCITT_T6 = 4  # T6/Group 4 Fax
            LZW = 5
            OJPEG = 6  # old-style JPEG
            JPEG = 7
            ADOBE_DEFLATE = 8
            JBIG_BW = 9
            JBIG_COLOR = 10
            JPEG_99 = 99
            KODAK_262 = 262
            JPEGXR_NDPI = 22610
            NEXT = 32766
            SONY_ARW = 32767
            PACKED_RAW = 32769
            SAMSUNG_SRW = 32770
            CCIRLEW = 32771
            SAMSUNG_SRW2 = 32772
            PACKBITS = 32773
            THUNDERSCAN = 32809
            IT8CTPAD = 32895
            IT8LW = 32896
            IT8MP = 32897
            IT8BL = 32898
            PIXARFILM = 32908
            PIXARLOG = 32909
            DEFLATE = 32946
            DCS = 32947
            APERIO_JP2000_YCBC = 33003  # Leica Aperio
            JPEG_2000_LOSSY = 33004  # BioFormats
            APERIO_JP2000_RGB = 33005  # Leica Aperio
            ALT_JPEG = 33007  # BioFormats
            JBIG = 34661
            SGILOG = 34676
            SGILOG24 = 34677
            JPEG2000 = 34712
            NIKON_NEF = 34713
            JBIG2 = 34715
            MDI_BINARY = 34718  # Microsoft Document Imaging
            MDI_PROGRESSIVE = 34719  # Microsoft Document Imaging
            MDI_VECTOR = 34720  # Microsoft Document Imaging
            LERC = 34887  # ESRI Lerc
            JPEG_LOSSY = 34892  # DNG
            LZMA = 34925
            ZSTD_DEPRECATED = 34926
            WEBP_DEPRECATED = 34927
            PNG = 34933  # Objective Pathology Services
            JPEGXR = 34934  # Objective Pathology Services
            ZSTD = 50000
            WEBP = 50001
            JPEGXL = 50002  # JXL
            PIXTIFF = 50013
            # EER_V0 = 65000
            # EER_V1 = 65001
            # KODAK_DCR = 65000
            # PENTAX_PEF = 65535

            def __bool__(self):
                return self != 1

        return COMPRESSION

    def PHOTOMETRIC():
        class PHOTOMETRIC(enum.IntEnum):
            MINISWHITE = 0
            MINISBLACK = 1
            RGB = 2
            PALETTE = 3
            MASK = 4
            SEPARATED = 5  # CMYK
            YCBCR = 6
            CIELAB = 8
            ICCLAB = 9
            ITULAB = 10
            CFA = 32803  # Color Filter Array
            LOGL = 32844
            LOGLUV = 32845
            LINEAR_RAW = 34892
            DEPTH_MAP = 51177  # DNG 1.5
            SEMANTIC_MASK = 52527  # DNG 1.6

        return PHOTOMETRIC

    def PHOTOMETRIC_SAMPLES():
        return {
            0: 1,  # MINISWHITE
            1: 1,  # MINISBLACK
            2: 3,  # RGB
            3: 1,  # PALETTE
            4: 1,  # MASK
            5: 4,  # SEPARATED
            6: 3,  # YCBCR
            8: 3,  # CIELAB
            9: 3,  # ICCLAB
            10: 3,  # ITULAB
            32803: 1,  # CFA
            32844: 1,  # LOGL ?
            32845: 3,  # LOGLUV
            34892: 3,  # LINEAR_RAW ?
            51177: 1,  # DEPTH_MAP ?
            52527: 1,  # SEMANTIC_MASK ?
        }

    def THRESHHOLD():
        class THRESHHOLD(enum.IntEnum):
            BILEVEL = 1
            HALFTONE = 2
            ERRORDIFFUSE = 3

        return THRESHHOLD

    def FILLORDER():
        class FILLORDER(enum.IntEnum):
            MSB2LSB = 1
            LSB2MSB = 2

        return FILLORDER

    def ORIENTATION():
        class ORIENTATION(enum.IntEnum):
            TOPLEFT = 1
            TOPRIGHT = 2
            BOTRIGHT = 3
            BOTLEFT = 4
            LEFTTOP = 5
            RIGHTTOP = 6
            RIGHTBOT = 7
            LEFTBOT = 8

        return ORIENTATION

    def PLANARCONFIG():
        class PLANARCONFIG(enum.IntEnum):
            CONTIG = 1  # CHUNKY
            SEPARATE = 2

        return PLANARCONFIG

    def GRAYRESPONSEUNIT():
        class GRAYRESPONSEUNIT(enum.IntEnum):
            _10S = 1
            _100S = 2
            _1000S = 3
            _10000S = 4
            _100000S = 5

        return GRAYRESPONSEUNIT

    def GROUP4OPT():
        class GROUP4OPT(enum.IntEnum):
            UNCOMPRESSED = 2

        return GROUP4OPT

    def RESUNIT():
        class RESUNIT(enum.IntEnum):
            NONE = 1
            INCH = 2
            CENTIMETER = 3
            MILLIMETER = 4  # DNG
            MICROMETER = 5  # DNG

            def __bool__(self):
                return self != 1

        return RESUNIT

    def COLORRESPONSEUNIT():
        class COLORRESPONSEUNIT(enum.IntEnum):
            _10S = 1
            _100S = 2
            _1000S = 3
            _10000S = 4
            _100000S = 5

        return COLORRESPONSEUNIT

    def PREDICTOR():
        class PREDICTOR(enum.IntEnum):
            NONE = 1
            HORIZONTAL = 2
            FLOATINGPOINT = 3
            HORIZONTALX2 = 34892  # DNG
            HORIZONTALX4 = 34893
            FLOATINGPOINTX2 = 34894
            FLOATINGPOINTX4 = 34895

            def __bool__(self):
                return self != 1

        return PREDICTOR

    def EXTRASAMPLE():
        class EXTRASAMPLE(enum.IntEnum):
            UNSPECIFIED = 0
            ASSOCALPHA = 1
            UNASSALPHA = 2

        return EXTRASAMPLE

    def SAMPLEFORMAT():
        class SAMPLEFORMAT(enum.IntEnum):
            UINT = 1
            INT = 2
            IEEEFP = 3
            VOID = 4
            COMPLEXINT = 5
            COMPLEXIEEEFP = 6

        return SAMPLEFORMAT

    def DATATYPES():
        class DATATYPES(enum.IntEnum):
            BYTE = 1  # 8-bit unsigned integer
            ASCII = 2  # 8-bit byte that contains a 7-bit ASCII code;
            #            the last byte must be NULL (binary zero)
            SHORT = 3  # 16-bit (2-byte) unsigned integer
            LONG = 4  # 32-bit (4-byte) unsigned integer
            RATIONAL = 5  # two LONGs: the first represents the numerator
            #               of a fraction; the second, the denominator
            SBYTE = 6  # an 8-bit signed (twos-complement) integer
            UNDEFINED = 7  # an 8-bit byte that may contain anything,
            #                depending on the definition of the field
            SSHORT = 8  # A 16-bit (2-byte) signed (twos-complement) integer
            SLONG = 9  # a 32-bit (4-byte) signed (twos-complement) integer
            SRATIONAL = 10  # two SLONGs: the first represents the numerator
            #                 of a fraction, the second the denominator
            FLOAT = 11  # single precision (4-byte) IEEE format
            DOUBLE = 12  # double precision (8-byte) IEEE format
            IFD = 13  # unsigned 4 byte IFD offset
            UNICODE = 14
            COMPLEX = 15
            LONG8 = 16  # unsigned 8 byte integer (BigTiff)
            SLONG8 = 17  # signed 8 byte integer (BigTiff)
            IFD8 = 18  # unsigned 8 byte IFD offset (BigTiff)

        return DATATYPES

    def DATA_FORMATS():
        # map TIFF DATATYPES to Python struct formats
        return {
            1: '1B',
            2: '1s',
            3: '1H',
            4: '1I',
            5: '2I',
            6: '1b',
            7: '1B',
            8: '1h',
            9: '1i',
            10: '2i',
            11: '1f',
            12: '1d',
            13: '1I',
            # 14: '',
            # 15: '',
            16: '1Q',
            17: '1q',
            18: '1Q',
        }

    def DATA_DTYPES():
        # map numpy dtypes to TIFF DATATYPES
        return {
            'B': 1,
            's': 2,
            'H': 3,
            'I': 4,
            '2I': 5,
            'b': 6,
            'h': 8,
            'i': 9,
            '2i': 10,
            'f': 11,
            'd': 12,
            'Q': 16,
            'q': 17,
        }

    def SAMPLE_DTYPES():
        # map SampleFormat and BitsPerSample to numpy dtype
        return {
            # UINT
            (1, 1): '?',  # bitmap
            (1, 2): 'B',
            (1, 3): 'B',
            (1, 4): 'B',
            (1, 5): 'B',
            (1, 6): 'B',
            (1, 7): 'B',
            (1, 8): 'B',
            (1, 9): 'H',
            (1, 10): 'H',
            (1, 11): 'H',
            (1, 12): 'H',
            (1, 13): 'H',
            (1, 14): 'H',
            (1, 15): 'H',
            (1, 16): 'H',
            (1, 17): 'I',
            (1, 18): 'I',
            (1, 19): 'I',
            (1, 20): 'I',
            (1, 21): 'I',
            (1, 22): 'I',
            (1, 23): 'I',
            (1, 24): 'I',
            (1, 25): 'I',
            (1, 26): 'I',
            (1, 27): 'I',
            (1, 28): 'I',
            (1, 29): 'I',
            (1, 30): 'I',
            (1, 31): 'I',
            (1, 32): 'I',
            (1, 64): 'Q',
            # VOID : treat as UINT
            (4, 1): '?',  # bitmap
            (4, 2): 'B',
            (4, 3): 'B',
            (4, 4): 'B',
            (4, 5): 'B',
            (4, 6): 'B',
            (4, 7): 'B',
            (4, 8): 'B',
            (4, 9): 'H',
            (4, 10): 'H',
            (4, 11): 'H',
            (4, 12): 'H',
            (4, 13): 'H',
            (4, 14): 'H',
            (4, 15): 'H',
            (4, 16): 'H',
            (4, 17): 'I',
            (4, 18): 'I',
            (4, 19): 'I',
            (4, 20): 'I',
            (4, 21): 'I',
            (4, 22): 'I',
            (4, 23): 'I',
            (4, 24): 'I',
            (4, 25): 'I',
            (4, 26): 'I',
            (4, 27): 'I',
            (4, 28): 'I',
            (4, 29): 'I',
            (4, 30): 'I',
            (4, 31): 'I',
            (4, 32): 'I',
            (4, 64): 'Q',
            # INT
            (2, 8): 'b',
            (2, 16): 'h',
            (2, 32): 'i',
            (2, 64): 'q',
            # IEEEFP
            (3, 16): 'e',
            (3, 24): 'f',  # float24 bit not supported by numpy
            (3, 32): 'f',
            (3, 64): 'd',
            # COMPLEXIEEEFP
            (6, 64): 'F',
            (6, 128): 'D',
            # RGB565
            (1, (5, 6, 5)): 'B',
            # COMPLEXINT : not supported by numpy
            (5, 16): 'E',
            (5, 32): 'F',
            (5, 64): 'D',
        }

    def PREDICTORS():
        # map PREDICTOR to predictor encode functions

        class PREDICTORS:
            def __init__(self):
                self._codecs = {None: identityfunc, 1: identityfunc}
                if imagecodecs is None:
                    self._codecs[2] = delta_encode

            def __getitem__(self, key):
                if key in self._codecs:
                    return self._codecs[key]
                try:
                    if key == 2:
                        codec = imagecodecs.delta_encode
                    elif key == 3:
                        codec = imagecodecs.floatpred_encode
                    elif key == 34892:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.delta_encode(
                                data, axis=axis, out=out, dist=2
                            )

                    elif key == 34893:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.delta_encode(
                                data, axis=axis, out=out, dist=4
                            )

                    elif key == 34894:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.floatpred_encode(
                                data, axis=axis, out=out, dist=2
                            )

                    elif key == 34895:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.floatpred_encode(
                                data, axis=axis, out=out, dist=4
                            )

                    else:
                        raise KeyError(f'{key} is not a known PREDICTOR')
                except AttributeError:
                    raise KeyError(
                        f'{TIFF.PREDICTOR(key)!r}'
                        " requires the 'imagecodecs' package"
                    )
                self._codecs[key] = codec
                return codec

        return PREDICTORS()

    def UNPREDICTORS():
        # map PREDICTOR to predictor decode functions

        class UNPREDICTORS:
            def __init__(self):
                self._codecs = {None: identityfunc, 1: identityfunc}
                if imagecodecs is None:
                    self._codecs[2] = delta_decode

            def __getitem__(self, key):
                if key in self._codecs:
                    return self._codecs[key]
                try:
                    if key == 2:
                        codec = imagecodecs.delta_decode
                    elif key == 3:
                        codec = imagecodecs.floatpred_decode
                    elif key == 34892:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.delta_decode(
                                data, axis=axis, out=out, dist=2
                            )

                    elif key == 34893:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.delta_decode(
                                data, axis=axis, out=out, dist=4
                            )

                    elif key == 34894:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.floatpred_decode(
                                data, axis=axis, out=out, dist=2
                            )

                    elif key == 34895:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.floatpred_decode(
                                data, axis=axis, out=out, dist=4
                            )

                    else:
                        raise KeyError(f'{key} is not a known PREDICTOR')
                except AttributeError:
                    raise KeyError(
                        f'{TIFF.PREDICTOR(key)!r}'
                        " requires the 'imagecodecs' package"
                    )
                self._codecs[key] = codec
                return codec

        return UNPREDICTORS()

    def COMPRESSORS():
        # map COMPRESSION to compress functions

        class COMPRESSORS:
            def __init__(self):
                self._codecs = {None: identityfunc, 1: identityfunc}
                if imagecodecs is None:
                    self._codecs[8] = zlib_encode
                    self._codecs[32946] = zlib_encode
                    self._codecs[34925] = lzma_encode

            def __getitem__(self, key):
                if key in self._codecs:
                    return self._codecs[key]
                try:
                    if key == 5:
                        codec = imagecodecs.lzw_encode
                    elif key == 7:
                        codec = imagecodecs.jpeg_encode
                    elif key == 8 or key == 32946:
                        if (
                            hasattr(imagecodecs, 'DEFLATE')
                            and imagecodecs.DEFLATE
                        ):
                            codec = imagecodecs.deflate_encode
                        elif imagecodecs.ZLIB:
                            codec = imagecodecs.zlib_encode
                        else:
                            codec = zlib_encode
                    elif key == 32773:
                        codec = imagecodecs.packbits_encode
                    elif (
                        key == 33003
                        or key == 33004
                        or key == 33005
                        or key == 34712
                    ):
                        codec = imagecodecs.jpeg2k_encode
                    elif key == 34887:
                        codec = imagecodecs.lerc_encode
                    elif key == 34892:
                        codec = imagecodecs.jpeg8_encode  # DNG lossy
                    elif key == 34925:
                        if imagecodecs.LZMA:
                            codec = imagecodecs.lzma_encode
                        else:
                            codec = lzma_encode
                    elif key == 34933:
                        codec = imagecodecs.png_encode
                    elif key == 34934 or key == 22610:
                        codec = imagecodecs.jpegxr_encode
                    elif key == 50000:
                        codec = imagecodecs.zstd_encode
                    elif key == 50001:
                        codec = imagecodecs.webp_encode
                    elif key == 50002:
                        codec = imagecodecs.jpegxl_encode
                    else:
                        try:
                            msg = f'{TIFF.COMPRESSION(key)!r} not supported'
                        except ValueError:
                            msg = f'{key} is not a known COMPRESSION'
                        raise KeyError(msg)
                except AttributeError:
                    raise KeyError(
                        f'{TIFF.COMPRESSION(key)!r} '
                        "requires the 'imagecodecs' package"
                    )
                self._codecs[key] = codec
                return codec

        return COMPRESSORS()

    def DECOMPRESSORS():
        # map COMPRESSION to decompress functions

        class DECOMPRESSORS:
            def __init__(self):
                self._codecs = {None: identityfunc, 1: identityfunc}
                if imagecodecs is None:
                    self._codecs[8] = zlib_decode
                    self._codecs[32773] = packbits_decode
                    self._codecs[32946] = zlib_decode
                    self._codecs[34925] = lzma_decode

            def __getitem__(self, key):
                if key in self._codecs:
                    return self._codecs[key]
                try:
                    if key == 5:
                        codec = imagecodecs.lzw_decode
                    elif key == 6 or key == 7 or key == 33007:
                        codec = imagecodecs.jpeg_decode
                    elif key == 8 or key == 32946:
                        if (
                            hasattr(imagecodecs, 'DEFLATE')
                            and imagecodecs.DEFLATE
                        ):
                            codec = imagecodecs.deflate_decode
                        elif imagecodecs.ZLIB:
                            codec = imagecodecs.zlib_decode
                        else:
                            codec = zlib_decode
                    elif key == 32773:
                        codec = imagecodecs.packbits_decode
                    elif (
                        key == 33003
                        or key == 33004
                        or key == 33005
                        or key == 34712
                    ):
                        codec = imagecodecs.jpeg2k_decode
                    elif key == 34887:
                        codec = imagecodecs.lerc_decode
                    elif key == 34892:
                        codec = imagecodecs.jpeg8_decode  # DNG lossy
                    elif key == 34925:
                        if imagecodecs.LZMA:
                            codec = imagecodecs.lzma_decode
                        else:
                            codec = lzma_decode
                    elif key == 34933:
                        codec = imagecodecs.png_decode
                    elif key == 34934 or key == 22610:
                        codec = imagecodecs.jpegxr_decode
                    elif key == 50000 or key == 34926:  # 34926 deprecated
                        codec = imagecodecs.zstd_decode
                    elif key == 50001 or key == 34927:  # 34927 deprecated
                        codec = imagecodecs.webp_decode
                    elif key == 50002:
                        codec = imagecodecs.jpegxl_decode
                    else:
                        try:
                            msg = f'{TIFF.COMPRESSION(key)!r} not supported'
                        except ValueError:
                            msg = f'{key} is not a known COMPRESSION'
                        raise KeyError(msg)
                except AttributeError:
                    raise KeyError(
                        f'{TIFF.COMPRESSION(key)!r} '
                        "requires the 'imagecodecs' package"
                    )
                self._codecs[key] = codec
                return codec

            def __contains__(self, key):
                try:
                    self[key]
                except KeyError:
                    return False
                return True

        return DECOMPRESSORS()

    def FRAME_ATTRS():
        # attributes that a TiffFrame shares with its keyframe
        return {'shape', 'ndim', 'size', 'dtype', 'axes', 'is_final', 'decode'}

    def FILE_FLAGS():
        # TiffFile and TiffPage 'is_\*' attributes
        exclude = {
            'reduced',
            'mask',
            'final',
            'memmappable',
            'contiguous',
            'tiled',
            'subsampled',
        }
        return {
            a[3:]
            for a in dir(TiffPage)
            if a[:3] == 'is_' and a[3:] not in exclude
        }

    def FILE_PATTERNS():
        # predefined FileSequence patterns
        return {
            'axes': r"""(?ix)
                # matches Olympus OIF and Leica TIFF series
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                """
        }

    def FILE_EXTENSIONS():
        # TIFF file extensions
        return (
            'tif',
            'tiff',
            'ome.tif',
            'lsm',
            'stk',
            'qpi',
            'pcoraw',
            'qptiff',
            'gel',
            'seq',
            'svs',
            'scn',
            'zif',
            'ndpi',
            'bif',
            'tf8',
            'tf2',
            'btf',
            'eer',
        )

    def FILEOPEN_FILTER():
        # string for use in Windows File Open box
        return [
            (f'{ext.upper()} files', f'*.{ext}')
            for ext in TIFF.FILE_EXTENSIONS
        ] + [('allfiles', '*')]

    def AXES_LABELS():

        # TODO: is there a standard for character axes labels?
        axes = {
            'X': 'width',
            'Y': 'length',  # height
            'Z': 'depth',
            'S': 'sample',  # rgb(a), cmyk
            'I': 'series',  # general sequence of frames/planes/pages/IFDs
            'T': 'time',
            'C': 'channel',  # color, emission wavelength
            'A': 'angle',
            'P': 'phase',  # formerly F    # P is Position in LSM!
            'R': 'tile',  # region, point, mosaic
            'H': 'lifetime',  # histogram
            'E': 'lambda',  # excitation wavelength
            'L': 'exposure',  # lux
            'V': 'event',
            'Q': 'other',
            'M': 'mosaic',  # LSM 6
        }
        axes.update({v: k for k, v in axes.items()})
        return axes

    def NDPI_TAGS():
        # 65420 - 65458  Private Hamamatsu NDPI tags
        # TODO: obtain specification
        return TiffTagRegistry(
            (
                (65324, 'OffsetHighBytes'),
                (65325, 'ByteCountHighBytes'),
                (65420, 'FileFormat'),
                (65421, 'Magnification'),  # SourceLens
                (65422, 'XOffsetFromSlideCenter'),
                (65423, 'YOffsetFromSlideCenter'),
                (65424, 'ZOffsetFromSlideCenter'),  # FocalPlane
                (65425, 'TissueIndex'),
                (65426, 'McuStarts'),
                (65427, 'SlideLabel'),
                (65428, 'AuthCode'),  # ?
                (65429, '65429'),
                (65430, '65430'),
                (65431, '65431'),
                (65432, 'McuStartsHighBytes'),
                (65433, '65433'),
                (65434, 'Fluorescence'),  # FilterSetName
                (65435, 'ExposureRatio'),
                (65436, 'RedMultiplier'),
                (65437, 'GreenMultiplier'),
                (65438, 'BlueMultiplier'),
                (65439, 'FocusPoints'),
                (65440, 'FocusPointRegions'),
                (65441, 'CaptureMode'),
                (65442, 'ScannerSerialNumber'),
                (65443, '65443'),
                (65444, 'JpegQuality'),
                (65445, 'RefocusInterval'),
                (65446, 'FocusOffset'),
                (65447, 'BlankLines'),
                (65448, 'FirmwareVersion'),
                (65449, 'Comments'),  # PropertyMap, CalibrationInfo
                (65450, 'LabelObscured'),
                (65451, 'Wavelength'),
                (65452, '65452'),
                (65453, 'LampAge'),
                (65454, 'ExposureTime'),
                (65455, 'FocusTime'),
                (65456, 'ScanTime'),
                (65457, 'WriteTime'),
                (65458, 'FullyAutoFocus'),
                (65500, 'DefaultGamma'),
            )
        )

    def EXIF_TAGS():
        # 65000 - 65112  Photoshop Camera RAW EXIF tags
        tags = TiffTagRegistry(
            (
                (65000, 'OwnerName'),
                (65001, 'SerialNumber'),
                (65002, 'Lens'),
                (65100, 'RawFile'),
                (65101, 'Converter'),
                (65102, 'WhiteBalance'),
                (65105, 'Exposure'),
                (65106, 'Shadows'),
                (65107, 'Brightness'),
                (65108, 'Contrast'),
                (65109, 'Saturation'),
                (65110, 'Sharpness'),
                (65111, 'Smoothness'),
                (65112, 'MoireFilter'),
            )
        )
        tags.update(TIFF.TAGS)
        return tags

    def GPS_TAGS():
        return TiffTagRegistry(
            (
                (0, 'GPSVersionID'),
                (1, 'GPSLatitudeRef'),
                (2, 'GPSLatitude'),
                (3, 'GPSLongitudeRef'),
                (4, 'GPSLongitude'),
                (5, 'GPSAltitudeRef'),
                (6, 'GPSAltitude'),
                (7, 'GPSTimeStamp'),
                (8, 'GPSSatellites'),
                (9, 'GPSStatus'),
                (10, 'GPSMeasureMode'),
                (11, 'GPSDOP'),
                (12, 'GPSSpeedRef'),
                (13, 'GPSSpeed'),
                (14, 'GPSTrackRef'),
                (15, 'GPSTrack'),
                (16, 'GPSImgDirectionRef'),
                (17, 'GPSImgDirection'),
                (18, 'GPSMapDatum'),
                (19, 'GPSDestLatitudeRef'),
                (20, 'GPSDestLatitude'),
                (21, 'GPSDestLongitudeRef'),
                (22, 'GPSDestLongitude'),
                (23, 'GPSDestBearingRef'),
                (24, 'GPSDestBearing'),
                (25, 'GPSDestDistanceRef'),
                (26, 'GPSDestDistance'),
                (27, 'GPSProcessingMethod'),
                (28, 'GPSAreaInformation'),
                (29, 'GPSDateStamp'),
                (30, 'GPSDifferential'),
                (31, 'GPSHPositioningError'),
            )
        )

    def IOP_TAGS():
        return TiffTagRegistry(
            (
                (1, 'InteroperabilityIndex'),
                (2, 'InteroperabilityVersion'),
                (4096, 'RelatedImageFileFormat'),
                (4097, 'RelatedImageWidth'),
                (4098, 'RelatedImageLength'),
            )
        )

    def GEO_KEYS():
        try:
            from .tifffile_geodb import GeoKeys  # delayed import
        except ImportError:
            try:
                from tifffile_geodb import GeoKeys  # delayed import
            except ImportError:

                class GeoKeys(enum.IntEnum):
                    pass

        return GeoKeys

    def GEO_CODES():
        try:
            from .tifffile_geodb import GEO_CODES  # delayed import
        except ImportError:
            try:
                from tifffile_geodb import GEO_CODES  # delayed import
            except ImportError:
                GEO_CODES = {}
        return GEO_CODES

    def CZ_LSMINFO():
        return [
            ('MagicNumber', 'u4'),
            ('StructureSize', 'i4'),
            ('DimensionX', 'i4'),
            ('DimensionY', 'i4'),
            ('DimensionZ', 'i4'),
            ('DimensionChannels', 'i4'),
            ('DimensionTime', 'i4'),
            ('DataType', 'i4'),  # DATATYPES
            ('ThumbnailX', 'i4'),
            ('ThumbnailY', 'i4'),
            ('VoxelSizeX', 'f8'),
            ('VoxelSizeY', 'f8'),
            ('VoxelSizeZ', 'f8'),
            ('OriginX', 'f8'),
            ('OriginY', 'f8'),
            ('OriginZ', 'f8'),
            ('ScanType', 'u2'),
            ('SpectralScan', 'u2'),
            ('TypeOfData', 'u4'),  # TYPEOFDATA
            ('OffsetVectorOverlay', 'u4'),
            ('OffsetInputLut', 'u4'),
            ('OffsetOutputLut', 'u4'),
            ('OffsetChannelColors', 'u4'),
            ('TimeIntervall', 'f8'),
            ('OffsetChannelDataTypes', 'u4'),
            ('OffsetScanInformation', 'u4'),  # SCANINFO
            ('OffsetKsData', 'u4'),
            ('OffsetTimeStamps', 'u4'),
            ('OffsetEventList', 'u4'),
            ('OffsetRoi', 'u4'),
            ('OffsetBleachRoi', 'u4'),
            ('OffsetNextRecording', 'u4'),
            # LSM 2.0 ends here
            ('DisplayAspectX', 'f8'),
            ('DisplayAspectY', 'f8'),
            ('DisplayAspectZ', 'f8'),
            ('DisplayAspectTime', 'f8'),
            ('OffsetMeanOfRoisOverlay', 'u4'),
            ('OffsetTopoIsolineOverlay', 'u4'),
            ('OffsetTopoProfileOverlay', 'u4'),
            ('OffsetLinescanOverlay', 'u4'),
            ('ToolbarFlags', 'u4'),
            ('OffsetChannelWavelength', 'u4'),
            ('OffsetChannelFactors', 'u4'),
            ('ObjectiveSphereCorrection', 'f8'),
            ('OffsetUnmixParameters', 'u4'),
            # LSM 3.2, 4.0 end here
            ('OffsetAcquisitionParameters', 'u4'),
            ('OffsetCharacteristics', 'u4'),
            ('OffsetPalette', 'u4'),
            ('TimeDifferenceX', 'f8'),
            ('TimeDifferenceY', 'f8'),
            ('TimeDifferenceZ', 'f8'),
            ('InternalUse1', 'u4'),
            ('DimensionP', 'i4'),
            ('DimensionM', 'i4'),
            ('DimensionsReserved', '16i4'),
            ('OffsetTilePositions', 'u4'),
            ('', '9u4'),  # Reserved
            ('OffsetPositions', 'u4'),
            # ('', '21u4'),  # must be 0
        ]

    def CZ_LSMINFO_READERS():
        # import functions for CZ_LSMINFO sub-records
        # TODO: read more CZ_LSMINFO sub-records
        return {
            'ScanInformation': read_lsm_scaninfo,
            'TimeStamps': read_lsm_timestamps,
            'EventList': read_lsm_eventlist,
            'ChannelColors': read_lsm_channelcolors,
            'Positions': read_lsm_positions,
            'TilePositions': read_lsm_positions,
            'VectorOverlay': None,
            'InputLut': read_lsm_lookuptable,
            'OutputLut': read_lsm_lookuptable,
            'TimeIntervall': None,
            'ChannelDataTypes': read_lsm_channeldatatypes,
            'KsData': None,
            'Roi': None,
            'BleachRoi': None,
            'NextRecording': None,  # read with TiffFile(fh, offset=)
            'MeanOfRoisOverlay': None,
            'TopoIsolineOverlay': None,
            'TopoProfileOverlay': None,
            'ChannelWavelength': read_lsm_channelwavelength,
            'SphereCorrection': None,
            'ChannelFactors': None,
            'UnmixParameters': None,
            'AcquisitionParameters': None,
            'Characteristics': None,
        }

    def CZ_LSMINFO_SCANTYPE():
        # map CZ_LSMINFO.ScanType to dimension order
        return {
            0: 'XYZCT',  # 'Stack' normal x-y-z-scan
            1: 'XYZCT',  # 'Z-Scan' x-z-plane Y=1
            2: 'XYZCT',  # 'Line'
            3: 'XYTCZ',  # 'Time Series Plane' time series x-y  XYCTZ ? Z=1
            4: 'XYZTC',  # 'Time Series z-Scan' time series x-z
            5: 'XYTCZ',  # 'Time Series Mean-of-ROIs'
            6: 'XYZTC',  # 'Time Series Stack' time series x-y-z
            7: 'XYCTZ',  # Spline Scan
            8: 'XYCZT',  # Spline Plane x-z
            9: 'XYTCZ',  # Time Series Spline Plane x-z
            10: 'XYZCT',  # 'Time Series Point' point mode
        }

    def CZ_LSMINFO_DIMENSIONS():
        # map dimension codes to CZ_LSMINFO attribute
        return {
            'X': 'DimensionX',
            'Y': 'DimensionY',
            'Z': 'DimensionZ',
            'C': 'DimensionChannels',
            'T': 'DimensionTime',
            'P': 'DimensionP',
            'M': 'DimensionM',
        }

    def CZ_LSMINFO_DATATYPES():
        # description of CZ_LSMINFO.DataType
        return {
            0: 'varying data types',
            1: '8 bit unsigned integer',
            2: '12 bit unsigned integer',
            5: '32 bit float',
        }

    def CZ_LSMINFO_TYPEOFDATA():
        # description of CZ_LSMINFO.TypeOfData
        return {
            0: 'Original scan data',
            1: 'Calculated data',
            2: '3D reconstruction',
            3: 'Topography height map',
        }

    def CZ_LSMINFO_SCANINFO_ARRAYS():
        return {
            0x20000000: 'Tracks',
            0x30000000: 'Lasers',
            0x60000000: 'DetectionChannels',
            0x80000000: 'IlluminationChannels',
            0xA0000000: 'BeamSplitters',
            0xC0000000: 'DataChannels',
            0x11000000: 'Timers',
            0x13000000: 'Markers',
        }

    def CZ_LSMINFO_SCANINFO_STRUCTS():
        return {
            # 0x10000000: 'Recording',
            0x40000000: 'Track',
            0x50000000: 'Laser',
            0x70000000: 'DetectionChannel',
            0x90000000: 'IlluminationChannel',
            0xB0000000: 'BeamSplitter',
            0xD0000000: 'DataChannel',
            0x12000000: 'Timer',
            0x14000000: 'Marker',
        }

    def CZ_LSMINFO_SCANINFO_ATTRIBUTES():
        return {
            # Recording
            0x10000001: 'Name',
            0x10000002: 'Description',
            0x10000003: 'Notes',
            0x10000004: 'Objective',
            0x10000005: 'ProcessingSummary',
            0x10000006: 'SpecialScanMode',
            0x10000007: 'ScanType',
            0x10000008: 'ScanMode',
            0x10000009: 'NumberOfStacks',
            0x1000000A: 'LinesPerPlane',
            0x1000000B: 'SamplesPerLine',
            0x1000000C: 'PlanesPerVolume',
            0x1000000D: 'ImagesWidth',
            0x1000000E: 'ImagesHeight',
            0x1000000F: 'ImagesNumberPlanes',
            0x10000010: 'ImagesNumberStacks',
            0x10000011: 'ImagesNumberChannels',
            0x10000012: 'LinscanXySize',
            0x10000013: 'ScanDirection',
            0x10000014: 'TimeSeries',
            0x10000015: 'OriginalScanData',
            0x10000016: 'ZoomX',
            0x10000017: 'ZoomY',
            0x10000018: 'ZoomZ',
            0x10000019: 'Sample0X',
            0x1000001A: 'Sample0Y',
            0x1000001B: 'Sample0Z',
            0x1000001C: 'SampleSpacing',
            0x1000001D: 'LineSpacing',
            0x1000001E: 'PlaneSpacing',
            0x1000001F: 'PlaneWidth',
            0x10000020: 'PlaneHeight',
            0x10000021: 'VolumeDepth',
            0x10000023: 'Nutation',
            0x10000034: 'Rotation',
            0x10000035: 'Precession',
            0x10000036: 'Sample0time',
            0x10000037: 'StartScanTriggerIn',
            0x10000038: 'StartScanTriggerOut',
            0x10000039: 'StartScanEvent',
            0x10000040: 'StartScanTime',
            0x10000041: 'StopScanTriggerIn',
            0x10000042: 'StopScanTriggerOut',
            0x10000043: 'StopScanEvent',
            0x10000044: 'StopScanTime',
            0x10000045: 'UseRois',
            0x10000046: 'UseReducedMemoryRois',
            0x10000047: 'User',
            0x10000048: 'UseBcCorrection',
            0x10000049: 'PositionBcCorrection1',
            0x10000050: 'PositionBcCorrection2',
            0x10000051: 'InterpolationY',
            0x10000052: 'CameraBinning',
            0x10000053: 'CameraSupersampling',
            0x10000054: 'CameraFrameWidth',
            0x10000055: 'CameraFrameHeight',
            0x10000056: 'CameraOffsetX',
            0x10000057: 'CameraOffsetY',
            0x10000059: 'RtBinning',
            0x1000005A: 'RtFrameWidth',
            0x1000005B: 'RtFrameHeight',
            0x1000005C: 'RtRegionWidth',
            0x1000005D: 'RtRegionHeight',
            0x1000005E: 'RtOffsetX',
            0x1000005F: 'RtOffsetY',
            0x10000060: 'RtZoom',
            0x10000061: 'RtLinePeriod',
            0x10000062: 'Prescan',
            0x10000063: 'ScanDirectionZ',
            # Track
            0x40000001: 'MultiplexType',  # 0 After Line; 1 After Frame
            0x40000002: 'MultiplexOrder',
            0x40000003: 'SamplingMode',  # 0 Sample; 1 Line Avg; 2 Frame Avg
            0x40000004: 'SamplingMethod',  # 1 Mean; 2 Sum
            0x40000005: 'SamplingNumber',
            0x40000006: 'Acquire',
            0x40000007: 'SampleObservationTime',
            0x4000000B: 'TimeBetweenStacks',
            0x4000000C: 'Name',
            0x4000000D: 'Collimator1Name',
            0x4000000E: 'Collimator1Position',
            0x4000000F: 'Collimator2Name',
            0x40000010: 'Collimator2Position',
            0x40000011: 'IsBleachTrack',
            0x40000012: 'IsBleachAfterScanNumber',
            0x40000013: 'BleachScanNumber',
            0x40000014: 'TriggerIn',
            0x40000015: 'TriggerOut',
            0x40000016: 'IsRatioTrack',
            0x40000017: 'BleachCount',
            0x40000018: 'SpiCenterWavelength',
            0x40000019: 'PixelTime',
            0x40000021: 'CondensorFrontlens',
            0x40000023: 'FieldStopValue',
            0x40000024: 'IdCondensorAperture',
            0x40000025: 'CondensorAperture',
            0x40000026: 'IdCondensorRevolver',
            0x40000027: 'CondensorFilter',
            0x40000028: 'IdTransmissionFilter1',
            0x40000029: 'IdTransmission1',
            0x40000030: 'IdTransmissionFilter2',
            0x40000031: 'IdTransmission2',
            0x40000032: 'RepeatBleach',
            0x40000033: 'EnableSpotBleachPos',
            0x40000034: 'SpotBleachPosx',
            0x40000035: 'SpotBleachPosy',
            0x40000036: 'SpotBleachPosz',
            0x40000037: 'IdTubelens',
            0x40000038: 'IdTubelensPosition',
            0x40000039: 'TransmittedLight',
            0x4000003A: 'ReflectedLight',
            0x4000003B: 'SimultanGrabAndBleach',
            0x4000003C: 'BleachPixelTime',
            # Laser
            0x50000001: 'Name',
            0x50000002: 'Acquire',
            0x50000003: 'Power',
            # DetectionChannel
            0x70000001: 'IntegrationMode',
            0x70000002: 'SpecialMode',
            0x70000003: 'DetectorGainFirst',
            0x70000004: 'DetectorGainLast',
            0x70000005: 'AmplifierGainFirst',
            0x70000006: 'AmplifierGainLast',
            0x70000007: 'AmplifierOffsFirst',
            0x70000008: 'AmplifierOffsLast',
            0x70000009: 'PinholeDiameter',
            0x7000000A: 'CountingTrigger',
            0x7000000B: 'Acquire',
            0x7000000C: 'PointDetectorName',
            0x7000000D: 'AmplifierName',
            0x7000000E: 'PinholeName',
            0x7000000F: 'FilterSetName',
            0x70000010: 'FilterName',
            0x70000013: 'IntegratorName',
            0x70000014: 'ChannelName',
            0x70000015: 'DetectorGainBc1',
            0x70000016: 'DetectorGainBc2',
            0x70000017: 'AmplifierGainBc1',
            0x70000018: 'AmplifierGainBc2',
            0x70000019: 'AmplifierOffsetBc1',
            0x70000020: 'AmplifierOffsetBc2',
            0x70000021: 'SpectralScanChannels',
            0x70000022: 'SpiWavelengthStart',
            0x70000023: 'SpiWavelengthStop',
            0x70000026: 'DyeName',
            0x70000027: 'DyeFolder',
            # IlluminationChannel
            0x90000001: 'Name',
            0x90000002: 'Power',
            0x90000003: 'Wavelength',
            0x90000004: 'Aquire',
            0x90000005: 'DetchannelName',
            0x90000006: 'PowerBc1',
            0x90000007: 'PowerBc2',
            # BeamSplitter
            0xB0000001: 'FilterSet',
            0xB0000002: 'Filter',
            0xB0000003: 'Name',
            # DataChannel
            0xD0000001: 'Name',
            0xD0000003: 'Acquire',
            0xD0000004: 'Color',
            0xD0000005: 'SampleType',
            0xD0000006: 'BitsPerSample',
            0xD0000007: 'RatioType',
            0xD0000008: 'RatioTrack1',
            0xD0000009: 'RatioTrack2',
            0xD000000A: 'RatioChannel1',
            0xD000000B: 'RatioChannel2',
            0xD000000C: 'RatioConst1',
            0xD000000D: 'RatioConst2',
            0xD000000E: 'RatioConst3',
            0xD000000F: 'RatioConst4',
            0xD0000010: 'RatioConst5',
            0xD0000011: 'RatioConst6',
            0xD0000012: 'RatioFirstImages1',
            0xD0000013: 'RatioFirstImages2',
            0xD0000014: 'DyeName',
            0xD0000015: 'DyeFolder',
            0xD0000016: 'Spectrum',
            0xD0000017: 'Acquire',
            # Timer
            0x12000001: 'Name',
            0x12000002: 'Description',
            0x12000003: 'Interval',
            0x12000004: 'TriggerIn',
            0x12000005: 'TriggerOut',
            0x12000006: 'ActivationTime',
            0x12000007: 'ActivationNumber',
            # Marker
            0x14000001: 'Name',
            0x14000002: 'Description',
            0x14000003: 'TriggerIn',
            0x14000004: 'TriggerOut',
        }

    def CZ_LSM_LUTTYPE():
        class CZ_LSM_LUTTYPE(enum.IntEnum):
            NORMAL = 0
            ORIGINAL = 1
            RAMP = 2
            POLYLINE = 3
            SPLINE = 4
            GAMMA = 5

        return CZ_LSM_LUTTYPE

    def CZ_LSM_SUBBLOCK_TYPE():
        class CZ_LSM_SUBBLOCK_TYPE(enum.IntEnum):
            END = 0
            GAMMA = 1
            BRIGHTNESS = 2
            CONTRAST = 3
            RAMP = 4
            KNOTS = 5
            PALETTE_12_TO_12 = 6

        return CZ_LSM_SUBBLOCK_TYPE

    def NIH_IMAGE_HEADER():
        return [
            ('FileID', 'a8'),
            ('nLines', 'i2'),
            ('PixelsPerLine', 'i2'),
            ('Version', 'i2'),
            ('OldLutMode', 'i2'),
            ('OldnColors', 'i2'),
            ('Colors', 'u1', (3, 32)),
            ('OldColorStart', 'i2'),
            ('ColorWidth', 'i2'),
            ('ExtraColors', 'u2', (6, 3)),
            ('nExtraColors', 'i2'),
            ('ForegroundIndex', 'i2'),
            ('BackgroundIndex', 'i2'),
            ('XScale', 'f8'),
            ('Unused2', 'i2'),
            ('Unused3', 'i2'),
            ('UnitsID', 'i2'),  # NIH_UNITS_TYPE
            ('p1', [('x', 'i2'), ('y', 'i2')]),
            ('p2', [('x', 'i2'), ('y', 'i2')]),
            ('CurveFitType', 'i2'),  # NIH_CURVEFIT_TYPE
            ('nCoefficients', 'i2'),
            ('Coeff', 'f8', 6),
            ('UMsize', 'u1'),
            ('UM', 'a15'),
            ('UnusedBoolean', 'u1'),
            ('BinaryPic', 'b1'),
            ('SliceStart', 'i2'),
            ('SliceEnd', 'i2'),
            ('ScaleMagnification', 'f4'),
            ('nSlices', 'i2'),
            ('SliceSpacing', 'f4'),
            ('CurrentSlice', 'i2'),
            ('FrameInterval', 'f4'),
            ('PixelAspectRatio', 'f4'),
            ('ColorStart', 'i2'),
            ('ColorEnd', 'i2'),
            ('nColors', 'i2'),
            ('Fill1', '3u2'),
            ('Fill2', '3u2'),
            ('Table', 'u1'),  # NIH_COLORTABLE_TYPE
            ('LutMode', 'u1'),  # NIH_LUTMODE_TYPE
            ('InvertedTable', 'b1'),
            ('ZeroClip', 'b1'),
            ('XUnitSize', 'u1'),
            ('XUnit', 'a11'),
            ('StackType', 'i2'),  # NIH_STACKTYPE_TYPE
            # ('UnusedBytes', 'u1', 200)
        ]

    def NIH_COLORTABLE_TYPE():
        return (
            'CustomTable',
            'AppleDefault',
            'Pseudo20',
            'Pseudo32',
            'Rainbow',
            'Fire1',
            'Fire2',
            'Ice',
            'Grays',
            'Spectrum',
        )

    def NIH_LUTMODE_TYPE():
        return (
            'PseudoColor',
            'OldAppleDefault',
            'OldSpectrum',
            'GrayScale',
            'ColorLut',
            'CustomGrayscale',
        )

    def NIH_CURVEFIT_TYPE():
        return (
            'StraightLine',
            'Poly2',
            'Poly3',
            'Poly4',
            'Poly5',
            'ExpoFit',
            'PowerFit',
            'LogFit',
            'RodbardFit',
            'SpareFit1',
            'Uncalibrated',
            'UncalibratedOD',
        )

    def NIH_UNITS_TYPE():
        return (
            'Nanometers',
            'Micrometers',
            'Millimeters',
            'Centimeters',
            'Meters',
            'Kilometers',
            'Inches',
            'Feet',
            'Miles',
            'Pixels',
            'OtherUnits',
        )

    def TVIPS_HEADER_V1():
        # TVIPS TemData structure from EMMENU Help file
        return [
            ('Version', 'i4'),
            ('CommentV1', 'a80'),
            ('HighTension', 'i4'),
            ('SphericalAberration', 'i4'),
            ('IlluminationAperture', 'i4'),
            ('Magnification', 'i4'),
            ('PostMagnification', 'i4'),
            ('FocalLength', 'i4'),
            ('Defocus', 'i4'),
            ('Astigmatism', 'i4'),
            ('AstigmatismDirection', 'i4'),
            ('BiprismVoltage', 'i4'),
            ('SpecimenTiltAngle', 'i4'),
            ('SpecimenTiltDirection', 'i4'),
            ('IlluminationTiltDirection', 'i4'),
            ('IlluminationTiltAngle', 'i4'),
            ('ImageMode', 'i4'),
            ('EnergySpread', 'i4'),
            ('ChromaticAberration', 'i4'),
            ('ShutterType', 'i4'),
            ('DefocusSpread', 'i4'),
            ('CcdNumber', 'i4'),
            ('CcdSize', 'i4'),
            ('OffsetXV1', 'i4'),
            ('OffsetYV1', 'i4'),
            ('PhysicalPixelSize', 'i4'),
            ('Binning', 'i4'),
            ('ReadoutSpeed', 'i4'),
            ('GainV1', 'i4'),
            ('SensitivityV1', 'i4'),
            ('ExposureTimeV1', 'i4'),
            ('FlatCorrected', 'i4'),
            ('DeadPxCorrected', 'i4'),
            ('ImageMean', 'i4'),
            ('ImageStd', 'i4'),
            ('DisplacementX', 'i4'),
            ('DisplacementY', 'i4'),
            ('DateV1', 'i4'),
            ('TimeV1', 'i4'),
            ('ImageMin', 'i4'),
            ('ImageMax', 'i4'),
            ('ImageStatisticsQuality', 'i4'),
        ]

    def TVIPS_HEADER_V2():
        return [
            ('ImageName', 'V160'),  # utf16
            ('ImageFolder', 'V160'),
            ('ImageSizeX', 'i4'),
            ('ImageSizeY', 'i4'),
            ('ImageSizeZ', 'i4'),
            ('ImageSizeE', 'i4'),
            ('ImageDataType', 'i4'),
            ('Date', 'i4'),
            ('Time', 'i4'),
            ('Comment', 'V1024'),
            ('ImageHistory', 'V1024'),
            ('Scaling', '16f4'),
            ('ImageStatistics', '16c16'),
            ('ImageType', 'i4'),
            ('ImageDisplaType', 'i4'),
            ('PixelSizeX', 'f4'),  # distance between two px in x, [nm]
            ('PixelSizeY', 'f4'),  # distance between two px in y, [nm]
            ('ImageDistanceZ', 'f4'),
            ('ImageDistanceE', 'f4'),
            ('ImageMisc', '32f4'),
            ('TemType', 'V160'),
            ('TemHighTension', 'f4'),
            ('TemAberrations', '32f4'),
            ('TemEnergy', '32f4'),
            ('TemMode', 'i4'),
            ('TemMagnification', 'f4'),
            ('TemMagnificationCorrection', 'f4'),
            ('PostMagnification', 'f4'),
            ('TemStageType', 'i4'),
            ('TemStagePosition', '5f4'),  # x, y, z, a, b
            ('TemImageShift', '2f4'),
            ('TemBeamShift', '2f4'),
            ('TemBeamTilt', '2f4'),
            ('TilingParameters', '7f4'),  # 0: tiling? 1:x 2:y 3: max x
            #                               4: max y 5: overlap x 6: overlap y
            ('TemIllumination', '3f4'),  # 0: spotsize 1: intensity
            ('TemShutter', 'i4'),
            ('TemMisc', '32f4'),
            ('CameraType', 'V160'),
            ('PhysicalPixelSizeX', 'f4'),
            ('PhysicalPixelSizeY', 'f4'),
            ('OffsetX', 'i4'),
            ('OffsetY', 'i4'),
            ('BinningX', 'i4'),
            ('BinningY', 'i4'),
            ('ExposureTime', 'f4'),
            ('Gain', 'f4'),
            ('ReadoutRate', 'f4'),
            ('FlatfieldDescription', 'V160'),
            ('Sensitivity', 'f4'),
            ('Dose', 'f4'),
            ('CamMisc', '32f4'),
            ('FeiMicroscopeInformation', 'V1024'),
            ('FeiSpecimenInformation', 'V1024'),
            ('Magic', 'u4'),
        ]

    def MM_HEADER():
        # Olympus FluoView MM_Header
        MM_DIMENSION = [
            ('Name', 'a16'),
            ('Size', 'i4'),
            ('Origin', 'f8'),
            ('Resolution', 'f8'),
            ('Unit', 'a64'),
        ]
        return [
            ('HeaderFlag', 'i2'),
            ('ImageType', 'u1'),
            ('ImageName', 'a257'),
            ('OffsetData', 'u4'),
            ('PaletteSize', 'i4'),
            ('OffsetPalette0', 'u4'),
            ('OffsetPalette1', 'u4'),
            ('CommentSize', 'i4'),
            ('OffsetComment', 'u4'),
            ('Dimensions', MM_DIMENSION, 10),
            ('OffsetPosition', 'u4'),
            ('MapType', 'i2'),
            ('MapMin', 'f8'),
            ('MapMax', 'f8'),
            ('MinValue', 'f8'),
            ('MaxValue', 'f8'),
            ('OffsetMap', 'u4'),
            ('Gamma', 'f8'),
            ('Offset', 'f8'),
            ('GrayChannel', MM_DIMENSION),
            ('OffsetThumbnail', 'u4'),
            ('VoiceField', 'i4'),
            ('OffsetVoiceField', 'u4'),
        ]

    def MM_DIMENSIONS():
        # map FluoView MM_Header.Dimensions to axes characters
        return {
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'T': 'T',
            'CH': 'C',
            'WAVELENGTH': 'C',
            'TIME': 'T',
            'XY': 'R',
            'EVENT': 'V',
            'EXPOSURE': 'L',
        }

    def UIC_TAGS():
        # map Universal Imaging Corporation MetaMorph internal tag ids to
        # name and type
        from fractions import Fraction  # delayed import

        return [
            ('AutoScale', int),
            ('MinScale', int),
            ('MaxScale', int),
            ('SpatialCalibration', int),
            ('XCalibration', Fraction),
            ('YCalibration', Fraction),
            ('CalibrationUnits', str),
            ('Name', str),
            ('ThreshState', int),
            ('ThreshStateRed', int),
            ('tagid_10', None),  # undefined
            ('ThreshStateGreen', int),
            ('ThreshStateBlue', int),
            ('ThreshStateLo', int),
            ('ThreshStateHi', int),
            ('Zoom', int),
            ('CreateTime', julian_datetime),
            ('LastSavedTime', julian_datetime),
            ('currentBuffer', int),
            ('grayFit', None),
            ('grayPointCount', None),
            ('grayX', Fraction),
            ('grayY', Fraction),
            ('grayMin', Fraction),
            ('grayMax', Fraction),
            ('grayUnitName', str),
            ('StandardLUT', int),
            ('wavelength', int),
            ('StagePosition', '(%i,2,2)u4'),  # N xy positions as fract
            ('CameraChipOffset', '(%i,2,2)u4'),  # N xy offsets as fract
            ('OverlayMask', None),
            ('OverlayCompress', None),
            ('Overlay', None),
            ('SpecialOverlayMask', None),
            ('SpecialOverlayCompress', None),
            ('SpecialOverlay', None),
            ('ImageProperty', read_uic_image_property),
            ('StageLabel', '%ip'),  # N str
            ('AutoScaleLoInfo', Fraction),
            ('AutoScaleHiInfo', Fraction),
            ('AbsoluteZ', '(%i,2)u4'),  # N fractions
            ('AbsoluteZValid', '(%i,)u4'),  # N long
            ('Gamma', 'I'),  # 'I' uses offset
            ('GammaRed', 'I'),
            ('GammaGreen', 'I'),
            ('GammaBlue', 'I'),
            ('CameraBin', '2I'),
            ('NewLUT', int),
            ('ImagePropertyEx', None),
            ('PlaneProperty', int),
            ('UserLutTable', '(256,3)u1'),
            ('RedAutoScaleInfo', int),
            ('RedAutoScaleLoInfo', Fraction),
            ('RedAutoScaleHiInfo', Fraction),
            ('RedMinScaleInfo', int),
            ('RedMaxScaleInfo', int),
            ('GreenAutoScaleInfo', int),
            ('GreenAutoScaleLoInfo', Fraction),
            ('GreenAutoScaleHiInfo', Fraction),
            ('GreenMinScaleInfo', int),
            ('GreenMaxScaleInfo', int),
            ('BlueAutoScaleInfo', int),
            ('BlueAutoScaleLoInfo', Fraction),
            ('BlueAutoScaleHiInfo', Fraction),
            ('BlueMinScaleInfo', int),
            ('BlueMaxScaleInfo', int),
            # ('OverlayPlaneColor', read_uic_overlay_plane_color),
        ]

    def PILATUS_HEADER():
        # PILATUS CBF Header Specification, Version 1.4
        # map key to [value_indices], type
        return {
            'Detector': ([slice(1, None)], str),
            'Pixel_size': ([1, 4], float),
            'Silicon': ([3], float),
            'Exposure_time': ([1], float),
            'Exposure_period': ([1], float),
            'Tau': ([1], float),
            'Count_cutoff': ([1], int),
            'Threshold_setting': ([1], float),
            'Gain_setting': ([1, 2], str),
            'N_excluded_pixels': ([1], int),
            'Excluded_pixels': ([1], str),
            'Flat_field': ([1], str),
            'Trim_file': ([1], str),
            'Image_path': ([1], str),
            # optional
            'Wavelength': ([1], float),
            'Energy_range': ([1, 2], float),
            'Detector_distance': ([1], float),
            'Detector_Voffset': ([1], float),
            'Beam_xy': ([1, 2], float),
            'Flux': ([1], str),
            'Filter_transmission': ([1], float),
            'Start_angle': ([1], float),
            'Angle_increment': ([1], float),
            'Detector_2theta': ([1], float),
            'Polarization': ([1], float),
            'Alpha': ([1], float),
            'Kappa': ([1], float),
            'Phi': ([1], float),
            'Phi_increment': ([1], float),
            'Chi': ([1], float),
            'Chi_increment': ([1], float),
            'Oscillation_axis': ([slice(1, None)], str),
            'N_oscillations': ([1], int),
            'Start_position': ([1], float),
            'Position_increment': ([1], float),
            'Shutter_time': ([1], float),
            'Omega': ([1], float),
            'Omega_increment': ([1], float),
        }

    def ALLOCATIONGRANULARITY():
        # alignment for writing contiguous data to TIFF
        import mmap  # delayed import

        return mmap.ALLOCATIONGRANULARITY

    def MAXWORKERS():
        # half of CPU cores
        import multiprocessing  # delayed import

        return max(multiprocessing.cpu_count() // 2, 1)

    def CHUNKMODE():
        class CHUNKMODE(enum.IntEnum):
            NONE = 0
            PLANE = 1
            PAGE = 2
            FILE = 3

        return CHUNKMODE


def read_tags(
    fh, byteorder, offsetsize, tagnames, customtags=None, maxifds=None
):
    """Read tags from chain of IFDs and return as list of dicts.

    The file handle position must be at a valid IFD header.
    Does not work with NDPI.

    """
    if offsetsize == 4:
        offsetformat = byteorder + 'I'
        tagnosize = 2
        tagnoformat = byteorder + 'H'
        tagsize = 12
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'I4s'
    elif offsetsize == 8:
        offsetformat = byteorder + 'Q'
        tagnosize = 8
        tagnoformat = byteorder + 'Q'
        tagsize = 20
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'Q8s'
    else:
        raise ValueError('invalid offset size')

    if customtags is None:
        customtags = {}
    if maxifds is None:
        maxifds = 2 ** 32

    result = []
    unpack = struct.unpack
    offset = fh.tell()
    while len(result) < maxifds:
        # loop over IFDs
        try:
            tagno = unpack(tagnoformat, fh.read(tagnosize))[0]
            if tagno > 4096:
                raise TiffFileError(f'suspicious number of tags {tagno}')
        except Exception as exc:
            log_warning(
                f'<tifffile.read_tags> corrupted tag list @{offset} ({exc})'
            )
            break

        tags = {}
        data = fh.read(tagsize * tagno)
        pos = fh.tell()
        index = 0

        for _ in range(tagno):
            code, dtype = unpack(tagformat1, data[index : index + 4])
            count, value = unpack(
                tagformat2, data[index + 4 : index + tagsize]
            )
            index += tagsize
            name = tagnames.get(code, str(code))
            try:
                valueformat = TIFF.DATA_FORMATS[dtype]
            except KeyError:
                raise TiffFileError(
                    f'invalid data type {dtype!r} for tag #{code}'
                )

            valuesize = count * struct.calcsize(valueformat)
            if valuesize > offsetsize or code in customtags:
                valueoffset = unpack(offsetformat, value)[0]
                if valueoffset < 8 or valueoffset + valuesize > fh.size:
                    raise TiffFileError(
                        f'invalid value offset {valueoffset} for tag #{code}'
                    )
                fh.seek(valueoffset)
                if code in customtags:
                    readfunc = customtags[code][1]
                    value = readfunc(fh, byteorder, dtype, count, offsetsize)
                elif dtype == 1 or dtype == 2 or dtype == 7:
                    # BYTES, ASCII, UNDEFINED
                    value = fh.read(valuesize)
                    if len(value) != valuesize:
                        log_warning(
                            '<tifffile.read_tags> '
                            f'could not read all values for tag #{code}'
                        )
                elif code in tagnames:
                    fmt = '{}{}{}'.format(
                        byteorder, count * int(valueformat[0]), valueformat[1]
                    )
                    value = unpack(fmt, fh.read(valuesize))
                else:
                    value = read_numpy(fh, byteorder, dtype, count, offsetsize)
            elif dtype == 1 or dtype == 2 or dtype == 7:
                # BYTES, ASCII, UNDEFINED
                value = value[:valuesize]
            else:
                fmt = '{}{}{}'.format(
                    byteorder, count * int(valueformat[0]), valueformat[1]
                )
                value = unpack(fmt, value[:valuesize])

            process = (
                code not in customtags
                and code not in TIFF.TAG_TUPLE
                and dtype != 7  # UNDEFINED
            )
            if process and dtype == 2:
                # TIFF ASCII fields can contain multiple strings,
                #   each terminated with a NUL
                try:
                    value = bytes2str(stripnull(value, first=False).strip())
                except UnicodeDecodeError:
                    log_warning(
                        '<tifffile.read_tags> '
                        f'coercing invalid ASCII to bytes for tag #{code}'
                    )
            else:
                if code in TIFF.TAG_ENUM:
                    t = TIFF.TAG_ENUM[code]
                    try:
                        value = tuple(t(v) for v in value)
                    except ValueError as exc:
                        if code not in (259, 317):
                            # ignore compression/predictor
                            log_warning(
                                '<tifffile.read_tags> '
                                f'failed for tag #{code}: {exc}'
                            )
                if process and len(value) == 1:
                    value = value[0]
            tags[name] = value

        result.append(tags)

        # read offset to next page
        fh.seek(pos)
        offset = unpack(offsetformat, fh.read(offsetsize))[0]
        if offset == 0:
            break
        if offset >= fh.size:
            log_warning(
                f'<tifffile.read_tags> invalid next page offset {offset}'
            )
            break
        fh.seek(offset)

    if result and maxifds == 1:
        result = result[0]
    return result


def read_exif_ifd(fh, byteorder, dtype, count, offsetsize):
    """Read EXIF tags from file and return as dict."""
    exif = read_tags(fh, byteorder, offsetsize, TIFF.EXIF_TAGS, maxifds=1)
    for name in ('ExifVersion', 'FlashpixVersion'):
        try:
            exif[name] = bytes2str(exif[name])
        except Exception:
            pass
    if 'UserComment' in exif:
        idcode = exif['UserComment'][:8]
        try:
            if idcode == b'ASCII\x00\x00\x00':
                exif['UserComment'] = bytes2str(exif['UserComment'][8:])
            elif idcode == b'UNICODE\x00':
                exif['UserComment'] = exif['UserComment'][8:].decode('utf-16')
        except Exception:
            pass
    return exif


def read_gps_ifd(fh, byteorder, dtype, count, offsetsize):
    """Read GPS tags from file and return as dict."""
    return read_tags(fh, byteorder, offsetsize, TIFF.GPS_TAGS, maxifds=1)


def read_interoperability_ifd(fh, byteorder, dtype, count, offsetsize):
    """Read Interoperability tags from file and return as dict."""
    return read_tags(fh, byteorder, offsetsize, TIFF.IOP_TAGS, maxifds=1)


def read_bytes(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as bytes."""
    dtype = 'B' if dtype == 2 else byteorder + TIFF.DATA_FORMATS[dtype][-1]
    count *= numpy.dtype(dtype).itemsize
    data = fh.read(count)
    if len(data) != count:
        log_warning(
            '<tifffile.read_bytes> '
            f'failed to read {count} bytes, got {len(data)})'
        )
    return data


def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as Unicode string."""
    return fh.read(count).decode()


def read_numpy(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as numpy array."""
    dtype = 'b' if dtype == 2 else byteorder + TIFF.DATA_FORMATS[dtype][-1]
    return fh.read_array(dtype, count)


def read_colormap(fh, byteorder, dtype, count, offsetsize):
    """Read ColorMap/TransferFunction from file and return as numpy array."""
    cmap = fh.read_array(byteorder + TIFF.DATA_FORMATS[dtype][-1], count)
    if count % 3 == 0:
        cmap.shape = (3, -1)
    return cmap


def read_json(fh, byteorder, dtype, count, offsetsize):
    """Read JSON tag data from file and return as object."""
    data = fh.read(count)
    try:
        return json.loads(stripnull(data).decode())
    except ValueError as exc:
        log_warning(f'<tifffile.read_json> {exc.__class__.__name__}: {exc}')
    return None


def read_mm_header(fh, byteorder, dtype, count, offsetsize):
    """Read FluoView mm_header tag from file and return as dict."""
    mmh = fh.read_record(TIFF.MM_HEADER, byteorder=byteorder)
    mmh = recarray2dict(mmh)
    mmh['Dimensions'] = [
        (bytes2str(d[0]).strip(), d[1], d[2], d[3], bytes2str(d[4]).strip())
        for d in mmh['Dimensions']
    ]
    d = mmh['GrayChannel']
    mmh['GrayChannel'] = (
        bytes2str(d[0]).strip(),
        d[1],
        d[2],
        d[3],
        bytes2str(d[4]).strip(),
    )
    return mmh


def read_mm_stamp(fh, byteorder, dtype, count, offsetsize):
    """Read FluoView mm_stamp tag from file and return as numpy.ndarray."""
    return fh.read_array(byteorder + 'f8', 8)


def read_uic1tag(fh, byteorder, dtype, count, offsetsize, planecount=None):
    """Read MetaMorph STK UIC1Tag from file and return as dict.

    Return empty dictionary if planecount is unknown.

    """
    if dtype not in (4, 5) or byteorder != '<':
        raise ValueError(f'invalid UIC1Tag {byteorder}{dtype}')
    result = {}
    if dtype == 5:
        # pre MetaMorph 2.5 (not tested)
        values = fh.read_array('<u4', 2 * count).reshape(count, 2)
        result = {'ZDistance': values[:, 0] / values[:, 1]}
    else:
        for _ in range(count):
            tagid = struct.unpack('<I', fh.read(4))[0]
            if tagid in (28, 29, 37, 40, 41):
                # silently skip unexpected tags
                fh.read(4)
                continue
            name, value = read_uic_tag(fh, tagid, planecount, offset=True)
            result[name] = value
    return result


def read_uic2tag(fh, byteorder, dtype, planecount, offsetsize):
    """Read MetaMorph STK UIC2Tag from file and return as dict."""
    if dtype != 5 or byteorder != '<':
        raise ValueError('invalid UIC2Tag')
    values = fh.read_array('<u4', 6 * planecount).reshape(planecount, 6)
    return {
        'ZDistance': values[:, 0] / values[:, 1],
        'DateCreated': values[:, 2],  # julian days
        'TimeCreated': values[:, 3],  # milliseconds
        'DateModified': values[:, 4],  # julian days
        'TimeModified': values[:, 5],  # milliseconds
    }


def read_uic3tag(fh, byteorder, dtype, planecount, offsetsize):
    """Read MetaMorph STK UIC3Tag from file and return as dict."""
    if dtype != 5 or byteorder != '<':
        raise ValueError('invalid UIC3Tag')
    values = fh.read_array('<u4', 2 * planecount).reshape(planecount, 2)
    return {'Wavelengths': values[:, 0] / values[:, 1]}


def read_uic4tag(fh, byteorder, dtype, planecount, offsetsize):
    """Read MetaMorph STK UIC4Tag from file and return as dict."""
    if dtype != 4 or byteorder != '<':
        raise ValueError('invalid UIC4Tag')
    result = {}
    while True:
        tagid = struct.unpack('<H', fh.read(2))[0]
        if tagid == 0:
            break
        name, value = read_uic_tag(fh, tagid, planecount, offset=False)
        result[name] = value
    return result


def read_uic_tag(fh, tagid, planecount, offset):
    """Read a single UIC tag value from file and return tag name and value.

    UIC1Tags use an offset.

    """

    def read_int(count=1):
        value = struct.unpack(f'<{count}I', fh.read(4 * count))
        return value[0] if count == 1 else value

    try:
        name, dtype = TIFF.UIC_TAGS[tagid]
    except IndexError:
        # unknown tag
        return f'_TagId{tagid}', read_int()

    Fraction = TIFF.UIC_TAGS[4][1]

    if offset:
        pos = fh.tell()
        if dtype not in (int, None):
            off = read_int()
            if off < 8:
                if dtype is str:
                    return name, ''
                log_warning(
                    '<tifffile.read_uic_tag> '
                    f'invalid offset for tag {name!r} @{off}'
                )
                return name, off
            fh.seek(off)

    if dtype is None:
        # skip
        name = '_' + name
        value = read_int()
    elif dtype is int:
        # int
        value = read_int()
    elif dtype is Fraction:
        # fraction
        value = read_int(2)
        value = value[0] / value[1]
    elif dtype is julian_datetime:
        # datetime
        value = julian_datetime(*read_int(2))
    elif dtype is read_uic_image_property:
        # ImagePropertyEx
        value = read_uic_image_property(fh)
    elif dtype is str:
        # pascal string
        size = read_int()
        if 0 <= size < 2 ** 10:
            value = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
            value = bytes2str(stripnull(value))
        elif offset:
            value = ''
            log_warning(
                f'<tifffile.read_uic_tag> invalid string in tag {name!r}'
            )
        else:
            raise ValueError(f'invalid string size {size}')
    elif planecount is None:
        value = None
    elif dtype == '%ip':
        # sequence of pascal strings
        value = []
        for _ in range(planecount):
            size = read_int()
            if 0 <= size < 2 ** 10:
                string = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
                string = bytes2str(stripnull(string))
                value.append(string)
            elif offset:
                log_warning(
                    f'<tifffile.read_uic_tag> invalid string in tag {name!r}'
                )
            else:
                raise ValueError(f'invalid string size: {size}')
    else:
        # struct or numpy type
        dtype = '<' + dtype
        if '%i' in dtype:
            dtype = dtype % planecount
        if '(' in dtype:
            # numpy type
            value = fh.read_array(dtype, 1)[0]
            if value.shape[-1] == 2:
                # assume fractions
                value = value[..., 0] / value[..., 1]
        else:
            # struct format
            value = struct.unpack(dtype, fh.read(struct.calcsize(dtype)))
            if len(value) == 1:
                value = value[0]

    if offset:
        fh.seek(pos + 4)

    return name, value


def read_uic_image_property(fh):
    """Read UIC ImagePropertyEx tag from file and return as dict."""
    # TODO: test this
    size = struct.unpack('B', fh.read(1))[0]
    name = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
    flags, prop = struct.unpack('<IB', fh.read(5))
    if prop == 1:
        value = struct.unpack('II', fh.read(8))
        value = value[0] / value[1]
    else:
        size = struct.unpack('B', fh.read(1))[0]
        value = struct.unpack(f'{size}s', fh.read(size))[0]
    return dict(name=name, flags=flags, value=value)


def read_cz_lsminfo(fh, byteorder, dtype, count, offsetsize):
    """Read CZ_LSMINFO tag from file and return as dict."""
    if byteorder != '<':
        raise ValueError('invalid CZ_LSMINFO structure')
    magic_number, structure_size = struct.unpack('<II', fh.read(8))
    if magic_number not in (50350412, 67127628):
        raise ValueError('invalid CZ_LSMINFO structure')
    fh.seek(-8, os.SEEK_CUR)

    if structure_size < numpy.dtype(TIFF.CZ_LSMINFO).itemsize:
        # adjust structure according to structure_size
        lsminfo = []
        size = 0
        for name, dtype in TIFF.CZ_LSMINFO:
            size += numpy.dtype(dtype).itemsize
            if size > structure_size:
                break
            lsminfo.append((name, dtype))
    else:
        lsminfo = TIFF.CZ_LSMINFO

    lsminfo = fh.read_record(lsminfo, byteorder=byteorder)
    lsminfo = recarray2dict(lsminfo)

    # read LSM info subrecords at offsets
    for name, reader in TIFF.CZ_LSMINFO_READERS.items():
        if reader is None:
            continue
        offset = lsminfo.get('Offset' + name, 0)
        if offset < 8:
            continue
        fh.seek(offset)
        try:
            lsminfo[name] = reader(fh)
        except ValueError:
            pass
    return lsminfo


def read_lsm_channeldatatypes(fh):
    """Read LSM channel data type."""
    size = struct.unpack('<I', fh.read(4))[0]
    return fh.read_array('<u4', count=size)


def read_lsm_channelwavelength(fh):
    """Read LSM channel wavelength ranges from file and return as list."""
    size = struct.unpack('<i', fh.read(4))[0]
    return fh.read_array('<2f8', count=size)


def read_lsm_positions(fh):
    """Read LSM positions from file and return as list."""
    size = struct.unpack('<I', fh.read(4))[0]
    return fh.read_array('<3f8', count=size)


def read_lsm_timestamps(fh):
    """Read LSM time stamps from file and return as list."""
    size, count = struct.unpack('<ii', fh.read(8))
    if size != (8 + 8 * count):
        log_warning(
            '<tifffile.read_lsm_timestamps> invalid LSM TimeStamps block'
        )
        return []
    # return struct.unpack(f'<{count}d', fh.read(8 * count))
    return fh.read_array('<f8', count=count)


def read_lsm_eventlist(fh):
    """Read LSM events from file and return as list of (time, type, text)."""
    count = struct.unpack('<II', fh.read(8))[1]
    events = []
    while count > 0:
        esize, etime, etype = struct.unpack('<IdI', fh.read(16))
        etext = bytes2str(stripnull(fh.read(esize - 16)))
        events.append((etime, etype, etext))
        count -= 1
    return events


def read_lsm_channelcolors(fh):
    """Read LSM ChannelColors structure from file and return as dict."""
    result = {'Mono': False, 'Colors': [], 'ColorNames': []}
    pos = fh.tell()
    (size, ncolors, nnames, coffset, noffset, mono) = struct.unpack(
        '<IIIIII', fh.read(24)
    )
    if ncolors != nnames:
        log_warning(
            '<tifffile.read_lsm_channelcolors> '
            'invalid LSM ChannelColors structure'
        )
        return result
    result['Mono'] = bool(mono)
    # Colors
    fh.seek(pos + coffset)
    colors = fh.read_array('uint8', count=ncolors * 4).reshape((ncolors, 4))
    result['Colors'] = colors.tolist()
    # ColorNames
    fh.seek(pos + noffset)
    buffer = fh.read(size - noffset)
    names = []
    while len(buffer) > 4:
        size = struct.unpack('<I', buffer[:4])[0]
        names.append(bytes2str(buffer[4 : 3 + size]))
        buffer = buffer[4 + size :]
    result['ColorNames'] = names
    return result


def read_lsm_lookuptable(fh):
    """Read LSM lookup tables from file and return as dict."""
    result = {}
    (
        size,
        nsubblocks,
        nchannels,
        luttype,
        advanced,
        currentchannel,
    ) = struct.unpack('<iiiiii', fh.read(24))
    if size < 60:
        log_warning(
            '<tifffile.read_lsm_lookuptable> '
            'invalid LSM LookupTables structure'
        )
        return result
    fh.read(9 * 4)  # reserved
    result['LutType'] = TIFF.CZ_LSM_LUTTYPE(luttype)
    result['Advanced'] = advanced
    result['NumberChannels'] = nchannels
    result['CurrentChannel'] = currentchannel
    result['SubBlocks'] = subblocks = []
    for i in range(nsubblocks):
        sbtype = struct.unpack('<i', fh.read(4))[0]
        if sbtype <= 0:
            break
        size = struct.unpack('<i', fh.read(4))[0] - 8
        if sbtype == 1:
            data = fh.read_array('<f8', count=nchannels)
        elif sbtype == 2:
            data = fh.read_array('<f8', count=nchannels)
        elif sbtype == 3:
            data = fh.read_array('<f8', count=nchannels)
        elif sbtype == 4:
            # the data type is wrongly documented as f8
            data = fh.read_array('<i4', count=nchannels * 4)
            data = data.reshape((-1, 2, 2))
        elif sbtype == 5:
            # the data type is wrongly documented as f8
            nknots = struct.unpack('<i', fh.read(4))[0]  # undocumented
            data = fh.read_array('<i4', count=nchannels * nknots * 2)
            data = data.reshape((nchannels, nknots, 2))
        elif sbtype == 6:
            data = fh.read_array('<i2', count=nchannels * 4096)
            data = data.reshape((-1, 4096))
        else:
            log_warning(
                '<tifffile.read_lsm_lookuptable> '
                f'invalid LSM SubBlock type {sbtype}'
            )
            break
        subblocks.append(
            {'Type': TIFF.CZ_LSM_SUBBLOCK_TYPE(sbtype), 'Data': data}
        )
    return result


def read_lsm_scaninfo(fh):
    """Read LSM ScanInfo structure from file and return as dict."""
    block = {}
    blocks = [block]
    unpack = struct.unpack
    if struct.unpack('<I', fh.read(4))[0] != 0x10000000:
        # not a Recording sub block
        log_warning(
            '<tifffile.read_lsm_scaninfo> invalid LSM ScanInfo structure'
        )
        return block
    fh.read(8)
    while True:
        entry, dtype, size = unpack('<III', fh.read(12))
        if dtype == 2:
            # ascii
            value = bytes2str(stripnull(fh.read(size)))
        elif dtype == 4:
            # long
            value = unpack('<i', fh.read(4))[0]
        elif dtype == 5:
            # rational
            value = unpack('<d', fh.read(8))[0]
        else:
            value = 0
        if entry in TIFF.CZ_LSMINFO_SCANINFO_ARRAYS:
            blocks.append(block)
            name = TIFF.CZ_LSMINFO_SCANINFO_ARRAYS[entry]
            newobj = []
            block[name] = newobj
            block = newobj
        elif entry in TIFF.CZ_LSMINFO_SCANINFO_STRUCTS:
            blocks.append(block)
            newobj = {}
            block.append(newobj)
            block = newobj
        elif entry in TIFF.CZ_LSMINFO_SCANINFO_ATTRIBUTES:
            name = TIFF.CZ_LSMINFO_SCANINFO_ATTRIBUTES[entry]
            block[name] = value
        elif entry == 0xFFFFFFFF:
            # end sub block
            block = blocks.pop()
        else:
            # unknown entry
            block[f'Entry0x{entry:x}'] = value
        if not blocks:
            break
    return block


def read_sis(fh, byteorder, dtype, count, offsetsize):
    """Read OlympusSIS structure and return as dict.

    No specification is avaliable. Only few fields are known.

    """
    result = {}

    (magic, minute, hour, day, month, year, name, tagcount) = struct.unpack(
        '<4s6xhhhhh6x32sh', fh.read(60)
    )

    if magic != b'SIS0':
        raise ValueError('invalid OlympusSIS structure')

    result['name'] = bytes2str(stripnull(name))
    try:
        result['datetime'] = datetime.datetime(
            1900 + year, month + 1, day, hour, minute
        )
    except ValueError:
        pass

    data = fh.read(8 * tagcount)
    for i in range(0, tagcount * 8, 8):
        tagtype, count, offset = struct.unpack('<hhI', data[i : i + 8])
        fh.seek(offset)
        if tagtype == 1:
            # general data
            (lenexp, xcal, ycal, mag, camname, pictype) = struct.unpack(
                '<10xhdd8xd2x34s32s', fh.read(112)  # 220
            )
            m = math.pow(10, lenexp)
            result['pixelsizex'] = xcal * m
            result['pixelsizey'] = ycal * m
            result['magnification'] = mag
            result['cameraname'] = bytes2str(stripnull(camname))
            result['picturetype'] = bytes2str(stripnull(pictype))
        elif tagtype == 10:
            # channel data
            continue
            # TODO: does not seem to work?
            # (length, _, exptime, emv, _, camname, _, mictype,
            #  ) = struct.unpack('<h22sId4s32s48s32s', fh.read(152))  # 720
            # result['exposuretime'] = exptime
            # result['emvoltage'] = emv
            # result['cameraname2'] = bytes2str(stripnull(camname))
            # result['microscopename'] = bytes2str(stripnull(mictype))

    return result


def read_sis_ini(fh, byteorder, dtype, count, offsetsize):
    """Read OlympusSIS INI string and return as dict."""
    inistr = fh.read(count)
    inistr = bytes2str(stripnull(inistr))
    try:
        return olympusini_metadata(inistr)
    except Exception as exc:
        log_warning(
            f'<tifffile.olympusini_metadata> {exc.__class__.__name__}: {exc}'
        )
        return {}


def read_tvips_header(fh, byteorder, dtype, count, offsetsize):
    """Read TVIPS EM-MENU headers and return as dict."""
    result = {}
    header = fh.read_record(TIFF.TVIPS_HEADER_V1, byteorder=byteorder)
    for name, typestr in TIFF.TVIPS_HEADER_V1:
        result[name] = header[name].tolist()
    if header['Version'] == 2:
        header = fh.read_record(TIFF.TVIPS_HEADER_V2, byteorder=byteorder)
        if header['Magic'] != int(0xAAAAAAAA):
            log_warning(
                '<tifffile.read_tvips_header> invalid TVIPS v2 magic number'
            )
            return {}
        # decode utf16 strings
        for name, typestr in TIFF.TVIPS_HEADER_V2:
            if typestr.startswith('V'):
                s = header[name].tobytes().decode('utf-16', errors='ignore')
                result[name] = stripnull(s, null='\x00')
            else:
                result[name] = header[name].tolist()
        # convert nm to m
        for axis in 'XY':
            header['PhysicalPixelSize' + axis] /= 1e9
            header['PixelSize' + axis] /= 1e9
    elif header.version != 1:
        log_warning(
            '<tifffile.read_tvips_header> unknown TVIPS header version'
        )
        return {}
    return result


def read_fei_metadata(fh, byteorder, dtype, count, offsetsize):
    """Read FEI SFEG/HELIOS headers and return as dict."""
    result = {}
    section = {}
    data = bytes2str(stripnull(fh.read(count)))
    for line in data.splitlines():
        line = line.strip()
        if line.startswith('['):
            section = {}
            result[line[1:-1]] = section
            continue
        try:
            key, value = line.split('=')
        except ValueError:
            continue
        section[key] = astype(value)
    return result


def read_cz_sem(fh, byteorder, dtype, count, offsetsize):
    """Read Zeiss SEM tag and return as dict.

    See https://sourceforge.net/p/gwyddion/mailman/message/29275000/ for
    unnamed values.

    """
    result = {'': ()}
    key = None
    data = bytes2str(stripnull(fh.read(count)))
    for line in data.splitlines():
        if line.isupper():
            key = line.lower()
        elif key:
            try:
                name, value = line.split('=')
            except ValueError:
                try:
                    name, value = line.split(':', 1)
                except Exception:
                    continue
            value = value.strip()
            unit = ''
            try:
                v, u = value.split()
                number = astype(v, (int, float))
                if number != v:
                    value = number
                    unit = u
            except Exception:
                number = astype(value, (int, float))
                if number != value:
                    value = number
                if value in ('No', 'Off'):
                    value = False
                elif value in ('Yes', 'On'):
                    value = True
            result[key] = (name.strip(), value)
            if unit:
                result[key] += (unit,)
            key = None
        else:
            result[''] += (astype(line, (int, float)),)
    return result


def read_nih_image_header(fh, byteorder, dtype, count, offsetsize):
    """Read NIH_IMAGE_HEADER tag from file and return as dict."""
    a = fh.read_record(TIFF.NIH_IMAGE_HEADER, byteorder=byteorder)
    a = a.newbyteorder(byteorder)
    a = recarray2dict(a)
    a['XUnit'] = a['XUnit'][: a['XUnitSize']]
    a['UM'] = a['UM'][: a['UMsize']]
    return a


def read_scanimage_metadata(fh):
    """Read ScanImage BigTIFF v3 or v4 static and ROI metadata from open file.

    Return non-varying frame data, ROI group data, and version as
    tuple(dict, dict, int).

    The settings can be used to read image data and metadata without parsing
    the TIFF file.

    Raise ValueError if file does not contain valid ScanImage metadata.

    Frame data and ROI groups can alternatively be obtained from the Software
    and Artist tags of any TIFF page.

    """
    fh.seek(0)
    try:
        byteorder, version = struct.unpack('<2sH', fh.read(4))
        if byteorder != b'II' or version != 43:
            raise ValueError('not a BigTIFF file')
        fh.seek(16)
        magic, version, size0, size1 = struct.unpack('<IIII', fh.read(16))
        if magic != 117637889 or version not in (3, 4):
            raise ValueError(
                f'invalid magic {magic} or version {version} number'
            )
    except UnicodeDecodeError as exc:
        raise ValueError('file must be opened in binary mode') from exc
    except Exception as exc:
        raise ValueError('not a ScanImage BigTIFF v3 or v4 file') from exc

    frame_data = matlabstr2py(bytes2str(fh.read(size0)[:-1]))
    roi_data = read_json(fh, '<', None, size1, None) if size1 > 1 else {}
    return frame_data, roi_data, version


def read_micromanager_metadata(fh):
    """Read MicroManager non-TIFF settings from open file and return as dict.

    The settings can be used to read image data without parsing the TIFF file.

    """
    fh.seek(0)
    try:
        byteorder = {b'II': '<', b'MM': '>'}[fh.read(2)]
    except IndexError:
        raise ValueError('not a MicroManager TIFF file')

    result = {}
    fh.seek(8)
    (
        index_header,  # Index map
        index_offset,
        display_header,
        display_offset,
        comments_header,
        comments_offset,
        summary_header,
        summary_length,
    ) = struct.unpack(byteorder + 'IIIIIIII', fh.read(32))

    if summary_header == 2355492:
        result['Summary'] = read_json(
            fh, byteorder, None, summary_length, None
        )
    else:
        log_warning(
            '<tifffile.read_micromanager_metadata> '
            'invalid MicroManager summary header'
        )

    if index_header == 54773648:
        fh.seek(index_offset)
        header, count = struct.unpack(byteorder + 'II', fh.read(8))
        if header == 3453623:
            data = struct.unpack(
                byteorder + 'IIIII' * count, fh.read(20 * count)
            )
            result['IndexMap'] = {
                'Channel': data[::5],
                'Slice': data[1::5],
                'Frame': data[2::5],
                'Position': data[3::5],
                'Offset': data[4::5],
            }
        else:
            log_warning(
                '<tifffile.read_micromanager_metadata> '
                'invalid MicroManager index header'
            )
    else:
        log_warning(
            '<tifffile.read_micromanager_metadata> '
            'invalid MicroManager index header'
        )

    if display_header == 483765892:
        fh.seek(display_offset)
        header, count = struct.unpack(byteorder + 'II', fh.read(8))
        if header == 347834724:
            result['DisplaySettings'] = read_json(
                fh, byteorder, None, count, None
            )
        else:
            log_warning(
                '<tifffile.read_micromanager_metadata> '
                'invalid MicroManager display header'
            )
    else:
        log_warning(
            '<tifffile.read_micromanager_metadata> '
            'invalid MicroManager display header'
        )

    result['MajorVersion'] = 0
    if comments_header == 99384722:
        # Micro-Manager multipage TIFF
        fh.seek(comments_offset)
        header, count = struct.unpack(byteorder + 'II', fh.read(8))
        if header == 84720485:
            result['Comments'] = read_json(fh, byteorder, None, count, None)
        else:
            log_warning(
                '<tifffile.read_micromanager_metadata> '
                'invalid MicroManager comments header'
            )
    elif comments_header == 483729:
        # NDTiffStorage
        result['MajorVersion'] = comments_offset
    else:
        log_warning(
            '<tifffile.read_micromanager_metadata> '
            'invalid MicroManager comments header'
        )

    return result


def read_metaseries_catalog(fh):
    """Read MetaSeries non-TIFF hint catalog from file.

    Raise ValueError if the file does not contain a valid hint catalog.

    """
    # TODO: implement read_metaseries_catalog
    raise NotImplementedError


def imagej_metadata_tag(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to TiffWriter.write() as extratags.

    The metadata dict may contain the following keys and values:

        Info : str
            Human-readable information as string.
        Labels : sequence of str
            Human-readable labels for each channel.
        Ranges : sequence of doubles
            Lower and upper values for each channel.
        LUTs : sequence of (3, 256) uint8 ndarrays
            Color palettes for each channel.
        Plot : bytes
            Undocumented ImageJ internal format.
        ROI: bytes
            Undocumented ImageJ internal region of interest format.
        Overlays : bytes
            Undocumented ImageJ internal format.
        Properties : {str: str}
            Map of key, value items as strings.

    """
    if not metadata:
        return ()
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def _string(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def _doubles(data, byteorder):
        return struct.pack(byteorder + ('d' * len(data)), *data)

    def _ndarray(data, byteorder):
        return data.tobytes()

    def _bytes(data, byteorder):
        return data

    metadata_types = (
        ('Info', b'info', _string),
        ('Labels', b'labl', _string),
        ('Ranges', b'rang', _doubles),
        ('LUTs', b'luts', _ndarray),
        ('Plot', b'plot', _bytes),
        ('ROI', b'roi ', _bytes),
        ('Overlays', b'over', _bytes),
        ('Properties', b'prop', _string),
    )

    for key, mtype, func in metadata_types:
        if key.lower() in metadata:
            key = key.lower()
        elif key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if isinstance(values, dict):
            values = [str(i) for item in values.items() for i in item]
            count = len(values)
        elif isinstance(values, list):
            count = len(values)
        else:
            values = [values]
            count = 1
        header.append(mtype + struct.pack(byteorder + 'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    if not body:
        return ()
    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder + ('I' * len(bytecounts)), *bytecounts)
    return (
        (50839, 1, len(data), data, True),
        (50838, 4, len(bytecounts) // 4, bytecounts, True),
    )


def imagej_metadata(data, bytecounts, byteorder):
    """Return IJMetadata tag value as dict.

    The 'Info' string can have multiple formats, e.g. OIF or ScanImage,
    that might be parsed into dicts using the matlabstr2py or
    oiffile.SettingsFile functions.
    'ROI' and 'Overlays' are returned as bytes, which can be parsed with the
    ImagejRoi.frombytes() function of the roifile package.

    """

    def _string(data, byteorder):
        return data.decode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def _doubles(data, byteorder):
        return struct.unpack(byteorder + ('d' * (len(data) // 8)), data)

    def _lut(data, byteorder):
        return numpy.frombuffer(data, 'uint8').reshape(-1, 256)

    def _bytes(data, byteorder):
        return data

    # big-endian
    metadata_types = {
        b'info': ('Info', _string),
        b'labl': ('Labels', _string),
        b'rang': ('Ranges', _doubles),
        b'luts': ('LUTs', _lut),
        b'plot': ('Plot', _bytes),
        b'roi ': ('ROI', _bytes),
        b'over': ('Overlays', _bytes),
        b'prop': ('Properties', _string),
    }
    # little-endian
    metadata_types.update({k[::-1]: v for k, v in metadata_types.items()})

    if not bytecounts:
        raise ValueError('no ImageJ metadata')

    if not data[:4] in (b'IJIJ', b'JIJI'):
        raise ValueError('invalid ImageJ metadata')

    header_size = bytecounts[0]
    if header_size < 12 or header_size > 804:
        raise ValueError('invalid ImageJ metadata header size')

    ntypes = (header_size - 4) // 8
    header = struct.unpack(
        byteorder + '4sI' * ntypes, data[4 : 4 + ntypes * 8]
    )
    pos = 4 + ntypes * 8
    counter = 0
    result = {}
    for mtype, count in zip(header[::2], header[1::2]):
        values = []
        name, func = metadata_types.get(mtype, (bytes2str(mtype), _bytes))
        for _ in range(count):
            counter += 1
            pos1 = pos + bytecounts[counter]
            values.append(func(data[pos:pos1], byteorder))
            pos = pos1
        result[name.strip()] = values[0] if count == 1 else values
    prop = result.get('Properties')
    if prop and len(prop) % 2 == 0:
        result['Properties'] = dict(
            prop[i : i + 2] for i in range(0, len(prop), 2)
        )
    return result


def imagej_description_metadata(description):
    r"""Return metatata from ImageJ image description as dict.

    Raise ValueError if not a valid ImageJ description.

    >>> description = 'ImageJ=1.11a\nimages=510\nhyperstack=true\n'
    >>> imagej_description_metadata(description)  # doctest: +SKIP
    {'ImageJ': '1.11a', 'images': 510, 'hyperstack': True}

    """

    def _bool(val):
        return {'true': True, 'false': False}[val.lower()]

    result = {}
    for line in description.splitlines():
        try:
            key, val = line.split('=')
        except Exception:
            continue
        key = key.strip()
        val = val.strip()
        for dtype in (int, float, _bool):
            try:
                val = dtype(val)
                break
            except Exception:
                pass
        result[key] = val

    if 'ImageJ' not in result:
        raise ValueError('not an ImageJ image description')
    return result


def imagej_description(
    shape,
    rgb=None,
    colormaped=False,
    version=None,
    hyperstack=None,
    mode=None,
    loop=None,
    **kwargs,
):
    """Return ImageJ image description from data shape.

    ImageJ can handle up to 6 dimensions in order TZCYXS.

    >>> imagej_description((51, 5, 2, 196, 171))  # doctest: +SKIP
    ImageJ=1.11a
    images=510
    channels=2
    slices=5
    frames=51
    hyperstack=true
    mode=grayscale
    loop=false

    """
    if colormaped:
        hyperstack = False
        rgb = False
    if version is None:
        version = kwargs.pop('ImageJ', '1.11a')
    axes = kwargs.pop('axes', None)
    shape = imagej_shape(shape, rgb=rgb, axes=axes)
    rgb = shape[-1] in (3, 4)

    append = []
    result = [f'ImageJ={version}']
    result.append(f'images={product(shape[:-3])}')
    if hyperstack is None:
        hyperstack = True
        append.append('hyperstack=true')
    else:
        append.append(f'hyperstack={bool(hyperstack)}')
    if shape[2] > 1:
        result.append(f'channels={shape[2]}')
    if mode is None and not rgb and not colormaped:
        mode = 'grayscale'
    if hyperstack and mode:
        append.append(f'mode={mode}')
    if shape[1] > 1:
        result.append(f'slices={shape[1]}')
    if shape[0] > 1:
        result.append(f'frames={shape[0]}')
        if loop is None:
            append.append('loop=false')
    if loop is not None:
        append.append(f'loop={bool(loop)}'.lower())

    for key, value in kwargs.items():
        if key not in ('images', 'channels', 'slices', 'frames'):
            append.append(f'{key.lower()}={value}')

    return '\n'.join(result + append + [''])


def imagej_shape(shape, rgb=None, axes=None):
    """Return shape normalized to 6D ImageJ hyperstack TZCYXS.

    Raise ValueError if not a valid ImageJ hyperstack shape or axes order.

    >>> imagej_shape((2, 3, 4, 5, 3), False)
    (2, 3, 4, 5, 3, 1)

    """
    shape = tuple(int(i) for i in shape)
    ndim = len(shape)
    if 1 > ndim > 6:
        raise ValueError('ImageJ hyperstack must be 2-6 dimensional')

    if axes:
        if len(axes) != ndim:
            raise ValueError('ImageJ hyperstack shape and axes do not match')
        i = 0
        axes = axes.upper()
        for ax in axes:
            j = 'TZCYXS'.find(ax)
            if j < i:
                raise ValueError(
                    'ImageJ hyperstack axes must be in TZCYXS order'
                )
            i = j
        ndims = len(axes)
        newshape = []
        i = 0
        for ax in 'TZCYXS':
            if i < ndims and ax == axes[i]:
                newshape.append(shape[i])
                i += 1
            else:
                newshape.append(1)
        if newshape[-1] not in (1, 3, 4):
            raise ValueError(
                'ImageJ hyperstack must contain 1, 3, or 4 samples'
            )
        return tuple(newshape)

    if rgb is None:
        rgb = shape[-1] in (3, 4) and ndim > 2
    if rgb and shape[-1] not in (3, 4):
        raise ValueError('ImageJ hyperstack is not a RGB image')
    if not rgb and ndim == 6 and shape[-1] != 1:
        raise ValueError('ImageJ hyperstack is not a grayscale image')
    if rgb or shape[-1] == 1:
        return (1,) * (6 - ndim) + shape
    return (1,) * (5 - ndim) + shape + (1,)


def jpeg_decode_colorspace(photometric, planarconfig, extrasamples):
    """Return JPEG and output colorspace for jpeg_decode function."""
    colorspace = None
    outcolorspace = None
    if extrasamples:
        pass
    elif photometric == 6:
        # YCBCR -> RGB
        outcolorspace = 2  # RGB
    elif photometric == 2:
        if planarconfig == 1:
            colorspace = outcolorspace = 2  # RGB
    elif photometric == 5:
        # CMYK
        outcolorspace = 4
    elif photometric > 3:
        outcolorspace = TIFF.PHOTOMETRIC(photometric).name
    return colorspace, outcolorspace


def jpeg_shape(jpeg):
    """Return bitdepth and shape of JPEG image."""
    i = 0
    while True and i < len(jpeg):
        marker = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if marker == 0xFFD8:
            # start of image
            continue
        if marker == 0xFFD9:
            # end of image
            break
        if 0xFFD0 <= marker <= 0xFFD7:
            # restart marker
            continue
        if marker == 0xFF01:
            # private marker
            continue

        length = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if 0xFFC0 <= marker <= 0xFFC3:
            # start of frame
            return struct.unpack('>BHHB', jpeg[i : i + 6])
        if marker == 0xFFDA:
            # start of scan
            break

        # skip to next marker
        i += length - 2

    raise ValueError('no SOF marker found')


def ndpi_jpeg_tile(jpeg):
    """Return tile shape and JPEG header from JPEG with restart markers."""
    restartinterval = 0
    sofoffset = 0
    sosoffset = 0
    i = 0
    while True and i < len(jpeg):
        marker = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if marker == 0xFFD8:
            # start of image
            continue
        if marker == 0xFFD9:
            # end of image
            break
        if 0xFFD0 <= marker <= 0xFFD7:
            # restart marker
            continue
        if marker == 0xFF01:
            # private marker
            continue

        length = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if marker == 0xFFDD:
            # define restart interval
            restartinterval = struct.unpack('>H', jpeg[i : i + 2])[0]

        elif marker == 0xFFC0:
            # start of frame
            sofoffset = i + 1
            precision, imlength, imwidth, ncomponents = struct.unpack(
                '>BHHB', jpeg[i : i + 6]
            )
            i += 6
            mcuwidth = 1
            mcuheight = 1
            for _ in range(ncomponents):
                cid, factor, table = struct.unpack('>BBB', jpeg[i : i + 3])
                i += 3
                if factor >> 4 > mcuwidth:
                    mcuwidth = factor >> 4
                if factor & 0b00001111 > mcuheight:
                    mcuheight = factor & 0b00001111
            mcuwidth *= 8
            mcuheight *= 8
            i = sofoffset - 1

        elif marker == 0xFFDA:
            # start of scan
            sosoffset = i + length - 2
            break

        # skip to next marker
        i += length - 2

    if restartinterval == 0 or sofoffset == 0 or sosoffset == 0:
        raise ValueError('missing required JPEG markers')

    # patch jpeg header for tile size
    tilelength = mcuheight
    tilewidth = restartinterval * mcuwidth
    jpegheader = (
        jpeg[:sofoffset]
        + struct.pack('>HH', tilelength, tilewidth)
        + jpeg[sofoffset + 4 : sosoffset]
    )
    return tilelength, tilewidth, jpegheader


def json_description(shape, **metadata):
    """Return JSON image description from data shape and other metadata.

    Return UTF-8 encoded JSON.

    >>> json_description((256, 256, 3), axes='YXS')  # doctest: +SKIP
    b'{"shape": [256, 256, 3], "axes": "YXS"}'

    """
    metadata.update(shape=shape)
    return json.dumps(metadata)  # .encode()


def json_description_metadata(description):
    """Return metatata from JSON formated image description as dict.

    Raise ValuError if description is of unknown format.

    >>> description = '{"shape": [256, 256, 3], "axes": "YXS"}'
    >>> json_description_metadata(description)  # doctest: +SKIP
    {'shape': [256, 256, 3], 'axes': 'YXS'}
    >>> json_description_metadata('shape=(256, 256, 3)')
    {'shape': (256, 256, 3)}

    """
    if description[:6] == 'shape=':
        # old-style 'shaped' description; not JSON
        shape = tuple(int(i) for i in description[7:-1].split(','))
        return dict(shape=shape)
    if description[:1] == '{' and description[-1:] == '}':
        # JSON description
        return json.loads(description)
    raise ValueError('invalid JSON image description', description)


def fluoview_description_metadata(description, ignoresections=None):
    r"""Return metatata from FluoView image description as dict.

    The FluoView image description format is unspecified. Expect failures.

    >>> descr = ('[Intensity Mapping]\nMap Ch0: Range=00000 to 02047\n'
    ...          '[Intensity Mapping End]')
    >>> fluoview_description_metadata(descr)
    {'Intensity Mapping': {'Map Ch0: Range': '00000 to 02047'}}

    """
    if not description.startswith('['):
        raise ValueError('invalid FluoView image description')
    if ignoresections is None:
        ignoresections = {'Region Info (Fields)', 'Protocol Description'}

    result = {}
    sections = [result]
    comment = False
    for line in description.splitlines():
        if not comment:
            line = line.strip()
        if not line:
            continue
        if line[0] == '[':
            if line[-5:] == ' End]':
                # close section
                del sections[-1]
                section = sections[-1]
                name = line[1:-5]
                if comment:
                    section[name] = '\n'.join(section[name])
                if name[:4] == 'LUT ':
                    a = numpy.array(section[name], dtype=numpy.uint8)
                    a.shape = -1, 3
                    section[name] = a
                continue
            # new section
            comment = False
            name = line[1:-1]
            if name[:4] == 'LUT ':
                section = []
            elif name in ignoresections:
                section = []
                comment = True
            else:
                section = {}
            sections.append(section)
            result[name] = section
            continue
        # add entry
        if comment:
            section.append(line)
            continue
        line = line.split('=', 1)
        if len(line) == 1:
            section[line[0].strip()] = None
            continue
        key, value = line
        if key[:4] == 'RGB ':
            section.extend(int(rgb) for rgb in value.split())
        else:
            section[key.strip()] = astype(value.strip())
    return result


def pilatus_description_metadata(description):
    """Return metatata from Pilatus image description as dict.

    Return metadata from Pilatus pixel array detectors by Dectris, created
    by camserver or TVX software.

    >>> pilatus_description_metadata('# Pixel_size 172e-6 m x 172e-6 m')
    {'Pixel_size': (0.000172, 0.000172)}

    """
    result = {}
    if not description.startswith('# '):
        return result
    for c in '#:=,()':
        description = description.replace(c, ' ')
    for line in description.split('\n'):
        if line[:2] != '  ':
            continue
        line = line.split()
        name = line[0]
        if line[0] not in TIFF.PILATUS_HEADER:
            try:
                result['DateTime'] = datetime.datetime.strptime(
                    ' '.join(line), '%Y-%m-%dT%H %M %S.%f'
                )
            except Exception:
                result[name] = ' '.join(line[1:])
            continue
        indices, dtype = TIFF.PILATUS_HEADER[line[0]]
        if isinstance(indices[0], slice):
            # assumes one slice
            values = line[indices[0]]
        else:
            values = [line[i] for i in indices]
        if dtype is float and values[0] == 'not':
            values = ['NaN']
        values = tuple(dtype(v) for v in values)
        if dtype == str:
            values = ' '.join(values)
        elif len(values) == 1:
            values = values[0]
        result[name] = values
    return result


def svs_description_metadata(description):
    """Return metatata from Aperio image description as dict.

    The Aperio image description format is unspecified. Expect failures.

    >>> svs_description_metadata('Aperio Image Library v1.0')
    {'Header': 'Aperio Image Library v1.0'}

    """
    if not description.startswith('Aperio '):
        raise ValueError('invalid Aperio image description')
    result = {}
    items = description.split('|')
    result['Header'] = items[0]
    if len(items) == 1:
        return result
    for item in items[1:]:
        key, value = item.split(' = ')
        result[key.strip()] = astype(value.strip())
    return result


def stk_description_metadata(description):
    """Return metadata from MetaMorph image description as list of dict.

    The MetaMorph image description format is unspecified. Expect failures.

    """
    description = description.strip()
    if not description:
        return []
    try:
        description = bytes2str(description)
    except UnicodeDecodeError as exc:
        log_warning(
            '<tifffile.stk_description_metadata> '
            f'{exc.__class__.__name__}: {exc}'
        )
        return []
    result = []
    for plane in description.split('\x00'):
        d = {}
        for line in plane.split('\r\n'):
            line = line.split(':', 1)
            if len(line) > 1:
                name, value = line
                d[name.strip()] = astype(value.strip())
            else:
                value = line[0].strip()
                if value:
                    if '' in d:
                        d[''].append(value)
                    else:
                        d[''] = [value]
        result.append(d)
    return result


def metaseries_description_metadata(description):
    """Return metatata from MetaSeries image description as dict."""
    if not description.startswith('<MetaData>'):
        raise ValueError('invalid MetaSeries image description')

    from xml.etree import ElementTree as etree  # delayed import

    root = etree.fromstring(description)
    types = {
        'float': float,
        'int': int,
        'bool': lambda x: asbool(x, 'on', 'off'),
    }

    def parse(root, result):
        # recursive
        for child in root:
            attrib = child.attrib
            if not attrib:
                result[child.tag] = parse(child, {})
                continue
            if 'id' in attrib:
                i = attrib['id']
                t = attrib['type']
                v = attrib['value']
                if t in types:
                    result[i] = types[t](v)
                else:
                    result[i] = v
        return result

    adict = parse(root, {})
    if 'Description' in adict:
        adict['Description'] = adict['Description'].replace('&#13;&#10;', '\n')
    return adict


def scanimage_description_metadata(description):
    """Return metatata from ScanImage image description as dict."""
    return matlabstr2py(description)


def scanimage_artist_metadata(artist):
    """Return metatata from ScanImage artist tag as dict."""
    try:
        return json.loads(artist)
    except ValueError as exc:
        log_warning(
            '<tifffile.scanimage_artist_metadata> '
            f'{exc.__class__.__name__}: {exc}'
        )
    return None


def olympusini_metadata(inistr):
    """Return OlympusSIS metadata from INI string.

    No documentation is available.

    """

    def keyindex(key):
        # split key into name and index
        index = 0
        i = len(key.rstrip('0123456789'))
        if i < len(key):
            index = int(key[i:]) - 1
            key = key[:i]
        return key, index

    result = {}
    bands = []
    zpos = None
    tpos = None
    for line in inistr.splitlines():
        line = line.strip()
        if line == '' or line[0] == ';':
            continue
        if line[0] == '[' and line[-1] == ']':
            section_name = line[1:-1]
            result[section_name] = section = {}
            if section_name == 'Dimension':
                result['axes'] = axes = []
                result['shape'] = shape = []
            elif section_name == 'ASD':
                result[section_name] = []
            elif section_name == 'Z':
                if 'Dimension' in result:
                    result[section_name]['ZPos'] = zpos = []
            elif section_name == 'Time':
                if 'Dimension' in result:
                    result[section_name]['TimePos'] = tpos = []
            elif section_name == 'Band':
                nbands = result['Dimension']['Band']
                bands = [{'LUT': []} for _ in range(nbands)]
                result[section_name] = bands
                iband = 0
        else:
            key, value = line.split('=')
            if value.strip() == '':
                value = None
            elif ',' in value:
                value = tuple(astype(v) for v in value.split(','))
            else:
                value = astype(value)

            if section_name == 'Dimension':
                section[key] = value
                axes.append(key)
                shape.append(value)
            elif section_name == 'ASD':
                if key == 'Count':
                    result['ASD'] = [{}] * value
                else:
                    key, index = keyindex(key)
                    result['ASD'][index][key] = value
            elif section_name == 'Band':
                if key[:3] == 'LUT':
                    lut = bands[iband]['LUT']
                    value = struct.pack('<I', value)
                    lut.append(
                        [ord(value[0:1]), ord(value[1:2]), ord(value[2:3])]
                    )
                else:
                    key, iband = keyindex(key)
                    bands[iband][key] = value
            elif key[:4] == 'ZPos' and zpos is not None:
                zpos.append(value)
            elif key[:7] == 'TimePos' and tpos is not None:
                tpos.append(value)
            else:
                section[key] = value

    if 'axes' in result:
        sisaxes = {'Band': 'C'}
        axes = []
        shape = []
        for i, x in zip(result['shape'], result['axes']):
            if i > 1:
                axes.append(sisaxes.get(x, x[0].upper()))
                shape.append(i)
        result['axes'] = ''.join(axes)
        result['shape'] = tuple(shape)
    try:
        result['Z']['ZPos'] = numpy.array(
            result['Z']['ZPos'][: result['Dimension']['Z']], 'float64'
        )
    except Exception:
        pass
    try:
        result['Time']['TimePos'] = numpy.array(
            result['Time']['TimePos'][: result['Dimension']['Time']], 'int32'
        )
    except Exception:
        pass
    for band in bands:
        band['LUT'] = numpy.array(band['LUT'], 'uint8')
    return result


def unpack_rgb(data, dtype=None, bitspersample=None, rescale=True):
    """Return array from bytes containing packed samples.

    Use to unpack RGB565 or RGB555 to RGB888 format.
    Works on little-endian platforms only.

    Parameters
    ----------
    data : byte str
        The data to be decoded. Samples in each pixel are stored consecutively.
        Pixels are aligned to 8, 16, or 32 bit boundaries.
    dtype : numpy.dtype
        The sample data type. The byteorder applies also to the data stream.
    bitspersample : tuple
        Number of bits for each sample in a pixel.
    rescale : bool
        Upscale samples to the number of bits in dtype.

    Returns
    -------
    numpy.ndarray
        Flattened array of unpacked samples of native dtype.

    Examples
    --------
    >>> data = struct.pack('BBBB', 0x21, 0x08, 0xff, 0xff)
    >>> print(unpack_rgb(data, '<B', (5, 6, 5), False))
    [ 1  1  1 31 63 31]
    >>> print(unpack_rgb(data, '<B', (5, 6, 5)))
    [  8   4   8 255 255 255]
    >>> print(unpack_rgb(data, '<B', (5, 5, 5)))
    [ 16   8   8 255 255 255]

    """
    if bitspersample is None:
        bitspersample = (5, 6, 5)
    if dtype is None:
        dtype = '<B'
    dtype = numpy.dtype(dtype)
    bits = int(numpy.sum(bitspersample))
    if not (
        bits <= 32 and all(i <= dtype.itemsize * 8 for i in bitspersample)
    ):
        raise ValueError(f'sample size not supported: {bitspersample}')
    dt = next(i for i in 'BHI' if numpy.dtype(i).itemsize * 8 >= bits)
    data = numpy.frombuffer(data, dtype.byteorder + dt)
    result = numpy.empty((data.size, len(bitspersample)), dtype.char)
    for i, bps in enumerate(bitspersample):
        t = data >> int(numpy.sum(bitspersample[i + 1 :]))
        t &= int('0b' + '1' * bps, 2)
        if rescale:
            o = ((dtype.itemsize * 8) // bps + 1) * bps
            if o > data.dtype.itemsize * 8:
                t = t.astype('I')
            t *= (2 ** o - 1) // (2 ** bps - 1)
            t //= 2 ** (o - (dtype.itemsize * 8))
        result[:, i] = t
    return result.reshape(-1)


def float24_decode(data, byteorder):
    """Return float32 array from float24."""
    raise NotImplementedError('float24_decode')


def zlib_encode(data, level=None, out=None):
    """Compress Zlib DEFLATE."""
    import zlib

    return zlib.compress(data, 6 if level is None else level)


def zlib_decode(data, out=None):
    """Decompress Zlib DEFLATE."""
    import zlib

    return zlib.decompress(data)


def lzma_encode(data, level=None, out=None):
    """Compress LZMA."""
    import lzma

    return lzma.compress(data)


def lzma_decode(data, out=None):
    """Decompress LZMA."""
    import lzma

    return lzma.decompress(data)


if imagecodecs is None:

    def delta_encode(data, axis=-1, dist=1, out=None):
        """Encode Delta."""
        if dist != 1:
            raise NotImplementedError(f'dist {dist} not implemented')
        if isinstance(data, (bytes, bytearray)):
            data = numpy.frombuffer(data, dtype=numpy.uint8)
            diff = numpy.diff(data, axis=0)
            return numpy.insert(diff, 0, data[0]).tobytes()

        dtype = data.dtype
        if dtype.kind == 'f':
            data = data.view(f'u{dtype.itemsize}')

        diff = numpy.diff(data, axis=axis)
        key = [slice(None)] * data.ndim
        key[axis] = 0
        diff = numpy.insert(diff, 0, data[tuple(key)], axis=axis)

        if dtype.kind == 'f':
            return diff.view(dtype)
        return diff

    def delta_decode(data, axis=-1, dist=1, out=None):
        """Decode Delta."""
        if dist != 1:
            raise NotImplementedError(f'dist {dist} not implemented')
        if out is not None and not out.flags.writeable:
            out = None
        if isinstance(data, (bytes, bytearray)):
            data = numpy.frombuffer(data, dtype=numpy.uint8)
            return numpy.cumsum(
                data, axis=0, dtype=numpy.uint8, out=out
            ).tobytes()
        if data.dtype.kind == 'f':
            view = data.view(f'u{data.dtype.itemsize}')
            view = numpy.cumsum(view, axis=axis, dtype=view.dtype)
            return view.view(data.dtype)
        return numpy.cumsum(data, axis=axis, dtype=data.dtype, out=out)

    def bitorder_decode(data, out=None, _bitorder=[]):
        r"""Reverse bits in each byte of bytes or numpy array.

        Decode data where pixels with lower column values are stored in the
        lower-order bits of the bytes (TIFF FillOrder is LSB2MSB).

        Parameters
        ----------
        data : bytes or ndarray
            The data to be bit reversed. If bytes, a new bit-reversed
            bytes is returned. Numpy arrays are bit-reversed in-place.

        Examples
        --------
        >>> bitorder_decode(b'\x01\x64')
        b'\x80&'
        >>> data = numpy.array([1, 666], dtype='uint16')
        >>> bitorder_decode(data)
        >>> data
        array([  128, 16473], dtype=uint16)

        """
        if not _bitorder:
            _bitorder.append(
                b'\x00\x80@\xc0 \xa0`\xe0\x10\x90P\xd00\xb0p\xf0\x08\x88H'
                b'\xc8(\xa8h\xe8\x18\x98X\xd88\xb8x\xf8\x04\x84D\xc4$\xa4d'
                b'\xe4\x14\x94T\xd44\xb4t\xf4\x0c\x8cL\xcc,\xacl\xec\x1c\x9c'
                b'\\\xdc<\xbc|\xfc\x02\x82B\xc2"\xa2b\xe2\x12\x92R\xd22'
                b'\xb2r\xf2\n\x8aJ\xca*\xaaj\xea\x1a\x9aZ\xda:\xbaz\xfa'
                b'\x06\x86F\xc6&\xa6f\xe6\x16\x96V\xd66\xb6v\xf6\x0e\x8eN'
                b'\xce.\xaen\xee\x1e\x9e^\xde>\xbe~\xfe\x01\x81A\xc1!\xa1a'
                b'\xe1\x11\x91Q\xd11\xb1q\xf1\t\x89I\xc9)\xa9i\xe9\x19'
                b'\x99Y\xd99\xb9y\xf9\x05\x85E\xc5%\xa5e\xe5\x15\x95U\xd55'
                b'\xb5u\xf5\r\x8dM\xcd-\xadm\xed\x1d\x9d]\xdd=\xbd}\xfd'
                b'\x03\x83C\xc3#\xa3c\xe3\x13\x93S\xd33\xb3s\xf3\x0b\x8bK'
                b'\xcb+\xabk\xeb\x1b\x9b[\xdb;\xbb{\xfb\x07\x87G\xc7\'\xa7g'
                b'\xe7\x17\x97W\xd77\xb7w\xf7\x0f\x8fO\xcf/\xafo\xef\x1f\x9f_'
                b'\xdf?\xbf\x7f\xff'
            )
            _bitorder.append(numpy.frombuffer(_bitorder[0], dtype=numpy.uint8))
        try:
            view = data.view('uint8')
            numpy.take(_bitorder[1], view, out=view)
            return data
        except AttributeError:
            return data.translate(_bitorder[0])
        except ValueError:
            raise NotImplementedError('slices of arrays not supported')
        return None

    def packints_encode(data, bitspersample, axis=-1, out=None):
        """Tightly pack integers."""
        raise NotImplementedError('packints_encode')

    def packints_decode(data, dtype, bitspersample, runlen=0, out=None):
        """Decompress bytes to array of integers.

        This implementation only handles itemsizes 1, 8, 16, 32, and 64 bits.
        Install the imagecodecs package for decoding other integer sizes.

        Parameters
        ----------
        data : byte str
            Data to decompress.
        dtype : numpy.dtype or str
            A numpy boolean or integer type.
        bitspersample : int
            Number of bits per integer.
        runlen : int
            Number of consecutive integers, after which to start at next byte.

        Examples
        --------
        >>> packints_decode(b'a', 'B', 1)
        array([0, 1, 1, 0, 0, 0, 0, 1], dtype=uint8)

        """
        if bitspersample == 1:  # bitarray
            data = numpy.frombuffer(data, '|B')
            data = numpy.unpackbits(data)
            if runlen % 8:
                data = data.reshape(-1, runlen + (8 - runlen % 8))
                data = data[:, :runlen].reshape(-1)
            return data.astype(dtype)
        if bitspersample in (8, 16, 32, 64):
            return numpy.frombuffer(data, dtype)
        raise NotImplementedError(
            f'unpacking {bitspersample}-bit integers '
            f'to {numpy.dtype(dtype)} not supported'
        )

    def packbits_decode(encoded, out=None):
        r"""Decompress PackBits encoded byte string.

        >>> packbits_decode(b'\x80\x80')  # NOP
        b''
        >>> packbits_decode(b'\x02123')
        b'123'
        >>> packbits_decode(
        ...   b'\xfe\xaa\x02\x80\x00\x2a\xfd\xaa\x03\x80\x00\x2a\x22\xf7\xaa'
        ...     )[:-5]
        b'\xaa\xaa\xaa\x80\x00*\xaa\xaa\xaa\xaa\x80\x00*"\xaa\xaa\xaa\xaa\xaa'

        """
        out = []
        out_extend = out.extend
        i = 0
        try:
            while True:
                n = ord(encoded[i : i + 1]) + 1
                i += 1
                if n > 129:
                    # replicate
                    out_extend(encoded[i : i + 1] * (258 - n))
                    i += 1
                elif n < 129:
                    # literal
                    out_extend(encoded[i : i + n])
                    i += n
        except TypeError:
            pass
        return bytes(out)


else:
    bitorder_decode = imagecodecs.bitorder_decode  # noqa
    packints_decode = imagecodecs.packints_decode  # noqa
    packints_encode = imagecodecs.packints_encode  # noqa
    try:
        float24_decode = imagecodecs.float24_decode  # noqa
    except AttributeError:
        pass


def apply_colormap(image, colormap, contig=True):
    """Return palette-colored image.

    The image values are used to index the colormap on axis 1. The returned
    image is of shape image.shape+colormap.shape[0] and dtype colormap.dtype.

    Parameters
    ----------
    image : numpy.ndarray
        Indexes into the colormap.
    colormap : numpy.ndarray
        RGB lookup table aka palette of shape (3, 2**bits_per_sample).
    contig : bool
        If True, return a contiguous array.

    Examples
    --------
    >>> image = numpy.arange(256, dtype='uint8')
    >>> colormap = numpy.vstack([image, image, image]).astype('uint16') * 256
    >>> apply_colormap(image, colormap)[-1]
    array([65280, 65280, 65280], dtype=uint16)

    """
    image = numpy.take(colormap, image, axis=1)
    image = numpy.rollaxis(image, 0, image.ndim)
    if contig:
        image = numpy.ascontiguousarray(image)
    return image


def parse_filenames(
    files, pattern, axesorder=None, categories=None, _shape=None
):
    r"""Return shape and axes from sequence of file names matching pattern.

    Parameters
    ----------
    files : sequence of str
        Sequence of file names to parse.
    pattern : str
        Regular expression pattern matching axes labels and chunk indices
        in file names. By default, no pattern matching is performed.
        Axes labels can be specified by matching groups preceding the index
        groups in the file name, be provided as group names for the index
        groups, or be omitted.
        The predefined 'axes' pattern matches Olympus OIF and Leica TIFF
        series.
    axesorder : sequence of int (optional)
        Indices of axes in pattern. By default axes are returned in the order
        they appear in pattern.
    categories : dict of dicts (optional)
        Map of index group matches to integer indices.
        {'axislabel': {'category': index}}
    _shape : tuple of int (optional)
        Shape of the file sequence. If None (default), the shape is
        maximum-minimum+1 of the parsed indices for each dimension.

    Returns
    -------
    labels : tuple of str
        Axes labels for each dimension.
    shape : tuple of int
        Shape of file series.
    indices : sequence of tuples
        Index of each file in shape.
    files : sequence of str
        Filtered sequence of file names.

    Examples
    --------
    >>> parse_filenames(
    ...     ['c1001.ext', 'c2002.ext'], r'([^\d])(\d)(?P<t>\d+)\.ext'
    ... )
    (('c', 't'), (2, 2), [(0, 0), (1, 1)], ['c1001.ext', 'c2002.ext'])

    """
    # TODO: add option to filter files that do not match pattern

    shape = _shape
    if pattern is None:
        if shape is not None and (len(shape) != 1 or shape[0] < len(files)):
            raise ValueError(
                f'shape {(len(files),)} does not fit provided shape {shape}'
            )
        return (
            ('I',),
            (len(files),),
            tuple((i,) for i in range(len(files))),
            files,
        )

    pattern = TIFF.FILE_PATTERNS.get(pattern, pattern)
    if not pattern:
        raise ValueError('invalid pattern')
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    if categories is None:
        categories = {}

    def parse(fname):
        # return axes labels and indices from file name
        labels = []
        indices = []
        groupindex = {v: k for k, v in pattern.groupindex.items()}
        match = pattern.search(fname)
        if not match:
            raise ValueError(f'pattern does not match file name {fname!r}')
        ax = None
        for i, m in enumerate(match.groups()):
            if m is None:
                continue
            if i + 1 in groupindex:
                ax = groupindex[i + 1]
            elif m[0].isalpha():
                ax = m  # axis label for next index
                continue
            if ax is None:
                ax = 'Q'  # no preceding axis letter
            try:
                if ax in categories:
                    m = categories[ax][m]
                m = int(m)
            except Exception as exc:
                raise ValueError(f'invalid index {m!r}') from exc
            indices.append(m)
            labels.append(ax)
            ax = None
        return tuple(labels), indices

    normpaths = [os.path.normpath(f) for f in files]
    if len(normpaths) == 1:
        prefix = os.path.dirname(normpaths[0])
    else:
        prefix = os.path.commonpath(normpaths)
    prefix = len(prefix)

    labels = None
    indices = []
    for fname in normpaths:
        lbl, idx = parse(fname[prefix:])
        if labels is None:
            labels = lbl
            if axesorder is not None and (
                len(axesorder) != len(labels)
                or any(i not in axesorder for i in range(len(labels)))
            ):
                raise ValueError(
                    f'invalid axesorder {axesorder!r} for {labels!r}'
                )
        elif labels != lbl:
            raise ValueError('axes labels do not match within image sequence')
        if axesorder is not None:
            idx = [idx[i] for i in axesorder]
        indices.append(idx)

    if axesorder is None:
        labels = tuple(labels)
    else:
        labels = tuple(labels[i] for i in axesorder)

    # determine shape
    indices = numpy.array(indices, dtype=numpy.intp)
    parsedshape = numpy.max(indices, axis=0)

    if shape is None:
        startindex = numpy.min(indices, axis=0)
        indices -= startindex
        parsedshape -= startindex
        parsedshape += 1
        shape = tuple(parsedshape.tolist())
    elif len(parsedshape) != len(shape) or any(
        i > j for i, j in zip(shape, parsedshape)
    ):
        raise ValueError(
            f'parsed shape {parsedshape} does not fit provided shape {shape}'
        )

    indices = [tuple(index) for index in indices.tolist()]

    return labels, shape, indices, files


def iter_images(data):
    """Return iterator over pages in data array of normalized shape."""
    yield from data


def iter_tiles(data, tile, tiles):
    """Return iterator over tiles in data array of normalized shape."""
    shape = data.shape
    chunk = numpy.empty(tile + (shape[-1],), dtype=data.dtype)
    if not 1 < len(tile) < 4:
        raise ValueError('invalid tile shape')
    if len(tile) == 2:
        for page in data:
            for plane in page:
                for ty in range(tiles[0]):
                    for tx in range(tiles[1]):
                        c1 = min(tile[0], shape[3] - ty * tile[0])
                        c2 = min(tile[1], shape[4] - tx * tile[1])
                        chunk[c1:, c2:] = 0
                        chunk[:c1, :c2] = plane[
                            0,
                            ty * tile[0] : ty * tile[0] + c1,
                            tx * tile[1] : tx * tile[1] + c2,
                        ]
                        yield chunk
    else:
        for page in data:
            for plane in page:
                for tz in range(tiles[0]):
                    for ty in range(tiles[1]):
                        for tx in range(tiles[2]):
                            c0 = min(tile[0], shape[2] - tz * tile[0])
                            c1 = min(tile[1], shape[3] - ty * tile[1])
                            c2 = min(tile[2], shape[4] - tx * tile[2])
                            chunk[c0:, c1:, c2:] = 0
                            chunk[:c0, :c1, :c2] = plane[
                                tz * tile[0] : tz * tile[0] + c0,
                                ty * tile[1] : ty * tile[1] + c1,
                                tx * tile[2] : tx * tile[2] + c2,
                            ]
                            if tile[0] == 1:
                                # squeeze for image compressors
                                yield chunk[0]
                            else:
                                yield chunk


def pad_tile(tile, shape, dtype):
    """Return tile padded to tile shape."""
    if tile.dtype != dtype or tile.nbytes > product(shape) * dtype.itemsize:
        raise ValueError('invalid tile shape or dtype')
    pad = tuple((0, i - j) for i, j in zip(shape, tile.shape))
    return numpy.pad(tile, pad)


def reorient(image, orientation):
    """Return reoriented view of image array.

    Parameters
    ----------
    image : numpy.ndarray
        Non-squeezed output of asarray() functions.
        Axes -3 and -2 must be image length and width respectively.
    orientation : int or str
        One of TIFF.ORIENTATION names or values.

    """
    orient = TIFF.ORIENTATION
    orientation = enumarg(orient, orientation)

    if orientation == orient.TOPLEFT:
        return image
    if orientation == orient.TOPRIGHT:
        return image[..., ::-1, :]
    if orientation == orient.BOTLEFT:
        return image[..., ::-1, :, :]
    if orientation == orient.BOTRIGHT:
        return image[..., ::-1, ::-1, :]
    if orientation == orient.LEFTTOP:
        return numpy.swapaxes(image, -3, -2)
    if orientation == orient.RIGHTTOP:
        return numpy.swapaxes(image, -3, -2)[..., ::-1, :]
    if orientation == orient.RIGHTBOT:
        return numpy.swapaxes(image, -3, -2)[..., ::-1, :, :]
    if orientation == orient.LEFTBOT:
        return numpy.swapaxes(image, -3, -2)[..., ::-1, ::-1, :]
    return image


def repeat_nd(a, repeats):
    """Return read-only view into input array with elements repeated.

    Zoom nD image by integer factors using nearest neighbor interpolation
    (box filter).

    Parameters
    ----------
    a : array-like
        Input array.
    repeats : sequence of int
        The number of repetitions to apply along each dimension of input array.

    Examples
    --------
    >>> repeat_nd([[1, 2], [3, 4]], (2, 2))
    array([[1, 1, 2, 2],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 4, 4]])

    """
    a = numpy.asarray(a)
    reshape = []
    shape = []
    strides = []
    for i, j, k in zip(a.strides, a.shape, repeats):
        shape.extend((j, k))
        strides.extend((i, 0))
        reshape.append(j * k)
    return numpy.lib.stride_tricks.as_strided(
        a, shape, strides, writeable=False
    ).reshape(reshape)


def reshape_nd(data_or_shape, ndim):
    """Return image array or shape with at least ndim dimensions.

    Prepend 1s to image shape as necessary.

    >>> reshape_nd(numpy.empty(0), 1).shape
    (0,)
    >>> reshape_nd(numpy.empty(1), 2).shape
    (1, 1)
    >>> reshape_nd(numpy.empty((2, 3)), 3).shape
    (1, 2, 3)
    >>> reshape_nd(numpy.empty((3, 4, 5)), 3).shape
    (3, 4, 5)
    >>> reshape_nd((2, 3), 3)
    (1, 2, 3)

    """
    is_shape = isinstance(data_or_shape, tuple)
    shape = data_or_shape if is_shape else data_or_shape.shape
    if len(shape) >= ndim:
        return data_or_shape
    shape = (1,) * (ndim - len(shape)) + shape
    return shape if is_shape else data_or_shape.reshape(shape)


def squeeze_axes(shape, axes, skip=None):
    """Return shape and axes with single-dimensional entries removed.

    Remove unused dimensions unless their axes are listed in 'skip'.

    >>> squeeze_axes((5, 1, 2, 1, 1), 'TZYXC')
    ((5, 2, 1), 'TYX')

    >>> squeeze_axes((1,), 'Q')
    ((1,), 'Q')

    """
    if len(shape) != len(axes):
        raise ValueError('dimensions of axes and shape do not match')
    if skip is None:
        skip = 'XY'
    try:
        shape_squeezed, axes_squeezed = zip(
            *(i for i in zip(shape, axes) if i[0] > 1 or i[1] in skip)
        )
    except ValueError:
        # not enough values to unpack, return last axis
        shape_squeezed = shape[-1:]
        axes_squeezed = axes[-1:]
    return tuple(shape_squeezed), ''.join(axes_squeezed)


def transpose_axes(image, axes, asaxes=None):
    """Return image with its axes permuted to match specified axes.

    A view is returned if possible.

    >>> transpose_axes(numpy.zeros((2, 3, 4, 5)), 'TYXC', asaxes='CTZYX').shape
    (5, 2, 1, 3, 4)

    """
    for ax in axes:
        if ax not in asaxes:
            raise ValueError(f'unknown axis {ax}')
    # add missing axes to image
    if asaxes is None:
        asaxes = 'CTZYX'
    shape = image.shape
    for ax in reversed(asaxes):
        if ax not in axes:
            axes = ax + axes
            shape = (1,) + shape
    image = image.reshape(shape)
    # transpose axes
    image = image.transpose([axes.index(ax) for ax in asaxes])
    return image


def reshape_axes(axes, shape, newshape, unknown=None):
    """Return axes matching new shape.

    By default, unknown dimensions are labelled 'Q'.

    >>> reshape_axes('YXS', (219, 301, 1), (219, 301))
    'YX'
    >>> reshape_axes('IYX', (12, 219, 301), (3, 4, 219, 1, 301, 1))
    'QQYQXQ'

    """
    shape = tuple(shape)
    newshape = tuple(newshape)
    if len(axes) != len(shape):
        raise ValueError('axes do not match shape')

    size = product(shape)
    newsize = product(newshape)
    if size != newsize:
        raise ValueError(f'cannot reshape {shape} to {newshape}')
    if not axes or not newshape:
        return ''

    lendiff = max(0, len(shape) - len(newshape))
    if lendiff:
        newshape = newshape + (1,) * lendiff

    i = len(shape) - 1
    prodns = 1
    prods = 1
    result = []
    for ns in newshape[::-1]:
        prodns *= ns
        while i > 0 and shape[i] == 1 and ns != 1:
            i -= 1
        if ns == shape[i] and prodns == prods * shape[i]:
            prods *= shape[i]
            result.append(axes[i])
            i -= 1
        elif unknown:
            result.append(unknown)
        else:
            unknown = 'Q'
            result.append(unknown)

    return ''.join(reversed(result[lendiff:]))


def subresolution(a, b, p=2, n=16):
    """Return level of subresolution of series or page b vs a."""
    if a.axes != b.axes or a.dtype != b.dtype:
        return None
    level = None
    for ax, i, j in zip(a.axes.lower(), a.shape, b.shape):
        if ax in 'xyz':
            if level is None:
                for r in range(n):
                    d = p ** r
                    if d > i:
                        return None
                    if abs((i / d) - j) < 1.0:
                        level = r
                        break
                else:
                    return None
            else:
                d = p ** level
                if d > i:
                    return None
                if abs((i / d) - j) >= 1.0:
                    return None
        elif i != j:
            return None
    return level


def pyramidize_series(series, isreduced=False):
    """Pyramidize list of TiffPageSeries in-place.

    TiffPageSeries that are a subresolution of another TiffPageSeries are
    appended to the other's TiffPageSeries levels and removed from the list.
    Levels are to be ordered by size using the same downsampling factor.
    TiffPageSeries of subifds cannot be pyramid top levels.

    """
    samplingfactors = (2, 3, 4)
    i = 0
    while i < len(series):
        a = series[i]
        p = None
        j = i + 1
        if isinstance(a.keyframe.index, tuple):
            # subifds cannot be pyramid top levels
            i += 1
            continue
        while j < len(series):
            b = series[j]
            if isreduced and not b.keyframe.is_reduced:
                # pyramid levels must be reduced
                j += 1
                continue  # not a pyramid level
            if p is None:
                for f in samplingfactors:
                    if subresolution(a.levels[-1], b, p=f) == 1:
                        p = f
                        break  # not a pyramid level
                else:
                    j += 1
                    continue  # not a pyramid level
            elif subresolution(a.levels[-1], b, p=p) != 1:
                j += 1
                continue
            a.levels.append(b)
            del series[j]
        i += 1


def stack_pages(pages, out=None, maxworkers=None, **kwargs):
    """Read data from sequence of TiffPage/Frame and stack them vertically.

    Additional parameters are passsed to the TiffPage.asarray function.

    """
    npages = len(pages)
    if npages == 0:
        raise ValueError('no pages')

    if npages == 1:
        kwargs['maxworkers'] = maxworkers
        return pages[0].asarray(out=out, **kwargs)

    page0 = next(p.keyframe for p in pages if p is not None)
    shape = (npages,) + page0.shape
    dtype = page0.dtype
    out = create_output(out, shape, dtype)

    # TODO: benchmark and optimize this
    if maxworkers is None or maxworkers < 1:
        # auto-detect
        page_maxworkers = page0.maxworkers
        maxworkers = min(npages, TIFF.MAXWORKERS)
        if maxworkers == 1 or page_maxworkers < 1:
            maxworkers = page_maxworkers = 1
        elif npages < 3:
            maxworkers = 1
        elif (
            page_maxworkers <= 2
            and page0.compression == 1
            and page0.fillorder == 1
            and page0.predictor == 1
        ):
            maxworkers = 1
        else:
            page_maxworkers = 1
    elif maxworkers == 1:
        maxworkers = page_maxworkers = 1
    elif npages > maxworkers or page0.maxworkers < 2:
        page_maxworkers = 1
    else:
        page_maxworkers = maxworkers
        maxworkers = 1

    kwargs['maxworkers'] = page_maxworkers

    filehandle = page0.parent.filehandle
    haslock = filehandle.has_lock
    if not haslock and maxworkers > 1 or page_maxworkers > 1:
        filehandle.lock = True
    filecache = FileCache(size=max(4, maxworkers), lock=filehandle.lock)

    def func(page, index, out=out, filecache=filecache, kwargs=kwargs):
        # read, decode, and copy page data
        if page is not None:
            filecache.open(page.parent.filehandle)
            page.asarray(lock=filecache.lock, out=out[index], **kwargs)
            filecache.close(page.parent.filehandle)

    if maxworkers < 2:
        for i, page in enumerate(pages):
            func(page, i)
    else:
        page0.decode  # init TiffPage.decode function
        with ThreadPoolExecutor(maxworkers) as executor:
            for _ in executor.map(func, pages, range(npages)):
                pass

    filecache.clear()
    if not haslock:
        filehandle.lock = False
    return out


def create_output(out, shape, dtype, mode='w+', suffix=None, fillvalue=0):
    """Return numpy array where image data of shape and dtype can be copied.

    The 'out' parameter may have the following values or types:

    None
        A zeroed array of shape and dtype is created and returned.
    numpy.ndarray
        An existing writable array of compatible dtype and shape. A view of
        the same array is returned after verification.
    'memmap' or 'memmap:tempdir'
        A memory-map to an array stored in a temporary binary file on disk
        is created and returned.
    str or open file
        The file name or file object used to create a memory-map to an array
        stored in a binary file on disk. The created memory-mapped array is
        returned.

    """
    if out is None:
        if fillvalue is None:
            return numpy.empty(shape, dtype)
        if fillvalue:
            out = numpy.empty(shape, dtype)
            out[:] = fillvalue
            return out
        return numpy.zeros(shape, dtype)
    if isinstance(out, numpy.ndarray):
        if product(shape) != product(out.shape):
            raise ValueError('incompatible output shape')
        if not numpy.can_cast(dtype, out.dtype):
            raise ValueError('incompatible output dtype')
        return out.reshape(shape)
    if isinstance(out, str) and out[:6] == 'memmap':
        import tempfile

        tempdir = out[7:] if len(out) > 7 else None
        if suffix is None:
            suffix = '.memmap'
        with tempfile.NamedTemporaryFile(dir=tempdir, suffix=suffix) as fh:
            return numpy.memmap(fh, shape=shape, dtype=dtype, mode=mode)
    return numpy.memmap(out, shape=shape, dtype=dtype, mode=mode)


def matlabstr2py(string):
    r"""Return Python object from Matlab string representation.

    Return str, bool, int, float, list (Matlab arrays or cells), or
    dict (Matlab structures) types.

    Use to access ScanImage metadata.

    >>> matlabstr2py('1')
    1
    >>> matlabstr2py("['x y z' true false; 1 2.0 -3e4; NaN Inf @class]")
    [['x y z', True, False], [1, 2.0, -30000.0], [nan, inf, '@class']]
    >>> d = matlabstr2py(
    ...     "SI.hChannels.channelType = {'stripe' 'stripe'}\n"
    ...     "SI.hChannels.channelsActive = 2"
    ... )
    >>> d['SI.hChannels.channelType']
    ['stripe', 'stripe']

    """
    # TODO: handle invalid input
    # TODO: review unboxing of multidimensional arrays

    def lex(s):
        # return sequence of tokens from matlab string representation
        tokens = ['[']
        while True:
            t, i = next_token(s)
            if t is None:
                break
            if t == ';':
                tokens.extend((']', '['))
            elif t == '[':
                tokens.extend(('[', '['))
            elif t == ']':
                tokens.extend((']', ']'))
            else:
                tokens.append(t)
            s = s[i:]
        tokens.append(']')
        return tokens

    def next_token(s):
        # return next token in matlab string
        length = len(s)
        if length == 0:
            return None, 0
        i = 0
        while i < length and s[i] == ' ':
            i += 1
        if i == length:
            return None, i
        if s[i] in '{[;]}':
            return s[i], i + 1
        if s[i] == "'":
            j = i + 1
            while j < length and s[j] != "'":
                j += 1
            return s[i : j + 1], j + 1
        if s[i] == '<':
            j = i + 1
            while j < length and s[j] != '>':
                j += 1
            return s[i : j + 1], j + 1
        j = i
        while j < length and not s[j] in ' {[;]}':
            j += 1
        return s[i:j], j

    def value(s, fail=False):
        # return Python value of token
        s = s.strip()
        if not s:
            return s
        if len(s) == 1:
            try:
                return int(s)
            except Exception:
                if fail:
                    raise ValueError
                return s
        if s[0] == "'":
            if fail and s[-1] != "'" or "'" in s[1:-1]:
                raise ValueError
            return s[1:-1]
        if s[0] == '<':
            if fail and s[-1] != '>' or '<' in s[1:-1]:
                raise ValueError
            return s
        if fail and any(i in s for i in " ';[]{}"):
            raise ValueError
        if s[0] == '@':
            return s
        if s == 'true' or s == 'True':
            return True
        if s == 'false' or s == 'False':
            return False
        if s[:6] == 'zeros(':
            return numpy.zeros([int(i) for i in s[6:-1].split(',')]).tolist()
        if s[:5] == 'ones(':
            return numpy.ones([int(i) for i in s[5:-1].split(',')]).tolist()
        if '.' in s or 'e' in s:
            try:
                return float(s)
            except Exception:
                pass
        try:
            return int(s)
        except Exception:
            pass
        try:
            return float(s)  # nan, inf
        except Exception:
            if fail:
                raise ValueError
        return s

    def parse(s):
        # return Python value from string representation of Matlab value
        s = s.strip()
        try:
            return value(s, fail=True)
        except ValueError:
            pass
        result = add2 = []
        levels = [add2]
        for t in lex(s):
            if t in '[{':
                add2 = []
                levels.append(add2)
            elif t in ']}':
                x = levels.pop()
                if len(x) == 1 and isinstance(x[0], (list, str)):
                    x = x[0]
                add2 = levels[-1]
                add2.append(x)
            else:
                add2.append(value(t))
        if len(result) == 1 and isinstance(result[0], (list, str)):
            result = result[0]
        return result

    if '\r' in string or '\n' in string:
        # structure
        d = {}
        for line in string.splitlines():
            line = line.strip()
            if not line or line[0] == '%':
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            if any(c in k for c in " ';[]{}<>"):
                continue
            d[k] = parse(v)
        return d
    return parse(string)


def stripnull(string, null=b'\x00', first=True):
    r"""Return string truncated at first null character.

    Clean NULL terminated C strings. For Unicode strings use null='\0'.

    >>> stripnull(b'string\x00\x00')
    b'string'
    >>> stripnull(b'string\x00string\x00\x00', first=False)
    b'string\x00string'
    >>> stripnull('string\x00', null='\0')
    'string'

    """
    if first:
        i = string.find(null)
        return string if i < 0 else string[:i]
    null = null[0]
    i = len(string)
    while i:
        i -= 1
        if string[i] != null:
            break
    else:
        i = -1
    return string[: i + 1]


def stripascii(string):
    r"""Return string truncated at last byte that is 7-bit ASCII.

    Clean NULL separated and terminated TIFF strings.

    >>> stripascii(b'string\x00string\n\x01\x00')
    b'string\x00string\n'
    >>> stripascii(b'\x00')
    b''

    """
    # TODO: pythonize this
    i = len(string)
    while i:
        i -= 1
        if 8 < string[i] < 127:
            break
    else:
        i = -1
    return string[: i + 1]


def asbool(value, true=None, false=None):
    """Return string as bool if possible, else raise TypeError.

    >>> asbool(b' False ')
    False
    >>> asbool('ON', ['on'], ['off'])
    True

    """
    value = value.strip().lower()
    isbytes = False
    if true is None:
        if isinstance(value, bytes):
            if value == b'true':
                return True
            isbytes = True
        elif value == 'true':
            return True
    if false is None:
        if isbytes or isinstance(value, bytes):
            if value == b'false':
                return False
        elif value == 'false':
            return False
    if value in true:
        return True
    if value in false:
        return False
    raise TypeError


def astype(value, types=None):
    """Return argument as one of types if possible.

    >>> astype('42')
    42
    >>> astype('3.14')
    3.14
    >>> astype('True')
    True
    >>> astype(b'Neee-Wom')
    'Neee-Wom'

    """
    if types is None:
        types = int, float, asbool, bytes2str
    for typ in types:
        try:
            return typ(value)
        except (ValueError, AttributeError, TypeError, UnicodeEncodeError):
            pass
    return value


def format_size(size, threshold=1536):
    """Return file size as string from byte size.

    >>> format_size(1234)
    '1234 B'
    >>> format_size(12345678901)
    '11.50 GiB'

    """
    if size < threshold:
        return f'{size} B'
    for unit in ('KiB', 'MiB', 'GiB', 'TiB', 'PiB'):
        size /= 1024.0
        if size < threshold:
            return f'{size:.2f} {unit}'
    return 'ginormous'


def identityfunc(arg, *args, **kwargs):
    """Single argument identity function.

    >>> identityfunc('arg')
    'arg'

    """
    return arg


def nullfunc(*args, **kwargs):
    """Null function.

    >>> nullfunc('arg', kwarg='kwarg')

    """
    return


def sequence(value):
    """Return tuple containing value if value is not a tuple or list.

    >>> sequence(1)
    (1,)
    >>> sequence([1])
    [1]
    >>> sequence('ab')
    ('ab',)

    """
    return value if isinstance(value, (tuple, list)) else (value,)


def product(iterable):
    """Return product of sequence of numbers.

    Equivalent of functools.reduce(operator.mul, iterable, 1).
    Multiplying numpy integers might overflow.

    >>> product([2**8, 2**30])
    274877906944
    >>> product([])
    1

    """
    prod = 1
    for i in iterable:
        prod *= i
    return prod


def natural_sorted(iterable):
    """Return human sorted list of strings.

    E.g. for sorting file names.

    >>> natural_sorted(['f1', 'f2', 'f10'])
    ['f1', 'f2', 'f10']

    """

    def sortkey(x):
        return [(int(c) if c.isdigit() else c) for c in re.split(numbers, x)]

    numbers = re.compile(r'(\d+)')
    return sorted(iterable, key=sortkey)


def epics_datetime(sec, nsec):
    """Return datetime object from epicsTSSec and epicsTSNsec tag values."""
    return datetime.datetime.fromtimestamp(sec + 631152000 + nsec / 1e9)


def excel_datetime(timestamp, epoch=None):
    """Return datetime object from timestamp in Excel serial format.

    Convert LSM time stamps.

    >>> excel_datetime(40237.029999999795)
    datetime.datetime(2010, 2, 28, 0, 43, 11, 999982)

    """
    if epoch is None:
        epoch = datetime.datetime.fromordinal(693594)
    return epoch + datetime.timedelta(timestamp)


def julian_datetime(julianday, milisecond=0):
    """Return datetime from days since 1/1/4713 BC and ms since midnight.

    Convert Julian dates according to MetaMorph.

    >>> julian_datetime(2451576, 54362783)
    datetime.datetime(2000, 2, 2, 15, 6, 2, 783)

    """
    if julianday <= 1721423:
        # no datetime before year 1
        return None

    a = julianday + 1
    if a > 2299160:
        alpha = math.trunc((a - 1867216.25) / 36524.25)
        a += 1 + alpha - alpha // 4
    b = a + (1524 if a > 1721423 else 1158)
    c = math.trunc((b - 122.1) / 365.25)
    d = math.trunc(365.25 * c)
    e = math.trunc((b - d) / 30.6001)

    day = b - d - math.trunc(30.6001 * e)
    month = e - (1 if e < 13.5 else 13)
    year = c - (4716 if month > 2.5 else 4715)

    hour, milisecond = divmod(milisecond, 1000 * 60 * 60)
    minute, milisecond = divmod(milisecond, 1000 * 60)
    second, milisecond = divmod(milisecond, 1000)

    return datetime.datetime(
        year, month, day, hour, minute, second, milisecond
    )


def byteorder_isnative(byteorder):
    """Return if byteorder matches the system's byteorder.

    >>> byteorder_isnative('=')
    True

    """
    if byteorder == '=' or byteorder == sys.byteorder:
        return True
    keys = {'big': '>', 'little': '<'}
    return keys.get(byteorder, byteorder) == keys[sys.byteorder]


def byteorder_compare(byteorder, byteorder2):
    """Return if byteorders match.

    >>> byteorder_compare('<', '<')
    True
    >>> byteorder_compare('>', '<')
    False

    """
    if byteorder == byteorder2 or byteorder == '|' or byteorder2 == '|':
        return True
    if byteorder == '=':
        byteorder = {'big': '>', 'little': '<'}[sys.byteorder]
    elif byteorder2 == '=':
        byteorder2 = {'big': '>', 'little': '<'}[sys.byteorder]
    return byteorder == byteorder2


def recarray2dict(recarray):
    """Return numpy.recarray as dict."""
    # TODO: subarrays
    result = {}
    for descr, value in zip(recarray.dtype.descr, recarray):
        name, dtype = descr[:2]
        if dtype[1] == 'S':
            value = bytes2str(stripnull(value))
        elif value.ndim < 2:
            value = value.tolist()
        result[name] = value
    return result


def xml2dict(xml, sanitize=True, prefix=None):
    """Return XML as dict.

    >>> xml2dict('<?xml version="1.0" ?><root attr="name"><key>1</key></root>')
    {'root': {'key': 1, 'attr': 'name'}}
    >>> xml2dict('<level1><level2>3.5322</level2></level1>')
    {'level1': {'level2': 3.5322}}

    """
    from xml.etree import ElementTree as etree  # delayed import

    at = tx = ''
    if prefix:
        at, tx = prefix

    def astype(value):
        # return string value as int, float, bool, or unchanged
        if not isinstance(value, (str, bytes)):
            return value
        for t in (int, float, asbool):
            try:
                return t(value)
            except Exception:
                pass
        return value

    def etree2dict(t):
        # adapted from https://stackoverflow.com/a/10077069/453463
        key = t.tag
        if sanitize:
            key = key.rsplit('}', 1)[-1]
        d = {key: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = collections.defaultdict(list)
            for dc in map(etree2dict, children):
                for k, v in dc.items():
                    dd[k].append(astype(v))
            d = {
                key: {
                    k: astype(v[0]) if len(v) == 1 else astype(v)
                    for k, v in dd.items()
                }
            }
        if t.attrib:
            d[key].update((at + k, astype(v)) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[key][tx + 'value'] = astype(text)
            else:
                d[key] = astype(text)
        return d

    return etree2dict(etree.fromstring(xml))


def hexdump(bytestr, width=75, height=24, snipat=-2, modulo=2, ellipsis=None):
    """Return hexdump representation of bytes.

    >>> hexdump(binascii.unhexlify('49492a00080000000e00fe0004000100'))
    '49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 II*.............'

    """
    size = len(bytestr)
    if size < 1 or width < 2 or height < 1:
        return ''
    if height == 1:
        addr = b''
        bytesperline = min(
            modulo * (((width - len(addr)) // 4) // modulo), size
        )
        if bytesperline < 1:
            return ''
        nlines = 1
    else:
        addr = b'%%0%ix: ' % len(b'%x' % size)
        bytesperline = min(
            modulo * (((width - len(addr % 1)) // 4) // modulo), size
        )
        if bytesperline < 1:
            return ''
        width = 3 * bytesperline + len(addr % 1)
        nlines = (size - 1) // bytesperline + 1

    if snipat is None or snipat == 1:
        snipat = height
    elif 0 < abs(snipat) < 1:
        snipat = int(math.floor(height * snipat))
    if snipat < 0:
        snipat += height

    if height == 1 or nlines == 1:
        blocks = [(0, bytestr[:bytesperline])]
        addr = b''
        height = 1
        width = 3 * bytesperline
    elif height is None or nlines <= height:
        blocks = [(0, bytestr)]
    elif snipat <= 0:
        start = bytesperline * (nlines - height)
        blocks = [(start, bytestr[start:])]  # (start, None)
    elif snipat >= height or height < 3:
        end = bytesperline * height
        blocks = [(0, bytestr[:end])]  # (end, None)
    else:
        end1 = bytesperline * snipat
        end2 = bytesperline * (height - snipat - 1)
        blocks = [
            (0, bytestr[:end1]),
            (size - end1 - end2, None),
            (size - end2, bytestr[size - end2 :]),
        ]

    ellipsis = b'...' if ellipsis is None else ellipsis.encode('cp1252')
    result = []
    for start, bytestr in blocks:
        if bytestr is None:
            result.append(ellipsis)  # 'skip %i bytes' % start)
            continue
        hexstr = binascii.hexlify(bytestr)
        strstr = re.sub(br'[^\x20-\x7f]', b'.', bytestr)
        for i in range(0, len(bytestr), bytesperline):
            h = hexstr[2 * i : 2 * i + bytesperline * 2]
            r = (addr % (i + start)) if height > 1 else addr
            r += b' '.join(h[i : i + 2] for i in range(0, 2 * bytesperline, 2))
            r += b' ' * (width - len(r))
            r += strstr[i : i + bytesperline]
            result.append(r)
    result = b'\n'.join(result)
    result = result.decode('ascii')
    return result


def isprintable(string):
    r"""Return if all characters in string are printable.

    >>> isprintable('abc')
    True
    >>> isprintable(b'\01')
    False

    """
    string = string.strip()
    if not string:
        return True
    try:
        return string.isprintable()
    except Exception:
        pass
    try:
        return string.decode().isprintable()
    except Exception:
        pass


def clean_whitespace(string, compact=False):
    """Return string with compressed whitespace."""
    string = (
        string.replace('\r\n', '\n')
        .replace('\r', '\n')
        .replace('\n\n', '\n')
        .replace('\t', ' ')
        .replace('  ', ' ')
    )
    if compact:
        string = (
            string.replace('\n', ' ')
            .replace('[ ', '[')
            .replace('  ', ' ')
            .replace('  ', ' ')
            .replace('  ', ' ')
        )
    return string.strip()


def pformat_xml(xml):
    """Return pretty formatted XML."""
    try:
        from lxml import etree  # delayed import

        if not isinstance(xml, bytes):
            xml = xml.encode()
        tree = etree.parse(io.BytesIO(xml))
        xml = etree.tostring(
            tree,
            pretty_print=True,
            xml_declaration=True,
            encoding=tree.docinfo.encoding,
        )
        xml = bytes2str(xml)
    except Exception:
        if isinstance(xml, bytes):
            xml = bytes2str(xml)
        xml = xml.replace('><', '>\n<')
    return xml.replace('  ', ' ').replace('\t', ' ')


def pformat(arg, width=79, height=24, compact=True):
    """Return pretty formatted representation of object as string.

    Whitespace might be altered.

    """
    if height is None or height < 1:
        height = 1024
    if width is None or width < 1:
        width = 256

    npopt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=100, linewidth=width)

    if isinstance(arg, bytes):
        if arg[:5].lower() == b'<?xml' or arg[-4:] == b'OME>':
            arg = bytes2str(arg)

    if isinstance(arg, bytes):
        if isprintable(arg):
            arg = bytes2str(arg)
            arg = clean_whitespace(arg)
        else:
            numpy.set_printoptions(**npopt)
            return hexdump(arg, width=width, height=height, modulo=1)
        arg = arg.rstrip()
    elif isinstance(arg, str):
        if arg[:5].lower() == '<?xml' or arg[-4:] == 'OME>':
            arg = arg[: 4 * width] if height == 1 else pformat_xml(arg)
        # too slow
        # else:
        #    import textwrap  # delayed import
        #    return '\n'.join(
        #        textwrap.wrap(arg, width=width, max_lines=height, tabsize=2)
        #    )
        arg = arg.rstrip()
    elif isinstance(arg, numpy.record):
        arg = arg.pprint()
    else:
        import pprint  # delayed import

        arg = pprint.pformat(arg, width=width, compact=compact)

    numpy.set_printoptions(**npopt)

    if height == 1:
        arg = arg[: width * width]
        arg = clean_whitespace(arg, compact=True)
        return arg[:width]

    argl = list(arg.splitlines())
    if len(argl) > height:
        arg = '\n'.join(
            line[:width]
            for line in argl[: height // 2] + ['...'] + argl[-height // 2 :]
        )
    else:
        arg = '\n'.join(line[:width] for line in argl[:height])
    return arg


def snipstr(string, width=79, snipat=None, ellipsis=None):
    """Return string cut to specified length.

    >>> snipstr('abcdefghijklmnop', 8)
    'abc...op'

    """
    if snipat is None:
        snipat = 0.5
    if ellipsis is None:
        if isinstance(string, bytes):
            ellipsis = b'...'
        else:
            ellipsis = '\u2026'
    esize = len(ellipsis)

    splitlines = string.splitlines()
    # TODO: finish and test multiline snip

    result = []
    for line in splitlines:
        if line is None:
            result.append(ellipsis)
            continue
        linelen = len(line)
        if linelen <= width:
            result.append(string)
            continue

        split = snipat
        if split is None or split == 1:
            split = linelen
        elif 0 < abs(split) < 1:
            split = int(math.floor(linelen * split))
        if split < 0:
            split += linelen
            if split < 0:
                split = 0

        if esize == 0 or width < esize + 1:
            if split <= 0:
                result.append(string[-width:])
            else:
                result.append(string[:width])
        elif split <= 0:
            result.append(ellipsis + string[esize - width :])
        elif split >= linelen or width < esize + 4:
            result.append(string[: width - esize] + ellipsis)
        else:
            splitlen = linelen - width + esize
            end1 = split - splitlen // 2
            end2 = end1 + splitlen
            result.append(string[:end1] + ellipsis + string[end2:])

    if isinstance(string, bytes):
        return b'\n'.join(result)
    return '\n'.join(result)


def enumstr(enum):
    """Return short string representation of Enum instance."""
    name = enum.name
    if name is None:
        name = str(enum)
    return name


def enumarg(enum, arg):
    """Return enum member from its name or value.

    >>> enumarg(TIFF.PHOTOMETRIC, 2)
    <PHOTOMETRIC.RGB: 2>
    >>> enumarg(TIFF.PHOTOMETRIC, 'RGB')
    <PHOTOMETRIC.RGB: 2>

    """
    try:
        return enum(arg)
    except Exception:
        try:
            return enum[arg.upper()]
        except Exception:
            raise ValueError(f'invalid argument {arg}')


def parse_kwargs(kwargs, *keys, **keyvalues):
    """Return dict with keys from keys|keyvals and values from kwargs|keyvals.

    Existing keys are deleted from kwargs.

    >>> kwargs = {'one': 1, 'two': 2, 'four': 4}
    >>> kwargs2 = parse_kwargs(kwargs, 'two', 'three', four=None, five=5)
    >>> kwargs == {'one': 1}
    True
    >>> kwargs2 == {'two': 2, 'four': 4, 'five': 5}
    True

    """
    result = {}
    for key in keys:
        if key in kwargs:
            result[key] = kwargs[key]
            del kwargs[key]
    for key, value in keyvalues.items():
        if key in kwargs:
            result[key] = kwargs[key]
            del kwargs[key]
        else:
            result[key] = value
    return result


def update_kwargs(kwargs, **keyvalues):
    """Update dict with keys and values if keys do not already exist.

    >>> kwargs = {'one': 1, }
    >>> update_kwargs(kwargs, one=None, two=2)
    >>> kwargs == {'one': 1, 'two': 2}
    True

    """
    for key, value in keyvalues.items():
        if key not in kwargs:
            kwargs[key] = value


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)


def validate_jhove(filename, jhove=None, ignore=None):
    """Validate TIFF file using jhove -m TIFF-hul.

    Raise ValueError if jhove outputs an error message unless the message
    contains one of the strings in 'ignore'.

    JHOVE does not support bigtiff or more than 50 IFDs.

    See `JHOVE TIFF-hul Module <http://jhove.sourceforge.net/tiff-hul.html>`_

    """
    import subprocess

    if ignore is None:
        ignore = ['More than 50 IFDs']
    if jhove is None:
        jhove = 'jhove'
    out = subprocess.check_output([jhove, filename, '-m', 'TIFF-hul'])
    if b'ErrorMessage: ' in out:
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(b'ErrorMessage: '):
                error = line[14:].decode()
                for i in ignore:
                    if i in error:
                        break
                else:
                    raise ValueError(error)
                break


def tiffcomment(arg, comment=None, index=None, code=None):
    """Return or replace ImageDescription value in first page of TIFF file."""
    if index is None:
        index = 0
    if code is None:
        code = 270
    mode = None if comment is None else 'r+b'
    with TiffFile(arg, mode=mode) as tif:
        tag = tif.pages[index].tags.get(code, None)
        if tag is None:
            raise ValueError(f'no {TIFF.TAGS[code]} tag found')
        if comment is None:
            return tag.value
        tag.overwrite(comment)


def tiff2fsspec(
    filename,
    url,
    out=None,
    key=None,
    series=None,
    level=None,
    chunkmode=None,
    version=None,
):
    """Write fsspec ReferenceFileSystem JSON from TIFF file."""
    if out is None:
        out = filename + '.json'
    with TiffFile(filename) as tif:
        with tif.aszarr(
            key=key, series=series, level=level, chunkmode=chunkmode
        ) as store:
            store.write_fsspec(out, url, version=version)


def lsm2bin(lsmfile, binfile=None, tile=None, verbose=True):
    """Convert [MP]TZCYX LSM file to series of BIN files.

    One BIN file containing 'ZCYX' data are created for each position, time,
    and tile. The position, time, and tile indices are encoded at the end
    of the filenames.

    """
    verbose = print if verbose else nullfunc

    if tile is None:
        tile = (256, 256)

    if binfile is None:
        binfile = lsmfile
    elif binfile.lower() == 'none':
        binfile = None
    if binfile:
        binfile += '_(z%ic%iy%ix%i)_m%%ip%%it%%03iy%%ix%%i.bin'

    verbose('\nOpening LSM file... ', end='', flush=True)
    timer = Timer()

    with TiffFile(lsmfile) as lsm:
        if not lsm.is_lsm:
            verbose('\n', lsm, flush=True)
            raise ValueError('not a LSM file')
        series = lsm.series[0]  # first series contains the image data
        shape = series.get_shape(False)
        axes = series.get_axes(False)
        dtype = series.dtype
        size = product(shape) * dtype.itemsize

        verbose(timer)
        # verbose(lsm, flush=True)
        verbose(
            'Image\n  axes:  {}\n  shape: {}\n  dtype: {}\n  size:  {}'.format(
                axes, shape, dtype, format_size(size)
            ),
            flush=True,
        )
        if not series.axes.endswith('TZCYX'):
            raise ValueError('not a *TZCYX LSM file')

        verbose('Copying image from LSM to BIN files', end='', flush=True)
        timer.start()
        tiles = shape[-2] // tile[-2], shape[-1] // tile[-1]
        if binfile:
            binfile = binfile % (shape[-4], shape[-3], tile[0], tile[1])
        shape = (1,) * (7 - len(shape)) + shape
        # cache for ZCYX stacks and output files
        data = numpy.empty(shape[3:], dtype=dtype)
        out = numpy.empty(
            (shape[-4], shape[-3], tile[0], tile[1]), dtype=dtype
        )
        # iterate over Tiff pages containing data
        pages = iter(series.pages)
        for m in range(shape[0]):  # mosaic axis
            for p in range(shape[1]):  # position axis
                for t in range(shape[2]):  # time axis
                    for z in range(shape[3]):  # z slices
                        data[z] = next(pages).asarray()
                    for y in range(tiles[0]):  # tile y
                        for x in range(tiles[1]):  # tile x
                            out[:] = data[
                                ...,
                                y * tile[0] : (y + 1) * tile[0],
                                x * tile[1] : (x + 1) * tile[1],
                            ]
                            if binfile:
                                out.tofile(binfile % (m, p, t, y, x))
                            verbose('.', end='', flush=True)
        verbose(timer, flush=True)


def imshow(
    data,
    photometric=None,
    planarconfig=None,
    bitspersample=None,
    nodata=0,
    interpolation=None,
    cmap=None,
    vmin=None,
    vmax=None,
    figure=None,
    title=None,
    dpi=96,
    subplot=None,
    maxdim=None,
    **kwargs,
):
    """Plot n-dimensional images using matplotlib.pyplot.

    Return figure, subplot, and plot axis.
    Requires pyplot already imported C{from matplotlib import pyplot}.

    Parameters
    ----------
    data : nd array
        The image data.
    photometric : {'MINISWHITE', 'MINISBLACK', 'RGB', or 'PALETTE'}
        The color space of the image data.
    planarconfig : {'CONTIG' or 'SEPARATE'}
        Defines how components of each pixel are stored.
    bitspersample : int
        Number of bits per channel in integer RGB images.
    interpolation : str
        The image interpolation method used in matplotlib.imshow. By default,
        'nearest' is used for image dimensions <= 512, else 'bilinear'.
    cmap : str or matplotlib.colors.Colormap
        The colormap maps non-RGBA scalar data to colors.
    vmin, vmax : scalar
        Data range covered by the colormap. By default, the complete
        range of the data is covered.
    figure : matplotlib.figure.Figure
        Matplotlib figure to use for plotting.
    title : str
        Window and subplot title.
    subplot : int
        A matplotlib.pyplot.subplot axis.
    maxdim : int
        Maximum image width and length.
    kwargs : dict
        Additional arguments for matplotlib.pyplot.imshow.

    """
    # TODO: rewrite detection of isrgb, iscontig
    # TODO: use planarconfig
    if photometric is None:
        photometric = 'RGB'
    if maxdim is None:
        maxdim = 2 ** 16
    isrgb = photometric in ('RGB', 'YCBCR')  # 'PALETTE', 'YCBCR'

    if data.dtype == 'float16':
        data = data.astype('float32')

    if data.dtype.kind == 'b':
        isrgb = False

    if isrgb and not (
        data.shape[-1] in (3, 4)
        or (data.ndim > 2 and data.shape[-3] in (3, 4))
    ):
        isrgb = False
        photometric = 'MINISBLACK'

    data = data.squeeze()
    if photometric in ('MINISWHITE', 'MINISBLACK', None):
        data = reshape_nd(data, 2)
    else:
        data = reshape_nd(data, 3)

    dims = data.ndim
    if dims < 2:
        raise ValueError('not an image')
    if dims == 2:
        dims = 0
        isrgb = False
    else:
        if isrgb and data.shape[-3] in (3, 4):
            data = numpy.swapaxes(data, -3, -2)
            data = numpy.swapaxes(data, -2, -1)
        elif not isrgb and (
            data.shape[-1] < data.shape[-2] // 8
            and data.shape[-1] < data.shape[-3] // 8
        ):
            data = numpy.swapaxes(data, -3, -1)
            data = numpy.swapaxes(data, -2, -1)
        isrgb = isrgb and data.shape[-1] in (3, 4)
        dims -= 3 if isrgb else 2

    if interpolation is None:
        threshold = 512
    elif isinstance(interpolation, int):
        threshold = interpolation
    else:
        threshold = 0

    if isrgb:
        data = data[..., :maxdim, :maxdim, :maxdim]
        if threshold:
            if data.shape[-2] > threshold or data.shape[-3] > threshold:
                interpolation = 'bilinear'
            else:
                interpolation = 'nearest'
    else:
        data = data[..., :maxdim, :maxdim]
        if threshold:
            if data.shape[-1] > threshold or data.shape[-2] > threshold:
                interpolation = 'bilinear'
            else:
                interpolation = 'nearest'

    if photometric == 'PALETTE' and isrgb:
        try:
            datamax = numpy.max(data)
        except ValueError:
            datamax = 1
        if datamax > 255:
            data = data >> 8  # possible precision loss
        data = data.astype('B')
    elif data.dtype.kind in 'ui':
        if not (isrgb and data.dtype.itemsize <= 1) or bitspersample is None:
            try:
                bitspersample = int(math.ceil(math.log(data.max(), 2)))
            except Exception:
                bitspersample = data.dtype.itemsize * 8
        elif not isinstance(bitspersample, (int, numpy.integer)):
            # bitspersample can be tuple, e.g. (5, 6, 5)
            bitspersample = data.dtype.itemsize * 8
        datamax = 2 ** bitspersample
        if isrgb:
            if bitspersample < 8:
                data = data << (8 - bitspersample)
            elif bitspersample > 8:
                data = data >> (bitspersample - 8)  # precision loss
            data = data.astype('B')
    elif data.dtype.kind == 'f':
        if nodata:
            data = data.copy()
            data[data > 1e30] = 0.0
        try:
            datamax = numpy.max(data)
        except ValueError:
            datamax = 1
        if isrgb and datamax > 1.0:
            if data.dtype.char == 'd':
                data = data.astype('f')
                data /= datamax
            else:
                data = data / datamax
    elif data.dtype.kind == 'b':
        datamax = 1
    elif data.dtype.kind == 'c':
        data = numpy.absolute(data)
        try:
            datamax = numpy.max(data)
        except ValueError:
            datamax = 1

    if isrgb:
        vmin = 0
    else:
        if vmax is None:
            vmax = datamax
        if vmin is None:
            if data.dtype.kind == 'i':
                dtmin = numpy.iinfo(data.dtype).min
                try:
                    vmin = numpy.min(data)
                except ValueError:
                    vmin = -1
                if vmin == dtmin:
                    vmin = numpy.min(data[data > dtmin])
            elif data.dtype.kind == 'f':
                dtmin = numpy.finfo(data.dtype).min
                try:
                    vmin = numpy.min(data)
                except ValueError:
                    vmin = 0.0
                if vmin == dtmin:
                    vmin = numpy.min(data[data > dtmin])
            else:
                vmin = 0

    pyplot = sys.modules['matplotlib.pyplot']

    if figure is None:
        pyplot.rc('font', family='sans-serif', weight='normal', size=8)
        figure = pyplot.figure(
            dpi=dpi,
            figsize=(10.3, 6.3),
            frameon=True,
            facecolor='1.0',
            edgecolor='w',
        )
        try:
            figure.canvas.manager.window.title(title)
        except Exception:
            pass
        size = len(title.splitlines()) if title else 1
        pyplot.subplots_adjust(
            bottom=0.03 * (dims + 2),
            top=0.98 - size * 0.03,
            left=0.1,
            right=0.95,
            hspace=0.05,
            wspace=0.0,
        )
    if subplot is None:
        subplot = 111
    subplot = pyplot.subplot(subplot)
    subplot.set_facecolor((0, 0, 0))

    if title:
        try:
            title = str(title, 'Windows-1252')
        except TypeError:
            pass
        pyplot.title(title, size=11)

    if cmap is None:
        if data.dtype.char == '?':
            cmap = 'gray'
        elif data.dtype.kind in 'buf' or vmin == 0:
            cmap = 'viridis'
        else:
            cmap = 'coolwarm'
        if photometric == 'MINISWHITE':
            cmap += '_r'

    image = pyplot.imshow(
        numpy.atleast_2d(data[(0,) * dims].squeeze()),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        **kwargs,
    )

    if not isrgb:
        pyplot.colorbar()  # panchor=(0.55, 0.5), fraction=0.05

    def format_coord(x, y):
        # callback function to format coordinate display in toolbar
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            if dims:
                return f'{curaxdat[1][y, x]} @ {current} [{y:4}, {x:4}]'
            return f'{data[y, x]} @ [{y:4}, {x:4}]'
        except IndexError:
            return ''

    def none(event):
        return ''

    subplot.format_coord = format_coord
    image.get_cursor_data = none
    image.format_cursor_data = none

    if dims:
        current = list((0,) * dims)
        curaxdat = [0, data[tuple(current)].squeeze()]
        sliders = [
            pyplot.Slider(
                pyplot.axes([0.125, 0.03 * (axis + 1), 0.725, 0.025]),
                f'Dimension {axis}',
                0,
                data.shape[axis] - 1,
                0,
                facecolor='0.5',
                valfmt=f'%.0f [{data.shape[axis]}]',
            )
            for axis in range(dims)
        ]
        for slider in sliders:
            slider.drawon = False

        def set_image(current, sliders=sliders, data=data):
            # change image and redraw canvas
            curaxdat[1] = data[tuple(current)].squeeze()
            image.set_data(curaxdat[1])
            for ctrl, index in zip(sliders, current):
                ctrl.eventson = False
                ctrl.set_val(index)
                ctrl.eventson = True
            figure.canvas.draw()

        def on_changed(index, axis, data=data, current=current):
            # callback function for slider change event
            index = int(round(index))
            curaxdat[0] = axis
            if index == current[axis]:
                return
            if index >= data.shape[axis]:
                index = 0
            elif index < 0:
                index = data.shape[axis] - 1
            current[axis] = index
            set_image(current)

        def on_keypressed(event, data=data, current=current):
            # callback function for key press event
            key = event.key
            axis = curaxdat[0]
            if str(key) in '0123456789':
                on_changed(key, axis)
            elif key == 'right':
                on_changed(current[axis] + 1, axis)
            elif key == 'left':
                on_changed(current[axis] - 1, axis)
            elif key == 'up':
                curaxdat[0] = 0 if axis == len(data.shape) - 1 else axis + 1
            elif key == 'down':
                curaxdat[0] = len(data.shape) - 1 if axis == 0 else axis - 1
            elif key == 'end':
                on_changed(data.shape[axis] - 1, axis)
            elif key == 'home':
                on_changed(0, axis)

        figure.canvas.mpl_connect('key_press_event', on_keypressed)
        for axis, ctrl in enumerate(sliders):
            ctrl.on_changed(lambda k, a=axis: on_changed(k, a))

    return figure, subplot, image


def _app_show():
    """Block the GUI. For use as skimage plugin."""
    pyplot = sys.modules['matplotlib.pyplot']
    pyplot.show()


def askopenfilename(**kwargs):
    """Return file name(s) from Tkinter's file open dialog."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames


def main():
    """Tifffile command line usage main function."""
    import logging
    import optparse  # TODO: use argparse

    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = optparse.OptionParser(
        usage='usage: %prog [options] path',
        description='Display image data in TIFF files.',
        version=f'%prog {__version__}',
        prog='tifffile',
    )
    opt = parser.add_option
    opt(
        '-p',
        '--page',
        dest='page',
        type='int',
        default=-1,
        help='display single page',
    )
    opt(
        '-s',
        '--series',
        dest='series',
        type='int',
        default=-1,
        help='display series of pages of same shape',
    )
    opt(
        '-l',
        '--level',
        dest='level',
        type='int',
        default=-1,
        help='display pyramid level of series',
    )
    opt(
        '--nomultifile',
        dest='nomultifile',
        action='store_true',
        default=False,
        help='do not read OME series from multiple files',
    )
    opt(
        '--noplots',
        dest='noplots',
        type='int',
        default=10,
        help='maximum number of plots',
    )
    opt(
        '--interpol',
        dest='interpol',
        metavar='INTERPOL',
        default=None,
        help='image interpolation method',
    )
    opt('--dpi', dest='dpi', type='int', default=96, help='plot resolution')
    opt(
        '--vmin',
        dest='vmin',
        type='int',
        default=None,
        help='minimum value for colormapping',
    )
    opt(
        '--vmax',
        dest='vmax',
        type='int',
        default=None,
        help='maximum value for colormapping',
    )
    opt(
        '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='raise exception on failures',
    )
    opt(
        '--doctest',
        dest='doctest',
        action='store_true',
        default=False,
        help='runs the docstring examples',
    )
    opt('-v', '--detail', dest='detail', type='int', default=2)
    opt('-q', '--quiet', dest='quiet', action='store_true')

    settings, path = parser.parse_args()
    path = ' '.join(path)

    if settings.doctest:
        import doctest

        try:
            import tifffile.tifffile as m
        except ImportError:
            m = None
        doctest.testmod(m, optionflags=doctest.ELLIPSIS)
        return 0
    if not path:
        path = askopenfilename(
            title='Select a TIFF file', filetypes=TIFF.FILEOPEN_FILTER
        )
        if not path:
            parser.error('No file specified')

    if any(i in path for i in '?*'):
        path = glob.glob(path)
        if not path:
            print('No files match the pattern')
            return 0
        # TODO: handle image sequences
        path = path[0]

    if not settings.quiet:
        print('\nReading TIFF header:', end=' ', flush=True)
    timer = Timer()
    try:
        tif = TiffFile(path, _multifile=not settings.nomultifile)
    except Exception as exc:
        if settings.debug:
            raise
        print(f'\n\n{exc.__class__.__name__}: {exc}')
        sys.exit(0)

    if not settings.quiet:
        print(timer)

    if tif.is_ome:
        settings.norgb = True

    images = []
    if settings.noplots > 0:
        if not settings.quiet:
            print('Reading image data: ', end=' ', flush=True)

        def notnone(x):
            return next(i for i in x if i is not None)

        timer.start()
        try:
            if settings.page >= 0:
                images = [
                    (
                        tif.asarray(key=settings.page),
                        tif.pages[settings.page],
                        None,
                    )
                ]
            elif settings.series >= 0:
                series = tif.series[settings.series]
                if settings.level >= 0:
                    level = settings.level
                elif series.is_pyramidal and product(series.shape) > 2 ** 32:
                    level = -1
                    for r in series.levels:
                        level += 1
                        if product(r.shape) < 2 ** 32:
                            break
                else:
                    level = 0
                images = [
                    (
                        tif.asarray(series=settings.series, level=level),
                        notnone(tif.series[settings.series]._pages),
                        tif.series[settings.series],
                    )
                ]
            else:
                for i, s in enumerate(tif.series[: settings.noplots]):
                    if settings.level < 0:
                        level = -1
                        for r in s.levels:
                            level += 1
                            if product(r.shape) < 2 ** 31:
                                break
                    else:
                        level = 0
                    try:
                        images.append(
                            (
                                tif.asarray(series=i, level=level),
                                notnone(s._pages),
                                tif.series[i],
                            )
                        )
                    except Exception as exc:
                        images.append((None, notnone(s.pages), None))
                        if settings.debug:
                            raise
                        print(
                            '\nSeries {} failed with {}: {}... '.format(
                                i, exc.__class__.__name__, exc
                            ),
                            end='',
                        )
        except Exception as exc:
            if settings.debug:
                raise
            print(f'{exc.__class__.__name__}: {exc}')

        if not settings.quiet:
            print(timer)

    if not settings.quiet:
        print('Generating report:', end='   ', flush=True)
        timer.start()
        info = TiffFile.__str__(tif, detail=int(settings.detail))
        print(timer)
        print()
        print(info)
        print()
    tif.close()

    if images and settings.noplots > 0:
        try:
            import matplotlib

            matplotlib.use('TkAgg')
            from matplotlib import pyplot
        except ImportError as exc:
            log_warning(f'<tifffile.main> {exc.__class__.__name__}: {exc}')
        else:
            for img, page, series in images:
                if img is None:
                    continue
                keyframe = page.keyframe
                vmin, vmax = settings.vmin, settings.vmax
                if keyframe.nodata:
                    try:
                        vmin = numpy.min(img[img > keyframe.nodata])
                    except ValueError:
                        pass
                if tif.is_stk:
                    try:
                        vmin = tif.stk_metadata['MinScale']
                        vmax = tif.stk_metadata['MaxScale']
                    except KeyError:
                        pass
                    else:
                        if vmax <= vmin:
                            vmin, vmax = settings.vmin, settings.vmax
                if series:
                    title = f'{tif}\n{page}\n{series}'
                else:
                    title = f'{tif}\n {page}'
                photometric = 'MINISBLACK'
                if keyframe.photometric not in (3,):
                    photometric = TIFF.PHOTOMETRIC(keyframe.photometric).name
                imshow(
                    img,
                    title=title,
                    vmin=vmin,
                    vmax=vmax,
                    bitspersample=keyframe.bitspersample,
                    nodata=keyframe.nodata,
                    photometric=photometric,
                    interpolation=settings.interpol,
                    dpi=settings.dpi,
                )
            pyplot.show()
    return 0


def bytes2str(b, encoding=None, errors='strict'):
    """Return Unicode string from encoded bytes."""
    if encoding is not None:
        return b.decode(encoding, errors)
    try:
        return b.decode('utf-8', errors)
    except UnicodeDecodeError:
        return b.decode('cp1252', errors)


def bytestr(s, encoding='cp1252'):
    """Return bytes from Unicode string, else pass through."""
    return s.encode(encoding) if isinstance(s, str) else s


# aliases and deprecated
imsave = imwrite
TiffWriter.save = TiffWriter.write
TiffReader = TiffFile

if __name__ == '__main__':
    sys.exit(main())
