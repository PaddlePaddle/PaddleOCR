"""
A set of objects representing each file extension recognized by ImageIO. If an
extension is not listed here it is still supported, as long as there exists a
supporting backend.

"""

from typing import List, Dict


class FileExtension:
    """File Extension Metadata

    This class holds information about a image file format associated with a
    given extension. This information is used to track plugins that are known to
    be able to handle a particular format. It also contains additional
    information about a format, which is used when creating the supported format
    docs.

    Plugins known to be able to handle this format are ordered by a ``priority``
    list. This list is used to determine the ideal plugin to use when choosing a
    plugin based on file extension.

    Parameters
    ----------
    extension : str
        The name of the extension including the initial dot, e.g. ".png".
    priority : List
        A list of plugin names (entries in config.known_plugins) that can handle
        this format. The position of a plugin expresses a preference, e.g.
        ["plugin1", "plugin2"] indicates that, if available, plugin1 should be
        preferred over plugin2 when handling a request related to this format.
    name : str
        The full name of the format.
    description : str
        A description of the format.
    external_link : str
        A link to further information about the format. Typically, the format's
        specification.

    Examples
    --------
    >>> FileExtension(
            name="Bitmap",
            extension=".bmp",
            priority=["pillow", "BMP-PIL", "BMP-FI", "ITK"],
            external_link="https://en.wikipedia.org/wiki/BMP_file_format",
        )

    """

    def __init__(
        self,
        *,
        extension: str,
        priority: List[str],
        name: str = None,
        description: str = None,
        external_link: str = None
    ) -> None:
        self.extension = extension
        self.priority = priority
        self.name = name
        self.description = description
        self.external_link = external_link
        self.default_priority = priority.copy()

    def reset(self) -> None:
        self.priority = self.default_priority.copy()


extension_list = [
    FileExtension(
        name="Hasselblad raw",
        extension=".3fr",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Sony alpha",
        extension=".arw",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Audio Video Interleave",
        extension=".avi",
        priority=["FFMPEG"],
    ),
    FileExtension(
        name="Casio raw format",
        extension=".bay",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Bitmap",
        extension=".bmp",
        priority=["pillow", "BMP-PIL", "BMP-FI", "ITK"],
        external_link="https://en.wikipedia.org/wiki/BMP_file_format",
    ),
    FileExtension(
        name="Re-Volt mipmap",
        extension=".bmq",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Binary Structured Data Format",
        extension=".bsdf",
        priority=["BSDF"],
        external_link="http://bsdf.io/",
    ),
    FileExtension(
        name="Binary Universal Form for the Representation of meteorological data",
        extension=".bufr",
        priority=["pillow", "BUFR-PIL"],
    ),
    FileExtension(
        name="Silicon Graphics Image",
        extension=".bw",
        priority=["pillow", "SGI-PIL", "SGI-FI"],
    ),
    FileExtension(
        name="Scirra Construct",
        extension=".cap",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="AMETEK High Speed Camera Format",
        extension=".cine",
        priority=["RAW-FI"],
        external_link="https://phantomhighspeed-knowledge.secure.force.com/servlet/fileField?id=0BE1N000000kD2i#:~:text=Cine%20is%20a%20video%20file,camera%20model%20and%20image%20resolution",
    ),
    FileExtension(extension=".cr2", priority=["RAW-FI"]),
    FileExtension(
        extension=".crw",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".cs1",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Computerized Tomography",
        extension=".ct",
        priority=["DICOM"],
    ),
    FileExtension(
        name="Windows Cursor Icons",
        extension=".cur",
        priority=["pillow", "CUR-PIL"],
    ),
    FileExtension(
        name="Dr. Halo",
        extension=".cut",
        priority=["CUT-FI"],
    ),
    FileExtension(
        extension=".dc2",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="DICOM file format",
        extension=".dcm",
        priority=["DICOM", "ITK"],
    ),
    FileExtension(
        extension=".dcr",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Intel DCX",
        extension=".dcx",
        priority=["pillow", "DCX-PIL"],
    ),
    FileExtension(
        name="DirectX Texture Container",
        extension=".dds",
        priority=["pillow", "DDS-FI", "DDS-PIL"],
    ),
    FileExtension(
        name="Windows Bitmap",
        extension=".DIB",
        priority=["pillow", "DIB-PIL"],
    ),
    FileExtension(
        name="DICOM file format",
        extension=".dicom",
        priority=["ITK"],
    ),
    FileExtension(
        extension=".dng",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".drf",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".dsc",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Enhanced Compression Wavelet",
        extension=".ecw",
        priority=["GDAL"],
    ),
    FileExtension(
        name="Windows Metafile",
        extension=".emf",
        priority=["pillow", "WMF-PIL"],
    ),
    FileExtension(
        name="Encapsulated Postscript",
        extension=".eps",
        priority=["pillow", "EPS-PIL"],
    ),
    FileExtension(
        extension=".erf",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="ILM OpenEXR",
        extension=".exr",
        priority=["EXR-FI"],
    ),
    FileExtension(
        extension=".fff",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Flexible Image Transport System File",
        extension=".fit",
        priority=["pillow", "FITS-PIL", "FITS"],
    ),
    FileExtension(
        name="Flexible Image Transport System File",
        extension=".fits",
        priority=["pillow", "FITS-PIL", "FITS"],
    ),
    FileExtension(
        name="Autodesk FLC Animation",
        extension=".flc",
        priority=["pillow", "FLI-PIL"],
    ),
    FileExtension(
        name="Autodesk FLI Animation",
        extension=".fli",
        priority=["pillow", "FLI-PIL"],
    ),
    FileExtension(
        name="Kodak FlashPix",
        extension=".fpx",
        priority=["pillow", "FPX-PIL"],
    ),
    FileExtension(
        name="Independence War 2: Edge Of Chaos Texture Format",
        extension=".ftc",
        priority=["pillow", "FTEX-PIL"],
    ),
    FileExtension(
        name="Flexible Image Transport System File",
        extension=".fts",
        priority=["FITS"],
    ),
    FileExtension(
        name="Independence War 2: Edge Of Chaos Texture Format",
        extension=".ftu",
        priority=["pillow", "FTEX-PIL"],
    ),
    FileExtension(
        name="Flexible Image Transport System File",
        extension=".fz",
        priority=["FITS"],
    ),
    FileExtension(
        name="Raw fax format CCITT G.3",
        extension=".g3",
        priority=["G3-FI"],
    ),
    FileExtension(
        name="GIMP brush file",
        extension=".gbr",
        priority=["pillow", "GBR-PIL"],
    ),
    FileExtension(
        name="Grassroots DICOM",
        extension=".gdcm",
        priority=["ITK"],
    ),
    FileExtension(
        name="Graphics Interchange Format",
        extension=".gif",
        priority=["pillow", "GIF-PIL"],
    ),
    FileExtension(
        name="UMDS GIPL",
        extension=".gipl",
        priority=["ITK"],
    ),
    FileExtension(
        name="gridded meteorological data",
        extension=".grib",
        priority=["pillow", "GRIB-PIL"],
    ),
    FileExtension(
        name="Hierarchical Data Format 5",
        extension=".h5",
        priority=["pillow", "HDF5-PIL"],
    ),
    FileExtension(
        name="Hierarchical Data Format 5",
        extension=".hdf",
        priority=["pillow", "HDF5-PIL"],
    ),
    FileExtension(
        name="Hierarchical Data Format 5",
        extension=".hdf5",
        priority=["ITK"],
    ),
    FileExtension(
        name="JPEG Extended Range",
        extension=".hdp",
        priority=["JPEG-XR-FI"],
    ),
    FileExtension(
        name="High Dynamic Range Image",
        extension=".hdr",
        priority=["HDR-FI", "ITK"],
    ),
    FileExtension(
        extension=".ia",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Mac OS Icon File",
        extension=".icns",
        priority=["pillow", "ICNS-PIL"],
    ),
    FileExtension(
        name="Windows Icon File",
        extension=".ico",
        priority=["pillow", "ICO-FI", "ICO-PIL"],
    ),
    FileExtension(
        name="ILBM Interleaved Bitmap",
        extension=".iff",
        priority=["IFF-FI"],
    ),
    FileExtension(
        name="IPTC/NAA",
        extension=".iim",
        priority=["pillow", "IPTC-PIL"],
    ),
    FileExtension(
        extension=".iiq",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="IFUNC Image Memory",
        extension=".im",
        priority=["pillow", "IM-PIL"],
    ),
    FileExtension(
        extension=".img",
        priority=["ITK", "GDAL"],
    ),
    FileExtension(
        extension=".img.gz",
        priority=["ITK"],
    ),
    FileExtension(
        name="IM Tools",
        extension=".IMT",
        priority=["pillow", "IMT-PIL"],
    ),
    FileExtension(
        name="Image Processing Lab",
        extension=".ipl",
        priority=["ITK"],
    ),
    FileExtension(
        name="JPEG 2000",
        extension=".j2c",
        priority=["pillow", "J2K-FI", "JPEG2000-PIL"],
    ),
    FileExtension(
        name="JPEG 2000",
        extension=".j2k",
        priority=["pillow", "J2K-FI", "JPEG2000-PIL"],
    ),
    FileExtension(
        name="JPEG",
        extension=".jfif",
        priority=["pillow", "JPEG-PIL"],
    ),
    FileExtension(
        name="JPEG",
        extension=".jif",
        priority=["JPEG-FI"],
    ),
    FileExtension(
        name="JPEG Network Graphics",
        extension=".jng",
        priority=["JNG-FI"],
    ),
    FileExtension(
        name="JPEG 2000",
        extension=".jp2",
        priority=["pillow", "JP2-FI", "JPEG2000-PIL"],
    ),
    FileExtension(
        name="JPEG 2000",
        extension=".jpc",
        priority=["pillow", "JPEG2000-PIL"],
    ),
    FileExtension(
        name="JPEG",
        extension=".jpe",
        priority=["pillow", "JPEG-FI", "JPEG-PIL"],
    ),
    FileExtension(
        name="Joint Photographic Experts Group",
        extension=".jpeg",
        priority=["pillow", "JPEG-PIL", "JPEG-FI", "ITK", "GDAL"],
    ),
    FileExtension(
        name="JPEG 2000",
        extension=".jpf",
        priority=["pillow", "JPEG2000-PIL"],
    ),
    FileExtension(
        name="Joint Photographic Experts Group",
        extension=".jpg",
        priority=["pillow", "JPEG-PIL", "JPEG-FI", "ITK", "GDAL"],
    ),
    FileExtension(
        name="JPEG 2000",
        extension=".jpx",
        priority=["pillow", "JPEG2000-PIL"],
    ),
    FileExtension(
        name="JPEG Extended Range",
        extension=".jxr",
        priority=["JPEG-XR-FI"],
    ),
    FileExtension(
        extension=".k25",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".kc2",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".kdc",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="C64 Koala Graphics",
        extension=".koa",
        priority=["KOALA-FI"],
    ),
    FileExtension(
        name="ILBM Interleaved Bitmap",
        extension=".lbm",
        priority=["IFF-FI"],
    ),
    FileExtension(
        name="Lytro F01",
        extension=".lfp",
        priority=["LYTRO-LFP"],
    ),
    FileExtension(
        name="Lytro Illum",
        extension=".lfr",
        priority=["LYTRO-LFR"],
    ),
    FileExtension(
        name="ZEISS LSM",
        extension=".lsm",
        priority=["ITK", "TIFF"],
    ),
    FileExtension(
        name="McIdas area file",
        extension=".MCIDAS",
        priority=["pillow", "MCIDAS-PIL"],
        external_link="https://www.ssec.wisc.edu/mcidas/doc/prog_man/2003print/progman2003-formats.html",
    ),
    FileExtension(
        extension=".mdc",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".mef",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="FreeSurfer File Format",
        extension=".mgh",
        priority=["ITK"],
    ),
    FileExtension(
        name="ITK MetaImage",
        extension=".mha",
        priority=["ITK"],
    ),
    FileExtension(
        name="ITK MetaImage Header",
        extension=".mhd",
        priority=["ITK"],
    ),
    FileExtension(
        name="Microsoft Image Composer",
        extension=".mic",
        priority=["pillow", "MIC-PIL"],
    ),
    FileExtension(
        name="Matroska Multimedia Container",
        extension=".mkv",
        priority=["FFMPEG"],
    ),
    FileExtension(
        name="Medical Imaging NetCDF",
        extension=".mnc",
        priority=["ITK"],
    ),
    FileExtension(
        name="Medical Imaging NetCDF 2",
        extension=".mnc2",
        priority=["ITK"],
    ),
    FileExtension(
        name="Leaf Raw Image Format",
        extension=".mos",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="QuickTime File Format",
        extension=".mov",
        priority=["FFMPEG"],
    ),
    FileExtension(
        name="MPEG-4 Part 14",
        extension=".mp4",
        priority=["FFMPEG"],
    ),
    FileExtension(
        name="Moving Picture Experts Group",
        extension=".mpeg",
        priority=["FFMPEG"],
    ),
    FileExtension(
        name="Moving Picture Experts Group",
        extension=".mpg",
        priority=["FFMPEG"],
    ),
    FileExtension(
        name="JPEG Multi-Picture Format",
        extension=".mpo",
        priority=["pillow", "MPO-PIL"],
    ),
    FileExtension(
        name="Magnetic resonance imaging",
        extension=".mri",
        priority=["DICOM"],
    ),
    FileExtension(
        extension=".mrw",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Windows Paint",
        extension=".msp",
        priority=["pillow", "MSP-PIL"],
    ),
    FileExtension(
        extension=".nef",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".nhdr",
        priority=["ITK"],
    ),
    FileExtension(
        extension=".nia",
        priority=["ITK"],
    ),
    FileExtension(
        extension=".nii",
        priority=["ITK"],
    ),
    FileExtension(
        name="nii.gz",
        extension=".nii.gz",
        priority=["ITK"],
    ),
    FileExtension(
        name="Numpy Array",
        extension=".npz",
        priority=["NPZ"],
    ),
    FileExtension(
        extension=".nrrd",
        priority=["ITK"],
    ),
    FileExtension(
        extension=".nrw",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".orf",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Portable Bitmap",
        extension=".pbm",
        priority=["PGM-FI", "PGMRAW-FI"],
    ),
    FileExtension(
        name="Kodak PhotoCD",
        extension=".pcd",
        priority=["pillow", "PCD-FI", "PCD-PIL"],
    ),
    FileExtension(
        name="Macintosh PICT",
        extension=".pct",
        priority=["PICT-FI"],
    ),
    FileExtension(
        name="Zsoft Paintbrush",
        extension=".PCX",
        priority=["pillow", "PCX-FI", "PCX-PIL"],
    ),
    FileExtension(
        extension=".pef",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".pfm",
        priority=["PFM-FI"],
    ),
    FileExtension(
        name="Portable Greymap",
        extension=".pgm",
        priority=["PGM-FI", "PGMRAW-FI"],
    ),
    FileExtension(
        name="Macintosh PICT",
        extension=".pic",
        priority=["PICT-FI", "ITK"],
    ),
    FileExtension(
        name="Macintosh PICT",
        extension=".pict",
        priority=["PICT-FI"],
    ),
    FileExtension(
        name="Portable Network Graphics",
        extension=".png",
        priority=["pillow", "PNG-PIL", "PNG-FI", "ITK"],
    ),
    FileExtension(
        name="Pbmplus image",
        extension=".ppm",
        priority=["pillow", "PPM-PIL"],
    ),
    FileExtension(
        name="Pbmplus image",
        extension=".pbm",
        priority=["pillow", "PPM-PIL"],
    ),
    FileExtension(
        name="Pbmplus image",
        extension=".pbm",
        priority=["pillow", "PPM-PIL", "PPM-FI"],
    ),
    FileExtension(
        name="Portable Pixelmap (ASCII)",
        extension=".ppm",
        priority=["PPM-FI"],
    ),
    FileExtension(
        name="Portable Pixelmap (Raw)",
        extension=".ppm",
        priority=["PPMRAW-FI"],
    ),
    FileExtension(
        name="Ghostscript",
        extension=".ps",
        priority=["pillow", "EPS-PIL"],
    ),
    FileExtension(
        name="Adope Photoshop 2.5 and 3.0",
        extension=".psd",
        priority=["pillow", "PSD-PIL", "PSD-FI"],
    ),
    FileExtension(
        extension=".ptx",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".pxn",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="PIXAR raster image",
        extension=".pxr",
        priority=["pillow", "PIXAR-PIL"],
    ),
    FileExtension(
        extension=".qtk",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".raf",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Sun Raster File",
        extension=".ras",
        priority=["pillow", "SUN-PIL", "RAS-FI"],
    ),
    FileExtension(
        extension=".raw",
        priority=["RAW-FI", "LYTRO-ILLUM-RAW", "LYTRO-F01-RAW"],
    ),
    FileExtension(
        extension=".rdc",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Silicon Graphics Image",
        extension=".rgb",
        priority=["pillow", "SGI-PIL"],
    ),
    FileExtension(
        name="Silicon Graphics Image",
        extension=".rgba",
        priority=["pillow", "SGI-PIL"],
    ),
    FileExtension(
        extension=".rw2",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".rwl",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".rwz",
        priority=["RAW-FI"],
    ),
    FileExtension(
        name="Silicon Graphics Image",
        extension=".sgi",
        priority=["pillow", "SGI-PIL"],
    ),
    FileExtension(
        name="SPE File Format",
        extension=".spe",
        priority=["SPE"],
    ),
    FileExtension(
        extension=".SPIDER",
        priority=["pillow", "SPIDER-PIL"],
    ),
    FileExtension(
        extension=".sr2",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".srf",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".srw",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".sti",
        priority=["RAW-FI"],
    ),
    FileExtension(
        extension=".stk",
        priority=["TIFF"],
    ),
    FileExtension(
        name="Shockwave Flash",
        extension=".swf",
        priority=["SWF"],
    ),
    FileExtension(
        name="Truevision TGA",
        extension=".targa",
        priority=["TARGA-FI"],
    ),
    FileExtension(
        name="Truevision TGA",
        extension=".tga",
        priority=["pillow", "TGA-PIL", "TARGA-FI"],
    ),
    FileExtension(
        name="Tagged Image File",
        extension=".tif",
        priority=["TIFF", "pillow", "TIFF-PIL", "TIFF-FI", "FEI", "ITK", "GDAL"],
    ),
    FileExtension(
        name="Tagged Image File Format",
        extension=".tiff",
        priority=["TIFF", "pillow", "TIFF-PIL", "TIFF-FI", "FEI", "ITK", "GDAL"],
    ),
    FileExtension(
        extension=".vtk",
        priority=["ITK"],
    ),
    FileExtension(
        name="Wireless Bitmap",
        extension=".wap",
        priority=["WBMP-FI"],
    ),
    FileExtension(
        name="Wireless Bitmap",
        extension=".wbm",
        priority=["WBMP-FI"],
    ),
    FileExtension(
        name="Wireless Bitmap",
        extension=".wbmp",
        priority=["WBMP-FI"],
    ),
    FileExtension(
        name="JPEG Extended Range",
        extension=".wdp",
        priority=["JPEG-XR-FI"],
    ),
    FileExtension(
        extension=".webm",
        priority=["FFMPEG"],
    ),
    FileExtension(
        name="Google WebP",
        extension=".webp",
        priority=["WEBP-FI"],
    ),
    FileExtension(
        name="Windows Meta File",
        extension=".wmf",
        priority=["pillow", "WMF-PIL"],
    ),
    FileExtension(
        name="Windows Media Video",
        extension=".wmv",
        priority=["FFMPEG"],
    ),
    FileExtension(
        name="X11 Bitmap",
        extension=".xbm",
        priority=["pillow", "XBM-PIL", "XBM-FI"],
    ),
    FileExtension(
        name="X11 Pixel Map",
        extension=".xpm",
        priority=["pillow", "XPM-PIL", "XPM-FI"],
    ),
    FileExtension(
        name="Thumbnail Image",
        extension=".XVTHUMB",
        priority=["pillow", "XVTHUMB-PIL"],
    ),
]
extension_list.sort(key=lambda x: x.extension)


known_extensions: Dict[str, List[FileExtension]] = dict()
for ext in extension_list:
    if ext.extension not in known_extensions:
        known_extensions[ext.extension] = list()
    known_extensions[ext.extension].append(ext)

extension_list = [ext for ext_list in known_extensions.values() for ext in ext_list]

_video_extension_strings = [
    ".avi",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
    ".gif",
]
video_extensions: List[FileExtension] = list()
for ext_string in _video_extension_strings:
    formats = known_extensions[ext_string]
    video_extensions.append(formats[0])
video_extensions.sort(key=lambda x: x.extension)
