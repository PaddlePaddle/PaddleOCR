from typing import Any
import importlib

from ..core.legacy_plugin_wrapper import LegacyPlugin


class PluginConfig:
    """Plugin Configuration Metadata

    This class holds the information needed to lazy-import plugins.

    Parameters
    ----------
    name : str
        The name of the plugin.
    class_name : str
        The name of the plugin class inside the plugin module.
    module_name : str
        The name of the module/package from which to import the plugin.
    is_legacy : bool
        If True, this plugin is a v2 plugin and will be wrapped in a
        LegacyPlugin. Default: False.
    package_name : str
        If the given module name points to a relative module, then the package
        name determines the package it is relative to.
    install_name : str
        The name of the optional dependency that can be used to install this
        plugin if it is missing.
    legacy_args : Dict
        A dictionary of kwargs to pass to the v2 plugin (Format) upon construction.

    Examples
    --------
    >>> PluginConfig(
            name="TIFF",
            class_name="TiffFormat",
            module_name="imageio.plugins.tifffile",
            is_legacy=True,
            install_name="tifffile",
            legacy_args={
                "description": "TIFF format",
                "extensions": ".tif .tiff .stk .lsm",
                "modes": "iIvV",
            },
        )
    >>> PluginConfig(
            name="pillow",
            class_name="PillowPlugin",
            module_name="imageio.plugins.pillow"
        )

    """

    def __init__(
        self,
        name: str,
        class_name: str,
        module_name: str,
        *,
        is_legacy: bool = False,
        package_name: str = None,
        install_name: str = None,
        legacy_args: dict = None,
    ) -> None:
        legacy_args = legacy_args or dict()

        self.name = name
        self.class_name = class_name
        self.module_name = module_name
        self.package_name = package_name

        self.is_legacy = is_legacy
        self.install_name = install_name or self.name
        self.legacy_args = {"name": name, "description": "A legacy plugin"}
        self.legacy_args.update(legacy_args)

    @property
    def format(self) -> Any:
        """For backwards compatibility with FormatManager

        Delete when migrating to v3
        """
        if not self.is_legacy:
            raise RuntimeError("Can only get format for legacy plugins.")

        module = importlib.import_module(self.module_name, self.package_name)
        clazz = getattr(module, self.class_name)
        return clazz(**self.legacy_args)

    @property
    def plugin_class(self) -> Any:
        """Get the plugin class (import if needed)

        Returns
        -------
        plugin_class : Any
            The class that can be used to instantiate plugins.

        """

        module = importlib.import_module(self.module_name, self.package_name)
        clazz = getattr(module, self.class_name)

        if self.is_legacy:
            legacy_plugin = clazz(**self.legacy_args)

            def partial_legacy_plugin(request):
                return LegacyPlugin(request, legacy_plugin)

            clazz = partial_legacy_plugin

        return clazz


known_plugins = dict()
known_plugins["pillow"] = PluginConfig(
    name="pillow", class_name="PillowPlugin", module_name="imageio.plugins.pillow"
)

# Legacy plugins
# ==============
#
# Which are partly registered by format, partly by plugin, and partly by a mix
# of both. We keep the naming here for backwards compatibility.
# In v3 this should become a single entry per plugin named after the plugin
# We can choose extension-specific priority in ``config.extensions``.
#
# Note: Since python 3.7 order of insertion determines the order of dict().keys()
# This means that the order here determines the order by which plugins are
# checked during the full fallback search. We don't advertise this downstream,
# but it could be a useful thing to keep in mind to choose a sensible default
# search order.

known_plugins["TIFF"] = PluginConfig(
    name="TIFF",
    class_name="TiffFormat",
    module_name="imageio.plugins.tifffile",
    is_legacy=True,
    install_name="tifffile",
    legacy_args={
        "description": "TIFF format",
        "extensions": ".tif .tiff .stk .lsm",
        "modes": "iIvV",
    },
)

# PILLOW plugin formats (legacy)
PILLOW_FORMATS = [
    ("BMP", "Windows Bitmap", ".bmp", "PillowFormat"),
    ("BUFR", "BUFR", ".bufr", "PillowFormat"),
    ("CUR", "Windows Cursor", ".cur", "PillowFormat"),
    ("DCX", "Intel DCX", ".dcx", "PillowFormat"),
    ("DDS", "DirectDraw Surface", ".dds", "PillowFormat"),
    ("DIB", "Windows Bitmap", "", "PillowFormat"),
    ("EPS", "Encapsulated Postscript", ".ps .eps", "PillowFormat"),
    ("FITS", "FITS", ".fit .fits", "PillowFormat"),
    ("FLI", "Autodesk FLI/FLC Animation", ".fli .flc", "PillowFormat"),
    ("FPX", "FlashPix", ".fpx", "PillowFormat"),
    ("FTEX", "Texture File Format (IW2:EOC)", ".ftc .ftu", "PillowFormat"),
    ("GBR", "GIMP brush file", ".gbr", "PillowFormat"),
    ("GIF", "Compuserve GIF", ".gif", "GIFFormat"),
    ("GRIB", "GRIB", ".grib", "PillowFormat"),
    ("HDF5", "HDF5", ".h5 .hdf", "PillowFormat"),
    ("ICNS", "Mac OS icns resource", ".icns", "PillowFormat"),
    ("ICO", "Windows Icon", ".ico", "PillowFormat"),
    ("IM", "IFUNC Image Memory", ".im", "PillowFormat"),
    ("IMT", "IM Tools", "", "PillowFormat"),
    ("IPTC", "IPTC/NAA", ".iim", "PillowFormat"),
    ("JPEG", "JPEG (ISO 10918)", ".jfif .jpe .jpg .jpeg", "JPEGFormat"),
    (
        "JPEG2000",
        "JPEG 2000 (ISO 15444)",
        ".jp2 .j2k .jpc .jpf .jpx .j2c",
        "JPEG2000Format",
    ),
    ("MCIDAS", "McIdas area file", "", "PillowFormat"),
    ("MIC", "Microsoft Image Composer", ".mic", "PillowFormat"),
    # skipped in legacy pillow
    # ("MPEG", "MPEG", ".mpg .mpeg", "PillowFormat"),
    ("MPO", "MPO (CIPA DC-007)", ".mpo", "PillowFormat"),
    ("MSP", "Windows Paint", ".msp", "PillowFormat"),
    ("PCD", "Kodak PhotoCD", ".pcd", "PillowFormat"),
    ("PCX", "Paintbrush", ".pcx", "PillowFormat"),
    ("PIXAR", "PIXAR raster image", ".pxr", "PillowFormat"),
    ("PNG", "Portable network graphics", ".png", "PNGFormat"),
    ("PPM", "Pbmplus image", ".pbm .pgm .ppm", "PillowFormat"),
    ("PSD", "Adobe Photoshop", ".psd", "PillowFormat"),
    ("SGI", "SGI Image File Format", ".bw .rgb .rgba .sgi", "PillowFormat"),
    ("SPIDER", "Spider 2D image", "", "PillowFormat"),
    ("SUN", "Sun Raster File", ".ras", "PillowFormat"),
    ("TGA", "Targa", ".tga", "PillowFormat"),
    ("TIFF", "Adobe TIFF", ".tif .tiff", "TIFFFormat"),
    ("WMF", "Windows Metafile", ".wmf .emf", "PillowFormat"),
    ("XBM", "X11 Bitmap", ".xbm", "PillowFormat"),
    ("XPM", "X11 Pixel Map", ".xpm", "PillowFormat"),
    ("XVTHUMB", "XV thumbnail image", "", "PillowFormat"),
]
for id, summary, ext, class_name in PILLOW_FORMATS:
    config = PluginConfig(
        name=id.upper() + "-PIL",
        class_name=class_name,
        module_name="imageio.plugins.pillow_legacy",
        is_legacy=True,
        install_name="pillow",
        legacy_args={
            "description": summary + " via Pillow",
            "extensions": ext,
            "modes": "iI" if class_name == "GIFFormat" else "i",
            "plugin_id": id,
        },
    )
    known_plugins[config.name] = config

known_plugins["FFMPEG"] = PluginConfig(
    name="FFMPEG",
    class_name="FfmpegFormat",
    module_name="imageio.plugins.ffmpeg",
    is_legacy=True,
    install_name="ffmpeg",
    legacy_args={
        "description": "Many video formats and cameras (via ffmpeg)",
        "extensions": ".mov .avi .mpg .mpeg .mp4 .mkv .webm .wmv",
        "modes": "I",
    },
)

known_plugins["BSDF"] = PluginConfig(
    name="BSDF",
    class_name="BsdfFormat",
    module_name="imageio.plugins.bsdf",
    is_legacy=True,
    install_name="bsdf",
    legacy_args={
        "description": "Format based on the Binary Structured Data Format",
        "extensions": ".bsdf",
        "modes": "iIvV",
    },
)

known_plugins["DICOM"] = PluginConfig(
    name="DICOM",
    class_name="DicomFormat",
    module_name="imageio.plugins.dicom",
    is_legacy=True,
    install_name="dicom",
    legacy_args={
        "description": "Digital Imaging and Communications in Medicine",
        "extensions": ".dcm .ct .mri",
        "modes": "iIvV",
    },
)

known_plugins["FEI"] = PluginConfig(
    name="FEI",
    class_name="FEISEMFormat",
    module_name="imageio.plugins.feisem",
    is_legacy=True,
    install_name="feisem",
    legacy_args={
        "description": "FEI-SEM TIFF format",
        "extensions": [".tif", ".tiff"],
        "modes": "iv",
    },
)

known_plugins["FITS"] = PluginConfig(
    name="FITS",
    class_name="FitsFormat",
    module_name="imageio.plugins.fits",
    is_legacy=True,
    install_name="fits",
    legacy_args={
        "description": "Flexible Image Transport System (FITS) format",
        "extensions": ".fits .fit .fts .fz",
        "modes": "iIvV",
    },
)

known_plugins["GDAL"] = PluginConfig(
    name="GDAL",
    class_name="GdalFormat",
    module_name="imageio.plugins.gdal",
    is_legacy=True,
    install_name="gdal",
    legacy_args={
        "description": "Geospatial Data Abstraction Library",
        "extensions": ".tiff  .tif .img .ecw .jpg .jpeg",
        "modes": "iIvV",
    },
)

known_plugins["ITK"] = PluginConfig(
    name="ITK",
    class_name="ItkFormat",
    module_name="imageio.plugins.simpleitk",
    is_legacy=True,
    install_name="simpleitk",
    legacy_args={
        "description": "Insight Segmentation and Registration Toolkit (ITK) format",
        "extensions": " ".join(
            (
                ".gipl",
                ".ipl",
                ".mha",
                ".mhd",
                ".nhdr",
                ".nia",
                ".hdr",
                ".nrrd",
                ".nii",
                ".nii.gz",
                ".img",
                ".img.gz",
                ".vtk",
                ".hdf5",
                ".lsm",
                ".mnc",
                ".mnc2",
                ".mgh",
                ".mnc",
                ".pic",
                ".bmp",
                ".jpeg",
                ".jpg",
                ".png",
                ".tiff",
                ".tif",
                ".dicom",
                ".dcm",
                ".gdcm",
            )
        ),
        "modes": "iIvV",
    },
)

known_plugins["NPZ"] = PluginConfig(
    name="NPZ",
    class_name="NpzFormat",
    module_name="imageio.plugins.npz",
    is_legacy=True,
    install_name="numpy",
    legacy_args={
        "description": "Numpy's compressed array format",
        "extensions": ".npz",
        "modes": "iIvV",
    },
)

known_plugins["SPE"] = PluginConfig(
    name="SPE",
    class_name="SpeFormat",
    module_name="imageio.plugins.spe",
    is_legacy=True,
    install_name="spe",
    legacy_args={
        "description": "SPE file format",
        "extensions": ".spe",
        "modes": "iIvV",
    },
)

known_plugins["SWF"] = PluginConfig(
    name="SWF",
    class_name="SWFFormat",
    module_name="imageio.plugins.swf",
    is_legacy=True,
    install_name="swf",
    legacy_args={
        "description": "Shockwave flash",
        "extensions": ".swf",
        "modes": "I",
    },
)

known_plugins["SCREENGRAB"] = PluginConfig(
    name="SCREENGRAB",
    class_name="ScreenGrabFormat",
    module_name="imageio.plugins.grab",
    is_legacy=True,
    install_name="pillow",
    legacy_args={
        "description": "Grab screenshots (Windows and OS X only)",
        "extensions": [],
        "modes": "i",
    },
)

known_plugins["CLIPBOARDGRAB"] = PluginConfig(
    name="CLIPBOARDGRAB",
    class_name="ClipboardGrabFormat",
    module_name="imageio.plugins.grab",
    is_legacy=True,
    install_name="pillow",
    legacy_args={
        "description": "Grab from clipboard (Windows only)",
        "extensions": [],
        "modes": "i",
    },
)

# LYTRO plugin (legacy)
lytro_formats = [
    ("lytro-lfr", "Lytro Illum lfr image file", ".lfr", "i", "LytroLfrFormat"),
    (
        "lytro-illum-raw",
        "Lytro Illum raw image file",
        ".raw",
        "i",
        "LytroIllumRawFormat",
    ),
    ("lytro-lfp", "Lytro F01 lfp image file", ".lfp", "i", "LytroLfpFormat"),
    ("lytro-f01-raw", "Lytro F01 raw image file", ".raw", "i", "LytroF01RawFormat"),
]
for name, des, ext, mode, class_name in lytro_formats:
    config = PluginConfig(
        name=name.upper(),
        class_name=class_name,
        module_name="imageio.plugins.lytro",
        is_legacy=True,
        install_name="lytro",
        legacy_args={
            "description": des,
            "extensions": ext,
            "modes": mode,
        },
    )
    known_plugins[config.name] = config

# FreeImage plugin (legacy)
FREEIMAGE_FORMATS = [
    (
        "BMP",
        0,
        "Windows or OS/2 Bitmap",
        ".bmp",
        "i",
        "FreeimageBmpFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "CUT",
        21,
        "Dr. Halo",
        ".cut",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "DDS",
        24,
        "DirectX Surface",
        ".dds",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "EXR",
        29,
        "ILM OpenEXR",
        ".exr",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "G3",
        27,
        "Raw fax format CCITT G.3",
        ".g3",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "GIF",
        25,
        "Static and animated gif (FreeImage)",
        ".gif",
        "iI",
        "GifFormat",
        "imageio.plugins.freeimagemulti",
    ),
    (
        "HDR",
        26,
        "High Dynamic Range Image",
        ".hdr",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "ICO",
        1,
        "Windows Icon",
        ".ico",
        "iI",
        "IcoFormat",
        "imageio.plugins.freeimagemulti",
    ),
    (
        "IFF",
        5,
        "IFF Interleaved Bitmap",
        ".iff .lbm",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "J2K",
        30,
        "JPEG-2000 codestream",
        ".j2k .j2c",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "JNG",
        3,
        "JPEG Network Graphics",
        ".jng",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "JP2",
        31,
        "JPEG-2000 File Format",
        ".jp2",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "JPEG",
        2,
        "JPEG - JFIF Compliant",
        ".jpg .jif .jpeg .jpe",
        "i",
        "FreeimageJpegFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "JPEG-XR",
        36,
        "JPEG XR image format",
        ".jxr .wdp .hdp",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "KOALA",
        4,
        "C64 Koala Graphics",
        ".koa",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    # not registered in legacy pillow
    # ("MNG", 6, "Multiple-image Network Graphics", ".mng", "i", "FreeimageFormat", "imageio.plugins.freeimage"),
    (
        "PBM",
        7,
        "Portable Bitmap (ASCII)",
        ".pbm",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PBMRAW",
        8,
        "Portable Bitmap (RAW)",
        ".pbm",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PCD",
        9,
        "Kodak PhotoCD",
        ".pcd",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PCX",
        10,
        "Zsoft Paintbrush",
        ".pcx",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PFM",
        32,
        "Portable floatmap",
        ".pfm",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PGM",
        11,
        "Portable Greymap (ASCII)",
        ".pgm",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PGMRAW",
        12,
        "Portable Greymap (RAW)",
        ".pgm",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PICT",
        33,
        "Macintosh PICT",
        ".pct .pict .pic",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PNG",
        13,
        "Portable Network Graphics",
        ".png",
        "i",
        "FreeimagePngFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PPM",
        14,
        "Portable Pixelmap (ASCII)",
        ".ppm",
        "i",
        "FreeimagePnmFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PPMRAW",
        15,
        "Portable Pixelmap (RAW)",
        ".ppm",
        "i",
        "FreeimagePnmFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "PSD",
        20,
        "Adobe Photoshop",
        ".psd",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "RAS",
        16,
        "Sun Raster Image",
        ".ras",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "RAW",
        34,
        "RAW camera image",
        ".3fr .arw .bay .bmq .cap .cine .cr2 .crw .cs1 .dc2 "
        ".dcr .drf .dsc .dng .erf .fff .ia .iiq .k25 .kc2 .kdc .mdc .mef .mos .mrw .nef .nrw .orf "
        ".pef .ptx .pxn .qtk .raf .raw .rdc .rw2 .rwl .rwz .sr2 .srf .srw .sti",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "SGI",
        28,
        "SGI Image Format",
        ".sgi .rgb .rgba .bw",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "TARGA",
        17,
        "Truevision Targa",
        ".tga .targa",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "TIFF",
        18,
        "Tagged Image File Format",
        ".tif .tiff",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "WBMP",
        19,
        "Wireless Bitmap",
        ".wap .wbmp .wbm",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "WebP",
        35,
        "Google WebP image format",
        ".webp",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "XBM",
        22,
        "X11 Bitmap Format",
        ".xbm",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
    (
        "XPM",
        23,
        "X11 Pixmap Format",
        ".xpm",
        "i",
        "FreeimageFormat",
        "imageio.plugins.freeimage",
    ),
]
for name, i, des, ext, mode, class_name, module_name in FREEIMAGE_FORMATS:
    config = PluginConfig(
        name=name.upper() + "-FI",
        class_name=class_name,
        module_name=module_name,
        is_legacy=True,
        install_name="freeimage",
        legacy_args={
            "description": des,
            "extensions": ext,
            "modes": mode,
            "fif": i,
        },
    )
    known_plugins[config.name] = config

# exists for backwards compatibility with FormatManager
# delete in V3
_original_order = [x for x, config in known_plugins.items() if config.is_legacy]
