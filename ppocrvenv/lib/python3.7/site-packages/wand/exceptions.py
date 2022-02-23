""":mod:`wand.exceptions` --- Errors and warnings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module maps MagickWand API's errors and warnings to Python's native
exceptions and warnings. You can catch all MagickWand errors using Python's
natural way to catch errors.

.. seealso::

   `ImageMagick Exceptions <http://www.imagemagick.org/script/exception.php>`_

.. versionadded:: 0.1.1

.. versionchanged:: 0.5.8
   Warning & Error Exceptions are now explicitly defined. Previously
   ImageMagick domain-based errors were dynamically generated at runtime.
"""


class WandException(Exception):
    """All Wand-related exceptions are derived from this class."""


class BaseWarning(WandException, Warning):
    """Base class for Wand-related warnings.

    .. versionadded:: 0.4.4

    """


class BaseError(WandException):
    """Base class for Wand-related errors.

    .. versionadded:: 0.4.4

    """


class BaseFatalError(WandException):
    """Base class for Wand-related fatal errors.

    .. versionadded:: 0.4.4

    """


class WandLibraryVersionError(WandException):
    """Base class for Wand-related ImageMagick version errors.

    .. versionadded:: 0.3.2

    """


class WandRuntimeError(WandException, RuntimeError):
    """Generic class for Wand-related runtime errors.

    .. versionadded:: 0.5.2
    """


class ResourceLimitWarning(BaseWarning, MemoryError):
    """A program resource is exhausted e.g. not enough memory."""
    wand_error_code = 300


class ResourceLimitError(BaseError, MemoryError):
    """A program resource is exhausted e.g. not enough memory."""
    wand_error_code = 400


class ResourceLimitFatalError(BaseFatalError, MemoryError):
    """A program resource is exhausted e.g. not enough memory."""
    wand_error_code = 700


class TypeWarning(BaseWarning):
    """A font is unavailable; a substitution may have occurred."""
    wand_error_code = 305


class TypeError(BaseError):
    """A font is unavailable; a substitution may have occurred."""
    wand_error_code = 405


class TypeFatalError(BaseFatalError):
    """A font is unavailable; a substitution may have occurred."""
    wand_error_code = 705


class OptionWarning(BaseWarning):
    """A command-line option was malformed."""
    wand_error_code = 310


class OptionError(BaseError):
    """A command-line option was malformed."""
    wand_error_code = 410


class OptionFatalError(BaseFatalError):
    """A command-line option was malformed."""
    wand_error_code = 710


class DelegateWarning(BaseWarning):
    """An ImageMagick delegate failed to complete."""
    wand_error_code = 315


class DelegateError(BaseError):
    """An ImageMagick delegate failed to complete."""
    wand_error_code = 415


class DelegateFatalError(BaseFatalError):
    """An ImageMagick delegate failed to complete."""
    wand_error_code = 715


class MissingDelegateWarning(BaseWarning, ImportError):
    """The image type can not be read or written because the appropriate;
       delegate is missing."""
    wand_error_code = 320


class MissingDelegateError(BaseError, ImportError):
    """The image type can not be read or written because the appropriate;
       delegate is missing."""
    wand_error_code = 420


class MissingDelegateFatalError(BaseFatalError, ImportError):
    """The image type can not be read or written because the appropriate;
       delegate is missing."""
    wand_error_code = 720


class CorruptImageWarning(BaseWarning, ValueError):
    """The image file may be corrupt."""
    wand_error_code = 325


class CorruptImageError(BaseError, ValueError):
    """The image file may be corrupt."""
    wand_error_code = 425


class CorruptImageFatalError(BaseFatalError, ValueError):
    """The image file may be corrupt."""
    wand_error_code = 725


class FileOpenWarning(BaseWarning, IOError):
    """The image file could not be opened for reading or writing."""
    wand_error_code = 330


class FileOpenError(BaseError, IOError):
    """The image file could not be opened for reading or writing."""
    wand_error_code = 430


class FileOpenFatalError(BaseFatalError, IOError):
    """The image file could not be opened for reading or writing."""
    wand_error_code = 730


class BlobWarning(BaseWarning, IOError):
    """A binary large object could not be allocated, read, or written."""
    wand_error_code = 335


class BlobError(BaseError, IOError):
    """A binary large object could not be allocated, read, or written."""
    wand_error_code = 435


class BlobFatalError(BaseFatalError, IOError):
    """A binary large object could not be allocated, read, or written."""
    wand_error_code = 735


class StreamWarning(BaseWarning, IOError):
    """There was a problem reading or writing from a stream."""
    wand_error_code = 340


class StreamError(BaseError, IOError):
    """There was a problem reading or writing from a stream."""
    wand_error_code = 440


class StreamFatalError(BaseFatalError, IOError):
    """There was a problem reading or writing from a stream."""
    wand_error_code = 740


class CacheWarning(BaseWarning):
    """Pixels could not be read or written to the pixel cache."""
    wand_error_code = 345


class CacheError(BaseError):
    """Pixels could not be read or written to the pixel cache."""
    wand_error_code = 445


class CacheFatalError(BaseFatalError):
    """Pixels could not be read or written to the pixel cache."""
    wand_error_code = 745


class CoderWarning(BaseWarning):
    """There was a problem with an image coder."""
    wand_error_code = 350


class CoderError(BaseError):
    """There was a problem with an image coder."""
    wand_error_code = 450


class CoderFatalError(BaseFatalError):
    """There was a problem with an image coder."""
    wand_error_code = 750


class ModuleWarning(BaseWarning):
    """There was a problem with an image module."""
    wand_error_code = 355


class ModuleError(BaseError):
    """There was a problem with an image module."""
    wand_error_code = 455


class ModuleFatalError(BaseFatalError):
    """There was a problem with an image module."""
    wand_error_code = 755


class DrawWarning(BaseWarning):
    """A drawing operation failed."""
    wand_error_code = 360


class DrawError(BaseError):
    """A drawing operation failed."""
    wand_error_code = 460


class DrawFatalError(BaseFatalError):
    """A drawing operation failed."""
    wand_error_code = 760


class ImageWarning(BaseWarning):
    """The operation could not complete due to an incompatible image."""
    wand_error_code = 365


class ImageError(BaseError):
    """The operation could not complete due to an incompatible image."""
    wand_error_code = 465


class ImageFatalError(BaseFatalError):
    """The operation could not complete due to an incompatible image."""
    wand_error_code = 765


class WandWarning(BaseWarning):
    """There was a problem specific to the MagickWand API."""
    wand_error_code = 370


class WandError(BaseError):
    """There was a problem specific to the MagickWand API."""
    wand_error_code = 470


class WandFatalError(BaseFatalError):
    """There was a problem specific to the MagickWand API."""
    wand_error_code = 770


class RandomWarning(BaseWarning):
    """There is a problem generating a true or pseudo-random number."""
    wand_error_code = 375


class RandomError(BaseError):
    """There is a problem generating a true or pseudo-random number."""
    wand_error_code = 475


class RandomFatalError(BaseFatalError):
    """There is a problem generating a true or pseudo-random number."""
    wand_error_code = 775


class XServerWarning(BaseWarning):
    """An X resource is unavailable."""
    wand_error_code = 380


class XServerError(BaseError):
    """An X resource is unavailable."""
    wand_error_code = 480


class XServerFatalError(BaseFatalError):
    """An X resource is unavailable."""
    wand_error_code = 780


class MonitorWarning(BaseWarning):
    """There was a problem activating the progress monitor."""
    wand_error_code = 385


class MonitorError(BaseError):
    """There was a problem activating the progress monitor."""
    wand_error_code = 485


class MonitorFatalError(BaseFatalError):
    """There was a problem activating the progress monitor."""
    wand_error_code = 785


class RegistryWarning(BaseWarning):
    """There was a problem getting or setting the registry."""
    wand_error_code = 390


class RegistryError(BaseError):
    """There was a problem getting or setting the registry."""
    wand_error_code = 490


class RegistryFatalError(BaseFatalError):
    """There was a problem getting or setting the registry."""
    wand_error_code = 790


class ConfigureWarning(BaseWarning):
    """There was a problem getting a configuration file."""
    wand_error_code = 395


class ConfigureError(BaseError):
    """There was a problem getting a configuration file."""
    wand_error_code = 495


class ConfigureFatalError(BaseFatalError):
    """There was a problem getting a configuration file."""
    wand_error_code = 795


class PolicyWarning(BaseWarning):
    """A policy denies access to a delegate, coder, filter, path, or
       resource."""
    wand_error_code = 399


class PolicyError(BaseError):
    """A policy denies access to a delegate, coder, filter, path, or
       resource."""
    wand_error_code = 499


class PolicyFatalError(BaseFatalError):
    """A policy denies access to a delegate, coder, filter, path, or
       resource."""
    wand_error_code = 799


#: (:class:`dict`) The dictionary of (code, exc_type).
TYPE_MAP = {
    300: ResourceLimitWarning,
    305: TypeWarning,
    310: OptionWarning,
    315: DelegateWarning,
    320: MissingDelegateWarning,
    325: CorruptImageWarning,
    330: FileOpenWarning,
    335: BlobWarning,
    340: StreamWarning,
    345: CacheWarning,
    350: CoderWarning,
    355: ModuleWarning,
    360: DrawWarning,
    365: ImageWarning,
    370: WandWarning,
    375: RandomWarning,
    380: XServerWarning,
    385: MonitorWarning,
    390: RegistryWarning,
    395: ConfigureWarning,
    399: PolicyWarning,
    400: ResourceLimitError,
    405: TypeError,
    410: OptionError,
    415: DelegateError,
    420: MissingDelegateError,
    425: CorruptImageError,
    430: FileOpenError,
    435: BlobError,
    440: StreamError,
    445: CacheError,
    450: CoderError,
    455: ModuleError,
    460: DrawError,
    465: ImageError,
    470: WandError,
    475: RandomError,
    480: XServerError,
    485: MonitorError,
    490: RegistryError,
    495: ConfigureError,
    499: PolicyError,
    700: ResourceLimitFatalError,
    705: TypeFatalError,
    710: OptionFatalError,
    715: DelegateFatalError,
    720: MissingDelegateFatalError,
    725: CorruptImageFatalError,
    730: FileOpenFatalError,
    735: BlobFatalError,
    740: StreamFatalError,
    745: CacheFatalError,
    750: CoderFatalError,
    755: ModuleFatalError,
    760: DrawFatalError,
    765: ImageFatalError,
    770: WandFatalError,
    775: RandomFatalError,
    780: XServerFatalError,
    785: MonitorFatalError,
    790: RegistryFatalError,
    795: ConfigureFatalError,
    799: PolicyFatalError,
}
