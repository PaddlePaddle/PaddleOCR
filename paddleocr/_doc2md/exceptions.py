class Any2MDError(Exception):
    """Base exception for doc2md."""

    pass


class UnsupportedFormatError(Any2MDError):
    """Raised when the input file format is not supported."""

    pass


class ConversionError(Any2MDError):
    """Raised when an error occurs during conversion."""

    pass
