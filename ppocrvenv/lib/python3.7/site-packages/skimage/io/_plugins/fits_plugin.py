__all__ = ['imread', 'imread_collection']

import skimage.io as io

try:
    from astropy.io import fits
except ImportError:
    raise ImportError(
        "Astropy could not be found. It is needed to read FITS files.\n"
        "Please refer to https://www.astropy.org for installation\n"
        "instructions.")


def imread(fname):
    """Load an image from a FITS file.

    Parameters
    ----------
    fname : string
        Image file name, e.g. ``test.fits``.

    Returns
    -------
    img_array : ndarray
        Unlike plugins such as PIL, where different color bands/channels are
        stored in the third dimension, FITS images are grayscale-only and can
        be N-dimensional, so an array of the native FITS dimensionality is
        returned, without color channels.

        Currently if no image is found in the file, None will be returned

    Notes
    -----
    Currently FITS ``imread()`` always returns the first image extension when
    given a Multi-Extension FITS file; use ``imread_collection()`` (which does
    lazy loading) to get all the extensions at once.

    """

    hdulist = fits.open(fname)

    # Iterate over FITS image extensions, ignoring any other extension types
    # such as binary tables, and get the first image data array:
    img_array = None
    for hdu in hdulist:
        if isinstance(hdu, fits.ImageHDU) or \
           isinstance(hdu, fits.PrimaryHDU):
            if hdu.data is not None:
                img_array = hdu.data
                break
    hdulist.close()

    return img_array


def imread_collection(load_pattern, conserve_memory=True):
    """Load a collection of images from one or more FITS files

    Parameters
    ----------
    load_pattern : str or list
        List of extensions to load. Filename globbing is currently
        unsupported.
    conserve_memory : bool
        If True, never keep more than one in memory at a specific
        time. Otherwise, images will be cached once they are loaded.

    Returns
    -------
    ic : ImageCollection
        Collection of images.

    """

    intype = type(load_pattern)
    if intype is not list and intype is not str:
        raise TypeError("Input must be a filename or list of filenames")

    # Ensure we have a list, otherwise we'll end up iterating over the string:
    if intype is not list:
        load_pattern = [load_pattern]

    # Generate a list of filename/extension pairs by opening the list of
    # files and finding the image extensions in each one:
    ext_list = []
    for filename in load_pattern:
        hdulist = fits.open(filename)
        for n, hdu in zip(range(len(hdulist)), hdulist):
            if isinstance(hdu, fits.ImageHDU) or \
               isinstance(hdu, fits.PrimaryHDU):
                # Ignore (primary) header units with no data (use '.size'
                # rather than '.data' to avoid actually loading the image):
                try:
                    data_size = hdu.size  # size is int in Astropy 3.1.2
                except TypeError:
                    data_size = hdu.size()
                if data_size > 0:
                    ext_list.append((filename, n))
        hdulist.close()

    return io.ImageCollection(ext_list, load_func=FITSFactory,
                              conserve_memory=conserve_memory)


def FITSFactory(image_ext):
    """Load an image extension from a FITS file and return a NumPy array

    Parameters
    ----------
    image_ext : tuple
        FITS extension to load, in the format ``(filename, ext_num)``.
        The FITS ``(extname, extver)`` format is unsupported, since this
        function is not called directly by the user and
        ``imread_collection()`` does the work of figuring out which
        extensions need loading.

    """

    # Expect a length-2 tuple with a filename as the first element:
    if not isinstance(image_ext, tuple):
        raise TypeError("Expected a tuple")

    if len(image_ext) != 2:
        raise ValueError("Expected a tuple of length 2")

    filename = image_ext[0]
    extnum = image_ext[1]

    if type(filename) is not str or type(extnum) is not int:
        raise ValueError("Expected a (filename, extension) tuple")

    hdulist = fits.open(filename)

    data = hdulist[extnum].data

    hdulist.close()

    if data is None:
        raise RuntimeError(
            "Extension %d of %s has no data" % (extnum, filename))

    return data
