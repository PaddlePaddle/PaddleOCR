# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Example plugin. You can use this as a template for your own plugin.
"""

import numpy as np

from .. import formats
from ..core import Format


class DummyFormat(Format):
    """The dummy format is an example format that does nothing.
    It will never indicate that it can read or write a file. When
    explicitly asked to read, it will simply read the bytes. When
    explicitly asked to write, it will raise an error.

    This documentation is shown when the user does ``help('thisformat')``.

    Parameters for reading
    ----------------------
    Specify arguments in numpy doc style here.

    Parameters for saving
    ---------------------
    Specify arguments in numpy doc style here.

    """

    def _can_read(self, request):
        # This method is called when the format manager is searching
        # for a format to read a certain image. Return True if this format
        # can do it.
        #
        # The format manager is aware of the extensions and the modes
        # that each format can handle. It will first ask all formats
        # that *seem* to be able to read it whether they can. If none
        # can, it will ask the remaining formats if they can: the
        # extension might be missing, and this allows formats to provide
        # functionality for certain extensions, while giving preference
        # to other plugins.
        #
        # If a format says it can, it should live up to it. The format
        # would ideally check the request.firstbytes and look for a
        # header of some kind.
        #
        # The request object has:
        # request.filename: a representation of the source (only for reporting)
        # request.firstbytes: the first 256 bytes of the file.
        # request.mode[0]: read or write mode
        # request.mode[1]: what kind of data the user expects: one of 'iIvV?'

        if request.mode[1] in (self.modes + "?"):
            if request.extension in self.extensions:
                return True

    def _can_write(self, request):
        # This method is called when the format manager is searching
        # for a format to write a certain image. It will first ask all
        # formats that *seem* to be able to write it whether they can.
        # If none can, it will ask the remaining formats if they can.
        #
        # Return True if the format can do it.

        # In most cases, this code does suffice:
        if request.mode[1] in (self.modes + "?"):
            if request.extension in self.extensions:
                return True

    # -- reader

    class Reader(Format.Reader):
        def _open(self, some_option=False, length=1):
            # Specify kwargs here. Optionally, the user-specified kwargs
            # can also be accessed via the request.kwargs object.
            #
            # The request object provides two ways to get access to the
            # data. Use just one:
            #  - Use request.get_file() for a file object (preferred)
            #  - Use request.get_local_filename() for a file on the system
            self._fp = self.request.get_file()
            self._length = length  # passed as an arg in this case for testing
            self._data = None

        def _close(self):
            # Close the reader.
            # Note that the request object will close self._fp
            pass

        def _get_length(self):
            # Return the number of images. Can be np.inf
            return self._length

        def _get_data(self, index):
            # Return the data and meta data for the given index
            if index >= self._length:
                raise IndexError("Image index %i > %i" % (index, self._length))
            # Read all bytes
            if self._data is None:
                self._data = self._fp.read()
            # Put in a numpy array
            im = np.frombuffer(self._data, "uint8")
            im.shape = len(im), 1
            # Return array and dummy meta data
            return im, {}

        def _get_meta_data(self, index):
            # Get the meta data for the given index. If index is None, it
            # should return the global meta data.
            return {}  # This format does not support meta data

    # -- writer

    class Writer(Format.Writer):
        def _open(self, flags=0):
            # Specify kwargs here. Optionally, the user-specified kwargs
            # can also be accessed via the request.kwargs object.
            #
            # The request object provides two ways to write the data.
            # Use just one:
            #  - Use request.get_file() for a file object (preferred)
            #  - Use request.get_local_filename() for a file on the system
            self._fp = self.request.get_file()

        def _close(self):
            # Close the reader.
            # Note that the request object will close self._fp
            pass

        def _append_data(self, im, meta):
            # Process the given data and meta data.
            raise RuntimeError("The dummy format cannot write image data.")

        def set_meta_data(self, meta):
            # Process the given meta data (global for all images)
            # It is not mandatory to support this.
            raise RuntimeError("The dummy format cannot write meta data.")


# Register. You register an *instance* of a Format class. Here specify:
format = DummyFormat(
    "dummy",  # short name
    "An example format that does nothing.",  # one line descr.
    ".foobar .nonexistentext",  # list of extensions
    "iI",  # modes, characters in iIvV
)
formats.add_format(format)
