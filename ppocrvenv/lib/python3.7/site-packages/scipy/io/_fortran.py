"""
Module to read / write Fortran unformatted sequential files.

This is in the spirit of code written by Neil Martinsen-Burrell and Joe Zuntz.

"""
import warnings
import numpy as np

__all__ = ['FortranFile', 'FortranEOFError', 'FortranFormattingError']


class FortranEOFError(TypeError, IOError):
    """Indicates that the file ended properly.

    This error descends from TypeError because the code used to raise
    TypeError (and this was the only way to know that the file had
    ended) so users might have ``except TypeError:``.

    """
    pass


class FortranFormattingError(TypeError, IOError):
    """Indicates that the file ended mid-record.

    Descends from TypeError for backward compatibility.

    """
    pass


class FortranFile:
    """
    A file object for unformatted sequential files from Fortran code.

    Parameters
    ----------
    filename : file or str
        Open file object or filename.
    mode : {'r', 'w'}, optional
        Read-write mode, default is 'r'.
    header_dtype : dtype, optional
        Data type of the header. Size and endiness must match the input/output file.

    Notes
    -----
    These files are broken up into records of unspecified types. The size of
    each record is given at the start (although the size of this header is not
    standard) and the data is written onto disk without any formatting. Fortran
    compilers supporting the BACKSPACE statement will write a second copy of
    the size to facilitate backwards seeking.

    This class only supports files written with both sizes for the record.
    It also does not support the subrecords used in Intel and gfortran compilers
    for records which are greater than 2GB with a 4-byte header.

    An example of an unformatted sequential file in Fortran would be written as::

        OPEN(1, FILE=myfilename, FORM='unformatted')

        WRITE(1) myvariable

    Since this is a non-standard file format, whose contents depend on the
    compiler and the endianness of the machine, caution is advised. Files from
    gfortran 4.8.0 and gfortran 4.1.2 on x86_64 are known to work.

    Consider using Fortran direct-access files or files from the newer Stream
    I/O, which can be easily read by `numpy.fromfile`.

    Examples
    --------
    To create an unformatted sequential Fortran file:

    >>> from scipy.io import FortranFile
    >>> f = FortranFile('test.unf', 'w')
    >>> f.write_record(np.array([1,2,3,4,5], dtype=np.int32))
    >>> f.write_record(np.linspace(0,1,20).reshape((5,4)).T)
    >>> f.close()

    To read this file:

    >>> f = FortranFile('test.unf', 'r')
    >>> print(f.read_ints(np.int32))
    [1 2 3 4 5]
    >>> print(f.read_reals(float).reshape((5,4), order="F"))
    [[0.         0.05263158 0.10526316 0.15789474]
     [0.21052632 0.26315789 0.31578947 0.36842105]
     [0.42105263 0.47368421 0.52631579 0.57894737]
     [0.63157895 0.68421053 0.73684211 0.78947368]
     [0.84210526 0.89473684 0.94736842 1.        ]]
    >>> f.close()

    Or, in Fortran::

        integer :: a(5), i
        double precision :: b(5,4)
        open(1, file='test.unf', form='unformatted')
        read(1) a
        read(1) b
        close(1)
        write(*,*) a
        do i = 1, 5
            write(*,*) b(i,:)
        end do

    """
    def __init__(self, filename, mode='r', header_dtype=np.uint32):
        if header_dtype is None:
            raise ValueError('Must specify dtype')

        header_dtype = np.dtype(header_dtype)
        if header_dtype.kind != 'u':
            warnings.warn("Given a dtype which is not unsigned.")

        if mode not in 'rw' or len(mode) != 1:
            raise ValueError('mode must be either r or w')

        if hasattr(filename, 'seek'):
            self._fp = filename
        else:
            self._fp = open(filename, '%sb' % mode)

        self._header_dtype = header_dtype

    def _read_size(self, eof_ok=False):
        n = self._header_dtype.itemsize
        b = self._fp.read(n)
        if (not b) and eof_ok:
            raise FortranEOFError("End of file occurred at end of record")
        elif len(b) < n:
            raise FortranFormattingError(
                "End of file in the middle of the record size")
        return int(np.frombuffer(b, dtype=self._header_dtype, count=1))

    def write_record(self, *items):
        """
        Write a record (including sizes) to the file.

        Parameters
        ----------
        *items : array_like
            The data arrays to write.

        Notes
        -----
        Writes data items to a file::

            write_record(a.T, b.T, c.T, ...)

            write(1) a, b, c, ...

        Note that data in multidimensional arrays is written in
        row-major order --- to make them read correctly by Fortran
        programs, you need to transpose the arrays yourself when
        writing them.

        """
        items = tuple(np.asarray(item) for item in items)
        total_size = sum(item.nbytes for item in items)

        nb = np.array([total_size], dtype=self._header_dtype)

        nb.tofile(self._fp)
        for item in items:
            item.tofile(self._fp)
        nb.tofile(self._fp)

    def read_record(self, *dtypes, **kwargs):
        """
        Reads a record of a given type from the file.

        Parameters
        ----------
        *dtypes : dtypes, optional
            Data type(s) specifying the size and endiness of the data.

        Returns
        -------
        data : ndarray
            A 1-D array object.

        Raises
        ------
        FortranEOFError
            To signal that no further records are available
        FortranFormattingError
            To signal that the end of the file was encountered
            part-way through a record

        Notes
        -----
        If the record contains a multidimensional array, you can specify
        the size in the dtype. For example::

            INTEGER var(5,4)

        can be read with::

            read_record('(4,5)i4').T

        Note that this function does **not** assume the file data is in Fortran
        column major order, so you need to (i) swap the order of dimensions
        when reading and (ii) transpose the resulting array.

        Alternatively, you can read the data as a 1-D array and handle the
        ordering yourself. For example::

            read_record('i4').reshape(5, 4, order='F')

        For records that contain several variables or mixed types (as opposed
        to single scalar or array types), give them as separate arguments::

            double precision :: a
            integer :: b
            write(1) a, b

            record = f.read_record('<f4', '<i4')
            a = record[0]  # first number
            b = record[1]  # second number

        and if any of the variables are arrays, the shape can be specified as
        the third item in the relevant dtype::

            double precision :: a
            integer :: b(3,4)
            write(1) a, b

            record = f.read_record('<f4', np.dtype(('<i4', (4, 3))))
            a = record[0]
            b = record[1].T

        NumPy also supports a short syntax for this kind of type::

            record = f.read_record('<f4', '(3,3)<i4')

        See Also
        --------
        read_reals
        read_ints

        """
        dtype = kwargs.pop('dtype', None)
        if kwargs:
            raise ValueError("Unknown keyword arguments {}".format(tuple(kwargs.keys())))

        if dtype is not None:
            dtypes = dtypes + (dtype,)
        elif not dtypes:
            raise ValueError('Must specify at least one dtype')

        first_size = self._read_size(eof_ok=True)

        dtypes = tuple(np.dtype(dtype) for dtype in dtypes)
        block_size = sum(dtype.itemsize for dtype in dtypes)

        num_blocks, remainder = divmod(first_size, block_size)
        if remainder != 0:
            raise ValueError('Size obtained ({0}) is not a multiple of the '
                             'dtypes given ({1}).'.format(first_size, block_size))

        if len(dtypes) != 1 and first_size != block_size:
            # Fortran does not write mixed type array items in interleaved order,
            # and it's not possible to guess the sizes of the arrays that were written.
            # The user must specify the exact sizes of each of the arrays.
            raise ValueError('Size obtained ({0}) does not match with the expected '
                             'size ({1}) of multi-item record'.format(first_size, block_size))

        data = []
        for dtype in dtypes:
            r = np.fromfile(self._fp, dtype=dtype, count=num_blocks)
            if len(r) != num_blocks:
                raise FortranFormattingError(
                    "End of file in the middle of a record")
            if dtype.shape != ():
                # Squeeze outmost block dimension for array items
                if num_blocks == 1:
                    assert r.shape == (1,) + dtype.shape
                    r = r[0]

            data.append(r)

        second_size = self._read_size()
        if first_size != second_size:
            raise IOError('Sizes do not agree in the header and footer for '
                          'this record - check header dtype')

        # Unpack result
        if len(dtypes) == 1:
            return data[0]
        else:
            return tuple(data)

    def read_ints(self, dtype='i4'):
        """
        Reads a record of a given type from the file, defaulting to an integer
        type (``INTEGER*4`` in Fortran).

        Parameters
        ----------
        dtype : dtype, optional
            Data type specifying the size and endiness of the data.

        Returns
        -------
        data : ndarray
            A 1-D array object.

        See Also
        --------
        read_reals
        read_record

        """
        return self.read_record(dtype)

    def read_reals(self, dtype='f8'):
        """
        Reads a record of a given type from the file, defaulting to a floating
        point number (``real*8`` in Fortran).

        Parameters
        ----------
        dtype : dtype, optional
            Data type specifying the size and endiness of the data.

        Returns
        -------
        data : ndarray
            A 1-D array object.

        See Also
        --------
        read_ints
        read_record

        """
        return self.read_record(dtype)

    def close(self):
        """
        Closes the file. It is unsupported to call any other methods off this
        object after closing it. Note that this class supports the 'with'
        statement in modern versions of Python, to call this automatically

        """
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
