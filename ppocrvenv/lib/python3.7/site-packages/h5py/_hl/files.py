# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements high-level support for HDF5 file objects.
"""

import sys
import os
from warnings import warn

from .compat import filename_decode, filename_encode

from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version

mpi = h5.get_config().mpi
ros3 = h5.get_config().ros3
hdf5_version = version.hdf5_version_tuple[0:3]

swmr_support = False
if hdf5_version >= h5.get_config().swmr_min_hdf5_version:
    swmr_support = True


libver_dict = {'earliest': h5f.LIBVER_EARLIEST, 'latest': h5f.LIBVER_LATEST}
libver_dict_r = dict((y, x) for x, y in libver_dict.items())
if hdf5_version >= (1, 10, 2):
    libver_dict.update({'v108': h5f.LIBVER_V18, 'v110': h5f.LIBVER_V110})
    libver_dict_r.update({h5f.LIBVER_V18: 'v108', h5f.LIBVER_V110: 'v110'})

if hdf5_version >= (1, 11, 4):
    libver_dict.update({'v112': h5f.LIBVER_V112})
    libver_dict_r.update({h5f.LIBVER_V112: 'v112'})

if hdf5_version >= (1, 13, 0):
    libver_dict.update({'v114': h5f.LIBVER_V114})
    libver_dict_r.update({h5f.LIBVER_V114: 'v114'})


def _set_fapl_mpio(plist, **kwargs):
    """Set file access property list for mpio driver"""
    if not mpi:
        raise ValueError("h5py was built without MPI support, can't use mpio driver")

    import mpi4py.MPI
    kwargs.setdefault('info', mpi4py.MPI.Info())
    plist.set_fapl_mpio(**kwargs)


def _set_fapl_fileobj(plist, **kwargs):
    """Set the Python file object driver in a file access property list"""
    plist.set_fileobj_driver(h5fd.fileobj_driver, kwargs.get('fileobj'))


_drivers = {
    'sec2': lambda plist, **kwargs: plist.set_fapl_sec2(**kwargs),
    'stdio': lambda plist, **kwargs: plist.set_fapl_stdio(**kwargs),
    'core': lambda plist, **kwargs: plist.set_fapl_core(**kwargs),
    'family': lambda plist, **kwargs: plist.set_fapl_family(
        memb_fapl=plist.copy(),
        **kwargs
    ),
    'mpio': _set_fapl_mpio,
    'fileobj': _set_fapl_fileobj,
    'split': lambda plist, **kwargs: plist.set_fapl_split(**kwargs),
}

if ros3:
    _drivers['ros3'] = lambda plist, **kwargs: plist.set_fapl_ros3(**kwargs)


def register_driver(name, set_fapl):
    """Register a custom driver.

    Parameters
    ----------
    name : str
        The name of the driver.
    set_fapl : callable[PropFAID, **kwargs] -> NoneType
        The function to set the fapl to use your custom driver.
    """
    _drivers[name] = set_fapl


def unregister_driver(name):
    """Unregister a custom driver.

    Parameters
    ----------
    name : str
        The name of the driver.
    """
    del _drivers[name]


def registered_drivers():
    """Return a frozenset of the names of all of the registered drivers.
    """
    return frozenset(_drivers)


def make_fapl(driver, libver, rdcc_nslots, rdcc_nbytes, rdcc_w0, locking,
              page_buf_size, min_meta_keep, min_raw_keep, **kwds):
    """ Set up a file access property list """
    plist = h5p.create(h5p.FILE_ACCESS)

    if libver is not None:
        if libver in libver_dict:
            low = libver_dict[libver]
            high = h5f.LIBVER_LATEST
        else:
            low, high = (libver_dict[x] for x in libver)
    else:
        # we default to earliest
        low, high = h5f.LIBVER_EARLIEST, h5f.LIBVER_LATEST
    plist.set_libver_bounds(low, high)

    cache_settings = list(plist.get_cache())
    if rdcc_nslots is not None:
        cache_settings[1] = rdcc_nslots
    if rdcc_nbytes is not None:
        cache_settings[2] = rdcc_nbytes
    if rdcc_w0 is not None:
        cache_settings[3] = rdcc_w0
    plist.set_cache(*cache_settings)

    if page_buf_size:
        plist.set_page_buffer_size(int(page_buf_size), int(min_meta_keep),
                                   int(min_raw_keep))
    if locking is not None:
        if hdf5_version < (1, 12, 1) and (hdf5_version[:2] != (1, 10) or hdf5_version[2] < 7):
            raise ValueError(
                "HDF version >= 1.12.1 or 1.10.x >= 1.10.7 required for file locking.")

        if locking in ("false", False):
            plist.set_file_locking(False, ignore_when_disabled=False)
        elif locking in ("true", True):
            plist.set_file_locking(True, ignore_when_disabled=False)
        elif locking == "best-effort":
            plist.set_file_locking(True, ignore_when_disabled=True)
        else:
            raise ValueError(f"Unsupported locking value: {locking}")

    if driver is None or (driver == 'windows' and sys.platform == 'win32'):
        # Prevent swallowing unused key arguments
        if kwds:
            msg = "'{key}' is an invalid keyword argument for this function" \
                  .format(key=next(iter(kwds)))
            raise TypeError(msg)
        return plist

    try:
        set_fapl = _drivers[driver]
    except KeyError:
        raise ValueError('Unknown driver type "%s"' % driver)
    else:
        set_fapl(plist, **kwds)

    return plist


def make_fcpl(track_order=False, fs_strategy=None, fs_persist=False,
              fs_threshold=1, fs_page_size=None):
    """ Set up a file creation property list """
    if track_order or fs_strategy:
        plist = h5p.create(h5p.FILE_CREATE)
        if track_order:
            plist.set_link_creation_order(
                h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
            plist.set_attr_creation_order(
                h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
        if fs_strategy:
            strategies = {
                'fsm': h5f.FSPACE_STRATEGY_FSM_AGGR,
                'page': h5f.FSPACE_STRATEGY_PAGE,
                'aggregate': h5f.FSPACE_STRATEGY_AGGR,
                'none': h5f.FSPACE_STRATEGY_NONE
            }
            fs_strat_num = strategies.get(fs_strategy, -1)
            if fs_strat_num == -1:
                raise ValueError("Invalid file space strategy type")

            plist.set_file_space_strategy(fs_strat_num, fs_persist, fs_threshold)
            if fs_page_size and fs_strategy == 'page':
                plist.set_file_space_page_size(int(fs_page_size))
    else:
        plist = None
    return plist


def make_fid(name, mode, userblock_size, fapl, fcpl=None, swmr=False):
    """ Get a new FileID by opening or creating a file.
    Also validates mode argument."""

    if userblock_size is not None:
        if mode in ('r', 'r+'):
            raise ValueError("User block may only be specified "
                             "when creating a file")
        try:
            userblock_size = int(userblock_size)
        except (TypeError, ValueError):
            raise ValueError("User block size must be an integer")
        if fcpl is None:
            fcpl = h5p.create(h5p.FILE_CREATE)
        fcpl.set_userblock(userblock_size)

    if mode == 'r':
        flags = h5f.ACC_RDONLY
        if swmr and swmr_support:
            flags |= h5f.ACC_SWMR_READ
        fid = h5f.open(name, flags, fapl=fapl)
    elif mode == 'r+':
        fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
    elif mode in ['w-', 'x']:
        fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
    elif mode == 'w':
        fid = h5f.create(name, h5f.ACC_TRUNC, fapl=fapl, fcpl=fcpl)
    elif mode == 'a':
        # Open in append mode (read/write).
        # If that fails, create a new file only if it won't clobber an
        # existing one (ACC_EXCL)
        try:
            fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
        # Not all drivers raise FileNotFoundError (commented those that do not)
        except FileNotFoundError if fapl.get_driver() in (
            h5fd.SEC2,
            # h5fd.STDIO,
            # h5fd.CORE,
            h5fd.FAMILY,
            h5fd.WINDOWS,
            # h5fd.MPIO,
            # h5fd.MPIPOSIX,
            h5fd.fileobj_driver,
            h5fd.ROS3D if ros3 else -1,
        ) else OSError:
            fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
    else:
        raise ValueError("Invalid mode; must be one of r, r+, w, w-, x, a")

    try:
        if userblock_size is not None:
            existing_fcpl = fid.get_create_plist()
            if existing_fcpl.get_userblock() != userblock_size:
                raise ValueError("Requested userblock size (%d) does not match that of existing file (%d)" % (userblock_size, existing_fcpl.get_userblock()))
    except Exception as e:
        fid.close()
        raise e

    return fid


class File(Group):

    """
        Represents an HDF5 file.
    """

    @property
    def attrs(self):
        """ Attributes attached to this object """
        # hdf5 complains that a file identifier is an invalid location for an
        # attribute. Instead of self, pass the root group to AttributeManager:
        from . import attrs
        with phil:
            return attrs.AttributeManager(self['/'])

    @property
    @with_phil
    def filename(self):
        """File name on disk"""
        return filename_decode(h5f.get_name(self.id))

    @property
    @with_phil
    def driver(self):
        """Low-level HDF5 file driver used to open file"""
        drivers = {h5fd.SEC2: 'sec2',
                   h5fd.STDIO: 'stdio',
                   h5fd.CORE: 'core',
                   h5fd.FAMILY: 'family',
                   h5fd.WINDOWS: 'windows',
                   h5fd.MPIO: 'mpio',
                   h5fd.MPIPOSIX: 'mpiposix',
                   h5fd.fileobj_driver: 'fileobj'}
        if ros3:
            drivers[h5fd.ROS3D] = 'ros3'
        return drivers.get(self.id.get_access_plist().get_driver(), 'unknown')

    @property
    @with_phil
    def mode(self):
        """ Python mode used to open file """
        write_intent = h5f.ACC_RDWR
        if swmr_support:
            write_intent |= h5f.ACC_SWMR_WRITE
        return 'r+' if self.id.get_intent() & write_intent else 'r'

    @property
    @with_phil
    def libver(self):
        """File format version bounds (2-tuple: low, high)"""
        bounds = self.id.get_access_plist().get_libver_bounds()
        return tuple(libver_dict_r[x] for x in bounds)

    @property
    @with_phil
    def userblock_size(self):
        """ User block size (in bytes) """
        fcpl = self.id.get_create_plist()
        return fcpl.get_userblock()

    if mpi and hdf5_version >= (1, 8, 9):

        @property
        @with_phil
        def atomic(self):
            """ Set/get MPI-IO atomic mode
            """
            return self.id.get_mpi_atomicity()

        @atomic.setter
        @with_phil
        def atomic(self, value):
            # pylint: disable=missing-docstring
            self.id.set_mpi_atomicity(value)

    @property
    @with_phil
    def swmr_mode(self):
        """ Controls single-writer multiple-reader mode """
        return swmr_support and bool(self.id.get_intent() & (h5f.ACC_SWMR_READ | h5f.ACC_SWMR_WRITE))

    @swmr_mode.setter
    @with_phil
    def swmr_mode(self, value):
        # pylint: disable=missing-docstring
        if swmr_support:
            if value:
                self.id.start_swmr_write()
            else:
                raise ValueError("It is not possible to forcibly switch SWMR mode off.")
        else:
            raise RuntimeError('SWMR support is not available in HDF5 version {}.{}.{}.'.format(*hdf5_version))

    def __init__(self, name, mode='r', driver=None, libver=None, userblock_size=None, swmr=False,
                 rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, track_order=None,
                 fs_strategy=None, fs_persist=False, fs_threshold=1, fs_page_size=None,
                 page_buf_size=None, min_meta_keep=0, min_raw_keep=0, locking=None, **kwds):
        """Create a new file object.

        See the h5py user guide for a detailed explanation of the options.

        name
            Name of the file on disk, or file-like object.  Note: for files
            created with the 'core' driver, HDF5 still requires this be
            non-empty.
        mode
            r        Readonly, file must exist (default)
            r+       Read/write, file must exist
            w        Create file, truncate if exists
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise
        driver
            Name of the driver to use.  Legal values are None (default,
            recommended), 'core', 'sec2', 'stdio', 'mpio', 'ros3'.
        libver
            Library version bounds.  Supported values: 'earliest', 'v108',
            'v110', 'v112'  and 'latest'. The 'v108', 'v110' and 'v112'
            options can only be specified with the HDF5 1.10.2 library or later.
        userblock_size
            Desired size of user block.  Only allowed when creating a new
            file (mode w, w- or x).
        swmr
            Open the file in SWMR read mode. Only used when mode = 'r'.
        rdcc_nbytes
            Total size of the raw data chunk cache in bytes. The default size
            is 1024**2 (1 MB) per dataset.
        rdcc_w0
            The chunk preemption policy for all datasets.  This must be
            between 0 and 1 inclusive and indicates the weighting according to
            which chunks which have been fully read or written are penalized
            when determining which chunks to flush from cache.  A value of 0
            means fully read or written chunks are treated no differently than
            other chunks (the preemption is strictly LRU) while a value of 1
            means fully read or written chunks are always preempted before
            other chunks.  If your application only reads or writes data once,
            this can be safely set to 1.  Otherwise, this should be set lower
            depending on how often you re-read or re-write the same data.  The
            default value is 0.75.
        rdcc_nslots
            The number of chunk slots in the raw data chunk cache for this
            file. Increasing this value reduces the number of cache collisions,
            but slightly increases the memory used. Due to the hashing
            strategy, this value should ideally be a prime number. As a rule of
            thumb, this value should be at least 10 times the number of chunks
            that can fit in rdcc_nbytes bytes. For maximum performance, this
            value should be set approximately 100 times that number of
            chunks. The default value is 521.
        track_order
            Track dataset/group/attribute creation order under root group
            if True. If None use global default h5.get_config().track_order.
        fs_strategy
            The file space handling strategy to be used.  Only allowed when
            creating a new file (mode w, w- or x).  Defined as:
            "fsm"        FSM, Aggregators, VFD
            "page"       Paged FSM, VFD
            "aggregate"  Aggregators, VFD
            "none"       VFD
            If None use HDF5 defaults.
        fs_page_size
            File space page size in bytes. Only used when fs_strategy="page". If
            None use the HDF5 default (4096 bytes).
        fs_persist
            A boolean value to indicate whether free space should be persistent
            or not.  Only allowed when creating a new file.  The default value
            is False.
        fs_threshold
            The smallest free-space section size that the free space manager
            will track.  Only allowed when creating a new file.  The default
            value is 1.
        page_buf_size
            Page buffer size in bytes. Only allowed for HDF5 files created with
            fs_strategy="page". Must be a power of two value and greater or
            equal than the file space page size when creating the file. It is
            not used by default.
        min_meta_keep
            Minimum percentage of metadata to keep in the page buffer before
            allowing pages containing metadata to be evicted. Applicable only if
            page_buf_size is set. Default value is zero.
        min_raw_keep
            Minimum percentage of raw data to keep in the page buffer before
            allowing pages containing raw data to be evicted. Applicable only if
            page_buf_size is set. Default value is zero.
        locking
            The file locking behavior. Defined as:
            False (or "false")  Disable file locking
            True (or "true")    Enable file locking
            "best-effort"       Enable file locking but ignore some errors
            None                Use HDF5 defaults
            Warning: The HDF5_USE_FILE_LOCKING environment variable can override
            this parameter.
            Only available with HDF5 >= 1.12.1 or 1.10.x >= 1.10.7.
        Additional keywords
            Passed on to the selected file driver.
        """
        if (fs_strategy or page_buf_size) and hdf5_version < (1, 10, 1):
            raise ValueError("HDF version 1.10.1 or greater required for file space strategy or page buffering support.")

        if swmr and not swmr_support:
            raise ValueError("The SWMR feature is not available in this version of the HDF5 library")

        if driver == 'ros3' and not ros3:
            raise ValueError(
                "h5py was built without ROS3 support, can't use ros3 driver")

        if locking is not None and hdf5_version < (1, 12, 1) and (
                hdf5_version[:2] != (1, 10) or hdf5_version[2] < 7):
            raise ValueError("HDF version >= 1.12.1 or 1.10.x >= 1.10.7 required for file locking options.")

        if isinstance(name, _objects.ObjectID):
            if fs_strategy:
                raise ValueError("Unable to set file space strategy of an existing file")

            with phil:
                fid = h5i.get_file_id(name)
        else:
            if hasattr(name, 'read') and hasattr(name, 'seek'):
                if driver not in (None, 'fileobj'):
                    raise ValueError("Driver must be 'fileobj' for file-like object if specified.")
                driver = 'fileobj'
                if kwds.get('fileobj', name) != name:
                    raise ValueError("Invalid value of 'fileobj' argument; "
                                     "must equal to file-like object if specified.")
                kwds.update(fileobj=name)
                name = repr(name).encode('ASCII', 'replace')
            else:
                name = filename_encode(name)

            if track_order is None:
                track_order = h5.get_config().track_order

            if fs_strategy and mode not in ('w', 'w-', 'x'):
                raise ValueError("Unable to set file space strategy of an existing file")

            if swmr and mode != 'r':
                warn(
                    "swmr=True only affects read ('r') mode. For swmr write "
                    "mode, set f.swmr_mode = True after opening the file.",
                    stacklevel=2,
                )

            with phil:
                fapl = make_fapl(driver, libver, rdcc_nslots, rdcc_nbytes, rdcc_w0,
                                 locking, page_buf_size, min_meta_keep, min_raw_keep, **kwds)
                fcpl = make_fcpl(track_order=track_order, fs_strategy=fs_strategy,
                                 fs_persist=fs_persist, fs_threshold=fs_threshold,
                                 fs_page_size=fs_page_size)
                fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)

            if isinstance(libver, tuple):
                self._libver = libver
            else:
                self._libver = (libver, 'latest')

        super(File, self).__init__(fid)

    def close(self):
        """ Close the file.  All open objects become invalid """
        with phil:
            # Check that the file is still open, otherwise skip
            if self.id.valid:
                # We have to explicitly murder all open objects related to the file

                # Close file-resident objects first, then the files.
                # Otherwise we get errors in MPI mode.
                self.id._close_open_objects(h5f.OBJ_LOCAL | ~h5f.OBJ_FILE)
                self.id._close_open_objects(h5f.OBJ_LOCAL | h5f.OBJ_FILE)

                self.id.close()
                _objects.nonlocal_close()

    def flush(self):
        """ Tell the HDF5 library to flush its buffers.
        """
        with phil:
            h5f.flush(self.id)

    @with_phil
    def __enter__(self):
        return self

    @with_phil
    def __exit__(self, *args):
        if self.id:
            self.close()

    @with_phil
    def __repr__(self):
        if not self.id:
            r = '<Closed HDF5 file>'
        else:
            # Filename has to be forced to Unicode if it comes back bytes
            # Mode is always a "native" string
            filename = self.filename
            if isinstance(filename, bytes):  # Can't decode fname
                filename = filename.decode('utf8', 'replace')
            r = f'<HDF5 file "{os.path.basename(filename)}" (mode {self.mode})>'

        return r
