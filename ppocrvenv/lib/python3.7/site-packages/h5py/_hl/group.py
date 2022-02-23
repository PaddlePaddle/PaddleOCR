# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements support for high-level access to HDF5 groups.
"""

from contextlib import contextmanager
import posixpath as pp
import numpy


from .compat import filename_decode, filename_encode

from .. import h5, h5g, h5i, h5o, h5r, h5t, h5l, h5p
from . import base
from .base import HLObject, MutableMappingHDF5, phil, with_phil
from . import dataset
from . import datatype
from .vds import vds_support


class Group(HLObject, MutableMappingHDF5):

    """ Represents an HDF5 group.
    """

    def __init__(self, bind):
        """ Create a new Group object by binding to a low-level GroupID.
        """
        with phil:
            if not isinstance(bind, h5g.GroupID):
                raise ValueError("%s is not a GroupID" % bind)
            super(Group, self).__init__(bind)


    _gcpl_crt_order = h5p.create(h5p.GROUP_CREATE)
    _gcpl_crt_order.set_link_creation_order(
        h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
    _gcpl_crt_order.set_attr_creation_order(
        h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)


    def create_group(self, name, track_order=None):
        """ Create and return a new subgroup.

        Name may be absolute or relative.  Fails if the target name already
        exists.

        track_order
            Track dataset/group/attribute creation order under this group
            if True. If None use global default h5.get_config().track_order.
        """
        if track_order is None:
            track_order = h5.get_config().track_order

        with phil:
            name, lcpl = self._e(name, lcpl=True)
            gcpl = Group._gcpl_crt_order if track_order else None
            gid = h5g.create(self.id, name, lcpl=lcpl, gcpl=gcpl)
            return Group(gid)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        """ Create a new HDF5 dataset

        name
            Name of the dataset (absolute or relative).  Provide None to make
            an anonymous dataset.
        shape
            Dataset shape.  Use "()" for scalar datasets.  Required if "data"
            isn't provided.
        dtype
            Numpy dtype or string.  If omitted, dtype('f') will be used.
            Required if "data" isn't provided; otherwise, overrides data
            array's dtype.
        data
            Provide data to initialize the dataset.  If used, you can omit
            shape and dtype arguments.

        Keyword-only arguments:

        chunks
            (Tuple or int) Chunk shape, or True to enable auto-chunking. Integers can
            be used for 1D shape.

        maxshape
            (Tuple or int) Make the dataset resizable up to this shape. Use None for
            axes you want to be unlimited. Integers can be used for 1D shape.
        compression
            (String or int) Compression strategy.  Legal values are 'gzip',
            'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        compression_opts
            Compression settings.  This is an integer for gzip, 2-tuple for
            szip, etc. If specifying a dynamically loaded compression filter
            number, this must be a tuple of values.
        scaleoffset
            (Integer) Enable scale/offset filter for (usually) lossy
            compression of integer or floating-point data. For integer
            data, the value of scaleoffset is the number of bits to
            retain (pass 0 to let HDF5 determine the minimum number of
            bits necessary for lossless compression). For floating point
            data, scaleoffset is the number of digits after the decimal
            place to retain; stored values thus have absolute error
            less than 0.5*10**(-scaleoffset).
        shuffle
            (T/F) Enable shuffle filter.
        fletcher32
            (T/F) Enable fletcher32 error detection. Not permitted in
            conjunction with the scale/offset filter.
        fillvalue
            (Scalar) Use this value for uninitialized parts of the dataset.
        track_times
            (T/F) Enable dataset creation timestamps.
        track_order
            (T/F) Track attribute creation order if True. If omitted use
            global default h5.get_config().track_order.
        external
            (Iterable of tuples) Sets the external storage property, thus
            designating that the dataset will be stored in one or more
            non-HDF5 files external to the HDF5 file.  Adds each tuple
            of (name, offset, size) to the dataset's list of external files.
            Each name must be a str, bytes, or os.PathLike; each offset and
            size, an integer.  If only a name is given instead of an iterable
            of tuples, it is equivalent to [(name, 0, h5py.h5f.UNLIMITED)].
        allow_unknown_filter
            (T/F) Do not check that the requested filter is available for use.
            This should only be used with ``write_direct_chunk``, where the caller
            compresses the data before handing it to h5py.
        """
        if 'track_order' not in kwds:
            kwds['track_order'] = h5.get_config().track_order

        with phil:
            group = self
            if name:
                name = self._e(name)
                if b'/' in name.lstrip(b'/'):
                    parent_path, name = name.rsplit(b'/', 1)
                    group = self.require_group(parent_path)

            dsid = dataset.make_new_dset(group, shape, dtype, data, name, **kwds)
            dset = dataset.Dataset(dsid)
            return dset

    if vds_support:
        def create_virtual_dataset(self, name, layout, fillvalue=None):
            """Create a new virtual dataset in this group.

            See virtual datasets in the docs for more information.

            name
                (str) Name of the new dataset

            layout
                (VirtualLayout) Defines the sources for the virtual dataset

            fillvalue
                The value to use where there is no data.

            """
            with phil:
                group = self

                if name:
                    name = self._e(name)
                    if b'/' in name.lstrip(b'/'):
                        parent_path, name = name.rsplit(b'/', 1)
                        group = self.require_group(parent_path)

                dsid = layout.make_dataset(
                    group, name=name, fillvalue=fillvalue,
                )
                dset = dataset.Dataset(dsid)

            return dset

        @contextmanager
        def build_virtual_dataset(
                self, name, shape, dtype, maxshape=None, fillvalue=None
        ):
            """Assemble a virtual dataset in this group.

            This is used as a context manager::

                with f.build_virtual_dataset('virt', (10, 1000), np.uint32) as layout:
                    layout[0] = h5py.VirtualSource('foo.h5', 'data', (1000,))

            name
                (str) Name of the new dataset
            shape
                (tuple) Shape of the dataset
            dtype
                A numpy dtype for data read from the virtual dataset
            maxshape
                (tuple, optional) Maximum dimensions if the dataset can grow.
                Use None for unlimited dimensions.
            fillvalue
                The value used where no data is available.
            """
            from .vds import VirtualLayout
            layout = VirtualLayout(shape, dtype, maxshape, self.file.filename)
            yield layout

            self.create_virtual_dataset(name, layout, fillvalue)

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        """ Open a dataset, creating it if it doesn't exist.

        If keyword "exact" is False (default), an existing dataset must have
        the same shape and a conversion-compatible dtype to be returned.  If
        True, the shape and dtype must match exactly.

        Other dataset keywords (see create_dataset) may be provided, but are
        only used if a new dataset is to be created.

        Raises TypeError if an incompatible object already exists, or if the
        shape or dtype don't match according to the above rules.
        """
        with phil:
            if not name in self:
                return self.create_dataset(name, *(shape, dtype), **kwds)

            if isinstance(shape, int):
                shape = (shape,)

            dset = self[name]
            if not isinstance(dset, dataset.Dataset):
                raise TypeError("Incompatible object (%s) already exists" % dset.__class__.__name__)

            if not shape == dset.shape:
                raise TypeError("Shapes do not match (existing %s vs new %s)" % (dset.shape, shape))

            if exact:
                if not dtype == dset.dtype:
                    raise TypeError("Datatypes do not exactly match (existing %s vs new %s)" % (dset.dtype, dtype))
            elif not numpy.can_cast(dtype, dset.dtype):
                raise TypeError("Datatypes cannot be safely cast (existing %s vs new %s)" % (dset.dtype, dtype))

            return dset

    def create_dataset_like(self, name, other, **kwupdate):
        """ Create a dataset similar to `other`.

        name
            Name of the dataset (absolute or relative).  Provide None to make
            an anonymous dataset.
        other
            The dataset which the new dataset should mimic. All properties, such
            as shape, dtype, chunking, ... will be taken from it, but no data
            or attributes are being copied.

        Any dataset keywords (see create_dataset) may be provided, including
        shape and dtype, in which case the provided values take precedence over
        those from `other`.
        """
        for k in ('shape', 'dtype', 'chunks', 'compression',
                  'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
                  'fillvalue'):
            kwupdate.setdefault(k, getattr(other, k))
        # TODO: more elegant way to pass these (dcpl to create_dataset?)
        dcpl = other.id.get_create_plist()
        kwupdate.setdefault('track_times', dcpl.get_obj_track_times())
        kwupdate.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

        # Special case: the maxshape property always exists, but if we pass it
        # to create_dataset, the new dataset will automatically get chunked
        # layout. So we copy it only if it is different from shape.
        if other.maxshape != other.shape:
            kwupdate.setdefault('maxshape', other.maxshape)

        return self.create_dataset(name, **kwupdate)

    def require_group(self, name):
        # TODO: support kwargs like require_dataset
        """Return a group, creating it if it doesn't exist.

        TypeError is raised if something with that name already exists that
        isn't a group.
        """
        with phil:
            if not name in self:
                return self.create_group(name)
            grp = self[name]
            if not isinstance(grp, Group):
                raise TypeError("Incompatible object (%s) already exists" % grp.__class__.__name__)
            return grp

    @with_phil
    def __getitem__(self, name):
        """ Open an object in the file """

        if isinstance(name, h5r.Reference):
            oid = h5r.dereference(name, self.id)
            if oid is None:
                raise ValueError("Invalid HDF5 object reference")
        elif isinstance(name, (bytes, str)):
            oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
        else:
            raise TypeError("Accessing a group is done with bytes or str, "
                            " not {}".format(type(name)))

        otype = h5i.get_type(oid)
        if otype == h5i.GROUP:
            return Group(oid)
        elif otype == h5i.DATASET:
            return dataset.Dataset(oid, readonly=(self.file.mode == 'r'))
        elif otype == h5i.DATATYPE:
            return datatype.Datatype(oid)
        else:
            raise TypeError("Unknown object type")

    def get(self, name, default=None, getclass=False, getlink=False):
        """ Retrieve an item or other information.

        "name" given only:
            Return the item, or "default" if it doesn't exist

        "getclass" is True:
            Return the class of object (Group, Dataset, etc.), or "default"
            if nothing with that name exists

        "getlink" is True:
            Return HardLink, SoftLink or ExternalLink instances.  Return
            "default" if nothing with that name exists.

        "getlink" and "getclass" are True:
            Return HardLink, SoftLink and ExternalLink classes.  Return
            "default" if nothing with that name exists.

        Example:

        >>> cls = group.get('foo', getclass=True)
        >>> if cls == SoftLink:
        """
        # pylint: disable=arguments-differ

        with phil:
            if not (getclass or getlink):
                try:
                    return self[name]
                except KeyError:
                    return default

            if not name in self:
                return default

            elif getclass and not getlink:
                typecode = h5o.get_info(self.id, self._e(name)).type

                try:
                    return {h5o.TYPE_GROUP: Group,
                            h5o.TYPE_DATASET: dataset.Dataset,
                            h5o.TYPE_NAMED_DATATYPE: datatype.Datatype}[typecode]
                except KeyError:
                    raise TypeError("Unknown object type")

            elif getlink:
                typecode = self.id.links.get_info(self._e(name)).type

                if typecode == h5l.TYPE_SOFT:
                    if getclass:
                        return SoftLink
                    linkbytes = self.id.links.get_val(self._e(name))
                    return SoftLink(self._d(linkbytes))

                elif typecode == h5l.TYPE_EXTERNAL:
                    if getclass:
                        return ExternalLink
                    filebytes, linkbytes = self.id.links.get_val(self._e(name))
                    return ExternalLink(
                        filename_decode(filebytes), self._d(linkbytes)
                    )

                elif typecode == h5l.TYPE_HARD:
                    return HardLink if getclass else HardLink()

                else:
                    raise TypeError("Unknown link type")

    def __setitem__(self, name, obj):
        """ Add an object to the group.  The name must not already be in use.

        The action taken depends on the type of object assigned:

        Named HDF5 object (Dataset, Group, Datatype)
            A hard link is created at "name" which points to the
            given object.

        SoftLink or ExternalLink
            Create the corresponding link.

        Numpy ndarray
            The array is converted to a dataset object, with default
            settings (contiguous storage, etc.).

        Numpy dtype
            Commit a copy of the datatype as a named datatype in the file.

        Anything else
            Attempt to convert it to an ndarray and store it.  Scalar
            values are stored as scalar datasets. Raise ValueError if we
            can't understand the resulting array dtype.
        """
        with phil:
            name, lcpl = self._e(name, lcpl=True)

            if isinstance(obj, HLObject):
                h5o.link(obj.id, self.id, name, lcpl=lcpl, lapl=self._lapl)

            elif isinstance(obj, SoftLink):
                self.id.links.create_soft(name, self._e(obj.path),
                              lcpl=lcpl, lapl=self._lapl)

            elif isinstance(obj, ExternalLink):
                fn = filename_encode(obj.filename)
                self.id.links.create_external(name, fn, self._e(obj.path),
                                              lcpl=lcpl, lapl=self._lapl)

            elif isinstance(obj, numpy.dtype):
                htype = h5t.py_create(obj, logical=True)
                htype.commit(self.id, name, lcpl=lcpl)

            else:
                ds = self.create_dataset(None, data=obj)
                h5o.link(ds.id, self.id, name, lcpl=lcpl)

    @with_phil
    def __delitem__(self, name):
        """ Delete (unlink) an item from this group. """
        self.id.unlink(self._e(name))

    @with_phil
    def __len__(self):
        """ Number of members attached to this group """
        return self.id.get_num_objs()

    @with_phil
    def __iter__(self):
        """ Iterate over member names """
        for x in self.id.__iter__():
            yield self._d(x)

    @with_phil
    def __reversed__(self):
        """ Iterate over member names in reverse order. """
        for x in self.id.__reversed__():
            yield self._d(x)

    @with_phil
    def __contains__(self, name):
        """ Test if a member name exists """
        return self._e(name) in self.id

    def copy(self, source, dest, name=None,
             shallow=False, expand_soft=False, expand_external=False,
             expand_refs=False, without_attrs=False):
        """Copy an object or group.

        The source can be a path, Group, Dataset, or Datatype object.  The
        destination can be either a path or a Group object.  The source and
        destinations need not be in the same file.

        If the source is a Group object, all objects contained in that group
        will be copied recursively.

        When the destination is a Group object, by default the target will
        be created in that group with its current name (basename of obj.name).
        You can override that by setting "name" to a string.

        There are various options which all default to "False":

         - shallow: copy only immediate members of a group.

         - expand_soft: expand soft links into new objects.

         - expand_external: expand external links into new objects.

         - expand_refs: copy objects that are pointed to by references.

         - without_attrs: copy object without copying attributes.

       Example:

        >>> f = File('myfile.hdf5', 'w')
        >>> f.create_group("MyGroup")
        >>> list(f.keys())
        ['MyGroup']
        >>> f.copy('MyGroup', 'MyCopy')
        >>> list(f.keys())
        ['MyGroup', 'MyCopy']

        """
        with phil:
            if isinstance(source, HLObject):
                source_path = '.'
            else:
                # Interpret source as a path relative to this group
                source_path = source
                source = self

            if isinstance(dest, Group):
                if name is not None:
                    dest_path = name
                elif source_path == '.':
                    dest_path = pp.basename(h5i.get_name(source.id))
                else:
                    # copy source into dest group: dest_name/source_name
                    dest_path = pp.basename(h5i.get_name(source[source_path].id))

            elif isinstance(dest, HLObject):
                raise TypeError("Destination must be path or Group object")
            else:
                # Interpret destination as a path relative to this group
                dest_path = dest
                dest = self

            flags = 0
            if shallow:
                flags |= h5o.COPY_SHALLOW_HIERARCHY_FLAG
            if expand_soft:
                flags |= h5o.COPY_EXPAND_SOFT_LINK_FLAG
            if expand_external:
                flags |= h5o.COPY_EXPAND_EXT_LINK_FLAG
            if expand_refs:
                flags |= h5o.COPY_EXPAND_REFERENCE_FLAG
            if without_attrs:
                flags |= h5o.COPY_WITHOUT_ATTR_FLAG
            if flags:
                copypl = h5p.create(h5p.OBJECT_COPY)
                copypl.set_copy_object(flags)
            else:
                copypl = None

            h5o.copy(source.id, self._e(source_path), dest.id, self._e(dest_path),
                     copypl, base.dlcpl)

    def move(self, source, dest):
        """ Move a link to a new location in the file.

        If "source" is a hard link, this effectively renames the object.  If
        "source" is a soft or external link, the link itself is moved, with its
        value unmodified.
        """
        with phil:
            if source == dest:
                return
            self.id.links.move(self._e(source), self.id, self._e(dest),
                               lapl=self._lapl, lcpl=self._lcpl)

    def visit(self, func):
        """ Recursively visit all names in this group and subgroups (HDF5 1.8).

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:

            func(<member name>) => <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guaranteed.

        Example:

        >>> # List the entire contents of the file
        >>> f = File("foo.hdf5")
        >>> list_of_names = []
        >>> f.visit(list_of_names.append)
        """
        with phil:
            def proxy(name):
                """ Call the function with the text name, not bytes """
                return func(self._d(name))
            return h5o.visit(self.id, proxy)

    def visititems(self, func):
        """ Recursively visit names and objects in this group (HDF5 1.8).

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:

            func(<member name>, <object>) => <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guaranteed.

        Example:

        # Get a list of all datasets in the file
        >>> mylist = []
        >>> def func(name, obj):
        ...     if isinstance(obj, Dataset):
        ...         mylist.append(name)
        ...
        >>> f = File('foo.hdf5')
        >>> f.visititems(func)
        """
        with phil:
            def proxy(name):
                """ Use the text name of the object, not bytes """
                name = self._d(name)
                return func(name, self[name])
            return h5o.visit(self.id, proxy)

    @with_phil
    def __repr__(self):
        if not self:
            r = u"<Closed HDF5 group>"
        else:
            namestr = (
                '"%s"' % self.name
            ) if self.name is not None else u"(anonymous)"
            r = '<HDF5 group %s (%d members)>' % (namestr, len(self))

        return r


class HardLink(object):

    """
        Represents a hard link in an HDF5 file.  Provided only so that
        Group.get works in a sensible way.  Has no other function.
    """

    pass


class SoftLink(object):

    """
        Represents a symbolic ("soft") link in an HDF5 file.  The path
        may be absolute or relative.  No checking is performed to ensure
        that the target actually exists.
    """

    @property
    def path(self):
        """ Soft link value.  Not guaranteed to be a valid path. """
        return self._path

    def __init__(self, path):
        self._path = str(path)

    def __repr__(self):
        return '<SoftLink to "%s">' % self.path


class ExternalLink(object):

    """
        Represents an HDF5 external link.  Paths may be absolute or relative.
        No checking is performed to ensure either the target or file exists.
    """

    @property
    def path(self):
        """ Soft link path, i.e. the part inside the HDF5 file. """
        return self._path

    @property
    def filename(self):
        """ Path to the external HDF5 file in the filesystem. """
        return self._filename

    def __init__(self, filename, path):
        self._filename = filename_decode(filename_encode(filename))
        self._path = path

    def __repr__(self):
        return '<ExternalLink to "%s" in file "%s"' % (self.path,
                                                       self.filename)
