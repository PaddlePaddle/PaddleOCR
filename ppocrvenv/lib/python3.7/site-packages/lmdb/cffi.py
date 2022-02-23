#
# Copyright 2013 The py-lmdb authors, all rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted only as authorized by the OpenLDAP
# Public License.
#
# A copy of this license is available in the file LICENSE in the
# top-level directory of the distribution or, alternatively, at
# <http://www.OpenLDAP.org/license.html>.
#
# OpenLDAP is a registered trademark of the OpenLDAP Foundation.
#
# Individual files and/or contributed packages may be copyright by
# other parties and/or subject to additional restrictions.
#
# This work also contains materials derived from public sources.
#
# Additional information about OpenLDAP can be obtained at
# <http://www.openldap.org/>.
#

"""
CPython/CFFI wrapper for OpenLDAP's "Lightning" MDB database.

Please see https://lmdb.readthedocs.io/
"""

from __future__ import absolute_import
from __future__ import with_statement

import errno
import inspect
import os
import sys
import threading

is_win32 = sys.platform == 'win32'
if is_win32:
    import msvcrt

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__  # type: ignore

import lmdb
try:
    from lmdb import _config
except ImportError:
    _config = None  # type: ignore


__all__ = [
    'Cursor',
    'Environment',
    'Transaction',
    '_Database',
    'enable_drop_gil',
    'version',
]

__all__ += [
    'BadDbiError',
    'BadRslotError',
    'BadTxnError',
    'BadValsizeError',
    'CorruptedError',
    'CursorFullError',
    'DbsFullError',
    'DiskError',
    'Error',
    'IncompatibleError',
    'InvalidError',
    'InvalidParameterError',
    'KeyExistsError',
    'LockError',
    'MapFullError',
    'MapResizedError',
    'MemoryError',
    'NotFoundError',
    'PageFullError',
    'PageNotFoundError',
    'PanicError',
    'ReadersFullError',
    'ReadonlyError',
    'TlsFullError',
    'TxnFullError',
    'VersionMismatchError',
]


# Handle moronic Python 3 mess.
UnicodeType = getattr(__builtin__, 'unicode', str)
BytesType = getattr(__builtin__, 'bytes', str)

O_0755 = int('0755', 8)
O_0111 = int('0111', 8)
EMPTY_BYTES = UnicodeType().encode()


# Used to track context across CFFI callbacks.
_callbacks = threading.local()

_CFFI_CDEF = '''
    typedef int mode_t;
    typedef ... MDB_env;
    typedef struct MDB_txn MDB_txn;
    typedef struct MDB_cursor MDB_cursor;
    typedef unsigned int MDB_dbi;
    enum MDB_cursor_op {
        MDB_FIRST,
        MDB_FIRST_DUP,
        MDB_GET_BOTH,
        MDB_GET_BOTH_RANGE,
        MDB_GET_CURRENT,
        MDB_GET_MULTIPLE,
        MDB_LAST,
        MDB_LAST_DUP,
        MDB_NEXT,
        MDB_NEXT_DUP,
        MDB_NEXT_MULTIPLE,
        MDB_NEXT_NODUP,
        MDB_PREV,
        MDB_PREV_DUP,
        MDB_PREV_NODUP,
        MDB_SET,
        MDB_SET_KEY,
        MDB_SET_RANGE,
        ...
    };
    typedef enum MDB_cursor_op MDB_cursor_op;

    struct MDB_val {
        size_t mv_size;
        void *mv_data;
        ...;
    };
    typedef struct MDB_val MDB_val;

    struct MDB_stat {
        unsigned int ms_psize;
        unsigned int ms_depth;
        size_t ms_branch_pages;
        size_t ms_leaf_pages;
        size_t ms_overflow_pages;
        size_t ms_entries;
        ...;
    };
    typedef struct MDB_stat MDB_stat;

    struct MDB_envinfo {
        void *me_mapaddr;
        size_t me_mapsize;
        size_t me_last_pgno;
        size_t me_last_txnid;
        unsigned int me_maxreaders;
        unsigned int me_numreaders;
        ...;
    };
    typedef struct MDB_envinfo MDB_envinfo;

    typedef int (*MDB_cmp_func)(const MDB_val *a, const MDB_val *b);
    typedef void (*MDB_rel_func)(MDB_val *item, void *oldptr, void *newptr,
                   void *relctx);

    char *mdb_strerror(int err);
    int mdb_env_create(MDB_env **env);
    int mdb_env_open(MDB_env *env, const char *path, unsigned int flags,
                     mode_t mode);
    int mdb_env_copy2(MDB_env *env, const char *path, int flags);
    int mdb_env_copyfd2(MDB_env *env, int fd, int flags);
    int mdb_env_stat(MDB_env *env, MDB_stat *stat);
    int mdb_env_info(MDB_env *env, MDB_envinfo *stat);
    int mdb_env_get_maxkeysize(MDB_env *env);
    int mdb_env_sync(MDB_env *env, int force);
    void mdb_env_close(MDB_env *env);
    int mdb_env_set_flags(MDB_env *env, unsigned int flags, int onoff);
    int mdb_env_get_flags(MDB_env *env, unsigned int *flags);
    int mdb_env_get_path(MDB_env *env, const char **path);
    int mdb_env_set_mapsize(MDB_env *env, size_t size);
    int mdb_env_set_maxreaders(MDB_env *env, unsigned int readers);
    int mdb_env_get_maxreaders(MDB_env *env, unsigned int *readers);
    int mdb_env_set_maxdbs(MDB_env *env, MDB_dbi dbs);
    int mdb_txn_begin(MDB_env *env, MDB_txn *parent, unsigned int flags,
                      MDB_txn **txn);
    int mdb_txn_commit(MDB_txn *txn);
    void mdb_txn_reset(MDB_txn *txn);
    int mdb_txn_renew(MDB_txn *txn);
    void mdb_txn_abort(MDB_txn *txn);
    size_t mdb_txn_id(MDB_txn *txn);
    int mdb_dbi_open(MDB_txn *txn, const char *name, unsigned int flags,
                     MDB_dbi *dbi);
    int mdb_stat(MDB_txn *txn, MDB_dbi dbi, MDB_stat *stat);
    int mdb_drop(MDB_txn *txn, MDB_dbi dbi, int del_);
    int mdb_get(MDB_txn *txn, MDB_dbi dbi, MDB_val *key, MDB_val *data);
    int mdb_cursor_open(MDB_txn *txn, MDB_dbi dbi, MDB_cursor **cursor);
    void mdb_cursor_close(MDB_cursor *cursor);
    int mdb_cursor_del(MDB_cursor *cursor, unsigned int flags);
    int mdb_cursor_count(MDB_cursor *cursor, size_t *countp);
    int mdb_cursor_get(MDB_cursor *cursor, MDB_val *key, MDB_val*data, int op);

    typedef int (MDB_msg_func)(const char *msg, void *ctx);
    int mdb_reader_list(MDB_env *env, MDB_msg_func *func, void *ctx);
    int mdb_reader_check(MDB_env *env, int *dead);
    int mdb_dbi_flags(MDB_txn *txn, MDB_dbi dbi, unsigned int *flags);

    #define MDB_VERSION_MAJOR ...
    #define MDB_VERSION_MINOR ...
    #define MDB_VERSION_PATCH ...

    #define EACCES ...
    #define EAGAIN ...
    #define EINVAL ...
    #define ENOMEM ...
    #define ENOSPC ...

    #define MDB_BAD_RSLOT ...
    #define MDB_BAD_DBI ...
    #define MDB_BAD_TXN ...
    #define MDB_BAD_VALSIZE ...
    #define MDB_CORRUPTED ...
    #define MDB_CURSOR_FULL ...
    #define MDB_DBS_FULL ...
    #define MDB_INCOMPATIBLE ...
    #define MDB_INVALID ...
    #define MDB_KEYEXIST ...
    #define MDB_MAP_FULL ...
    #define MDB_MAP_RESIZED ...
    #define MDB_NOTFOUND ...
    #define MDB_PAGE_FULL ...
    #define MDB_PAGE_NOTFOUND ...
    #define MDB_PANIC ...
    #define MDB_READERS_FULL ...
    #define MDB_TLS_FULL ...
    #define MDB_TXN_FULL ...
    #define MDB_VERSION_MISMATCH ...

    #define MDB_APPEND ...
    #define MDB_APPENDDUP ...
    #define MDB_CP_COMPACT ...
    #define MDB_CREATE ...
    #define MDB_DUPFIXED ...
    #define MDB_DUPSORT ...
    #define MDB_INTEGERDUP ...
    #define MDB_INTEGERKEY ...
    #define MDB_MAPASYNC ...
    #define MDB_NODUPDATA ...
    #define MDB_NOLOCK ...
    #define MDB_NOMEMINIT ...
    #define MDB_NOMETASYNC ...
    #define MDB_NOOVERWRITE ...
    #define MDB_NORDAHEAD ...
    #define MDB_NOSUBDIR ...
    #define MDB_NOSYNC ...
    #define MDB_NOTLS ...
    #define MDB_RDONLY ...
    #define MDB_REVERSEKEY ...
    #define MDB_WRITEMAP ...

    // Helpers below inline MDB_vals. Avoids key alloc/dup on CPython, where
    // CFFI will use PyString_AS_STRING when passed as an argument.
    static int pymdb_del(MDB_txn *txn, MDB_dbi dbi,
                         char *key_s, size_t keylen,
                         char *val_s, size_t vallen);
    static int pymdb_put(MDB_txn *txn, MDB_dbi dbi,
                         char *key_s, size_t keylen,
                         char *val_s, size_t vallen,
                         unsigned int flags);
    static int pymdb_get(MDB_txn *txn, MDB_dbi dbi,
                         char *key_s, size_t keylen,
                         MDB_val *val_out);
    static int pymdb_cursor_get(MDB_cursor *cursor,
                                char *key_s, size_t key_len,
                                char *data_s, size_t data_len,
                                MDB_val *key, MDB_val *data, int op);
    static int pymdb_cursor_put(MDB_cursor *cursor,
                                char *key_s, size_t keylen,
                                char *val_s, size_t vallen, int flags);

    // Prefaults a range
    static void preload(int rc, void *x, size_t size);

'''
_CFFI_CDEF_PATCHED = '''
    int mdb_env_copy3(MDB_env *env, const char *path, unsigned int flags, MDB_txn *txn);
    int mdb_env_copyfd3(MDB_env *env, int fd, unsigned int flags, MDB_txn *txn);
'''

_CFFI_VERIFY = '''
    #include <sys/stat.h>
    #include "lmdb.h"
    #include "preload.h"

    // Helpers below inline MDB_vals. Avoids key alloc/dup on CPython, where
    // CFFI will use PyString_AS_STRING when passed as an argument.
    static int pymdb_get(MDB_txn *txn, MDB_dbi dbi, char *key_s, size_t keylen,
                         MDB_val *val_out)
    {
        MDB_val key = {keylen, key_s};
        int rc = mdb_get(txn, dbi, &key, val_out);
        return rc;
    }

    static int pymdb_put(MDB_txn *txn, MDB_dbi dbi, char *key_s, size_t keylen,
                         char *val_s, size_t vallen, unsigned int flags)
    {
        MDB_val key = {keylen, key_s};
        MDB_val val = {vallen, val_s};
        return mdb_put(txn, dbi, &key, &val, flags);
    }

    static int pymdb_del(MDB_txn *txn, MDB_dbi dbi, char *key_s, size_t keylen,
                         char *val_s, size_t vallen)
    {
        MDB_val key = {keylen, key_s};
        MDB_val val = {vallen, val_s};
        MDB_val *valptr;
        if(vallen == 0) {
            valptr = NULL;
        } else {
            valptr = &val;
        }
        return mdb_del(txn, dbi, &key, valptr);
    }

    static int pymdb_cursor_get(MDB_cursor *cursor,
                                char *key_s, size_t key_len,
                                char *data_s, size_t data_len,
                                MDB_val *key, MDB_val *data, int op)
    {
        MDB_val tmp_key = {key_len, key_s};
        MDB_val tmp_data = {data_len, data_s};
        int rc = mdb_cursor_get(cursor, &tmp_key, &tmp_data, op);
        if(! rc) {
            *key = tmp_key;
            *data = tmp_data;
        }
        return rc;
    }

    static int pymdb_cursor_put(MDB_cursor *cursor, char *key_s, size_t keylen,
                                char *val_s, size_t vallen, int flags)
    {
        MDB_val tmpkey = {keylen, key_s};
        MDB_val tmpval = {vallen, val_s};
        return mdb_cursor_put(cursor, &tmpkey, &tmpval, flags);
    }

'''

if not lmdb._reading_docs():
    import cffi

    # Try to use distutils-bundled CFFI configuration to avoid a recompile and
    # potential compile errors during first module import.
    _config_vars = _config.CONFIG if _config else {
        'extra_compile_args': ['-w'],
        'extra_sources': ['lib/mdb.c', 'lib/midl.c'],
        'extra_include_dirs': ['lib'],
        'extra_library_dirs': [],
        'libraries': []
    }

    _have_patched_lmdb = '-DHAVE_PATCHED_LMDB=1' in _config.CONFIG['extra_compile_args']  # type: ignore

    if _have_patched_lmdb:
        _CFFI_CDEF += _CFFI_CDEF_PATCHED

    _ffi = cffi.FFI()
    _ffi.cdef(_CFFI_CDEF)
    _lib = _ffi.verify(_CFFI_VERIFY,
                       modulename='lmdb_cffi',
                       ext_package='lmdb',
                       sources=_config_vars['extra_sources'],
                       extra_compile_args=_config_vars['extra_compile_args'],
                       include_dirs=_config_vars['extra_include_dirs'],
                       libraries=_config_vars['libraries'],
                       library_dirs=_config_vars['extra_library_dirs'])

    @_ffi.callback("int(char *, void *)")
    def _msg_func(s, _):
        """mdb_msg_func() callback. Appends `s` to _callbacks.msg_func list.
        """
        _callbacks.msg_func.append(_ffi.string(s).decode())
        return 0

class Error(Exception):
    """Raised when an LMDB-related error occurs, and no more specific
    :py:class:`lmdb.Error` subclass exists."""
    def __init__(self, what, code=0):
        self.what = what
        self.code = code
        self.reason = _ffi.string(_lib.mdb_strerror(code))
        msg = what
        if code:
            msg = '%s: %s' % (what, self.reason)
            hint = getattr(self, 'MDB_HINT', None)
            if hint:
                msg += ' (%s)' % (hint,)
        Exception.__init__(self, msg)

class KeyExistsError(Error):
    """Key/data pair already exists."""
    MDB_NAME = 'MDB_KEYEXIST'

class NotFoundError(Error):
    """No matching key/data pair found.

    Normally py-lmdb indicates a missing key by returning ``None``, or a
    user-supplied default value, however LMDB may return this error where
    py-lmdb does not know to convert it into a non-exceptional return.
    """
    MDB_NAME = 'MDB_NOTFOUND'

class PageNotFoundError(Error):
    """Request page not found."""
    MDB_NAME = 'MDB_PAGE_NOTFOUND'

class CorruptedError(Error):
    """Located page was of the wrong type."""
    MDB_NAME = 'MDB_CORRUPTED'

class PanicError(Error):
    """Update of meta page failed."""
    MDB_NAME = 'MDB_PANIC'

class VersionMismatchError(Error):
    """Database environment version mismatch."""
    MDB_NAME = 'MDB_VERSION_MISMATCH'

class InvalidError(Error):
    """File is not an MDB file."""
    MDB_NAME = 'MDB_INVALID'

class MapFullError(Error):
    """Environment map_size= limit reached."""
    MDB_NAME = 'MDB_MAP_FULL'
    MDB_HINT = 'Please use a larger Environment(map_size=) parameter'

class DbsFullError(Error):
    """Environment max_dbs= limit reached."""
    MDB_NAME = 'MDB_DBS_FULL'
    MDB_HINT = 'Please use a larger Environment(max_dbs=) parameter'

class ReadersFullError(Error):
    """Environment max_readers= limit reached."""
    MDB_NAME = 'MDB_READERS_FULL'
    MDB_HINT = 'Please use a larger Environment(max_readers=) parameter'

class TlsFullError(Error):
    """Thread-local storage keys full - too many environments open."""
    MDB_NAME = 'MDB_TLS_FULL'

class TxnFullError(Error):
    """Transaciton has too many dirty pages - transaction too big."""
    MDB_NAME = 'MDB_TXN_FULL'
    MDB_HINT = 'Please do less work within your transaction'

class CursorFullError(Error):
    """Internal error - cursor stack limit reached."""
    MDB_NAME = 'MDB_CURSOR_FULL'

class PageFullError(Error):
    """Internal error - page has no more space."""
    MDB_NAME = 'MDB_PAGE_FULL'

class MapResizedError(Error):
    """Database contents grew beyond environment map_size=."""
    MDB_NAME = 'MDB_MAP_RESIZED'

class IncompatibleError(Error):
    """Operation and DB incompatible, or DB flags changed."""
    MDB_NAME = 'MDB_INCOMPATIBLE'

class BadRslotError(Error):
    """Invalid reuse of reader locktable slot."""
    MDB_NAME = 'MDB_BAD_RSLOT'

class BadDbiError(Error):
    """The specified DBI was changed unexpectedly."""
    MDB_NAME = 'MDB_BAD_DBI'

class BadTxnError(Error):
    """Transaction cannot recover - it must be aborted."""
    MDB_NAME = 'MDB_BAD_TXN'

class BadValsizeError(Error):
    """Too big key/data, key is empty, or wrong DUPFIXED size."""
    MDB_NAME = 'MDB_BAD_VALSIZE'

class ReadonlyError(Error):
    """An attempt was made to modify a read-only database."""
    MDB_NAME = 'EACCES'

class InvalidParameterError(Error):
    """An invalid parameter was specified."""
    MDB_NAME = 'EINVAL'

class LockError(Error):
    """The environment was locked by another process."""
    MDB_NAME = 'EAGAIN'

class MemoryError(Error):
    """Out of memory."""
    MDB_NAME = 'ENOMEM'

class DiskError(Error):
    """No more disk space."""
    MDB_NAME = 'ENOSPC'

# Prepare _error_map, a mapping of integer MDB_ERROR_CODE to exception class.
if not lmdb._reading_docs():
    _error_map = {}
    for obj in list(globals().values()):
        if inspect.isclass(obj) and issubclass(obj, Error) and obj is not Error:
            _error_map[getattr(_lib, obj.MDB_NAME)] = obj
    del obj

def _error(what, rc):
    """Lookup and instantiate the correct exception class for the error code
    `rc`, using :py:class:`Error` if no better class exists."""
    return _error_map.get(rc, Error)(what, rc)

class Some_LMDB_Resource_That_Was_Deleted_Or_Closed(object):
    """We need this because CFFI on PyPy treats None as cffi.NULL, instead of
    throwing an exception it feeds LMDB null pointers. That means simply
    replacing native handles with None during _invalidate() will cause NULL
    pointer dereferences. Instead use this class, and its weird name to cause a
    TypeError, with a very obvious string in the exception text.

    The only alternatives to this are inserting a check around every single use
    of a native handle to ensure the handle is still valid prior to calling
    LMDB, or doing no crash-safety checking at all.
    """
    def __nonzero__(self):
        return 0

    def __bool__(self):
        return False

    def __repr__(self):
        return "<This used to be a LMDB resource but it was deleted or closed>"
_invalid = Some_LMDB_Resource_That_Was_Deleted_Or_Closed()

def _mvbuf(mv):
    """Convert a MDB_val cdata to a CFFI buffer object."""
    return _ffi.buffer(mv.mv_data, mv.mv_size)

def _mvstr(mv):
    """Convert a MDB_val cdata to Python bytes."""
    return _ffi.buffer(mv.mv_data, mv.mv_size)[:]

def preload(mv):
    _lib.preload(0, mv.mv_data, mv.mv_size)

def enable_drop_gil():
    """Deprecated."""

def version(subpatch=False):
    """
    Return a tuple of integers `(major, minor, patch)` describing the LMDB
    library version that the binding is linked against. The version of the
    binding itself is available from ``lmdb.__version__``.

        `subpatch`:
            If true, returns a 4 integer tuple consisting of the same plus
            an extra integer that represents any patches applied by py-lmdb
            itself (0 representing no patches).

    """
    if subpatch:
        return (_lib.MDB_VERSION_MAJOR,
                _lib.MDB_VERSION_MINOR,
                _lib.MDB_VERSION_PATCH,
                1 if _have_patched_lmdb else 0)

    return (_lib.MDB_VERSION_MAJOR,
            _lib.MDB_VERSION_MINOR,
            _lib.MDB_VERSION_PATCH)


class Environment(object):
    """
    Structure for a database environment. An environment may contain multiple
    databases, all residing in the same shared-memory map and underlying disk
    file.

    To write to the environment a :py:class:`Transaction` must be created. One
    simultaneous write transaction is allowed, however there is no limit on the
    number of read transactions even when a write transaction exists.

    This class is aliased to `lmdb.open`.

    It is a serious error to have open the same LMDB file in the same process at
    the same time.  Failure to heed this may lead to data corruption and
    interpreter crash.

    Equivalent to `mdb_env_open()
    <http://lmdb.tech/doc/group__mdb.html#ga1fe2740e25b1689dc412e7b9faadba1b>`_

        `path`:
            Location of directory (if `subdir=True`) or file prefix to store
            the database.

        `map_size`:
            Maximum size database may grow to; used to size the memory mapping.
            If database grows larger than ``map_size``, an exception will be
            raised and the user must close and reopen :py:class:`Environment`.
            On 64-bit there is no penalty for making this huge (say 1TB). Must
            be <2GB on 32-bit.

            .. note::

                **The default map size is set low to encourage a crash**, so
                users can figure out a good value before learning about this
                option too late.

        `subdir`:
            If ``True``, `path` refers to a subdirectory to store the data and
            lock files in, otherwise it refers to a filename prefix.

        `readonly`:
            If ``True``, disallow any write operations. Note the lock file is
            still modified. If specified, the ``write`` flag to
            :py:meth:`begin` or :py:class:`Transaction` is ignored.

        `metasync`:
            If ``False``, flush system buffers to disk only once per
            transaction, omit the metadata flush. Defer that until the system
            flushes files to disk, or next commit or :py:meth:`sync`.

            This optimization maintains database integrity, but a system crash
            may undo the last committed transaction. I.e. it preserves the ACI
            (atomicity, consistency, isolation) but not D (durability) database
            property.

        `sync`:
            If ``False``, don't flush system buffers to disk when committing a
            transaction. This optimization means a system crash can corrupt the
            database or lose the last transactions if buffers are not yet
            flushed to disk.

            The risk is governed by how often the system flushes dirty buffers
            to disk and how often :py:meth:`sync` is called.  However, if the
            filesystem preserves write order and `writemap=False`, transactions
            exhibit ACI (atomicity, consistency, isolation) properties and only
            lose D (durability).  I.e. database integrity is maintained, but a
            system crash may undo the final transactions.

            Note that `sync=False, writemap=True` leaves the system with no
            hint for when to write transactions to disk, unless :py:meth:`sync`
            is called. `map_async=True, writemap=True` may be preferable.

        `mode`:
            File creation mode.

        `create`:
            If ``False``, do not create the directory `path` if it is missing.

        `readahead`:
            If ``False``, LMDB will disable the OS filesystem readahead
            mechanism, which may improve random read performance when a
            database is larger than RAM.

        `writemap`:
            If ``True``, use a writeable memory map unless `readonly=True`.
            This is faster and uses fewer mallocs, but loses protection from
            application bugs like wild pointer writes and other bad updates
            into the database. Incompatible with nested transactions.

            Processes with and without `writemap` on the same environment do
            not cooperate well.

        `meminit`:
            If ``False`` LMDB will not zero-initialize buffers prior to writing
            them to disk. This improves performance but may cause old heap data
            to be written saved in the unused portion of the buffer. Do not use
            this option if your application manipulates confidential data (e.g.
            plaintext passwords) in memory. This option is only meaningful when
            `writemap=False`; new pages are always zero-initialized when
            `writemap=True`.

        `map_async`:
             When ``writemap=True``, use asynchronous flushes to disk. As with
             ``sync=False``, a system crash can then corrupt the database or
             lose the last transactions. Calling :py:meth:`sync` ensures
             on-disk database integrity until next commit.

        `max_readers`:
            Maximum number of simultaneous read transactions. Can only be set
            by the first process to open an environment, as it affects the size
            of the lock file and shared memory area. Attempts to simultaneously
            start more than this many *read* transactions will fail.

        `max_dbs`:
            Maximum number of databases available. If 0, assume environment
            will be used as a single database.

        `max_spare_txns`:
            Read-only transactions to cache after becoming unused. Caching
            transactions avoids two allocations, one lock and linear scan
            of the shared environment per invocation of :py:meth:`begin`,
            :py:class:`Transaction`, :py:meth:`get`, :py:meth:`gets`, or
            :py:meth:`cursor`. Should match the process's maximum expected
            concurrent transactions (e.g. thread count).

        `lock`:
            If ``False``, don't do any locking. If concurrent access is
            anticipated, the caller must manage all concurrency itself. For
            proper operation the caller must enforce single-writer semantics,
            and must ensure that no readers are using old transactions while a
            writer is active. The simplest approach is to use an exclusive lock
            so that no readers may be active at all when a writer begins.
    """
    def __init__(self, path, map_size=10485760, subdir=True,
                 readonly=False, metasync=True, sync=True, map_async=False,
                 mode=O_0755, create=True, readahead=True, writemap=False,
                 meminit=True, max_readers=126, max_dbs=0, max_spare_txns=1,
                 lock=True):
        self._max_spare_txns = max_spare_txns
        self._spare_txns = []

        envpp = _ffi.new('MDB_env **')

        rc = _lib.mdb_env_create(envpp)
        if rc:
            raise _error("mdb_env_create", rc)
        self._env = envpp[0]
        self._deps = set()
        self._creating_db_in_readonly = False

        self.set_mapsize(map_size)

        rc = _lib.mdb_env_set_maxreaders(self._env, max_readers)
        if rc:
            raise _error("mdb_env_set_maxreaders", rc)

        rc = _lib.mdb_env_set_maxdbs(self._env, max_dbs)
        if rc:
            raise _error("mdb_env_set_maxdbs", rc)

        if create and subdir and not readonly:
            try:
                os.mkdir(path, mode)
            except EnvironmentError as e:
                if e.errno != errno.EEXIST:
                    raise

        flags = _lib.MDB_NOTLS
        if not subdir:
            flags |= _lib.MDB_NOSUBDIR
        if readonly:
            flags |= _lib.MDB_RDONLY
        self.readonly = readonly
        if not metasync:
            flags |= _lib.MDB_NOMETASYNC
        if not sync:
            flags |= _lib.MDB_NOSYNC
        if map_async:
            flags |= _lib.MDB_MAPASYNC
        if not readahead:
            flags |= _lib.MDB_NORDAHEAD
        if writemap:
            flags |= _lib.MDB_WRITEMAP
        if not meminit:
            flags |= _lib.MDB_NOMEMINIT
        if not lock:
            flags |= _lib.MDB_NOLOCK

        if isinstance(path, UnicodeType):
            path = path.encode(sys.getfilesystemencoding())

        rc = _lib.mdb_env_open(self._env, path, flags, mode & ~O_0111)
        if rc:
            raise _error(path, rc)

        with self.begin(db=object()) as txn:
            self._db = _Database(
                env=self,
                txn=txn,
                name=None,
                reverse_key=False,
                dupsort=False,
                create=True,
                integerkey=False,
                integerdup=False,
                dupfixed=False
            )

        self._dbs = {None: self._db}

    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        self.close()

    def __del__(self):
        self.close()

    _env = None
    _deps = None
    _spare_txns = None
    _dbs = None

    def set_mapsize(self, map_size):
        """Change the maximum size of the map file. This function will fail if
        any transactions are active in the current process.

        `map_size`:
            The new size in bytes.

        Equivalent to `mdb_env_set_mapsize()
        <http://lmdb.tech/doc/group__mdb.html#gaa2506ec8dab3d969b0e609cd82e619e5>`_

        Warning:
        There's a data race in the underlying library that may cause
        catastrophic loss of data if you use this method.

        You are safe if one of the following are true:
            * Only one process accessing a particular LMDB file ever calls
              this method.

            * You use locking external to this library to ensure that only one
              process accessing the current LMDB file can be inside this function.
        """
        rc = _lib.mdb_env_set_mapsize(self._env, map_size)
        if rc:
            raise _error("mdb_env_set_mapsize", rc)

    def close(self):
        """Close the environment, invalidating any open iterators, cursors, and
        transactions. Repeat calls to :py:meth:`close` have no effect.

        Equivalent to `mdb_env_close()
        <http://lmdb.tech/doc/group__mdb.html#ga4366c43ada8874588b6a62fbda2d1e95>`_
        """
        if self._env:
            if self._deps:
                while self._deps:
                    self._deps.pop()._invalidate()
            self._deps = None

            if self._spare_txns:
                while self._spare_txns:
                    _lib.mdb_txn_abort(self._spare_txns.pop())
            self._spare_txns = None

            if self._dbs:
                self._dbs.clear()
            self._dbs = None
            self._db = None

            _lib.mdb_env_close(self._env)
            self._env = _invalid

    def path(self):
        """Directory path or file name prefix where this environment is
        stored.

        Equivalent to `mdb_env_get_path()
        <http://lmdb.tech/doc/group__mdb.html#gac699fdd8c4f8013577cb933fb6a757fe>`_
        """
        path = _ffi.new('char **')
        rc = _lib.mdb_env_get_path(self._env, path)
        if rc:
            raise _error("mdb_env_get_path", rc)
        return _ffi.string(path[0]).decode(sys.getfilesystemencoding())

    def copy(self, path, compact=False, txn=None):
        """Make a consistent copy of the environment in the given destination
        directory.

        `compact`:
            If ``True``, perform compaction while copying: omit free pages and
            sequentially renumber all pages in output. This option consumes
            more CPU and runs more slowly than the default, but may produce a
            smaller output database.

        `txn`:
            If provided, the backup will be taken from the database with
            respect to that transaction, otherwise a temporary read-only
            transaction will be created.  Note:  this parameter being non-None
            is not available if the module was built with LMDB_PURE.  Note:
            this parameter may be set only if compact=True.

        Equivalent to `mdb_env_copy2() or mdb_env_copy3()
        <http://lmdb.tech/doc/group__mdb.html#ga5d51d6130325f7353db0955dbedbc378>`_
        """
        flags = _lib.MDB_CP_COMPACT if compact else 0
        if txn and not _have_patched_lmdb:
            raise TypeError("Non-patched LMDB doesn't support transaction with env.copy")

        if txn and not flags:
            raise TypeError("txn argument only compatible with compact=True")

        encoded = path.encode(sys.getfilesystemencoding())
        if _have_patched_lmdb:
            rc = _lib.mdb_env_copy3(self._env, encoded, flags, txn._txn if txn else _ffi.NULL)
            if rc:
                raise _error("mdb_env_copy3", rc)
        else:
            rc = _lib.mdb_env_copy2(self._env, encoded, flags)
            if rc:
                raise _error("mdb_env_copy2", rc)

    def copyfd(self, fd, compact=False, txn=None):
        """Copy a consistent version of the environment to file descriptor
        `fd`.

        `compact`:
            If ``True``, perform compaction while copying: omit free pages and
            sequentially renumber all pages in output. This option consumes
            more CPU and runs more slowly than the default, but may produce a
            smaller output database.

        `txn`:
            If provided, the backup will be taken from the database with
            respect to that transaction, otherwise a temporary read-only
            transaction will be created.  Note:  this parameter being non-None
            is not available if the module was built with LMDB_PURE.

        Equivalent to `mdb_env_copyfd2() or mdb_env_copyfd3
        <http://lmdb.tech/doc/group__mdb.html#ga5d51d6130325f7353db0955dbedbc378>`_
        """
        if txn and not _have_patched_lmdb:
            raise TypeError("Non-patched LMDB doesn't support transaction with env.copy")
        if is_win32:
            # Convert C library handle to kernel handle.
            fd = msvcrt.get_osfhandle(fd)
        flags = _lib.MDB_CP_COMPACT if compact else 0

        if txn and not flags:
            raise TypeError("txn argument only compatible with compact=True")

        if _have_patched_lmdb:
            rc = _lib.mdb_env_copyfd3(self._env, fd, flags, txn._txn if txn else _ffi.NULL)
            if rc:
                raise _error("mdb_env_copyfd3", rc)
        else:
            rc = _lib.mdb_env_copyfd2(self._env, fd, flags)
            if rc:
                raise _error("mdb_env_copyfd2", rc)

    def sync(self, force=False):
        """Flush the data buffers to disk.

        Equivalent to `mdb_env_sync()
        <http://lmdb.tech/doc/group__mdb.html#ga85e61f05aa68b520cc6c3b981dba5037>`_

        Data is always written to disk when :py:meth:`Transaction.commit` is
        called, but the operating system may keep it buffered. MDB always
        flushes the OS buffers upon commit as well, unless the environment was
        opened with `sync=False` or `metasync=False`.

        `force`:
            If ``True``, force a synchronous flush. Otherwise if the
            environment was opened with `sync=False` the flushes will be
            omitted, and with `map_async=True` they will be asynchronous.
        """
        rc = _lib.mdb_env_sync(self._env, force)
        if rc:
            raise _error("mdb_env_sync", rc)

    def _convert_stat(self, st):
        """Convert a MDB_stat to a dict.
        """
        return {
            "psize": st.ms_psize,
            "depth": st.ms_depth,
            "branch_pages": st.ms_branch_pages,
            "leaf_pages": st.ms_leaf_pages,
            "overflow_pages": st.ms_overflow_pages,
            "entries": st.ms_entries
        }

    def stat(self):
        """stat()

        Return some environment statistics for the default database as a dict:

        +--------------------+---------------------------------------+
        | ``psize``          | Size of a database page in bytes.     |
        +--------------------+---------------------------------------+
        | ``depth``          | Height of the B-tree.                 |
        +--------------------+---------------------------------------+
        | ``branch_pages``   | Number of internal (non-leaf) pages.  |
        +--------------------+---------------------------------------+
        | ``leaf_pages``     | Number of leaf pages.                 |
        +--------------------+---------------------------------------+
        | ``overflow_pages`` | Number of overflow pages.             |
        +--------------------+---------------------------------------+
        | ``entries``        | Number of data items.                 |
        +--------------------+---------------------------------------+

        Equivalent to `mdb_env_stat()
        <http://lmdb.tech/doc/group__mdb.html#gaf881dca452050efbd434cd16e4bae255>`_
        """
        st = _ffi.new('MDB_stat *')
        rc = _lib.mdb_env_stat(self._env, st)
        if rc:
            raise _error("mdb_env_stat", rc)
        return self._convert_stat(st)

    def info(self):
        """Return some nice environment information as a dict:

        +--------------------+---------------------------------------------+
        | ``map_addr``       | Address of database map in RAM.             |
        +--------------------+---------------------------------------------+
        | ``map_size``       | Size of database map in RAM.                |
        +--------------------+---------------------------------------------+
        | ``last_pgno``      | ID of last used page.                       |
        +--------------------+---------------------------------------------+
        | ``last_txnid``     | ID of last committed transaction.           |
        +--------------------+---------------------------------------------+
        | ``max_readers``    | Number of reader slots allocated in the     |
        |                    | lock file. Equivalent to the value of       |
        |                    | `maxreaders=` specified by the first        |
        |                    | process opening the Environment.            |
        +--------------------+---------------------------------------------+
        | ``num_readers``    | Maximum number of reader slots in           |
        |                    | simultaneous use since the lock file was    |
        |                    | initialized.                                |
        +--------------------+---------------------------------------------+

        Equivalent to `mdb_env_info()
        <http://lmdb.tech/doc/group__mdb.html#ga18769362c7e7d6cf91889a028a5c5947>`_
        """
        info = _ffi.new('MDB_envinfo *')
        rc = _lib.mdb_env_info(self._env, info)
        if rc:
            raise _error("mdb_env_info", rc)
        return {
            "map_addr": int(_ffi.cast('long', info.me_mapaddr)),
            "map_size": info.me_mapsize,
            "last_pgno": info.me_last_pgno,
            "last_txnid": info.me_last_txnid,
            "max_readers": info.me_maxreaders,
            "num_readers": info.me_numreaders
        }

    def flags(self):
        """Return a dict describing Environment constructor flags used to
        instantiate this environment."""
        flags_ = _ffi.new('unsigned int[]', 1)
        rc = _lib.mdb_env_get_flags(self._env, flags_)
        if rc:
            raise _error("mdb_env_get_flags", rc)
        flags = flags_[0]
        return {
            'subdir': not (flags & _lib.MDB_NOSUBDIR),
            'readonly': bool(flags & _lib.MDB_RDONLY),
            'metasync': not (flags & _lib.MDB_NOMETASYNC),
            'sync': not (flags & _lib.MDB_NOSYNC),
            'map_async': bool(flags & _lib.MDB_MAPASYNC),
            'readahead': not (flags & _lib.MDB_NORDAHEAD),
            'writemap': bool(flags & _lib.MDB_WRITEMAP),
            'meminit': not (flags & _lib.MDB_NOMEMINIT),
            'lock': not (flags & _lib.MDB_NOLOCK),
        }

    def max_key_size(self):
        """Return the maximum size in bytes of a record's key part. This
        matches the ``MDB_MAXKEYSIZE`` constant set at compile time."""
        return _lib.mdb_env_get_maxkeysize(self._env)

    def max_readers(self):
        """Return the maximum number of readers specified during open of the
        environment by the first process. This is the same as `max_readers=`
        specified to the constructor if this process was the first to open the
        environment."""
        readers_ = _ffi.new('unsigned int[]', 1)
        rc = _lib.mdb_env_get_maxreaders(self._env, readers_)
        if rc:
            raise _error("mdb_env_get_maxreaders", rc)
        return readers_[0]

    def readers(self):
        """Return a multi line Unicode string describing the current state of
        the reader lock table."""
        _callbacks.msg_func = []
        try:
            rc = _lib.mdb_reader_list(self._env, _msg_func, _ffi.NULL)
            if rc:
                raise _error("mdb_reader_list", rc)
            return UnicodeType().join(_callbacks.msg_func)
        finally:
            del _callbacks.msg_func

    def reader_check(self):
        """Search the reader lock table for stale entries, for example due to a
        crashed process. Returns the number of stale entries that were cleared.
        """
        reaped = _ffi.new('int[]', 1)
        rc = _lib.mdb_reader_check(self._env, reaped)
        if rc:
            raise _error('mdb_reader_check', rc)
        return reaped[0]

    def open_db(self, key=None, txn=None, reverse_key=False, dupsort=False,
                create=True, integerkey=False, integerdup=False,
                dupfixed=False):
        """
        Open a database, returning an instance of :py:class:`_Database`. Repeat
        :py:meth:`Environment.open_db` calls for the same name will return the
        same handle. As a special case, the main database is always open.

        Equivalent to `mdb_dbi_open()
        <http://lmdb.tech/doc/group__mdb.html#gac08cad5b096925642ca359a6d6f0562a>`_

        Named databases are implemented by *storing a special descriptor in the
        main database*. All databases in an environment *share the same file*.
        Because the descriptor is present in the main database, attempts to
        create a named database will fail if a key matching the database's name
        already exists. Furthermore *the key is visible to lookups and
        enumerations*. If your main database keyspace conflicts with the names
        you use for named databases, then move the contents of your main
        database to another named database.

        ::

            >>> env = lmdb.open('/tmp/test', max_dbs=2)
            >>> with env.begin(write=True) as txn:
            ...     txn.put('somename', 'somedata')

            >>> # Error: database cannot share name of existing key!
            >>> subdb = env.open_db('somename')

        A newly created database will not exist if the transaction that created
        it aborted, nor if another process deleted it. The handle resides in
        the shared environment, it is not owned by the current transaction or
        process. Only one thread should call this function; it is not
        mutex-protected in a read-only transaction.

        The `dupsort`, `integerkey`, `integerdup`, and `dupfixed` parameters are
        ignored if the database already exists.  The state of those settings are
        persistent and immutable per database.  See :py:meth:`_Database.flags`
        to view the state of those options for an opened database.  A consequence
        of the immutability of these flags is that the default non-named database
        will never have these flags set.

        Preexisting transactions, other than the current transaction and any
        parents, must not use the new handle, nor must their children.

            `key`:
                Bytestring database name. If ``None``, indicates the main
                database should be returned, otherwise indicates a named
                database should be created inside the main database.

                In other words, *a key representing the database will be
                visible in the main database, and the database name cannot
                conflict with any existing key.*

            `txn`:
                Transaction used to create the database if it does not exist.
                If unspecified, a temporarily write transaction is used. Do not
                call :py:meth:`open_db` from inside an existing transaction
                without supplying it here. Note the passed transaction must
                have `write=True`.

            `reverse_key`:
                If ``True``, keys are compared from right to left (e.g. DNS
                names).

            `dupsort`:
                Duplicate keys may be used in the database. (Or, from another
                perspective, keys may have multiple data items, stored in
                sorted order.) By default keys must be unique and may have only
                a single data item.

            `create`:
                If ``True``, create the database if it doesn't exist, otherwise
                raise an exception.

            `integerkey`:
                If ``True``, indicates keys in the database are C unsigned
                or ``size_t`` integers encoded in native byte order. Keys must
                all be either unsigned or ``size_t``, they cannot be mixed in a
                single database.

            `integerdup`:
                If ``True``, values in the
                database are C unsigned or ``size_t`` integers encode din
                native byte order.  Implies `dupsort` and `dupfixed` are
                ``True``.

            `dupfixed`:
                If ``True``, values for each key
                in database are of fixed size, allowing each additional
                duplicate value for a key to be stored without a header
                indicating its size.  Implies `dupsort` is ``True``.
        """
        if isinstance(key, UnicodeType):
            raise TypeError('key must be bytes')

        if key is None and (reverse_key or dupsort or integerkey or integerdup
                            or dupfixed):
            raise ValueError('May not set flags on the main database')

        db = self._dbs.get(key)
        if db:
            return db

        if integerdup:
            dupfixed = True

        if dupfixed:
            dupsort = True

        if txn:
            db = _Database(self, txn, key, reverse_key, dupsort, create,
                           integerkey, integerdup, dupfixed)
        else:
            try:
                self._creating_db_in_readonly = True
                with self.begin(write=not self.readonly) as txn:
                    db = _Database(self, txn, key, reverse_key, dupsort, create,
                                   integerkey, integerdup, dupfixed)
            finally:
                self._creating_db_in_readonly = False
        self._dbs[key] = db
        return db

    def begin(self, db=None, parent=None, write=False, buffers=False):
        """Shortcut for :py:class:`lmdb.Transaction`"""
        return Transaction(self, db, parent, write, buffers)


class _Database(object):
    """
    Internal database handle.  This class is opaque, save a single method.

    Should not be constructed directly.  Use :py:meth:`Environment.open_db`
    instead.
    """
    def __init__(self, env, txn, name, reverse_key, dupsort, create,
                 integerkey, integerdup, dupfixed):
        env._deps.add(self)
        self._deps = set()
        self._name = name

        flags = 0
        if reverse_key:
            flags |= _lib.MDB_REVERSEKEY
        if dupsort:
            flags |= _lib.MDB_DUPSORT
        if create:
            flags |= _lib.MDB_CREATE
        if integerkey:
            flags |= _lib.MDB_INTEGERKEY
        if integerdup:
            flags |= _lib.MDB_INTEGERDUP
        if dupfixed:
            flags |= _lib.MDB_DUPFIXED
        dbipp = _ffi.new('MDB_dbi *')
        self._dbi = None
        rc = _lib.mdb_dbi_open(txn._txn, name or _ffi.NULL, flags, dbipp)
        if rc:
            raise _error("mdb_dbi_open", rc)
        self._dbi = dbipp[0]
        self._load_flags(txn)

    def _load_flags(self, txn):
        """Load MDB's notion of the database flags."""
        flags_ = _ffi.new('unsigned int[]', 1)
        rc = _lib.mdb_dbi_flags(txn._txn, self._dbi, flags_)
        if rc:
            raise _error("mdb_dbi_flags", rc)
        self._flags = flags_[0]

    def flags(self, *args):
        """Return the database's associated flags as a dict of _Database
        constructor kwargs."""
        if len(args) > 1:
            raise TypeError('flags takes 0 or 1 arguments')

        return {
            'reverse_key': bool(self._flags & _lib.MDB_REVERSEKEY),
            'dupsort': bool(self._flags & _lib.MDB_DUPSORT),
            'integerkey': bool(self._flags & _lib.MDB_INTEGERKEY),
            'integerdup': bool(self._flags & _lib.MDB_INTEGERDUP),
            'dupfixed': bool(self._flags & _lib.MDB_DUPFIXED),
        }

    def _invalidate(self):
        self._dbi = _invalid

open = Environment


class Transaction(object):
    """
    A transaction object. All operations require a transaction handle,
    transactions may be read-only or read-write. Write transactions may not
    span threads. Transaction objects implement the context manager protocol,
    so that reliable release of the transaction happens even in the face of
    unhandled exceptions:

        .. code-block:: python

            # Transaction aborts correctly:
            with env.begin(write=True) as txn:
                crash()

            # Transaction commits automatically:
            with env.begin(write=True) as txn:
                txn.put('a', 'b')

    Equivalent to `mdb_txn_begin()
    <http://lmdb.tech/doc/group__mdb.html#gad7ea55da06b77513609efebd44b26920>`_

        `env`:
            Environment the transaction should be on.

        `db`:
            Default named database to operate on. If unspecified, defaults to
            the environment's main database. Can be overridden on a per-call
            basis below.

        `parent`:
            ``None``, or a parent transaction (see lmdb.h).

        `write`:
            Transactions are read-only by default. To modify the database, you
            must pass `write=True`. This flag is ignored if
            :py:class:`Environment` was opened with ``readonly=True``.

        `buffers`:
            If ``True``, indicates :py:func:`buffer` objects should be yielded
            instead of bytestrings. This setting applies to the
            :py:class:`Transaction` instance itself and any :py:class:`Cursors
            <Cursor>` created within the transaction.

            This feature significantly improves performance, since MDB has a
            zero-copy design, but it requires care when manipulating the
            returned buffer objects. The benefit of this facility is diminished
            when using small keys and values.
    """

    # If constructor fails, then __del__ will attempt to access these
    # attributes.
    _env = _invalid
    _txn = _invalid
    _parent = None
    _write = False

    # Mutations occurred since transaction start. Required to know when Cursor
    # key/value must be refreshed.
    _mutations = 0

    def __init__(self, env, db=None, parent=None, write=False, buffers=False):
        env._deps.add(self)
        self.env = env  # hold ref
        self._db = db or env._db
        self._env = env._env
        self._key = _ffi.new('MDB_val *')
        self._val = _ffi.new('MDB_val *')
        self._to_py = _mvbuf if buffers else _mvstr
        self._deps = set()

        if parent:
            self._parent = parent
            parent_txn = parent._txn
            parent._deps.add(self)
        else:
            parent_txn = _ffi.NULL

        if write:
            if env.readonly:
                msg = 'Cannot start write transaction with read-only env'
                raise _error(msg, _lib.EACCES)

            txnpp = _ffi.new('MDB_txn **')
            rc = _lib.mdb_txn_begin(self._env, parent_txn, 0, txnpp)
            if rc:
                raise _error("mdb_txn_begin", rc)
            self._txn = txnpp[0]
            self._write = True
        else:
            try:  # Exception catch in order to avoid racy 'if txns:' test
                if env._creating_db_in_readonly:  # Don't use spare txns for creating a DB when read-only
                    raise IndexError
                self._txn = env._spare_txns.pop()
                env._max_spare_txns += 1
                rc = _lib.mdb_txn_renew(self._txn)
                if rc:
                    while self._deps:
                        self._deps.pop()._invalidate()
                    _lib.mdb_txn_abort(self._txn)
                    self._txn = _invalid
                    self._invalidate()
                    raise _error("mdb_txn_renew", rc)
            except IndexError:
                txnpp = _ffi.new('MDB_txn **')
                flags = _lib.MDB_RDONLY
                rc = _lib.mdb_txn_begin(self._env, parent_txn, flags, txnpp)
                if rc:
                    raise _error("mdb_txn_begin", rc)
                self._txn = txnpp[0]

    def _invalidate(self):
        if self._txn:
            self.abort()
        self.env._deps.discard(self)
        self._parent = None
        self._env = _invalid

    def __del__(self):
        self.abort()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.abort()
        else:
            self.commit()

    def id(self):
        """id()

        Return the transaction's ID.

        This returns the identifier associated with this transaction. For a
        read-only transaction, this corresponds to the snapshot being read;
        concurrent readers will frequently have the same transaction ID.
        """
        return _lib.mdb_txn_id(self._txn)

    def stat(self, db):
        """stat(db)

        Return statistics like :py:meth:`Environment.stat`, except for a single
        DBI. `db` must be a database handle returned by :py:meth:`open_db`.
        """
        st = _ffi.new('MDB_stat *')
        rc = _lib.mdb_stat(self._txn, db._dbi, st)
        if rc:
            raise _error('mdb_stat', rc)
        return self.env._convert_stat(st)

    def drop(self, db, delete=True):
        """Delete all keys in a named database and optionally delete the named
        database itself. Deleting the named database causes it to become
        unavailable, and invalidates existing cursors.

        Equivalent to `mdb_drop()
        <http://lmdb.tech/doc/group__mdb.html#gab966fab3840fc54a6571dfb32b00f2db>`_
        """
        while db._deps:
            db._deps.pop()._invalidate()
        rc = _lib.mdb_drop(self._txn, db._dbi, delete)
        self._mutations += 1
        if rc:
            raise _error("mdb_drop", rc)
        if db._name in self.env._dbs:
            del self.env._dbs[db._name]

    def _cache_spare(self):
        # In order to avoid taking and maintaining a lock, a race is allowed
        # below which may result in more spare txns than desired. It seems
        # unlikely the race could ever result in a large amount of spare txns,
        # and in any case a correctly configured program should not be opening
        # more read-only transactions than there are configured spares.
        if self.env._max_spare_txns > 0:
            _lib.mdb_txn_reset(self._txn)
            self.env._spare_txns.append(self._txn)
            self.env._max_spare_txns -= 1
            self._txn = _invalid
            self._invalidate()
            return True

        return False

    def commit(self):
        """Commit the pending transaction.

        Equivalent to `mdb_txn_commit()
        <http://lmdb.tech/doc/group__mdb.html#ga846fbd6f46105617ac9f4d76476f6597>`_
        """
        while self._deps:
            self._deps.pop()._invalidate()
        if self._write or not self._cache_spare():
            rc = _lib.mdb_txn_commit(self._txn)
            self._txn = _invalid
            if rc:
                raise _error("mdb_txn_commit", rc)
            self._invalidate()

    def abort(self):
        """Abort the pending transaction. Repeat calls to :py:meth:`abort` have
        no effect after a previously successful :py:meth:`commit` or
        :py:meth:`abort`, or after the associated :py:class:`Environment` has
        been closed.

        Equivalent to `mdb_txn_abort()
        <http://lmdb.tech/doc/group__mdb.html#ga73a5938ae4c3239ee11efa07eb22b882>`_
        """
        if self._txn:
            while self._deps:
                self._deps.pop()._invalidate()
            if self._write or not self._cache_spare():
                rc = _lib.mdb_txn_abort(self._txn)
                self._txn = _invalid
                if rc:
                    raise _error("mdb_txn_abort", rc)
            self._invalidate()

    def get(self, key, default=None, db=None):
        """Fetch the first value matching `key`, returning `default` if `key`
        does not exist. A cursor must be used to fetch all values for a key in
        a `dupsort=True` database.

        Equivalent to `mdb_get()
        <http://lmdb.tech/doc/group__mdb.html#ga8bf10cd91d3f3a83a34d04ce6b07992d>`_
        """
        rc = _lib.pymdb_get(self._txn, (db or self._db)._dbi,
                            key, len(key), self._val)
        if rc:
            if rc == _lib.MDB_NOTFOUND:
                return default
            raise _error("mdb_cursor_get", rc)

        preload(self._val)
        return self._to_py(self._val)

    def put(self, key, value, dupdata=True, overwrite=True, append=False,
            db=None):
        """Store a record, returning ``True`` if it was written, or ``False``
        to indicate the key was already present and `overwrite=False`.
        On success, the cursor is positioned on the new record.

        Equivalent to `mdb_put()
        <http://lmdb.tech/doc/group__mdb.html#ga4fa8573d9236d54687c61827ebf8cac0>`_

            `key`:
                Bytestring key to store.

            `value`:
                Bytestring value to store.

            `dupdata`:
                If ``False`` and database was opened with `dupsort=True`, will return
                ``False`` if the key already has that value.  In other words, this only
                affects the return value.

            `overwrite`:
                If ``False``, do not overwrite any existing matching key.  If
                False and writing to a dupsort=True database, this will not add a value
                to the key and this function will return ``False``.

            `append`:
                If ``True``, append the pair to the end of the database without
                comparing its order first. Appending a key that is not greater
                than the highest existing key will fail and return ``False``.

            `db`:
                Named database to operate on. If unspecified, defaults to the
                database given to the :py:class:`Transaction` constructor.
        """
        flags = 0
        if not dupdata:
            flags |= _lib.MDB_NODUPDATA
        if not overwrite:
            flags |= _lib.MDB_NOOVERWRITE
        if append:
            flags |= _lib.MDB_APPEND

        rc = _lib.pymdb_put(self._txn, (db or self._db)._dbi,
                            key, len(key), value, len(value), flags)
        self._mutations += 1
        if rc:
            if rc == _lib.MDB_KEYEXIST:
                return False
            raise _error("mdb_put", rc)
        return True

    def replace(self, key, value, db=None):
        """Use a temporary cursor to invoke :py:meth:`Cursor.replace`.

            `db`:
                Named database to operate on. If unspecified, defaults to the
                database given to the :py:class:`Transaction` constructor.
        """
        with Cursor(db or self._db, self) as curs:
            return curs.replace(key, value)

    def pop(self, key, db=None):
        """Use a temporary cursor to invoke :py:meth:`Cursor.pop`.

            `db`:
                Named database to operate on. If unspecified, defaults to the
                database given to the :py:class:`Transaction` constructor.
        """
        with Cursor(db or self._db, self) as curs:
            return curs.pop(key)

    def delete(self, key, value=EMPTY_BYTES, db=None):
        """Delete a key from the database.

        Equivalent to `mdb_del()
        <http://lmdb.tech/doc/group__mdb.html#gab8182f9360ea69ac0afd4a4eaab1ddb0>`_

            `key`:
                The key to delete.

            value:
                If the database was opened with dupsort=True and value is not
                the empty bytestring, then delete elements matching only this
                `(key, value)` pair, otherwise all values for key are deleted.

        Returns True if at least one key was deleted.
        """
        if value is None:  # for bug-compatibility with cpython impl
            value = EMPTY_BYTES

        rc = _lib.pymdb_del(self._txn, (db or self._db)._dbi,
                            key, len(key), value, len(value))
        self._mutations += 1
        if rc:
            if rc == _lib.MDB_NOTFOUND:
                return False
            raise _error("mdb_del", rc)
        return True

    def cursor(self, db=None):
        """Shortcut for ``lmdb.Cursor(db, self)``"""
        return Cursor(db or self._db, self)


class Cursor(object):
    """
    Structure for navigating a database.

    Equivalent to `mdb_cursor_open()
    <http://lmdb.tech/doc/group__mdb.html#ga9ff5d7bd42557fd5ee235dc1d62613aa>`_

        `db`:
            :py:class:`_Database` to navigate.

        `txn`:
            :py:class:`Transaction` to navigate.

    As a convenience, :py:meth:`Transaction.cursor` can be used to quickly
    return a cursor:

        ::

            >>> env = lmdb.open('/tmp/foo')
            >>> child_db = env.open_db('child_db')
            >>> with env.begin() as txn:
            ...     cursor = txn.cursor()           # Cursor on main database.
            ...     cursor2 = txn.cursor(child_db)  # Cursor on child database.

    Cursors start in an unpositioned state. If :py:meth:`iternext` or
    :py:meth:`iterprev` are used in this state, iteration proceeds from the
    start or end respectively. Iterators directly position using the cursor,
    meaning strange behavior results when multiple iterators exist on the same
    cursor.

    .. note::

        From the perspective of the Python binding, cursors return to an
        'unpositioned' state once any scanning or seeking method (e.g.
        :py:meth:`next`, :py:meth:`prev_nodup`, :py:meth:`set_range`) returns
        ``False`` or raises an exception. This is primarily to ensure safe,
        consistent semantics in the face of any error condition.

        When the Cursor returns to an unpositioned state, its :py:meth:`key`
        and :py:meth:`value` return empty strings to indicate there is no
        active position, although internally the LMDB cursor may still have a
        valid position.

        This may lead to slightly surprising behaviour when iterating the
        values for a `dupsort=True` database's keys, since methods such as
        :py:meth:`iternext_dup` will cause Cursor to appear unpositioned,
        despite it returning ``False`` only to indicate there are no more
        values for the current key. In that case, simply calling
        :py:meth:`next` would cause iteration to resume at the next available
        key.

        This behaviour may change in future.

    Iterator methods such as :py:meth:`iternext` and :py:meth:`iterprev` accept
    `keys` and `values` arguments. If both are ``True``, then the value of
    :py:meth:`item` is yielded on each iteration. If only `keys` is ``True``,
    :py:meth:`key` is yielded, otherwise only :py:meth:`value` is yielded.

    Prior to iteration, a cursor can be positioned anywhere in the database:

        ::

            >>> with env.begin() as txn:
            ...     cursor = txn.cursor()
            ...     if not cursor.set_range('5'): # Position at first key >= '5'.
            ...         print('Not found!')
            ...     else:
            ...         for key, value in cursor: # Iterate from first key >= '5'.
            ...             print((key, value))

    Iteration is not required to navigate, and sometimes results in ugly or
    inefficient code. In cases where the iteration order is not obvious, or is
    related to the data being read, use of :py:meth:`set_key`,
    :py:meth:`set_range`, :py:meth:`key`, :py:meth:`value`, and :py:meth:`item`
    may be preferable:

        ::

            >>> # Record the path from a child to the root of a tree.
            >>> path = ['child14123']
            >>> while path[-1] != 'root':
            ...     assert cursor.set_key(path[-1]), \\
            ...         'Tree is broken! Path: %s' % (path,)
            ...     path.append(cursor.value())
    """
    def __init__(self, db, txn):
        db._deps.add(self)
        txn._deps.add(self)
        self.db = db # hold ref
        self.txn = txn # hold ref
        self._dbi = db._dbi
        self._txn = txn._txn
        self._key = _ffi.new('MDB_val *')
        self._val = _ffi.new('MDB_val *')
        self._valid = False
        self._to_py = txn._to_py
        curpp = _ffi.new('MDB_cursor **')
        self._cur = None
        rc = _lib.mdb_cursor_open(self._txn, self._dbi, curpp)
        if rc:
            raise _error("mdb_cursor_open", rc)
        self._cur = curpp[0]
        # If Transaction.mutations!=last_mutation, must MDB_GET_CURRENT to
        # refresh `key' and `val'.
        self._last_mutation = txn._mutations

    def _invalidate(self):
        if self._cur:
            _lib.mdb_cursor_close(self._cur)
            self.db._deps.discard(self)
            self.txn._deps.discard(self)
            self._cur = _invalid
            self._dbi = _invalid
            self._txn = _invalid

    def __del__(self):
        self._invalidate()

    def close(self):
        """Close the cursor, freeing its associated resources."""
        self._invalidate()

    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        self._invalidate()

    def key(self):
        """Return the current key."""
        # Must refresh `key` and `val` following mutation.
        if self._last_mutation != self.txn._mutations:
            self._cursor_get(_lib.MDB_GET_CURRENT)
        return self._to_py(self._key)

    def value(self):
        """Return the current value."""
        # Must refresh `key` and `val` following mutation.
        if self._last_mutation != self.txn._mutations:
            self._cursor_get(_lib.MDB_GET_CURRENT)
        preload(self._val)
        return self._to_py(self._val)

    def item(self):
        """Return the current `(key, value)` pair."""
        # Must refresh `key` and `val` following mutation.
        if self._last_mutation != self.txn._mutations:
            self._cursor_get(_lib.MDB_GET_CURRENT)
        preload(self._val)
        return self._to_py(self._key), self._to_py(self._val)

    def _iter(self, op, keys, values):
        if not values:
            get = self.key
        elif not keys:
            get = self.value
        else:
            get = self.item

        cur = self._cur
        key = self._key
        val = self._val
        rc = 0

        while self._valid:
            yield get()
            rc = _lib.mdb_cursor_get(cur, key, val, op)
            self._valid = not rc

        if rc:
            self._key.mv_size = 0
            self._val.mv_size = 0
            if rc != _lib.MDB_NOTFOUND:
                raise _error("mdb_cursor_get", rc)

    def iternext(self, keys=True, values=True):
        """Return a forward iterator that yields the current element before
        calling :py:meth:`next`, repeating until the end of the database is
        reached. As a convenience, :py:class:`Cursor` implements the iterator
        protocol by automatically returning a forward iterator when invoked:

            ::

                >>> # Equivalent:
                >>> it = iter(cursor)
                >>> it = cursor.iternext(keys=True, values=True)

        If the cursor is not yet positioned, it is moved to the first key in
        the database, otherwise iteration proceeds from the current position.
        """
        if not self._valid:
            self.first()
        return self._iter(_lib.MDB_NEXT, keys, values)
    __iter__ = iternext

    def iternext_dup(self, keys=False, values=True):
        """Return a forward iterator that yields the current value
        ("duplicate") of the current key before calling :py:meth:`next_dup`,
        repeating until the last value of the current key is reached.

        Only meaningful for databases opened with `dupsort=True`.

        .. code-block:: python

            if not cursor.set_key("foo"):
                print("No values found for 'foo'")
            else:
                for idx, data in enumerate(cursor.iternext_dup()):
                    print("%d'th value for 'foo': %s" % (idx, data))
        """
        return self._iter(_lib.MDB_NEXT_DUP, keys, values)

    def iternext_nodup(self, keys=True, values=False):
        """Return a forward iterator that yields the current value
        ("duplicate") of the current key before calling :py:meth:`next_nodup`,
        repeating until the end of the database is reached.

        Only meaningful for databases opened with `dupsort=True`.

        If the cursor is not yet positioned, it is moved to the first key in
        the database, otherwise iteration proceeds from the current position.

        .. code-block:: python

            for key in cursor.iternext_nodup():
                print("Key '%s' has %d values" % (key, cursor.count()))
        """
        if not self._valid:
            self.first()
        return self._iter(_lib.MDB_NEXT_NODUP, keys, values)

    def iterprev(self, keys=True, values=True):
        """Return a reverse iterator that yields the current element before
        calling :py:meth:`prev`, until the start of the database is reached.

        If the cursor is not yet positioned, it is moved to the last key in
        the database, otherwise iteration proceeds from the current position.

        ::

            >>> with env.begin() as txn:
            ...     for i, (key, value) in enumerate(txn.cursor().iterprev()):
            ...         print('%dth last item is (%r, %r)' % (1+i, key, value))
        """
        if not self._valid:
            self.last()
        return self._iter(_lib.MDB_PREV, keys, values)

    def iterprev_dup(self, keys=False, values=True):
        """Return a reverse iterator that yields the current value
        ("duplicate") of the current key before calling :py:meth:`prev_dup`,
        repeating until the first value of the current key is reached.

        Only meaningful for databases opened with `dupsort=True`.
        """
        return self._iter(_lib.MDB_PREV_DUP, keys, values)

    def iterprev_nodup(self, keys=True, values=False):
        """Return a reverse iterator that yields the current value
        ("duplicate") of the current key before calling :py:meth:`prev_nodup`,
        repeating until the start of the database is reached.

        If the cursor is not yet positioned, it is moved to the last key in
        the database, otherwise iteration proceeds from the current position.

        Only meaningful for databases opened with `dupsort=True`.
        """
        if not self._valid:
            self.last()
        return self._iter(_lib.MDB_PREV_NODUP, keys, values)

    def _cursor_get(self, op):
        rc = _lib.mdb_cursor_get(self._cur, self._key, self._val, op)
        self._valid = v = not rc
        self._last_mutation = self.txn._mutations
        if rc:
            self._key.mv_size = 0
            self._val.mv_size = 0
            if rc != _lib.MDB_NOTFOUND:
                if not (rc == _lib.EINVAL and op == _lib.MDB_GET_CURRENT):
                    raise _error("mdb_cursor_get", rc)
        return v

    def _cursor_get_kv(self, op, k, v):
        rc = _lib.pymdb_cursor_get(self._cur, k, len(k), v, len(v),
                                   self._key, self._val, op)
        self._valid = v = not rc
        if rc:
            self._key.mv_size = 0
            self._val.mv_size = 0
            if rc != _lib.MDB_NOTFOUND:
                if not (rc == _lib.EINVAL and op == _lib.MDB_GET_CURRENT):
                    raise _error("mdb_cursor_get", rc)
        return v

    def first(self):
        """Move to the first key in the database, returning ``True`` on success
        or ``False`` if the database is empty.

        If the database was opened with `dupsort=True` and the key contains
        duplicates, the cursor is positioned on the first value ("duplicate").

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_FIRST
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_FIRST)

    def first_dup(self):
        """Move to the first value ("duplicate") for the current key, returning
        ``True`` on success or ``False`` if the database is empty.

        Only meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_FIRST_DUP
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_FIRST_DUP)

    def last(self):
        """Move to the last key in the database, returning ``True`` on success
        or ``False`` if the database is empty.

        If the database was opened with `dupsort=True` and the key contains
        duplicates, the cursor is positioned on the last value ("duplicate").

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_LAST
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_LAST)

    def last_dup(self):
        """Move to the last value ("duplicate") for the current key, returning
        ``True`` on success or ``False`` if the database is empty.

        Only meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_LAST_DUP
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_LAST_DUP)

    def prev(self):
        """Move to the previous element, returning ``True`` on success or
        ``False`` if there is no previous item.

        For databases opened with `dupsort=True`, moves to the previous data
        item ("duplicate") for the current key if one exists, otherwise moves
        to the previous key.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_PREV
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_PREV)

    def prev_dup(self):
        """Move to the previous value ("duplicate") of the current key,
        returning ``True`` on success or ``False`` if there is no previous
        value.

        Only meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_PREV_DUP
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_PREV_DUP)

    def prev_nodup(self):
        """Move to the last value ("duplicate") of the previous key, returning
        ``True`` on success or ``False`` if there is no previous key.

        Only meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_PREV_NODUP
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_PREV_NODUP)

    def next(self):
        """Move to the next element, returning ``True`` on success or ``False``
        if there is no next element.

        For databases opened with `dupsort=True`, moves to the next value
        ("duplicate") for the current key if one exists, otherwise moves to the
        first value of the next key.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_NEXT
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_NEXT)

    def next_dup(self):
        """Move to the next value ("duplicate") of the current key, returning
        ``True`` on success or ``False`` if there is no next value.

        Only meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_NEXT_DUP
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_NEXT_DUP)

    def next_nodup(self):
        """Move to the first value ("duplicate") of the next key, returning
        ``True`` on success or ``False`` if there is no next key.

        Only meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_NEXT_NODUP
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get(_lib.MDB_NEXT_NODUP)

    def set_key(self, key):
        """Seek exactly to `key`, returning ``True`` on success or ``False`` if
        the exact key was not found. It is an error to :py:meth:`set_key` the
        empty bytestring.

        For databases opened with `dupsort=True`, moves to the first value
        ("duplicate") for the key.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_SET_KEY
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get_kv(_lib.MDB_SET_KEY, key, EMPTY_BYTES)

    def set_key_dup(self, key, value):
        """Seek exactly to `(key, value)`, returning ``True`` on success or
        ``False`` if the exact key and value was not found. It is an error
        to :py:meth:`set_key` the empty bytestring.

        Only meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_GET_BOTH
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        return self._cursor_get_kv(_lib.MDB_GET_BOTH, key, value)

    def get(self, key, default=None):
        """Equivalent to :py:meth:`set_key()`, except :py:meth:`value` is
        returned when `key` is found, otherwise `default`.
        """
        if self._cursor_get_kv(_lib.MDB_SET_KEY, key, EMPTY_BYTES):
            return self.value()
        return default

    def getmulti(self, keys, dupdata=False, dupfixed_bytes=None, keyfixed=False):
        """Returns an iterable of `(key, value)` 2-tuples containing results
        for each key in the iterable `keys`.

            `keys`:
                Iterable to read keys from.

            `dupdata`:
                If ``True`` and database was opened with `dupsort=True`, read
                all duplicate values for each matching key.

            `dupfixed_bytes`:
                If database was opened with `dupsort=True` and `dupfixed=True`,
                accepts the size of each value, in bytes, and applies an
                optimization reducing the number of database lookups.

            `keyfixed`:
                If `dupfixed_bytes` is set and database key size is fixed,
                setting keyfixed=True will result in this function returning
                a memoryview to the results as a structured array of bytes.
                The structured array can be instantiated by passing the
                memoryview buffer to NumPy:

                .. code-block:: python

                    key_bytes, val_bytes = 4, 8
                    dtype = np.dtype([(f'S{key_bytes}', f'S{val_bytes}}')])
                    arr = np.frombuffer(
                        cur.getmulti(keys, dupdata=True, dupfixed_bytes=val_bytes, keyfixed=True)
                    )

        """
        if dupfixed_bytes and dupfixed_bytes < 0:
            raise _error("dupfixed_bytes must be a positive integer.")
        elif (dupfixed_bytes or keyfixed) and not dupdata:
            raise _error("dupdata is required for dupfixed_bytes/key_bytes.")
        elif keyfixed and not dupfixed_bytes:
            raise _error("dupfixed_bytes is required for key_bytes.")

        if dupfixed_bytes:
            get_op = _lib.MDB_GET_MULTIPLE
            next_op = _lib.MDB_NEXT_MULTIPLE
        else:
            get_op = _lib.MDB_GET_CURRENT
            next_op = _lib.MDB_NEXT_DUP

        a = bytearray()
        lst = list()
        for key in keys:
            if self.set_key(key):
                while self._valid:
                    self._cursor_get(get_op)
                    preload(self._val)
                    key = self._to_py(self._key)
                    val = self._to_py(self._val)

                    if dupfixed_bytes:
                        gen = (
                            (key, val[i:i + dupfixed_bytes])
                            for i in range(0, len(val), dupfixed_bytes))
                        if keyfixed:
                            for k, v in gen:
                                a.extend(k + v)
                        else:
                            for k, v in gen:
                                lst.append((k, v))
                    else:
                        lst.append((key, val))

                    if dupdata:
                        self._cursor_get(next_op)
                    else:
                        break

        if keyfixed:
            return memoryview(a)
        else:
            return lst

    def set_range(self, key):
        """Seek to the first key greater than or equal to `key`, returning
        ``True`` on success, or ``False`` to indicate key was past end of
        database. Behaves like :py:meth:`first` if `key` is the empty
        bytestring.

        For databases opened with `dupsort=True`, moves to the first value
        ("duplicate") for the key.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_SET_RANGE
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        if not key:
            return self.first()
        return self._cursor_get_kv(_lib.MDB_SET_RANGE, key, EMPTY_BYTES)

    def set_range_dup(self, key, value):
        """Seek to the first key/value pair greater than or equal to `key`,
        returning ``True`` on success, or ``False`` to indicate that `value` was past the
        last value of `key` or that `(key, value)` was past the end end of database.

        Only meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_get()
        <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
        with `MDB_GET_BOTH_RANGE
        <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
        """
        rc = self._cursor_get_kv(_lib.MDB_GET_BOTH_RANGE, key, value)
        # issue #126: MDB_GET_BOTH_RANGE does not satisfy its documentation,
        # and fails to update `key` and `value` on success. Therefore
        # explicitly call MDB_GET_CURRENT after MDB_GET_BOTH_RANGE.
        self._cursor_get(_lib.MDB_GET_CURRENT)
        return rc

    def delete(self, dupdata=False):
        """Delete the current element and move to the next, returning ``True``
        on success or ``False`` if the database was empty.

        If `dupdata` is ``True``, delete all values ("duplicates") for the
        current key, otherwise delete only the currently positioned value. Only
        meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_del()
        <http://lmdb.tech/doc/group__mdb.html#ga26a52d3efcfd72e5bf6bd6960bf75f95>`_
        """
        v = self._valid
        if v:
            flags = _lib.MDB_NODUPDATA if dupdata else 0
            rc = _lib.mdb_cursor_del(self._cur, flags)
            self.txn._mutations += 1
            if rc:
                raise _error("mdb_cursor_del", rc)
            self._cursor_get(_lib.MDB_GET_CURRENT)
            v = rc == 0
        return v

    def count(self):
        """Return the number of values ("duplicates") for the current key.

        Only meaningful for databases opened with `dupsort=True`.

        Equivalent to `mdb_cursor_count()
        <http://lmdb.tech/doc/group__mdb.html#ga4041fd1e1862c6b7d5f10590b86ffbe2>`_
        """
        countp = _ffi.new('size_t *')
        rc = _lib.mdb_cursor_count(self._cur, countp)
        if rc:
            raise _error("mdb_cursor_count", rc)
        return countp[0]

    def put(self, key, val, dupdata=True, overwrite=True, append=False):
        """Store a record, returning ``True`` if it was written, or ``False``
        to indicate the key was already present and `overwrite=False`. On
        success, the cursor is positioned on the key.

        Equivalent to `mdb_cursor_put()
        <http://lmdb.tech/doc/group__mdb.html#ga1f83ccb40011837ff37cc32be01ad91e>`_

            `key`:
                Bytestring key to store.

            `val`:
                Bytestring value to store.

            `dupdata`:
                If ``False`` and database was opened with `dupsort=True`, will return
                ``False`` if the key already has that value.  In other words, this only
                affects the return value.

            `overwrite`:
                If ``False``, do not overwrite the value for the key if it
                exists, just return ``False``. For databases opened with
                `dupsort=True`, ``False`` will always be returned if a
                duplicate key/value pair is inserted, regardless of the setting
                for `overwrite`.

            `append`:
                If ``True``, append the pair to the end of the database without
                comparing its order first. Appending a key that is not greater
                than the highest existing key will fail and return ``False``.
        """
        flags = 0
        if not dupdata:
            flags |= _lib.MDB_NODUPDATA
        if not overwrite:
            flags |= _lib.MDB_NOOVERWRITE
        if append:
            if self.txn._db._flags & _lib.MDB_DUPSORT:
                flags |= _lib.MDB_APPENDDUP
            else:
                flags |= _lib.MDB_APPEND

        rc = _lib.pymdb_cursor_put(self._cur, key, len(key), val, len(val), flags)
        self.txn._mutations += 1
        if rc:
            if rc == _lib.MDB_KEYEXIST:
                return False
            raise _error("mdb_cursor_put", rc)
        self._cursor_get(_lib.MDB_GET_CURRENT)
        return True

    def putmulti(self, items, dupdata=True, overwrite=True, append=False):
        """Invoke :py:meth:`put` for each `(key, value)` 2-tuple from the
        iterable `items`. Elements must be exactly 2-tuples, they may not be of
        any other type, or tuple subclass.

        Returns a tuple `(consumed, added)`, where `consumed` is the number of
        elements read from the iterable, and `added` is the number of new
        entries added to the database. `added` may be less than `consumed` when
        `overwrite=False`.

            `items`:
                Iterable to read records from.

            `dupdata`:
                If ``True`` and database was opened with `dupsort=True`, add
                pair as a duplicate if the given key already exists. Otherwise
                overwrite any existing matching key.

            `overwrite`:
                If ``False``, do not overwrite the value for the key if it
                exists, just return ``False``. For databases opened with
                `dupsort=True`, ``False`` will always be returned if a
                duplicate key/value pair is inserted, regardless of the setting
                for `overwrite`.

            `append`:
                If ``True``, append records to the end of the database without
                comparing their order first. Appending a key that is not
                greater than the highest existing key will cause corruption.
        """
        flags = 0
        if not dupdata:
            flags |= _lib.MDB_NODUPDATA
        if not overwrite:
            flags |= _lib.MDB_NOOVERWRITE
        if append:
            if self.txn._db._flags & _lib.MDB_DUPSORT:
                flags |= _lib.MDB_APPENDDUP
            else:
                flags |= _lib.MDB_APPEND

        added = 0
        skipped = 0
        for key, value in items:
            rc = _lib.pymdb_cursor_put(self._cur, key, len(key),
                                       value, len(value), flags)
            self.txn._mutations += 1
            added += 1
            if rc:
                if rc == _lib.MDB_KEYEXIST:
                    skipped += 1
                else:
                    raise _error("mdb_cursor_put", rc)
        self._cursor_get(_lib.MDB_GET_CURRENT)
        return added, added - skipped

    def replace(self, key, val):
        """Store a record, returning its previous value if one existed. Returns
        ``None`` if no previous value existed. This uses the best available
        mechanism to minimize the cost of a `set-and-return-previous`
        operation.

        For databases opened with `dupsort=True`, only the first data element
        ("duplicate") is returned if it existed, all data elements are removed
        and the new `(key, data)` pair is inserted.

            `key`:
                Bytestring key to store.

            `value`:
                Bytestring value to store.
        """
        if self.db._flags & _lib.MDB_DUPSORT:
            if self._cursor_get_kv(_lib.MDB_SET_KEY, key, EMPTY_BYTES):
                preload(self._val)
                old = _mvstr(self._val)
                self.delete(True)
            else:
                old = None
            self.put(key, val)
            return old

        flags = _lib.MDB_NOOVERWRITE
        keylen = len(key)
        rc = _lib.pymdb_cursor_put(self._cur, key, keylen, val, len(val), flags)
        self.txn._mutations += 1
        if not rc:
            return
        if rc != _lib.MDB_KEYEXIST:
            raise _error("mdb_cursor_put", rc)

        self._cursor_get(_lib.MDB_GET_CURRENT)
        preload(self._val)
        old = _mvstr(self._val)
        rc = _lib.pymdb_cursor_put(self._cur, key, keylen, val, len(val), 0)
        self.txn._mutations += 1
        if rc:
            raise _error("mdb_cursor_put", rc)
        self._cursor_get(_lib.MDB_GET_CURRENT)
        return old

    def pop(self, key):
        """Fetch a record's value then delete it. Returns ``None`` if no
        previous value existed. This uses the best available mechanism to
        minimize the cost of a `delete-and-return-previous` operation.

        For databases opened with `dupsort=True`, the first data element
        ("duplicate") for the key will be popped.

            `key`:
                Bytestring key to delete.
        """
        if self._cursor_get_kv(_lib.MDB_SET_KEY, key, EMPTY_BYTES):
            preload(self._val)
            old = _mvstr(self._val)
            rc = _lib.mdb_cursor_del(self._cur, 0)
            self.txn._mutations += 1
            if rc:
                raise _error("mdb_cursor_del", rc)
            self._cursor_get(_lib.MDB_GET_CURRENT)
            return old

    def _iter_from(self, k, reverse):
        """Helper for centidb. Please do not rely on this interface, it may be
        removed in future.
        """
        if not k and not reverse:
            found = self.first()
        else:
            found = self.set_range(k)
        if reverse:
            if not found:
                self.last()
            return self.iterprev()
        else:
            if not found:
                return iter(())
            return self.iternext()
