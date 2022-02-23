"""
Test the memory module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import functools
import gc
import shutil
import os
import os.path
import pathlib
import pickle
import sys
import time
import datetime
import textwrap

import pytest

from joblib.memory import Memory
from joblib.memory import MemorizedFunc, NotMemorizedFunc
from joblib.memory import MemorizedResult, NotMemorizedResult
from joblib.memory import _FUNCTION_HASHES
from joblib.memory import register_store_backend, _STORE_BACKENDS
from joblib.memory import _build_func_identifier, _store_backend_factory
from joblib.memory import JobLibCollisionWarning
from joblib.parallel import Parallel, delayed
from joblib._store_backends import StoreBackendBase, FileSystemStoreBackend
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, raises, warns
from joblib.hashing import hash



###############################################################################
# Module-level variables for the tests
def f(x, y=1):
    """ A module-level function for testing purposes.
    """
    return x ** 2 + y


###############################################################################
# Helper function for the tests
def check_identity_lazy(func, accumulator, location):
    """ Given a function and an accumulator (a list that grows every
        time the function is called), check that the function can be
        decorated by memory to be a lazy identity.
    """
    # Call each function with several arguments, and check that it is
    # evaluated only once per argument.
    memory = Memory(location=location, verbose=0)
    func = memory.cache(func)
    for i in range(3):
        for _ in range(2):
            assert func(i) == i
            assert len(accumulator) == i + 1


def corrupt_single_cache_item(memory):
    single_cache_item, = memory.store_backend.get_items()
    output_filename = os.path.join(single_cache_item.path, 'output.pkl')
    with open(output_filename, 'w') as f:
        f.write('garbage')


def monkeypatch_cached_func_warn(func, monkeypatch_fixture):
    # Need monkeypatch because pytest does not
    # capture stdlib logging output (see
    # https://github.com/pytest-dev/pytest/issues/2079)

    recorded = []

    def append_to_record(item):
        recorded.append(item)
    monkeypatch_fixture.setattr(func, 'warn', append_to_record)
    return recorded


###############################################################################
# Tests
def test_memory_integration(tmpdir):
    """ Simple test of memory lazy evaluation.
    """
    accumulator = list()
    # Rmk: this function has the same name than a module-level function,
    # thus it serves as a test to see that both are identified
    # as different.

    def f(l):
        accumulator.append(1)
        return l

    check_identity_lazy(f, accumulator, tmpdir.strpath)

    # Now test clearing
    for compress in (False, True):
        for mmap_mode in ('r', None):
            memory = Memory(location=tmpdir.strpath, verbose=10,
                            mmap_mode=mmap_mode, compress=compress)
            # First clear the cache directory, to check that our code can
            # handle that
            # NOTE: this line would raise an exception, as the database file is
            # still open; we ignore the error since we want to test what
            # happens if the directory disappears
            shutil.rmtree(tmpdir.strpath, ignore_errors=True)
            g = memory.cache(f)
            g(1)
            g.clear(warn=False)
            current_accumulator = len(accumulator)
            out = g(1)

        assert len(accumulator) == current_accumulator + 1
        # Also, check that Memory.eval works similarly
        assert memory.eval(f, 1) == out
        assert len(accumulator) == current_accumulator + 1

    # Now do a smoke test with a function defined in __main__, as the name
    # mangling rules are more complex
    f.__module__ = '__main__'
    memory = Memory(location=tmpdir.strpath, verbose=0)
    memory.cache(f)(1)


@parametrize("call_before_reducing", [True, False])
def test_parallel_call_cached_function_defined_in_jupyter(
    tmpdir, call_before_reducing
):
    # Calling an interactively defined memory.cache()'d function inside a
    # Parallel call used to clear the existing cache related to the said
    # function (https://github.com/joblib/joblib/issues/1035)

    # This tests checks that this is no longer the case.

    # TODO: test that the cache related to the function cache persists across
    # ipython sessions (provided that no code change were made to the
    # function's source)?

    # The first part of the test makes the necessary low-level calls to emulate
    # the definition of a function in an jupyter notebook cell. Joblib has
    # some custom code to treat functions defined specifically in jupyter
    # notebooks/ipython session -- we want to test this code, which requires
    # the emulation to be rigorous.
    for session_no in [0, 1]:
        ipython_cell_source = '''
        def f(x):
            return x
        '''

        ipython_cell_id = '<ipython-input-{}-000000000000>'.format(session_no)

        exec(
            compile(
                textwrap.dedent(ipython_cell_source),
                filename=ipython_cell_id,
                mode='exec'
            )
        )
        # f is now accessible in the locals mapping - but for some unknown
        # reason, f = locals()['f'] throws a KeyError at runtime, we need to
        # bind locals()['f'] to a different name in the local namespace
        aliased_f = locals()['f']
        aliased_f.__module__ = "__main__"

        # Preliminary sanity checks, and tests checking that joblib properly
        # identified f as an interactive function defined in a jupyter notebook
        assert aliased_f(1) == 1
        assert aliased_f.__code__.co_filename == ipython_cell_id

        memory = Memory(location=tmpdir.strpath, verbose=0)
        cached_f = memory.cache(aliased_f)

        assert len(os.listdir(tmpdir / 'joblib')) == 1
        f_cache_relative_directory = os.listdir(tmpdir / 'joblib')[0]
        assert 'ipython-input' in f_cache_relative_directory

        f_cache_directory = tmpdir / 'joblib' / f_cache_relative_directory

        if session_no == 0:
            # The cache should be empty as cached_f has not been called yet.
            assert os.listdir(f_cache_directory) == ['f']
            assert os.listdir(f_cache_directory / 'f') == []

            if call_before_reducing:
                cached_f(3)
                # Two files were just created, func_code.py, and a folder
                # containing the informations (inputs hash/ouptput) of
                # cached_f(3)
                assert len(os.listdir(f_cache_directory / 'f')) == 2

                # Now, testing  #1035: when calling a cached function, joblib
                # used to dynamically inspect the underlying function to
                # extract its source code (to verify it matches the source code
                # of the function as last inspected by joblib) -- however,
                # source code introspection fails for dynamic functions sent to
                # child processes - which would eventually make joblib clear
                # the cache associated to f
                res = Parallel(n_jobs=2)(delayed(cached_f)(i) for i in [1, 2])
            else:
                # Submit the function to the joblib child processes, although
                # the function has never been called in the parent yet. This
                # triggers a specific code branch inside
                # MemorizedFunc.__reduce__.
                res = Parallel(n_jobs=2)(delayed(cached_f)(i) for i in [1, 2])
                assert len(os.listdir(f_cache_directory / 'f')) == 3

                cached_f(3)

            # Making sure f's cache does not get cleared after the parallel
            # calls, and contains ALL cached functions calls (f(1), f(2), f(3))
            # and 'func_code.py'
            assert len(os.listdir(f_cache_directory / 'f')) == 4
        else:
            # For the second session, there should be an already existing cache
            assert len(os.listdir(f_cache_directory / 'f')) == 4

            cached_f(3)

            # The previous cache should not be invalidated after calling the
            # function in a new session
            assert len(os.listdir(f_cache_directory / 'f')) == 4


def test_no_memory():
    """ Test memory with location=None: no memoize """
    accumulator = list()

    def ff(l):
        accumulator.append(1)
        return l

    memory = Memory(location=None, verbose=0)
    gg = memory.cache(ff)
    for _ in range(4):
        current_accumulator = len(accumulator)
        gg(1)
        assert len(accumulator) == current_accumulator + 1


def test_memory_kwarg(tmpdir):
    " Test memory with a function with keyword arguments."
    accumulator = list()

    def g(l=None, m=1):
        accumulator.append(1)
        return l

    check_identity_lazy(g, accumulator, tmpdir.strpath)

    memory = Memory(location=tmpdir.strpath, verbose=0)
    g = memory.cache(g)
    # Smoke test with an explicit keyword argument:
    assert g(l=30, m=2) == 30


def test_memory_lambda(tmpdir):
    " Test memory with a function with a lambda."
    accumulator = list()

    def helper(x):
        """ A helper function to define l as a lambda.
        """
        accumulator.append(1)
        return x

    l = lambda x: helper(x)

    check_identity_lazy(l, accumulator, tmpdir.strpath)


def test_memory_name_collision(tmpdir):
    " Check that name collisions with functions will raise warnings"
    memory = Memory(location=tmpdir.strpath, verbose=0)

    @memory.cache
    def name_collision(x):
        """ A first function called name_collision
        """
        return x

    a = name_collision

    @memory.cache
    def name_collision(x):
        """ A second function called name_collision
        """
        return x

    b = name_collision

    with warns(JobLibCollisionWarning) as warninfo:
        a(1)
        b(1)

    assert len(warninfo) == 1
    assert "collision" in str(warninfo[0].message)


def test_memory_warning_lambda_collisions(tmpdir):
    # Check that multiple use of lambda will raise collisions
    memory = Memory(location=tmpdir.strpath, verbose=0)
    a = lambda x: x
    a = memory.cache(a)
    b = lambda x: x + 1
    b = memory.cache(b)

    with warns(JobLibCollisionWarning) as warninfo:
        assert a(0) == 0
        assert b(1) == 2
        assert a(1) == 1

    # In recent Python versions, we can retrieve the code of lambdas,
    # thus nothing is raised
    assert len(warninfo) == 4


def test_memory_warning_collision_detection(tmpdir):
    # Check that collisions impossible to detect will raise appropriate
    # warnings.
    memory = Memory(location=tmpdir.strpath, verbose=0)
    a1 = eval('lambda x: x')
    a1 = memory.cache(a1)
    b1 = eval('lambda x: x+1')
    b1 = memory.cache(b1)

    with warns(JobLibCollisionWarning) as warninfo:
        a1(1)
        b1(1)
        a1(0)

    assert len(warninfo) == 2
    assert "cannot detect" in str(warninfo[0].message).lower()


def test_memory_partial(tmpdir):
    " Test memory with functools.partial."
    accumulator = list()

    def func(x, y):
        """ A helper function to define l as a lambda.
        """
        accumulator.append(1)
        return y

    import functools
    function = functools.partial(func, 1)

    check_identity_lazy(function, accumulator, tmpdir.strpath)


def test_memory_eval(tmpdir):
    " Smoke test memory with a function with a function defined in an eval."
    memory = Memory(location=tmpdir.strpath, verbose=0)

    m = eval('lambda x: x')
    mm = memory.cache(m)

    assert mm(1) == 1


def count_and_append(x=[]):
    """ A function with a side effect in its arguments.

        Return the length of its argument and append one element.
    """
    len_x = len(x)
    x.append(None)
    return len_x


def test_argument_change(tmpdir):
    """ Check that if a function has a side effect in its arguments, it
        should use the hash of changing arguments.
    """
    memory = Memory(location=tmpdir.strpath, verbose=0)
    func = memory.cache(count_and_append)
    # call the function for the first time, is should cache it with
    # argument x=[]
    assert func() == 0
    # the second time the argument is x=[None], which is not cached
    # yet, so the functions should be called a second time
    assert func() == 1


@with_numpy
@parametrize('mmap_mode', [None, 'r'])
def test_memory_numpy(tmpdir, mmap_mode):
    " Test memory with a function with numpy arrays."
    accumulator = list()

    def n(l=None):
        accumulator.append(1)
        return l

    memory = Memory(location=tmpdir.strpath, mmap_mode=mmap_mode,
                    verbose=0)
    cached_n = memory.cache(n)

    rnd = np.random.RandomState(0)
    for i in range(3):
        a = rnd.random_sample((10, 10))
        for _ in range(3):
            assert np.all(cached_n(a) == a)
            assert len(accumulator) == i + 1


@with_numpy
def test_memory_numpy_check_mmap_mode(tmpdir, monkeypatch):
    """Check that mmap_mode is respected even at the first call"""

    memory = Memory(location=tmpdir.strpath, mmap_mode='r', verbose=0)

    @memory.cache()
    def twice(a):
        return a * 2

    a = np.ones(3)

    b = twice(a)
    c = twice(a)

    assert isinstance(c, np.memmap)
    assert c.mode == 'r'

    assert isinstance(b, np.memmap)
    assert b.mode == 'r'

    # Corrupts the file,  Deleting b and c mmaps
    # is necessary to be able edit the file
    del b
    del c
    gc.collect()
    corrupt_single_cache_item(memory)

    # Make sure that corrupting the file causes recomputation and that
    # a warning is issued.
    recorded_warnings = monkeypatch_cached_func_warn(twice, monkeypatch)
    d = twice(a)
    assert len(recorded_warnings) == 1
    exception_msg = 'Exception while loading results'
    assert exception_msg in recorded_warnings[0]
    # Asserts that the recomputation returns a mmap
    assert isinstance(d, np.memmap)
    assert d.mode == 'r'


def test_memory_exception(tmpdir):
    """ Smoketest the exception handling of Memory.
    """
    memory = Memory(location=tmpdir.strpath, verbose=0)

    class MyException(Exception):
        pass

    @memory.cache
    def h(exc=0):
        if exc:
            raise MyException

    # Call once, to initialise the cache
    h()

    for _ in range(3):
        # Call 3 times, to be sure that the Exception is always raised
        with raises(MyException):
            h(1)


def test_memory_ignore(tmpdir):
    " Test the ignore feature of memory "
    memory = Memory(location=tmpdir.strpath, verbose=0)
    accumulator = list()

    @memory.cache(ignore=['y'])
    def z(x, y=1):
        accumulator.append(1)

    assert z.ignore == ['y']

    z(0, y=1)
    assert len(accumulator) == 1
    z(0, y=1)
    assert len(accumulator) == 1
    z(0, y=2)
    assert len(accumulator) == 1


def test_memory_ignore_decorated(tmpdir):
    " Test the ignore feature of memory on a decorated function "
    memory = Memory(location=tmpdir.strpath, verbose=0)
    accumulator = list()

    def decorate(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped

    @memory.cache(ignore=['y'])
    @decorate
    def z(x, y=1):
        accumulator.append(1)

    assert z.ignore == ['y']

    z(0, y=1)
    assert len(accumulator) == 1
    z(0, y=1)
    assert len(accumulator) == 1
    z(0, y=2)
    assert len(accumulator) == 1


def test_memory_args_as_kwargs(tmpdir):
    """Non-regression test against 0.12.0 changes.

    https://github.com/joblib/joblib/pull/751
    """
    memory = Memory(location=tmpdir.strpath, verbose=0)

    @memory.cache
    def plus_one(a):
        return a + 1

    # It's possible to call a positional arg as a kwarg.
    assert plus_one(1) == 2
    assert plus_one(a=1) == 2

    # However, a positional argument that joblib hadn't seen
    # before would cause a failure if it was passed as a kwarg.
    assert plus_one(a=2) == 3


@parametrize('ignore, verbose, mmap_mode', [(['x'], 100, 'r'),
                                            ([], 10, None)])
def test_partial_decoration(tmpdir, ignore, verbose, mmap_mode):
    "Check cache may be called with kwargs before decorating"
    memory = Memory(location=tmpdir.strpath, verbose=0)

    @memory.cache(ignore=ignore, verbose=verbose, mmap_mode=mmap_mode)
    def z(x):
        pass

    assert z.ignore == ignore
    assert z._verbose == verbose
    assert z.mmap_mode == mmap_mode


def test_func_dir(tmpdir):
    # Test the creation of the memory cache directory for the function.
    memory = Memory(location=tmpdir.strpath, verbose=0)
    path = __name__.split('.')
    path.append('f')
    path = tmpdir.join('joblib', *path).strpath

    g = memory.cache(f)
    # Test that the function directory is created on demand
    func_id = _build_func_identifier(f)
    location = os.path.join(g.store_backend.location, func_id)
    assert location == path
    assert os.path.exists(path)
    assert memory.location == os.path.dirname(g.store_backend.location)
    with warns(DeprecationWarning) as w:
        assert memory.cachedir == g.store_backend.location
    assert len(w) == 1
    assert "The 'cachedir' attribute has been deprecated" in str(w[-1].message)

    # Test that the code is stored.
    # For the following test to be robust to previous execution, we clear
    # the in-memory store
    _FUNCTION_HASHES.clear()
    assert not g._check_previous_func_code()
    assert os.path.exists(os.path.join(path, 'func_code.py'))
    assert g._check_previous_func_code()

    # Test the robustness to failure of loading previous results.
    func_id, args_id = g._get_output_identifiers(1)
    output_dir = os.path.join(g.store_backend.location, func_id, args_id)
    a = g(1)
    assert os.path.exists(output_dir)
    os.remove(os.path.join(output_dir, 'output.pkl'))
    assert a == g(1)


def test_persistence(tmpdir):
    # Test the memorized functions can be pickled and restored.
    memory = Memory(location=tmpdir.strpath, verbose=0)
    g = memory.cache(f)
    output = g(1)

    h = pickle.loads(pickle.dumps(g))

    func_id, args_id = h._get_output_identifiers(1)
    output_dir = os.path.join(h.store_backend.location, func_id, args_id)
    assert os.path.exists(output_dir)
    assert output == h.store_backend.load_item([func_id, args_id])
    memory2 = pickle.loads(pickle.dumps(memory))
    assert memory.store_backend.location == memory2.store_backend.location

    # Smoke test that pickling a memory with location=None works
    memory = Memory(location=None, verbose=0)
    pickle.loads(pickle.dumps(memory))
    g = memory.cache(f)
    gp = pickle.loads(pickle.dumps(g))
    gp(1)


def test_check_call_in_cache(tmpdir):
    for func in (MemorizedFunc(f, tmpdir.strpath),
                 Memory(location=tmpdir.strpath, verbose=0).cache(f)):
        result = func.check_call_in_cache(2)
        assert not result
        assert isinstance(result, bool)
        assert func(2) == 5
        result = func.check_call_in_cache(2)
        assert result
        assert isinstance(result, bool)
        func.clear()


def test_call_and_shelve(tmpdir):
    # Test MemorizedFunc outputting a reference to cache.

    for func, Result in zip((MemorizedFunc(f, tmpdir.strpath),
                             NotMemorizedFunc(f),
                             Memory(location=tmpdir.strpath,
                                    verbose=0).cache(f),
                             Memory(location=None).cache(f),
                             ),
                            (MemorizedResult, NotMemorizedResult,
                             MemorizedResult, NotMemorizedResult)):
        assert func(2) == 5
        result = func.call_and_shelve(2)
        assert isinstance(result, Result)
        assert result.get() == 5

        result.clear()
        with raises(KeyError):
            result.get()
        result.clear()  # Do nothing if there is no cache.


def test_call_and_shelve_argument_hash(tmpdir):
    # Verify that a warning is raised when accessing arguments_hash
    # attribute from MemorizedResult
    func = Memory(location=tmpdir.strpath, verbose=0).cache(f)
    result = func.call_and_shelve(2)
    assert isinstance(result, MemorizedResult)
    with warns(DeprecationWarning) as w:
        assert result.argument_hash == result.args_id
    assert len(w) == 1
    assert "The 'argument_hash' attribute has been deprecated" \
        in str(w[-1].message)


def test_call_and_shelve_lazily_load_stored_result(tmpdir):
    """Check call_and_shelve only load stored data if needed."""
    test_access_time_file = tmpdir.join('test_access')
    test_access_time_file.write('test_access')
    test_access_time = os.stat(test_access_time_file.strpath).st_atime
    # check file system access time stats resolution is lower than test wait
    # timings.
    time.sleep(0.5)
    assert test_access_time_file.read() == 'test_access'

    if test_access_time == os.stat(test_access_time_file.strpath).st_atime:
        # Skip this test when access time cannot be retrieved with enough
        # precision from the file system (e.g. NTFS on windows).
        pytest.skip("filesystem does not support fine-grained access time "
                    "attribute")

    memory = Memory(location=tmpdir.strpath, verbose=0)
    func = memory.cache(f)
    func_id, argument_hash = func._get_output_identifiers(2)
    result_path = os.path.join(memory.store_backend.location,
                               func_id, argument_hash, 'output.pkl')
    assert func(2) == 5
    first_access_time = os.stat(result_path).st_atime
    time.sleep(1)

    # Should not access the stored data
    result = func.call_and_shelve(2)
    assert isinstance(result, MemorizedResult)
    assert os.stat(result_path).st_atime == first_access_time
    time.sleep(1)

    # Read the stored data => last access time is greater than first_access
    assert result.get() == 5
    assert os.stat(result_path).st_atime > first_access_time


def test_memorized_pickling(tmpdir):
    for func in (MemorizedFunc(f, tmpdir.strpath), NotMemorizedFunc(f)):
        filename = tmpdir.join('pickling_test.dat').strpath
        result = func.call_and_shelve(2)
        with open(filename, 'wb') as fp:
            pickle.dump(result, fp)
        with open(filename, 'rb') as fp:
            result2 = pickle.load(fp)
        assert result2.get() == result.get()
        os.remove(filename)


def test_memorized_repr(tmpdir):
    func = MemorizedFunc(f, tmpdir.strpath)
    result = func.call_and_shelve(2)

    func2 = MemorizedFunc(f, tmpdir.strpath)
    result2 = func2.call_and_shelve(2)
    assert result.get() == result2.get()
    assert repr(func) == repr(func2)

    # Smoke test with NotMemorizedFunc
    func = NotMemorizedFunc(f)
    repr(func)
    repr(func.call_and_shelve(2))

    # Smoke test for message output (increase code coverage)
    func = MemorizedFunc(f, tmpdir.strpath, verbose=11, timestamp=time.time())
    result = func.call_and_shelve(11)
    result.get()

    func = MemorizedFunc(f, tmpdir.strpath, verbose=11)
    result = func.call_and_shelve(11)
    result.get()

    func = MemorizedFunc(f, tmpdir.strpath, verbose=5, timestamp=time.time())
    result = func.call_and_shelve(11)
    result.get()

    func = MemorizedFunc(f, tmpdir.strpath, verbose=5)
    result = func.call_and_shelve(11)
    result.get()


def test_memory_file_modification(capsys, tmpdir, monkeypatch):
    # Test that modifying a Python file after loading it does not lead to
    # Recomputation
    dir_name = tmpdir.mkdir('tmp_import').strpath
    filename = os.path.join(dir_name, 'tmp_joblib_.py')
    content = 'def f(x):\n    print(x)\n    return x\n'
    with open(filename, 'w') as module_file:
        module_file.write(content)

    # Load the module:
    monkeypatch.syspath_prepend(dir_name)
    import tmp_joblib_ as tmp

    memory = Memory(location=tmpdir.strpath, verbose=0)
    f = memory.cache(tmp.f)
    # First call f a few times
    f(1)
    f(2)
    f(1)

    # Now modify the module where f is stored without modifying f
    with open(filename, 'w') as module_file:
        module_file.write('\n\n' + content)

    # And call f a couple more times
    f(1)
    f(1)

    # Flush the .pyc files
    shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    # Now modify the module where f is stored, modifying f
    content = 'def f(x):\n    print("x=%s" % x)\n    return x\n'
    with open(filename, 'w') as module_file:
        module_file.write(content)

    # And call f more times prior to reloading: the cache should not be
    # invalidated at this point as the active function definition has not
    # changed in memory yet.
    f(1)
    f(1)

    # Now reload
    sys.stdout.write('Reloading\n')
    sys.modules.pop('tmp_joblib_')
    import tmp_joblib_ as tmp
    f = memory.cache(tmp.f)

    # And call f more times
    f(1)
    f(1)

    out, err = capsys.readouterr()
    assert out == '1\n2\nReloading\nx=1\n'


def _function_to_cache(a, b):
    # Just a place holder function to be mutated by tests
    pass


def _sum(a, b):
    return a + b


def _product(a, b):
    return a * b


def test_memory_in_memory_function_code_change(tmpdir):
    _function_to_cache.__code__ = _sum.__code__

    memory = Memory(location=tmpdir.strpath, verbose=0)
    f = memory.cache(_function_to_cache)

    assert f(1, 2) == 3
    assert f(1, 2) == 3

    with warns(JobLibCollisionWarning):
        # Check that inline function modification triggers a cache invalidation
        _function_to_cache.__code__ = _product.__code__
        assert f(1, 2) == 2
        assert f(1, 2) == 2


def test_clear_memory_with_none_location():
    memory = Memory(location=None)
    memory.clear()


def func_with_kwonly_args(a, b, *, kw1='kw1', kw2='kw2'):
    return a, b, kw1, kw2


def func_with_signature(a: int, b: float) -> float:
    return a + b


def test_memory_func_with_kwonly_args(tmpdir):
    memory = Memory(location=tmpdir.strpath, verbose=0)
    func_cached = memory.cache(func_with_kwonly_args)

    assert func_cached(1, 2, kw1=3) == (1, 2, 3, 'kw2')

    # Making sure that providing a keyword-only argument by
    # position raises an exception
    with raises(ValueError) as excinfo:
        func_cached(1, 2, 3, kw2=4)
    excinfo.match("Keyword-only parameter 'kw1' was passed as positional "
                  "parameter")

    # Keyword-only parameter passed by position with cached call
    # should still raise ValueError
    func_cached(1, 2, kw1=3, kw2=4)

    with raises(ValueError) as excinfo:
        func_cached(1, 2, 3, kw2=4)
    excinfo.match("Keyword-only parameter 'kw1' was passed as positional "
                  "parameter")

    # Test 'ignore' parameter
    func_cached = memory.cache(func_with_kwonly_args, ignore=['kw2'])
    assert func_cached(1, 2, kw1=3, kw2=4) == (1, 2, 3, 4)
    assert func_cached(1, 2, kw1=3, kw2='ignored') == (1, 2, 3, 4)


def test_memory_func_with_signature(tmpdir):
    memory = Memory(location=tmpdir.strpath, verbose=0)
    func_cached = memory.cache(func_with_signature)

    assert func_cached(1, 2.) == 3.


def _setup_toy_cache(tmpdir, num_inputs=10):
    memory = Memory(location=tmpdir.strpath, verbose=0)

    @memory.cache()
    def get_1000_bytes(arg):
        return 'a' * 1000

    inputs = list(range(num_inputs))
    for arg in inputs:
        get_1000_bytes(arg)

    func_id = _build_func_identifier(get_1000_bytes)
    hash_dirnames = [get_1000_bytes._get_output_identifiers(arg)[1]
                     for arg in inputs]

    full_hashdirs = [os.path.join(get_1000_bytes.store_backend.location,
                                  func_id, dirname)
                     for dirname in hash_dirnames]
    return memory, full_hashdirs, get_1000_bytes


def test__get_items(tmpdir):
    memory, expected_hash_dirs, _ = _setup_toy_cache(tmpdir)
    items = memory.store_backend.get_items()
    hash_dirs = [ci.path for ci in items]
    assert set(hash_dirs) == set(expected_hash_dirs)

    def get_files_size(directory):
        full_paths = [os.path.join(directory, fn)
                      for fn in os.listdir(directory)]
        return sum(os.path.getsize(fp) for fp in full_paths)

    expected_hash_cache_sizes = [get_files_size(hash_dir)
                                 for hash_dir in hash_dirs]
    hash_cache_sizes = [ci.size for ci in items]
    assert hash_cache_sizes == expected_hash_cache_sizes

    output_filenames = [os.path.join(hash_dir, 'output.pkl')
                        for hash_dir in hash_dirs]

    expected_last_accesses = [
        datetime.datetime.fromtimestamp(os.path.getatime(fn))
        for fn in output_filenames]
    last_accesses = [ci.last_access for ci in items]
    assert last_accesses == expected_last_accesses


def test__get_items_to_delete(tmpdir):
    memory, expected_hash_cachedirs, _ = _setup_toy_cache(tmpdir)
    items = memory.store_backend.get_items()
    # bytes_limit set to keep only one cache item (each hash cache
    # folder is about 1000 bytes + metadata)
    items_to_delete = memory.store_backend._get_items_to_delete('2K')
    nb_hashes = len(expected_hash_cachedirs)
    assert set.issubset(set(items_to_delete), set(items))
    assert len(items_to_delete) == nb_hashes - 1

    # Sanity check bytes_limit=2048 is the same as bytes_limit='2K'
    items_to_delete_2048b = memory.store_backend._get_items_to_delete(2048)
    assert sorted(items_to_delete) == sorted(items_to_delete_2048b)

    # bytes_limit greater than the size of the cache
    items_to_delete_empty = memory.store_backend._get_items_to_delete('1M')
    assert items_to_delete_empty == []

    # All the cache items need to be deleted
    bytes_limit_too_small = 500
    items_to_delete_500b = memory.store_backend._get_items_to_delete(
        bytes_limit_too_small)
    assert set(items_to_delete_500b), set(items)

    # Test LRU property: surviving cache items should all have a more
    # recent last_access that the ones that have been deleted
    items_to_delete_6000b = memory.store_backend._get_items_to_delete(6000)
    surviving_items = set(items).difference(items_to_delete_6000b)

    assert (max(ci.last_access for ci in items_to_delete_6000b) <=
            min(ci.last_access for ci in surviving_items))


def test_memory_reduce_size(tmpdir):
    memory, _, _ = _setup_toy_cache(tmpdir)
    ref_cache_items = memory.store_backend.get_items()

    # By default memory.bytes_limit is None and reduce_size is a noop
    memory.reduce_size()
    cache_items = memory.store_backend.get_items()
    assert sorted(ref_cache_items) == sorted(cache_items)

    # No cache items deleted if bytes_limit greater than the size of
    # the cache
    memory.bytes_limit = '1M'
    memory.reduce_size()
    cache_items = memory.store_backend.get_items()
    assert sorted(ref_cache_items) == sorted(cache_items)

    # bytes_limit is set so that only two cache items are kept
    memory.bytes_limit = '3K'
    memory.reduce_size()
    cache_items = memory.store_backend.get_items()
    assert set.issubset(set(cache_items), set(ref_cache_items))
    assert len(cache_items) == 2

    # bytes_limit set so that no cache item is kept
    bytes_limit_too_small = 500
    memory.bytes_limit = bytes_limit_too_small
    memory.reduce_size()
    cache_items = memory.store_backend.get_items()
    assert cache_items == []


def test_memory_clear(tmpdir):
    memory, _, _ = _setup_toy_cache(tmpdir)
    memory.clear()

    assert os.listdir(memory.store_backend.location) == []


def fast_func_with_complex_output():
    complex_obj = ['a' * 1000] * 1000
    return complex_obj


def fast_func_with_conditional_complex_output(complex_output=True):
    complex_obj = {str(i): i for i in range(int(1e5))}
    return complex_obj if complex_output else 'simple output'


@with_multiprocessing
def test_cached_function_race_condition_when_persisting_output(tmpdir, capfd):
    # Test race condition where multiple processes are writing into
    # the same output.pkl. See
    # https://github.com/joblib/joblib/issues/490 for more details.
    memory = Memory(location=tmpdir.strpath)
    func_cached = memory.cache(fast_func_with_complex_output)

    Parallel(n_jobs=2)(delayed(func_cached)() for i in range(3))

    stdout, stderr = capfd.readouterr()

    # Checking both stdout and stderr (ongoing PR #434 may change
    # logging destination) to make sure there is no exception while
    # loading the results
    exception_msg = 'Exception while loading results'
    assert exception_msg not in stdout
    assert exception_msg not in stderr


@with_multiprocessing
def test_cached_function_race_condition_when_persisting_output_2(tmpdir,
                                                                 capfd):
    # Test race condition in first attempt at solving
    # https://github.com/joblib/joblib/issues/490. The race condition
    # was due to the delay between seeing the cache directory created
    # (interpreted as the result being cached) and the output.pkl being
    # pickled.
    memory = Memory(location=tmpdir.strpath)
    func_cached = memory.cache(fast_func_with_conditional_complex_output)

    Parallel(n_jobs=2)(delayed(func_cached)(True if i % 2 == 0 else False)
                       for i in range(3))

    stdout, stderr = capfd.readouterr()

    # Checking both stdout and stderr (ongoing PR #434 may change
    # logging destination) to make sure there is no exception while
    # loading the results
    exception_msg = 'Exception while loading results'
    assert exception_msg not in stdout
    assert exception_msg not in stderr


def test_memory_recomputes_after_an_error_while_loading_results(
        tmpdir, monkeypatch):
    memory = Memory(location=tmpdir.strpath)

    def func(arg):
        # This makes sure that the timestamp returned by two calls of
        # func are different. This is needed on Windows where
        # time.time resolution may not be accurate enough
        time.sleep(0.01)
        return arg, time.time()

    cached_func = memory.cache(func)
    input_arg = 'arg'
    arg, timestamp = cached_func(input_arg)

    # Make sure the function is correctly cached
    assert arg == input_arg

    # Corrupting output.pkl to make sure that an error happens when
    # loading the cached result
    corrupt_single_cache_item(memory)

    # Make sure that corrupting the file causes recomputation and that
    # a warning is issued.
    recorded_warnings = monkeypatch_cached_func_warn(cached_func, monkeypatch)
    recomputed_arg, recomputed_timestamp = cached_func(arg)
    assert len(recorded_warnings) == 1
    exception_msg = 'Exception while loading results'
    assert exception_msg in recorded_warnings[0]
    assert recomputed_arg == arg
    assert recomputed_timestamp > timestamp

    # Corrupting output.pkl to make sure that an error happens when
    # loading the cached result
    corrupt_single_cache_item(memory)
    reference = cached_func.call_and_shelve(arg)
    try:
        reference.get()
        raise AssertionError(
            "It normally not possible to load a corrupted"
            " MemorizedResult"
        )
    except KeyError as e:
        message = "is corrupted"
        assert message in str(e.args)


def test_deprecated_cachedir_behaviour(tmpdir):
    # verify the right deprecation warnings are raised when using cachedir
    # option instead of new location parameter.
    with warns(None) as w:
        memory = Memory(cachedir=tmpdir.strpath, verbose=0)
        assert memory.store_backend.location.startswith(tmpdir.strpath)

    assert len(w) == 1
    assert "The 'cachedir' parameter has been deprecated" in str(w[-1].message)

    with warns(None) as w:
        memory = Memory()
        assert memory.cachedir is None

    assert len(w) == 1
    assert "The 'cachedir' attribute has been deprecated" in str(w[-1].message)

    error_regex = """You set both "location='.+ and "cachedir='.+"""
    with raises(ValueError, match=error_regex):
        memory = Memory(location=tmpdir.strpath, cachedir=tmpdir.strpath,
                        verbose=0)


class IncompleteStoreBackend(StoreBackendBase):
    """This backend cannot be instanciated and should raise a TypeError."""
    pass


class DummyStoreBackend(StoreBackendBase):
    """A dummy store backend that does nothing."""

    def _open_item(self, *args, **kwargs):
        """Open an item on store."""
        "Does nothing"

    def _item_exists(self, location):
        """Check if an item location exists."""
        "Does nothing"

    def _move_item(self, src, dst):
        """Move an item from src to dst in store."""
        "Does nothing"

    def create_location(self, location):
        """Create location on store."""
        "Does nothing"

    def exists(self, obj):
        """Check if an object exists in the store"""
        return False

    def clear_location(self, obj):
        """Clear object on store"""
        "Does nothing"

    def get_items(self):
        """Returns the whole list of items available in cache."""
        return []

    def configure(self, location, *args, **kwargs):
        """Configure the store"""
        "Does nothing"


@parametrize("invalid_prefix", [None, dict(), list()])
def test_register_invalid_store_backends_key(invalid_prefix):
    # verify the right exceptions are raised when passing a wrong backend key.
    with raises(ValueError) as excinfo:
        register_store_backend(invalid_prefix, None)
    excinfo.match(r'Store backend name should be a string*')


def test_register_invalid_store_backends_object():
    # verify the right exceptions are raised when passing a wrong backend
    # object.
    with raises(ValueError) as excinfo:
        register_store_backend("fs", None)
    excinfo.match(r'Store backend should inherit StoreBackendBase*')


def test_memory_default_store_backend():
    # test an unknown backend falls back into a FileSystemStoreBackend
    with raises(TypeError) as excinfo:
        Memory(location='/tmp/joblib', backend='unknown')
    excinfo.match(r"Unknown location*")


def test_warning_on_unknown_location_type():
    class NonSupportedLocationClass:
        pass
    unsupported_location = NonSupportedLocationClass()

    with warns(UserWarning) as warninfo:
        _store_backend_factory("local", location=unsupported_location)

    expected_mesage = ("Instanciating a backend using a "
                       "NonSupportedLocationClass as a location is not "
                       "supported by joblib")
    assert expected_mesage in str(warninfo[0].message)


def test_instanciate_incomplete_store_backend():
    # Verify that registering an external incomplete store backend raises an
    # exception when one tries to instanciate it.
    backend_name = "isb"
    register_store_backend(backend_name, IncompleteStoreBackend)
    assert (backend_name, IncompleteStoreBackend) in _STORE_BACKENDS.items()
    with raises(TypeError) as excinfo:
        _store_backend_factory(backend_name, "fake_location")
    excinfo.match(r"Can't instantiate abstract class "
                  "IncompleteStoreBackend with abstract methods*")


def test_dummy_store_backend():
    # Verify that registering an external store backend works.

    backend_name = "dsb"
    register_store_backend(backend_name, DummyStoreBackend)
    assert (backend_name, DummyStoreBackend) in _STORE_BACKENDS.items()

    backend_obj = _store_backend_factory(backend_name, "dummy_location")
    assert isinstance(backend_obj, DummyStoreBackend)


def test_instanciate_store_backend_with_pathlib_path():
    # Instanciate a FileSystemStoreBackend using a pathlib.Path object
    path = pathlib.Path("some_folder")
    backend_obj = _store_backend_factory("local", path)
    assert backend_obj.location == "some_folder"


def test_filesystem_store_backend_repr(tmpdir):
    # Verify string representation of a filesystem store backend.

    repr_pattern = 'FileSystemStoreBackend(location="{location}")'
    backend = FileSystemStoreBackend()
    assert backend.location is None

    repr(backend)  # Should not raise an exception

    assert str(backend) == repr_pattern.format(location=None)

    # backend location is passed explicitely via the configure method (called
    # by the internal _store_backend_factory function)
    backend.configure(tmpdir.strpath)

    assert str(backend) == repr_pattern.format(location=tmpdir.strpath)

    repr(backend)  # Should not raise an exception


def test_memory_objects_repr(tmpdir):
    # Verify printable reprs of MemorizedResult, MemorizedFunc and Memory.

    def my_func(a, b):
        return a + b

    memory = Memory(location=tmpdir.strpath, verbose=0)
    memorized_func = memory.cache(my_func)

    memorized_func_repr = 'MemorizedFunc(func={func}, location={location})'

    assert str(memorized_func) == memorized_func_repr.format(
        func=my_func,
        location=memory.store_backend.location)

    memorized_result = memorized_func.call_and_shelve(42, 42)

    memorized_result_repr = ('MemorizedResult(location="{location}", '
                             'func="{func}", args_id="{args_id}")')

    assert str(memorized_result) == memorized_result_repr.format(
        location=memory.store_backend.location,
        func=memorized_result.func_id,
        args_id=memorized_result.args_id)

    assert str(memory) == 'Memory(location={location})'.format(
        location=memory.store_backend.location)


def test_memorized_result_pickle(tmpdir):
    # Verify a MemoryResult object can be pickled/depickled. Non regression
    # test introduced following issue
    # https://github.com/joblib/joblib/issues/747

    memory = Memory(location=tmpdir.strpath)

    @memory.cache
    def g(x):
        return x**2

    memorized_result = g.call_and_shelve(4)
    memorized_result_pickle = pickle.dumps(memorized_result)
    memorized_result_loads = pickle.loads(memorized_result_pickle)

    assert memorized_result.store_backend.location == \
        memorized_result_loads.store_backend.location
    assert memorized_result.func == memorized_result_loads.func
    assert memorized_result.args_id == memorized_result_loads.args_id
    assert str(memorized_result) == str(memorized_result_loads)


def compare(left, right, ignored_attrs=None):
    if ignored_attrs is None:
        ignored_attrs = []

    left_vars = vars(left)
    right_vars = vars(right)
    assert set(left_vars.keys()) == set(right_vars.keys())
    for attr in left_vars.keys():
        if attr in ignored_attrs:
            continue
        assert left_vars[attr] == right_vars[attr]


@pytest.mark.parametrize('memory_kwargs',
                         [{'compress': 3, 'verbose': 2},
                          {'mmap_mode': 'r', 'verbose': 5, 'bytes_limit': 1e6,
                           'backend_options': {'parameter': 'unused'}}])
def test_memory_pickle_dump_load(tmpdir, memory_kwargs):
    memory = Memory(location=tmpdir.strpath, **memory_kwargs)

    memory_reloaded = pickle.loads(pickle.dumps(memory))

    # Compare Memory instance before and after pickle roundtrip
    compare(memory.store_backend, memory_reloaded.store_backend)
    compare(memory, memory_reloaded,
            ignored_attrs=set(['store_backend', 'timestamp', '_func_code_id']))
    assert hash(memory) == hash(memory_reloaded)

    func_cached = memory.cache(f)

    func_cached_reloaded = pickle.loads(pickle.dumps(func_cached))

    # Compare MemorizedFunc instance before/after pickle roundtrip
    compare(func_cached.store_backend, func_cached_reloaded.store_backend)
    compare(func_cached, func_cached_reloaded,
            ignored_attrs=set(['store_backend', 'timestamp', '_func_code_id']))
    assert hash(func_cached) == hash(func_cached_reloaded)

    # Compare MemorizedResult instance before/after pickle roundtrip
    memorized_result = func_cached.call_and_shelve(1)
    memorized_result_reloaded = pickle.loads(pickle.dumps(memorized_result))

    compare(memorized_result.store_backend,
            memorized_result_reloaded.store_backend)
    compare(memorized_result, memorized_result_reloaded,
            ignored_attrs=set(['store_backend', 'timestamp', '_func_code_id']))
    assert hash(memorized_result) == hash(memorized_result_reloaded)
