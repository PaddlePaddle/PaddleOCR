"""
from: https://bitbucket.org/haypo/misc/src/tip/python/pep418.py

Implementation of the PEP 418 in pure Python using ctypes.

Functions:

 - clock()
 - get_clock_info(name)
 - monotonic(): not always available
 - perf_frequency()
 - process_time()
 - sleep()
 - time()

Constants:

 - has_monotonic (bool): True if time.monotonic() is available
"""
# flake8: noqa
# TODO: gethrtime() for Solaris/OpenIndiana
# TODO: call GetSystemTimeAdjustment() to get the resolution
# TODO: other FIXME

import os
import sys
import time as python_time

has_mach_absolute_time = False
has_clock_gettime = False
has_gettimeofday = False
has_ftime = False
has_delay = False
has_libc_time = False
has_libc_clock = False
has_libc_sleep = False
has_GetTickCount64 = False
CLOCK_REALTIME = None
CLOCK_MONOTONIC = None
CLOCK_PROCESS_CPUTIME_ID = None
CLOCK_HIGHRES = None
CLOCK_PROF = None
try:
    import ctypes
    import ctypes.util
    from ctypes import POINTER
    from ctypes import byref
except ImportError as err:
    pass
else:
    def ctypes_oserror():
        errno = ctypes.get_errno()
        message = os.strerror(errno)
        return OSError(errno, message)

    time_t = ctypes.c_long

    if os.name == "nt":
        from ctypes import windll
        from ctypes.wintypes import BOOL
        from ctypes.wintypes import DWORD
        from ctypes.wintypes import FILETIME
        from ctypes.wintypes import HANDLE
        LARGEINTEGER = ctypes.c_int64
        LARGEINTEGER_p = POINTER(LARGEINTEGER)
        FILETIME_p = POINTER(FILETIME)
        ULONGLONG = ctypes.c_uint64

        def ctypes_winerror():
            errno = ctypes.get_errno()
            message = os.strerror(errno)
            return WindowsError(errno, message)

        _QueryPerformanceFrequency = windll.kernel32.QueryPerformanceFrequency
        _QueryPerformanceFrequency.restype = BOOL
        _QueryPerformanceFrequency.argtypes = (LARGEINTEGER_p,)
        def QueryPerformanceFrequency():
            frequency = LARGEINTEGER()
            ok = _QueryPerformanceFrequency(byref(frequency))
            if not ok:
                raise ctypes_winerror()
            return int(frequency.value)

        _QueryPerformanceCounter = windll.kernel32.QueryPerformanceCounter
        _QueryPerformanceCounter.restype = BOOL
        _QueryPerformanceCounter.argtypes = (LARGEINTEGER_p,)
        def QueryPerformanceCounter():
            frequency = LARGEINTEGER()
            ok = _QueryPerformanceCounter(byref(frequency))
            if not ok:
                raise ctypes_winerror()
            return int(frequency.value)

        GetTickCount = windll.kernel32.GetTickCount
        GetTickCount.restype = DWORD
        GetTickCount.argtypes = ()

        if hasattr(windll.kernel32, 'GetTickCount64'):
            GetTickCount64 = windll.kernel32.GetTickCount64
            GetTickCount64.restype = ULONGLONG
            GetTickCount64.argtypes = ()
            has_GetTickCount64 = True

        GetCurrentProcess = windll.kernel32.GetCurrentProcess
        GetCurrentProcess.argtypes = ()
        GetCurrentProcess.restype = HANDLE

        _GetProcessTimes = windll.kernel32.GetProcessTimes
        _GetProcessTimes.argtypes = (HANDLE, FILETIME_p, FILETIME_p, FILETIME_p, FILETIME_p)
        _GetProcessTimes.restype = BOOL

        def filetime2py(obj):
            return (obj.dwHighDateTime << 32) + obj.dwLowDateTime

        def GetProcessTimes(handle):
            creation_time = FILETIME()
            exit_time = FILETIME()
            kernel_time = FILETIME()
            user_time = FILETIME()
            ok = _GetProcessTimes(handle,
                        byref(creation_time), byref(exit_time),
                        byref(kernel_time), byref(user_time))
            if not ok:
                raise ctypes_winerror()
            return (filetime2py(creation_time), filetime2py(exit_time),
                    filetime2py(kernel_time), filetime2py(user_time))

        _GetSystemTimeAsFileTime = windll.kernel32.GetSystemTimeAsFileTime
        _GetSystemTimeAsFileTime.argtypes = (FILETIME_p,)
        _GetSystemTimeAsFileTime.restype = None

        def GetSystemTimeAsFileTime():
            system_time = FILETIME()
            _GetSystemTimeAsFileTime(byref(system_time))
            return filetime2py(system_time)

    libc_name = ctypes.util.find_library('c')
    if libc_name:
        libc = ctypes.CDLL(libc_name, use_errno=True)
        clock_t = ctypes.c_ulong

        if sys.platform == 'darwin':
            mach_absolute_time = libc.mach_absolute_time
            mach_absolute_time.argtypes = ()
            mach_absolute_time.restype = ctypes.c_uint64
            has_mach_absolute_time = True

            class mach_timebase_info_data_t(ctypes.Structure):
                _fields_ = (
                    ('numer', ctypes.c_uint32),
                    ('denom', ctypes.c_uint32),
                )
            mach_timebase_info_data_p = POINTER(mach_timebase_info_data_t)

            _mach_timebase_info = libc.mach_timebase_info
            _mach_timebase_info.argtypes = (mach_timebase_info_data_p,)
            _mach_timebase_info.restype = ctypes.c_int
            def mach_timebase_info():
                timebase = mach_timebase_info_data_t()
                _mach_timebase_info(byref(timebase))
                return (timebase.numer, timebase.denom)

        _libc_clock = libc.clock
        _libc_clock.argtypes = ()
        _libc_clock.restype = clock_t
        has_libc_clock = True

        if hasattr(libc, 'sleep'):
            _libc_sleep = libc.sleep
            _libc_sleep.argtypes = (ctypes.c_uint,)
            _libc_sleep.restype = ctypes.c_uint
            has_libc_sleep = True

        if hasattr(libc, 'gettimeofday'):
            class timeval(ctypes.Structure):
                _fields_ = (
                    ('tv_sec', time_t),
                    ('tv_usec', ctypes.c_long),
                )
            timeval_p = POINTER(timeval)
            timezone_p = ctypes.c_void_p

            _gettimeofday = libc.gettimeofday
            # FIXME: some platforms only expect one argument
            _gettimeofday.argtypes = (timeval_p, timezone_p)
            _gettimeofday.restype = ctypes.c_int
            def gettimeofday():
                tv = timeval()
                err = _gettimeofday(byref(tv), None)
                if err:
                    raise ctypes_oserror()
                return tv
            has_gettimeofday = True

        time_t_p = POINTER(time_t)
        if hasattr(libc, 'time'):
            _libc__time = libc.time
            _libc__time.argtypes = (time_t_p,)
            _libc__time.restype = time_t
            def _libc_time():
                return _libc__time(None)
            has_libc_time = True

    if sys.platform.startswith(("freebsd", "openbsd")):
        librt_name = libc_name
    else:
        librt_name = ctypes.util.find_library('rt')
    if librt_name:
        librt = ctypes.CDLL(librt_name, use_errno=True)
        if hasattr(librt, 'clock_gettime'):
            clockid_t = ctypes.c_int
            class timespec(ctypes.Structure):
                _fields_ = (
                    ('tv_sec', time_t),
                    ('tv_nsec', ctypes.c_long),
                )
            timespec_p = POINTER(timespec)

            _clock_gettime = librt.clock_gettime
            _clock_gettime.argtypes = (clockid_t, timespec_p)
            _clock_gettime.restype = ctypes.c_int
            def clock_gettime(clk_id):
                ts = timespec()
                err = _clock_gettime(clk_id, byref(ts))
                if err:
                    raise ctypes_oserror()
                return ts.tv_sec + ts.tv_nsec * 1e-9
            has_clock_gettime = True

            _clock_settime = librt.clock_settime
            _clock_settime.argtypes = (clockid_t, timespec_p)
            _clock_settime.restype = ctypes.c_int
            def clock_settime(clk_id, value):
                ts = timespec()
                ts.tv_sec = int(value)
                ts.tv_nsec = int(float(abs(value)) % 1.0 * 1e9)
                err = _clock_settime(clk_id, byref(ts))
                if err:
                    raise ctypes_oserror()
                return ts.tv_sec + ts.tv_nsec * 1e-9

            _clock_getres = librt.clock_getres
            _clock_getres.argtypes = (clockid_t, timespec_p)
            _clock_getres.restype = ctypes.c_int
            def clock_getres(clk_id):
                ts = timespec()
                err = _clock_getres(clk_id, byref(ts))
                if err:
                    raise ctypes_oserror()
                return ts.tv_sec + ts.tv_nsec * 1e-9

            if sys.platform.startswith("linux"):
                CLOCK_REALTIME = 0
                CLOCK_MONOTONIC = 1
                CLOCK_PROCESS_CPUTIME_ID = 2
            elif sys.platform.startswith("freebsd"):
                CLOCK_REALTIME = 0
                CLOCK_PROF = 2
                CLOCK_MONOTONIC = 4
            elif sys.platform.startswith("openbsd"):
                CLOCK_REALTIME = 0
                CLOCK_MONOTONIC = 3
            elif sys.platform.startswith("sunos"):
                CLOCK_REALTIME = 3
                CLOCK_HIGHRES = 4
                # clock_gettime(CLOCK_PROCESS_CPUTIME_ID) fails with errno 22
                # on OpenSolaris
                # CLOCK_PROCESS_CPUTIME_ID = 5

def _clock_gettime_info(use_info, clk_id):
    value = clock_gettime(clk_id)
    if use_info:
        name = {
            CLOCK_MONOTONIC: 'CLOCK_MONOTONIC',
            CLOCK_PROF: 'CLOCK_PROF',
            CLOCK_HIGHRES: 'CLOCK_HIGHRES',
            CLOCK_PROCESS_CPUTIME_ID: 'CLOCK_PROCESS_CPUTIME_ID',
            CLOCK_REALTIME: 'CLOCK_REALTIME',
        }[clk_id]
        try:
            resolution = clock_getres(clk_id)
        except OSError:
            resolution = 1e-9
        info = {
            'implementation': 'clock_gettime(%s)' % name,
            'resolution': resolution,
        }
        if clk_id in (CLOCK_MONOTONIC, CLOCK_PROF, CLOCK_HIGHRES, CLOCK_PROCESS_CPUTIME_ID):
            info['monotonic'] = True
            info['adjustable'] = False
        elif clk_id in (CLOCK_REALTIME,):
            info['monotonic'] = False
            info['adjustable'] = True
    else:
        info = None
    return (value, info)

has_monotonic = False
if os.name == 'nt':
    # GetTickCount64() requires Windows Vista, Server 2008 or later
    if has_GetTickCount64:
        def _monotonic(use_info):
            value = GetTickCount64() * 1e-3
            if use_info:
                info = {
                    'implementation': "GetTickCount64()",
                    "monotonic": True,
                    "resolution": 1e-3,
                    "adjustable": False,
                }
                # FIXME: call GetSystemTimeAdjustment() to get the resolution
            else:
                info = None
            return (value, info)
        has_monotonic = True
    else:
        def _monotonic(use_info):
            ticks = GetTickCount()
            if ticks < _monotonic.last:
                # Integer overflow detected
                _monotonic.delta += 2**32
            _monotonic.last = ticks
            value = (ticks + _monotonic.delta) * 1e-3

            if use_info:
                info = {
                    'implementation': "GetTickCount()",
                    "monotonic": True,
                    "resolution": 1e-3,
                    "adjustable": False,
                }
                # FIXME: call GetSystemTimeAdjustment() to get the resolution
            else:
                info = None
            return (value, info)
        _monotonic.last = 0
        _monotonic.delta = 0
    has_monotonic = True

elif has_mach_absolute_time:
    def _monotonic(use_info):
        if _monotonic.factor is None:
            timebase = mach_timebase_info()
            _monotonic.factor = timebase[0] / timebase[1] * 1e-9
        value = mach_absolute_time() * _monotonic.factor
        if use_info:
            info = {
                'implementation': "mach_absolute_time()",
                'resolution': _monotonic.factor,
                'monotonic': True,
                'adjustable': False,
            }
        else:
            info = None
        return (value, info)
    _monotonic.factor = None
    has_monotonic = True

elif has_clock_gettime and CLOCK_HIGHRES is not None:
    def _monotonic(use_info):
        return _clock_gettime_info(use_info, CLOCK_HIGHRES)
    has_monotonic = True

elif has_clock_gettime and CLOCK_MONOTONIC is not None:
    def _monotonic(use_info):
        return _clock_gettime_info(use_info, CLOCK_MONOTONIC)
    has_monotonic = True

if has_monotonic:
    def monotonic():
        return _monotonic(False)[0]


def _perf_counter(use_info):
    info = None
    if _perf_counter.use_performance_counter:
        if _perf_counter.performance_frequency is None:
            value, info = _win_perf_counter(use_info)
            if value is not None:
                return (value, info)
    if _perf_counter.use_monotonic:
        # The monotonic clock is preferred over the system time
        try:
            return _monotonic(use_info)
        except (OSError, WindowsError):
            _perf_counter.use_monotonic = False
    return _time(use_info)
_perf_counter.use_performance_counter = (os.name == 'nt')
if _perf_counter.use_performance_counter:
    _perf_counter.performance_frequency = None
_perf_counter.use_monotonic = has_monotonic

def perf_counter():
    return _perf_counter(False)[0]


if os.name == 'nt':
    def _process_time(use_info):
        handle = GetCurrentProcess()
        process_times = GetProcessTimes(handle)
        value = (process_times[2] + process_times[3]) * 1e-7
        if use_info:
            info = {
                "implementation": "GetProcessTimes()",
                "resolution": 1e-7,
                "monotonic": True,
                "adjustable": False,
                # FIXME: call GetSystemTimeAdjustment() to get the resolution
            }
        else:
            info = None
        return (value, info)
else:
    import os
    try:
        import resource
    except ImportError:
        has_resource = False
    else:
        has_resource = True

    def _process_time(use_info):
        info = None
        if _process_time.clock_id is not None:
            try:
                return _clock_gettime_info(use_info, _process_time.clock_id)
            except OSError:
                _process_time.clock_id = None
        if _process_time.use_getrusage:
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                value = usage[0] + usage[1]
            except OSError:
                _process_time.use_getrusage = False
            else:
                if use_info:
                    info = {
                        "implementation": "getrusage(RUSAGE_SELF)",
                        "resolution": 1e-6,
                        "monotonic": True,
                        "adjustable": False,
                    }
                return (value, info)
        if _process_time.use_times:
            try:
                times = os.times()
                value = times[0] + times[1]
            except OSError:
                _process_time.use_getrusage = False
            else:
                if use_info:
                    try:
                        ticks_per_second = os.sysconf("SC_CLK_TCK")
                    except ValueError:
                        ticks_per_second = 60 # FIXME: get HZ constant
                    info = {
                        "implementation": "times()",
                        "resolution": 1.0 / ticks_per_second,
                        "monotonic": True,
                        "adjustable": False,
                    }
                return (value, info)
        return _libc_clock_info(use_info)
    if has_clock_gettime and CLOCK_PROCESS_CPUTIME_ID is not None:
        _process_time.clock_id = CLOCK_PROCESS_CPUTIME_ID
    elif has_clock_gettime and CLOCK_PROF is not None:
        _process_time.clock_id = CLOCK_PROF
    else:
        _process_time.clock_id = None
    _process_time.use_getrusage = has_resource
    # On OS/2, only the 5th field of os.times() is set, others are zeros
    _process_time.use_times = (hasattr(os, 'times') and os.name != 'os2')

def process_time():
    return _process_time(False)[0]


if os.name == "nt":
    def _time(use_info):
        value = GetSystemTimeAsFileTime() * 1e-7
        if use_info:
            info = {
                'implementation': 'GetSystemTimeAsFileTime',
                'resolution': 1e-7,
                'monotonic': False,
                # FIXME: call GetSystemTimeAdjustment() to get the resolution
                # and adjustable
            }
        else:
            info = None
        return (value, info)
else:
    def _time(use_info):
        info = None
        if has_clock_gettime and CLOCK_REALTIME is not None:
            try:
                return _clock_gettime_info(use_info, CLOCK_REALTIME)
            except OSError:
                # CLOCK_REALTIME is not supported (unlikely)
                pass
        if has_gettimeofday:
            try:
                tv = gettimeofday()
            except OSError:
                # gettimeofday() should not fail
                pass
            else:
                if use_info:
                    info = {
                        'monotonic': False,
                        "implementation": "gettimeofday()",
                        "resolution": 1e-6,
                        'monotonic': False,
                        'adjustable': True,
                    }
                value = tv.tv_sec + tv.tv_usec * 1e-6
                return (value, info)
        # FIXME: implement ftime()
        if has_ftime:
            if use_info:
                info = {
                    "implementation": "ftime()",
                    "resolution": 1e-3,
                    'monotonic': False,
                    'adjustable': True,
                }
            value = ftime()
        elif has_libc_time:
            if use_info:
                info = {
                    "implementation": "time()",
                    "resolution": 1.0,
                    'monotonic': False,
                    'adjustable': True,
                }
            value = float(_libc_time())
        else:
            if use_info:
                info = {
                    "implementation": "time.time()",
                    'monotonic': False,
                    'adjustable': True,
                }
                if os.name == "nt":
                    # On Windows, time.time() uses ftime()
                    info["resolution"] = 1e-3
                else:
                    # guess that time.time() uses gettimeofday()
                    info["resolution"] = 1e-6
            value = python_time.time()
        return (value, info)

def time():
    return _time(False)[0]


try:
    import select
except ImportError:
    has_select = False
else:
    # FIXME: On Windows, select.select([], [], [], seconds) fails with
    #        select.error(10093)
    has_select = (hasattr(select, "select") and os.name != "nt")

if has_select:
    def _sleep(seconds):
        return select.select([], [], [], seconds)

elif has_delay:
    def _sleep(seconds):
        milliseconds = int(seconds * 1000)
        # FIXME
        delay(milliseconds)

#elif os.name == "nt":
#    def _sleep(seconds):
#        milliseconds = int(seconds * 1000)
#        # FIXME: use ctypes
#        win32api.ResetEvent(hInterruptEvent);
#        win32api.WaitForSingleObject(sleep.sigint_event, milliseconds)
#
#    sleep.sigint_event = win32api.CreateEvent(NULL, TRUE, FALSE, FALSE)
#    # SetEvent(sleep.sigint_event) will be called by the signal handler of SIGINT

elif os.name == "os2":
    def _sleep(seconds):
        milliseconds = int(seconds * 1000)
        # FIXME
        DosSleep(milliseconds)

elif has_libc_sleep:
    def _sleep(seconds):
        seconds = int(seconds)
        _libc_sleep(seconds)

else:
    def _sleep(seconds):
        python_time.sleep(seconds)

def sleep(seconds):
    if seconds < 0:
        raise ValueError("sleep length must be non-negative")
    _sleep(seconds)

def _libc_clock_info(use_info):
    if use_info:
        info = {
            'implementation': 'clock()',
            'resolution': 1.0,
            # FIXME: 'resolution': 1.0 / CLOCKS_PER_SEC,
            'monotonic': True,
            'adjustable': False,
        }
        if os.name != "nt":
            info['monotonic'] = True
    else:
        info = None
    if has_libc_clock:
        value = _libc_clock()
        if use_info:
            info['implementation'] = 'clock()'
    else:
        value = python_time.clock()
        if use_info:
            info['implementation'] = 'time.clock()'
    return (value, info)

def _win_perf_counter(use_info):
    if _win_perf_counter.perf_frequency is None:
        try:
            _win_perf_counter.perf_frequency = float(QueryPerformanceFrequency())
        except WindowsError:
            # QueryPerformanceFrequency() fails if the installed
            # hardware does not support a high-resolution performance
            # counter
            return (None, None)

    value = QueryPerformanceCounter() / _win_perf_counter.perf_frequency
    if use_info:
        info = {
            'implementation': 'QueryPerformanceCounter',
            'resolution': 1.0 / _win_perf_counter.perf_frequency,
            'monotonic': True,
            'adjustable': False,
        }
    else:
        info = None
    return (value, info)
_win_perf_counter.perf_frequency = None

if os.name == 'nt':
    def _clock(use_info):
        info = None
        if _clock.use_performance_counter:
            value, info = _win_perf_counter(use_info)
            if value is not None:
                return (value, info)
        return _libc_clock_info(use_info)
    _clock.use_performance_counter = True
else:
    def _clock(use_info):
        return _libc_clock_info(use_info)

def clock():
    return _clock(False)[0]


class clock_info(object):
    def __init__(self, implementation, monotonic, adjustable, resolution):
        self.implementation = implementation
        self.monotonic = monotonic
        self.adjustable = adjustable
        self.resolution = resolution

    def __repr__(self):
        return (
            'clockinfo(adjustable=%s, implementation=%r, monotonic=%s, resolution=%s'
            % (self.adjustable, self.implementation, self.monotonic, self.resolution))

def get_clock_info(name):
    if name == 'clock':
        info = _clock(True)[1]
    elif name == 'perf_counter':
        info = _perf_counter(True)[1]
    elif name == 'process_time':
        info = _process_time(True)[1]
    elif name == 'time':
        info = _time(True)[1]
    elif has_monotonic and name == 'monotonic':
        info = _monotonic(True)[1]
    else:
        raise ValueError("unknown clock: %s" % name)
    return clock_info(**info)

if __name__ == "__main__":
    import threading
    import unittest
    from errno import EPERM

    class TestPEP418(unittest.TestCase):
        if not hasattr(unittest.TestCase, 'assertIsInstance'):
            # Python < 2.7 or Python < 3.2
            def assertIsInstance(self, obj, klass):
                self.assertTrue(isinstance(obj, klass))
            def assertGreater(self, a, b):
                self.assertTrue(a > b)
            def assertLess(self, a, b):
                self.assertTrue(a < b)
            def assertLessEqual(self, a, b):
                self.assertTrue(a <= b)
            def assertAlmostEqual(self, first, second, delta):
                self.assertTrue(abs(first - second) <= delta)

        def test_clock(self):
            clock()

            info = get_clock_info('clock')
            self.assertEqual(info.monotonic, True)
            self.assertEqual(info.adjustable, False)

        def test_get_clock_info(self):
            clocks = ['clock', 'perf_counter', 'process_time', 'time']
            if has_monotonic:
                clocks.append('monotonic')

            for name in clocks:
                info = get_clock_info(name)
                self.assertIsInstance(info.implementation, str)
                self.assertNotEqual(info.implementation, '')
                self.assertIsInstance(info.monotonic, bool)
                self.assertIsInstance(info.resolution, float)
                # 0 < resolution <= 1.0
                self.assertGreater(info.resolution, 0)
                self.assertLessEqual(info.resolution, 1)
                self.assertIsInstance(info.adjustable, bool)

            self.assertRaises(ValueError, get_clock_info, 'xxx')

        if not has_monotonic:
            print("Skip test_monotonic: need time.monotonic")
        else:
            def test_monotonic(self):
                t1 = monotonic()
                python_time.sleep(0.1)
                t2 = monotonic()
                dt = t2 - t1
                self.assertGreater(t2, t1)
                self.assertAlmostEqual(dt, 0.1, delta=0.2)

                info = get_clock_info('monotonic')
                self.assertEqual(info.monotonic, True)
                self.assertEqual(info.adjustable, False)

        if not has_monotonic or not has_clock_gettime:
            if not has_monotonic:
                print('Skip test_monotonic_settime: need time.monotonic')
            elif not has_clock_gettime:
                print('Skip test_monotonic_settime: need time.clock_settime')
        else:
            def test_monotonic_settime(self):
                t1 = monotonic()
                realtime = clock_gettime(CLOCK_REALTIME)
                # jump backward with an offset of 1 hour
                try:
                    clock_settime(CLOCK_REALTIME, realtime - 3600)
                except OSError as err:
                    if err.errno == EPERM:
                        if hasattr(unittest, 'SkipTest'):
                            raise unittest.SkipTest(str(err))
                        else:
                            print("Skip test_monotonic_settime: %s" % err)
                            return
                    else:
                        raise
                t2 = monotonic()
                clock_settime(CLOCK_REALTIME, realtime)
                # monotonic must not be affected by system clock updates
                self.assertGreaterEqual(t2, t1)

        def test_perf_counter(self):
            perf_counter()

        def test_process_time(self):
            start = process_time()
            python_time.sleep(0.1)
            stop = process_time()
            self.assertLess(stop - start, 0.01)

            info = get_clock_info('process_time')
            self.assertEqual(info.monotonic, True)
            self.assertEqual(info.adjustable, False)


        def test_process_time_threads(self):
            class BusyThread(threading.Thread):
                def run(self):
                    while not self.stop:
                        pass

            thread = BusyThread()
            thread.stop = False
            t1 = process_time()
            thread.start()
            sleep(0.2)
            t2 = process_time()
            thread.stop = True
            thread.join()
            self.assertGreater(t2 - t1, 0.1)

        def test_sleep(self):
            self.assertRaises(ValueError, sleep, -2)
            self.assertRaises(ValueError, sleep, -1)
            sleep(1.2)

        def test_time(self):
            value = time()
            self.assertIsInstance(value, float)

            info = get_clock_info('time')
            self.assertEqual(info.monotonic, False)
            self.assertEqual(info.adjustable, True)


    if True:
        from pprint import pprint

        print("clock: %s" % clock())
        if has_monotonic:
            print("monotonic: %s" % monotonic())
        else:
            print("monotonic: <not available>")
        print("perf_counter: %s" % perf_counter())
        print("process_time: %s" % process_time())
        print("time: %s" % time())

        clocks = ['clock', 'perf_counter', 'process_time', 'time']
        if has_monotonic:
            clocks.append('monotonic')
        pprint(dict((name, get_clock_info(name)) for name in clocks))

    unittest.main()
