import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    # joblib imports may raise DeprecationWarning on certain Python
    # versions
    import joblib
    from joblib import logger
    from joblib import dump, load
    from joblib import __version__
    from joblib import effective_n_jobs
    from joblib import hash
    from joblib import cpu_count, Parallel, Memory, delayed
    from joblib import parallel_backend, register_parallel_backend


__all__ = [
    "parallel_backend",
    "register_parallel_backend",
    "cpu_count",
    "Parallel",
    "Memory",
    "delayed",
    "effective_n_jobs",
    "hash",
    "logger",
    "dump",
    "load",
    "joblib",
    "__version__",
]
