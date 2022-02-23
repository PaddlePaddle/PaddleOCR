class BaseLockException(Exception):
    # Error codes:
    LOCK_FAILED = 1

    def __init__(self, *args, fh=None, **kwargs):
        self.fh = fh
        Exception.__init__(self, *args, **kwargs)


class LockException(BaseLockException):
    pass


class AlreadyLocked(BaseLockException):
    pass


class FileToLarge(BaseLockException):
    pass
