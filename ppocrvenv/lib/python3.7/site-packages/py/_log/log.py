"""
basic logging functionality based on a producer/consumer scheme.

XXX implement this API: (maybe put it into slogger.py?)

        log = Logger(
                    info=py.log.STDOUT,
                    debug=py.log.STDOUT,
                    command=None)
        log.info("hello", "world")
        log.command("hello", "world")

        log = Logger(info=Logger(something=...),
                     debug=py.log.STDOUT,
                     command=None)
"""
import py
import sys


class Message(object):
    def __init__(self, keywords, args):
        self.keywords = keywords
        self.args = args

    def content(self):
        return " ".join(map(str, self.args))

    def prefix(self):
        return "[%s] " % (":".join(self.keywords))

    def __str__(self):
        return self.prefix() + self.content()


class Producer(object):
    """ (deprecated) Log producer API which sends messages to be logged
        to a 'consumer' object, which then prints them to stdout,
        stderr, files, etc. Used extensively by PyPy-1.1.
    """

    Message = Message  # to allow later customization
    keywords2consumer = {}

    def __init__(self, keywords, keywordmapper=None, **kw):
        if hasattr(keywords, 'split'):
            keywords = tuple(keywords.split())
        self._keywords = keywords
        if keywordmapper is None:
            keywordmapper = default_keywordmapper
        self._keywordmapper = keywordmapper

    def __repr__(self):
        return "<py.log.Producer %s>" % ":".join(self._keywords)

    def __getattr__(self, name):
        if '_' in name:
            raise AttributeError(name)
        producer = self.__class__(self._keywords + (name,))
        setattr(self, name, producer)
        return producer

    def __call__(self, *args):
        """ write a message to the appropriate consumer(s) """
        func = self._keywordmapper.getconsumer(self._keywords)
        if func is not None:
            func(self.Message(self._keywords, args))

class KeywordMapper:
    def __init__(self):
        self.keywords2consumer = {}

    def getstate(self):
        return self.keywords2consumer.copy()

    def setstate(self, state):
        self.keywords2consumer.clear()
        self.keywords2consumer.update(state)

    def getconsumer(self, keywords):
        """ return a consumer matching the given keywords.

            tries to find the most suitable consumer by walking, starting from
            the back, the list of keywords, the first consumer matching a
            keyword is returned (falling back to py.log.default)
        """
        for i in range(len(keywords), 0, -1):
            try:
                return self.keywords2consumer[keywords[:i]]
            except KeyError:
                continue
        return self.keywords2consumer.get('default', default_consumer)

    def setconsumer(self, keywords, consumer):
        """ set a consumer for a set of keywords. """
        # normalize to tuples
        if isinstance(keywords, str):
            keywords = tuple(filter(None, keywords.split()))
        elif hasattr(keywords, '_keywords'):
            keywords = keywords._keywords
        elif not isinstance(keywords, tuple):
            raise TypeError("key %r is not a string or tuple" % (keywords,))
        if consumer is not None and not py.builtin.callable(consumer):
            if not hasattr(consumer, 'write'):
                raise TypeError(
                    "%r should be None, callable or file-like" % (consumer,))
            consumer = File(consumer)
        self.keywords2consumer[keywords] = consumer


def default_consumer(msg):
    """ the default consumer, prints the message to stdout (using 'print') """
    sys.stderr.write(str(msg)+"\n")

default_keywordmapper = KeywordMapper()


def setconsumer(keywords, consumer):
    default_keywordmapper.setconsumer(keywords, consumer)


def setstate(state):
    default_keywordmapper.setstate(state)


def getstate():
    return default_keywordmapper.getstate()

#
# Consumers
#


class File(object):
    """ log consumer wrapping a file(-like) object """
    def __init__(self, f):
        assert hasattr(f, 'write')
        # assert isinstance(f, file) or not hasattr(f, 'open')
        self._file = f

    def __call__(self, msg):
        """ write a message to the log """
        self._file.write(str(msg) + "\n")
        if hasattr(self._file, 'flush'):
            self._file.flush()


class Path(object):
    """ log consumer that opens and writes to a Path """
    def __init__(self, filename, append=False,
                 delayed_create=False, buffering=False):
        self._append = append
        self._filename = str(filename)
        self._buffering = buffering
        if not delayed_create:
            self._openfile()

    def _openfile(self):
        mode = self._append and 'a' or 'w'
        f = open(self._filename, mode)
        self._file = f

    def __call__(self, msg):
        """ write a message to the log """
        if not hasattr(self, "_file"):
            self._openfile()
        self._file.write(str(msg) + "\n")
        if not self._buffering:
            self._file.flush()


def STDOUT(msg):
    """ consumer that writes to sys.stdout """
    sys.stdout.write(str(msg)+"\n")


def STDERR(msg):
    """ consumer that writes to sys.stderr """
    sys.stderr.write(str(msg)+"\n")


class Syslog:
    """ consumer that writes to the syslog daemon """

    def __init__(self, priority=None):
        if priority is None:
            priority = self.LOG_INFO
        self.priority = priority

    def __call__(self, msg):
        """ write a message to the log """
        import syslog
        syslog.syslog(self.priority, str(msg))


try:
    import syslog
except ImportError:
    pass
else:
    for _prio in "EMERG ALERT CRIT ERR WARNING NOTICE INFO DEBUG".split():
        _prio = "LOG_" + _prio
        try:
            setattr(Syslog, _prio, getattr(syslog, _prio))
        except AttributeError:
            pass
