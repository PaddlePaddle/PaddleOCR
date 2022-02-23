# -*- coding: utf-8 -*-
import logging
import os
import sys

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)


def setLogLevel(log_level):
    default_logger.setLevel(log_level)


check_paddle_install = {'is_paddle_installed': False}

try:
    import pkg_resources

    get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                                os.path.join(*res))
except ImportError:
    get_module_res = lambda *res: open(os.path.normpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__), *res)), 'rb')


def enable_paddle():
    try:
        import paddle
    except ImportError:
        default_logger.debug("Installing paddle-tiny, please wait a minute......")
        os.system("pip install paddlepaddle-tiny")
        try:
            import paddle
        except ImportError:
            default_logger.debug(
                "Import paddle error, please use command to install: pip install paddlepaddle-tiny==1.6.1."
                "Now, back to jieba basic cut......")
    if paddle.__version__ < '1.6.1':
        default_logger.debug("Find your own paddle version doesn't satisfy the minimum requirement (1.6.1), "
                             "please install paddle tiny by 'pip install --upgrade paddlepaddle-tiny', "
                             "or upgrade paddle full version by "
                             "'pip install --upgrade paddlepaddle (-gpu for GPU version)' ")
    else:
        try:
            import jieba.lac_small.predict as predict
            default_logger.debug("Paddle enabled successfully......")
            check_paddle_install['is_paddle_installed'] = True
        except ImportError:
            default_logger.debug("Import error, cannot find paddle.fluid and jieba.lac_small.predict module. "
                                 "Now, back to jieba basic cut......")


PY2 = sys.version_info[0] == 2

default_encoding = sys.getfilesystemencoding()

if PY2:
    text_type = unicode
    string_types = (str, unicode)

    iterkeys = lambda d: d.iterkeys()
    itervalues = lambda d: d.itervalues()
    iteritems = lambda d: d.iteritems()

else:
    text_type = str
    string_types = (str,)
    xrange = range

    iterkeys = lambda d: iter(d.keys())
    itervalues = lambda d: iter(d.values())
    iteritems = lambda d: iter(d.items())


def strdecode(sentence):
    if not isinstance(sentence, text_type):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            sentence = sentence.decode('gbk', 'ignore')
    return sentence


def resolve_filename(f):
    try:
        return f.name
    except AttributeError:
        return repr(f)
