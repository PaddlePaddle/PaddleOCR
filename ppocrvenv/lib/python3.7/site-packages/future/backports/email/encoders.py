# Copyright (C) 2001-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Encodings and related functions."""
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import str

__all__ = [
    'encode_7or8bit',
    'encode_base64',
    'encode_noop',
    'encode_quopri',
    ]


try:
    from base64 import encodebytes as _bencode
except ImportError:
    # Py2 compatibility. TODO: test this!
    from base64 import encodestring as _bencode
from quopri import encodestring as _encodestring


def _qencode(s):
    enc = _encodestring(s, quotetabs=True)
    # Must encode spaces, which quopri.encodestring() doesn't do
    return enc.replace(' ', '=20')


def encode_base64(msg):
    """Encode the message's payload in Base64.

    Also, add an appropriate Content-Transfer-Encoding header.
    """
    orig = msg.get_payload()
    encdata = str(_bencode(orig), 'ascii')
    msg.set_payload(encdata)
    msg['Content-Transfer-Encoding'] = 'base64'


def encode_quopri(msg):
    """Encode the message's payload in quoted-printable.

    Also, add an appropriate Content-Transfer-Encoding header.
    """
    orig = msg.get_payload()
    encdata = _qencode(orig)
    msg.set_payload(encdata)
    msg['Content-Transfer-Encoding'] = 'quoted-printable'


def encode_7or8bit(msg):
    """Set the Content-Transfer-Encoding header to 7bit or 8bit."""
    orig = msg.get_payload()
    if orig is None:
        # There's no payload.  For backwards compatibility we use 7bit
        msg['Content-Transfer-Encoding'] = '7bit'
        return
    # We play a trick to make this go fast.  If encoding/decode to ASCII
    # succeeds, we know the data must be 7bit, otherwise treat it as 8bit.
    try:
        if isinstance(orig, str):
            orig.encode('ascii')
        else:
            orig.decode('ascii')
    except UnicodeError:
        charset = msg.get_charset()
        output_cset = charset and charset.output_charset
        # iso-2022-* is non-ASCII but encodes to a 7-bit representation
        if output_cset and output_cset.lower().startswith('iso-2022-'):
            msg['Content-Transfer-Encoding'] = '7bit'
        else:
            msg['Content-Transfer-Encoding'] = '8bit'
    else:
        msg['Content-Transfer-Encoding'] = '7bit'
    if not isinstance(orig, str):
        msg.set_payload(orig.decode('ascii', 'surrogateescape'))


def encode_noop(msg):
    """Do nothing."""
    # Well, not quite *nothing*: in Python3 we have to turn bytes into a string
    # in our internal surrogateescaped form in order to keep the model
    # consistent.
    orig = msg.get_payload()
    if not isinstance(orig, str):
        msg.set_payload(orig.decode('ascii', 'surrogateescape'))
