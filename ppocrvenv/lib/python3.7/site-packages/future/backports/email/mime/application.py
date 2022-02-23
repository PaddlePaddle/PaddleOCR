# Copyright (C) 2001-2006 Python Software Foundation
# Author: Keith Dart
# Contact: email-sig@python.org

"""Class representing application/* type MIME documents."""
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

from future.backports.email import encoders
from future.backports.email.mime.nonmultipart import MIMENonMultipart

__all__ = ["MIMEApplication"]


class MIMEApplication(MIMENonMultipart):
    """Class for generating application/* MIME documents."""

    def __init__(self, _data, _subtype='octet-stream',
                 _encoder=encoders.encode_base64, **_params):
        """Create an application/* type MIME document.

        _data is a string containing the raw application data.

        _subtype is the MIME content type subtype, defaulting to
        'octet-stream'.

        _encoder is a function which will perform the actual encoding for
        transport of the application data, defaulting to base64 encoding.

        Any additional keyword arguments are passed to the base class
        constructor, which turns them into parameters on the Content-Type
        header.
        """
        if _subtype is None:
            raise TypeError('Invalid application MIME subtype')
        MIMENonMultipart.__init__(self, 'application', _subtype, **_params)
        self.set_payload(_data)
        _encoder(self)
