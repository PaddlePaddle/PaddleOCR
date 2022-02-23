# Copyright (C) 2002-2006 Python Software Foundation
# Author: Barry Warsaw
# Contact: email-sig@python.org

"""Base class for MIME type messages that are not multipart."""
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

__all__ = ['MIMENonMultipart']

from future.backports.email import errors
from future.backports.email.mime.base import MIMEBase


class MIMENonMultipart(MIMEBase):
    """Base class for MIME multipart/* type messages."""

    def attach(self, payload):
        # The public API prohibits attaching multiple subparts to MIMEBase
        # derived subtypes since none of them are, by definition, of content
        # type multipart/*
        raise errors.MultipartConversionError(
            'Cannot attach additional subparts to non-multipart/*')
