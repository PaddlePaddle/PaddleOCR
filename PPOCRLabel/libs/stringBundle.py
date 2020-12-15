# Copyright (c) <2015-Present> Tzutalin
# Copyright (C) 2013  MIT, Computer Science and Artificial Intelligence Laboratory. Bryan Russell, Antonio Torralba,
# William T. Freeman. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
# SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import sys
import locale
from libs.ustr import ustr

try:
    from PyQt5.QtCore import *
except ImportError:
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtCore import *


class StringBundle:

    __create_key = object()

    def __init__(self, create_key, localeStr):
        assert(create_key == StringBundle.__create_key), "StringBundle must be created using StringBundle.getBundle"
        self.idToMessage = {}
        paths = self.__createLookupFallbackList(localeStr)
        for path in paths:
            self.__loadBundle(path)

    @classmethod
    def getBundle(cls, localeStr=None):
        if localeStr is None:
            try:
                localeStr = locale.getlocale()[0] if locale.getlocale() and len(
                    locale.getlocale()) > 0 else os.getenv('LANG')
            except:
                print('Invalid locale')
                localeStr = 'en'

        return StringBundle(cls.__create_key, localeStr)

    def getString(self, stringId):
        assert(stringId in self.idToMessage), "Missing string id : " + stringId
        return self.idToMessage[stringId]

    def __createLookupFallbackList(self, localeStr):
        resultPaths = []
        basePath = ":/strings"
        resultPaths.append(basePath)
        if localeStr is not None:
            # Don't follow standard BCP47. Simple fallback
            tags = re.split('[^a-zA-Z]', localeStr)
            for tag in tags:
                lastPath = resultPaths[-1]
                resultPaths.append(lastPath + '-' + tag)

        return resultPaths

    def __loadBundle(self, path):
        PROP_SEPERATOR = '='
        f = QFile(path)
        if f.exists():
            if f.open(QIODevice.ReadOnly | QFile.Text):
                text = QTextStream(f)
                text.setCodec("UTF-8")

            while not text.atEnd():
                line = ustr(text.readLine())
                key_value = line.split(PROP_SEPERATOR)
                key = key_value[0].strip()
                value = PROP_SEPERATOR.join(key_value[1:]).strip().strip('"')
                self.idToMessage[key] = value

            f.close()
