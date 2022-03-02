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

import pickle
import os
import sys


class Settings(object):
    def __init__(self):
        # Be default, the home will be in the same folder as labelImg
        home = os.path.expanduser("~")
        self.data = {}
        # self.path = os.path.join(home, '.labelImgSettings.pkl')
        self.path = os.path.join(home, '.autoOCRSettings.pkl')

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default

    def save(self):
        if self.path:
            with open(self.path, 'wb') as f:
                pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
                return True
        return False

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'rb') as f:
                    self.data = pickle.load(f)
                    return True
        except:
            print('Loading setting failed')
        return False

    def reset(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            print('Remove setting pkl file ${0}'.format(self.path))
        self.data = {}
        self.path = None
