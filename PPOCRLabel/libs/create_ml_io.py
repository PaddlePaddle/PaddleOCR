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
# -*- coding: utf8 -*-
import json
from pathlib import Path

from libs.constants import DEFAULT_ENCODING
import os

JSON_EXT = '.json'
ENCODE_METHOD = DEFAULT_ENCODING


class CreateMLWriter:
    def __init__(self, foldername, filename, imgsize, shapes, outputfile, databasesrc='Unknown', localimgpath=None):
        self.foldername = foldername
        self.filename = filename
        self.databasesrc = databasesrc
        self.imgsize = imgsize
        self.boxlist = []
        self.localimgpath = localimgpath
        self.verified = False
        self.shapes = shapes
        self.outputfile = outputfile

    def write(self):
        if os.path.isfile(self.outputfile):
            with open(self.outputfile, "r") as file:
                input_data = file.read()
                outputdict = json.loads(input_data)
        else:
            outputdict = []

        outputimagedict = {
            "image": self.filename,
            "annotations": []
        }

        for shape in self.shapes:
            points = shape["points"]

            x1 = points[0][0]
            y1 = points[0][1]
            x2 = points[1][0]
            y2 = points[2][1]

            height, width, x, y = self.calculate_coordinates(x1, x2, y1, y2)

            shapedict = {
                "label": shape["label"],
                "coordinates": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
            }
            outputimagedict["annotations"].append(shapedict)

        # check if image already in output
        exists = False
        for i in range(0, len(outputdict)):
            if outputdict[i]["image"] == outputimagedict["image"]:
                exists = True
                outputdict[i] = outputimagedict
                break

        if not exists:
            outputdict.append(outputimagedict)

        Path(self.outputfile).write_text(json.dumps(outputdict), ENCODE_METHOD)

    def calculate_coordinates(self, x1, x2, y1, y2):
        if x1 < x2:
            xmin = x1
            xmax = x2
        else:
            xmin = x2
            xmax = x1
        if y1 < y2:
            ymin = y1
            ymax = y2
        else:
            ymin = y2
            ymax = y1
        width = xmax - xmin
        if width < 0:
            width = width * -1
        height = ymax - ymin
        # x and y from center of rect
        x = xmin + width / 2
        y = ymin + height / 2
        return height, width, x, y


class CreateMLReader:
    def __init__(self, jsonpath, filepath):
        self.jsonpath = jsonpath
        self.shapes = []
        self.verified = False
        self.filename = filepath.split("/")[-1:][0]
        try:
            self.parse_json()
        except ValueError:
            print("JSON decoding failed")

    def parse_json(self):
        with open(self.jsonpath, "r") as file:
            inputdata = file.read()

        outputdict = json.loads(inputdata)
        self.verified = True

        if len(self.shapes) > 0:
            self.shapes = []
        for image in outputdict:
            if image["image"] == self.filename:
                for shape in image["annotations"]:
                    self.add_shape(shape["label"], shape["coordinates"])

    def add_shape(self, label, bndbox):
        xmin = bndbox["x"] - (bndbox["width"] / 2)
        ymin = bndbox["y"] - (bndbox["height"] / 2)

        xmax = bndbox["x"] + (bndbox["width"] / 2)
        ymax = bndbox["y"] + (bndbox["height"] / 2)

        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, True))

    def get_shapes(self):
        return self.shapes
