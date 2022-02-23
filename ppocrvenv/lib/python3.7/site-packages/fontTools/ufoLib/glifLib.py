"""
glifLib.py -- Generic module for reading and writing the .glif format.

More info about the .glif format (GLyphInterchangeFormat) can be found here:

	http://unifiedfontobject.org

The main class in this module is GlyphSet. It manages a set of .glif files
in a folder. It offers two ways to read glyph data, and one way to write
glyph data. See the class doc string for details.
"""

from __future__ import annotations

import logging
import enum
from warnings import warn
from collections import OrderedDict
import fs
import fs.base
import fs.errors
import fs.osfs
import fs.path
from fontTools.misc.textTools import tobytes
from fontTools.misc import plistlib
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from fontTools.ufoLib.errors import GlifLibError
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.validators import (
	genericTypeValidator,
	colorValidator,
	guidelinesValidator,
	anchorsValidator,
	identifierValidator,
	imageValidator,
	glyphLibValidator,
)
from fontTools.misc import etree
from fontTools.ufoLib import _UFOBaseIO, UFOFormatVersion
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin


__all__ = [
	"GlyphSet",
	"GlifLibError",
	"readGlyphFromString", "writeGlyphToString",
	"glyphNameToFileName"
]

logger = logging.getLogger(__name__)


# ---------
# Constants
# ---------

CONTENTS_FILENAME = "contents.plist"
LAYERINFO_FILENAME = "layerinfo.plist"


class GLIFFormatVersion(tuple, _VersionTupleEnumMixin, enum.Enum):
	FORMAT_1_0 = (1, 0)
	FORMAT_2_0 = (2, 0)

	@classmethod
	def default(cls, ufoFormatVersion=None):
		if ufoFormatVersion is not None:
			return max(cls.supported_versions(ufoFormatVersion))
		return super().default()

	@classmethod
	def supported_versions(cls, ufoFormatVersion=None):
		if ufoFormatVersion is None:
			# if ufo format unspecified, return all the supported GLIF formats
			return super().supported_versions()
		# else only return the GLIF formats supported by the given UFO format
		versions = {cls.FORMAT_1_0}
		if ufoFormatVersion >= UFOFormatVersion.FORMAT_3_0:
			versions.add(cls.FORMAT_2_0)
		return frozenset(versions)


# ------------
# Simple Glyph
# ------------

class Glyph:

	"""
	Minimal glyph object. It has no glyph attributes until either
	the draw() or the drawPoints() method has been called.
	"""

	def __init__(self, glyphName, glyphSet):
		self.glyphName = glyphName
		self.glyphSet = glyphSet

	def draw(self, pen):
		"""
		Draw this glyph onto a *FontTools* Pen.
		"""
		pointPen = PointToSegmentPen(pen)
		self.drawPoints(pointPen)

	def drawPoints(self, pointPen):
		"""
		Draw this glyph onto a PointPen.
		"""
		self.glyphSet.readGlyph(self.glyphName, self, pointPen)


# ---------
# Glyph Set
# ---------

class GlyphSet(_UFOBaseIO):

	"""
	GlyphSet manages a set of .glif files inside one directory.

	GlyphSet's constructor takes a path to an existing directory as it's
	first argument. Reading glyph data can either be done through the
	readGlyph() method, or by using GlyphSet's dictionary interface, where
	the keys are glyph names and the values are (very) simple glyph objects.

	To write a glyph to the glyph set, you use the writeGlyph() method.
	The simple glyph objects returned through the dict interface do not
	support writing, they are just a convenient way to get at the glyph data.
	"""

	glyphClass = Glyph

	def __init__(
		self,
		path,
		glyphNameToFileNameFunc=None,
		ufoFormatVersion=None,
		validateRead=True,
		validateWrite=True,
		expectContentsFile=False,
	):
		"""
		'path' should be a path (string) to an existing local directory, or
		an instance of fs.base.FS class.

		The optional 'glyphNameToFileNameFunc' argument must be a callback
		function that takes two arguments: a glyph name and a list of all
		existing filenames (if any exist). It should return a file name
		(including the .glif extension). The glyphNameToFileName function
		is called whenever a file name is created for a given glyph name.

		``validateRead`` will validate read operations. Its default is ``True``.
		``validateWrite`` will validate write operations. Its default is ``True``.
		``expectContentsFile`` will raise a GlifLibError if a contents.plist file is
		not found on the glyph set file system. This should be set to ``True`` if you
		are reading an existing UFO and ``False`` if you create a fresh	glyph set.
		"""
		try:
			ufoFormatVersion = UFOFormatVersion(ufoFormatVersion)
		except ValueError as e:
			from fontTools.ufoLib.errors import UnsupportedUFOFormat

			raise UnsupportedUFOFormat(
				f"Unsupported UFO format: {ufoFormatVersion!r}"
			) from e

		if hasattr(path, "__fspath__"):  # support os.PathLike objects
			path = path.__fspath__()

		if isinstance(path, str):
			try:
				filesystem = fs.osfs.OSFS(path)
			except fs.errors.CreateFailed:
				raise GlifLibError("No glyphs directory '%s'" % path)
			self._shouldClose = True
		elif isinstance(path, fs.base.FS):
			filesystem = path
			try:
				filesystem.check()
			except fs.errors.FilesystemClosed:
				raise GlifLibError("the filesystem '%s' is closed" % filesystem)
			self._shouldClose = False
		else:
			raise TypeError(
				"Expected a path string or fs object, found %s"
				% type(path).__name__
			)
		try:
			path = filesystem.getsyspath("/")
		except fs.errors.NoSysPath:
			# network or in-memory FS may not map to the local one
			path = str(filesystem)
		# 'dirName' is kept for backward compatibility only, but it's DEPRECATED
		# as it's not guaranteed that it maps to an existing OSFS directory.
		# Client could use the FS api via the `self.fs` attribute instead.
		self.dirName = fs.path.parts(path)[-1]
		self.fs = filesystem
		# if glyphSet contains no 'contents.plist', we consider it empty
		self._havePreviousFile = filesystem.exists(CONTENTS_FILENAME)
		if expectContentsFile and not self._havePreviousFile:
			raise GlifLibError(f"{CONTENTS_FILENAME} is missing.")
		# attribute kept for backward compatibility
		self.ufoFormatVersion = ufoFormatVersion.major
		self.ufoFormatVersionTuple = ufoFormatVersion
		if glyphNameToFileNameFunc is None:
			glyphNameToFileNameFunc = glyphNameToFileName
		self.glyphNameToFileName = glyphNameToFileNameFunc
		self._validateRead = validateRead
		self._validateWrite = validateWrite
		self._existingFileNames: set[str] | None = None
		self._reverseContents = None

		self.rebuildContents()

	def rebuildContents(self, validateRead=None):
		"""
		Rebuild the contents dict by loading contents.plist.

		``validateRead`` will validate the data, by default it is set to the
		class's ``validateRead`` value, can be overridden.
		"""
		if validateRead is None:
			validateRead = self._validateRead
		contents = self._getPlist(CONTENTS_FILENAME, {})
		# validate the contents
		if validateRead:
			invalidFormat = False
			if not isinstance(contents, dict):
				invalidFormat = True
			else:
				for name, fileName in contents.items():
					if not isinstance(name, str):
						invalidFormat = True
					if not isinstance(fileName, str):
						invalidFormat = True
					elif not self.fs.exists(fileName):
						raise GlifLibError(
							"%s references a file that does not exist: %s"
							% (CONTENTS_FILENAME, fileName)
						)
			if invalidFormat:
				raise GlifLibError("%s is not properly formatted" % CONTENTS_FILENAME)
		self.contents = contents
		self._existingFileNames = None
		self._reverseContents = None

	def getReverseContents(self):
		"""
		Return a reversed dict of self.contents, mapping file names to
		glyph names. This is primarily an aid for custom glyph name to file
		name schemes that want to make sure they don't generate duplicate
		file names. The file names are converted to lowercase so we can
		reliably check for duplicates that only differ in case, which is
		important for case-insensitive file systems.
		"""
		if self._reverseContents is None:
			d = {}
			for k, v in self.contents.items():
				d[v.lower()] = k
			self._reverseContents = d
		return self._reverseContents

	def writeContents(self):
		"""
		Write the contents.plist file out to disk. Call this method when
		you're done writing glyphs.
		"""
		self._writePlist(CONTENTS_FILENAME, self.contents)

	# layer info

	def readLayerInfo(self, info, validateRead=None):
		"""
		``validateRead`` will validate the data, by default it is set to the
		class's ``validateRead`` value, can be overridden.
		"""
		if validateRead is None:
			validateRead = self._validateRead
		infoDict = self._getPlist(LAYERINFO_FILENAME, {})
		if validateRead:
			if not isinstance(infoDict, dict):
				raise GlifLibError("layerinfo.plist is not properly formatted.")
			infoDict = validateLayerInfoVersion3Data(infoDict)
		# populate the object
		for attr, value in infoDict.items():
			try:
				setattr(info, attr, value)
			except AttributeError:
				raise GlifLibError("The supplied layer info object does not support setting a necessary attribute (%s)." % attr)

	def writeLayerInfo(self, info, validateWrite=None):
		"""
		``validateWrite`` will validate the data, by default it is set to the
		class's ``validateWrite`` value, can be overridden.
		"""
		if validateWrite is None:
			validateWrite = self._validateWrite
		if self.ufoFormatVersionTuple.major < 3:
			raise GlifLibError(
				"layerinfo.plist is not allowed in UFO %d." % self.ufoFormatVersionTuple.major
			)
		# gather data
		infoData = {}
		for attr in layerInfoVersion3ValueData.keys():
			if hasattr(info, attr):
				try:
					value = getattr(info, attr)
				except AttributeError:
					raise GlifLibError("The supplied info object does not support getting a necessary attribute (%s)." % attr)
				if value is None or (attr == 'lib' and not value):
					continue
				infoData[attr] = value
		if infoData:
			# validate
			if validateWrite:
				infoData = validateLayerInfoVersion3Data(infoData)
			# write file
			self._writePlist(LAYERINFO_FILENAME, infoData)
		elif self._havePreviousFile and self.fs.exists(LAYERINFO_FILENAME):
			# data empty, remove existing file
			self.fs.remove(LAYERINFO_FILENAME)

	def getGLIF(self, glyphName):
		"""
		Get the raw GLIF text for a given glyph name. This only works
		for GLIF files that are already on disk.

		This method is useful in situations when the raw XML needs to be
		read from a glyph set for a particular glyph before fully parsing
		it into an object structure via the readGlyph method.

		Raises KeyError if 'glyphName' is not in contents.plist, or
		GlifLibError if the file associated with can't be found.
		"""
		fileName = self.contents[glyphName]
		try:
			return self.fs.readbytes(fileName)
		except fs.errors.ResourceNotFound:
			raise GlifLibError(
				"The file '%s' associated with glyph '%s' in contents.plist "
				"does not exist on %s" % (fileName, glyphName, self.fs)
			)

	def getGLIFModificationTime(self, glyphName):
		"""
		Returns the modification time for the GLIF file with 'glyphName', as
		a floating point number giving the number of seconds since the epoch.
		Return None if the associated file does not exist or the underlying
		filesystem does not support getting modified times.
		Raises KeyError if the glyphName is not in contents.plist.
		"""
		fileName = self.contents[glyphName]
		return self.getFileModificationTime(fileName)

	# reading/writing API

	def readGlyph(self, glyphName, glyphObject=None, pointPen=None, validate=None):
		"""
		Read a .glif file for 'glyphName' from the glyph set. The
		'glyphObject' argument can be any kind of object (even None);
		the readGlyph() method will attempt to set the following
		attributes on it:

		width
			the advance width of the glyph
		height
			the advance height of the glyph
		unicodes
			a list of unicode values for this glyph
		note
			a string
		lib
			a dictionary containing custom data
		image
			a dictionary containing image data
		guidelines
			a list of guideline data dictionaries
		anchors
			a list of anchor data dictionaries

		All attributes are optional, in two ways:

		1) An attribute *won't* be set if the .glif file doesn't
		   contain data for it. 'glyphObject' will have to deal
		   with default values itself.
		2) If setting the attribute fails with an AttributeError
		   (for example if the 'glyphObject' attribute is read-
		   only), readGlyph() will not propagate that exception,
		   but ignore that attribute.

		To retrieve outline information, you need to pass an object
		conforming to the PointPen protocol as the 'pointPen' argument.
		This argument may be None if you don't need the outline data.

		readGlyph() will raise KeyError if the glyph is not present in
		the glyph set.

		``validate`` will validate the data, by default it is set to the
		class's ``validateRead`` value, can be overridden.
		"""
		if validate is None:
			validate = self._validateRead
		text = self.getGLIF(glyphName)
		tree = _glifTreeFromString(text)
		formatVersions = GLIFFormatVersion.supported_versions(self.ufoFormatVersionTuple)
		_readGlyphFromTree(tree, glyphObject, pointPen, formatVersions=formatVersions, validate=validate)

	def writeGlyph(self, glyphName, glyphObject=None, drawPointsFunc=None, formatVersion=None, validate=None):
		"""
		Write a .glif file for 'glyphName' to the glyph set. The
		'glyphObject' argument can be any kind of object (even None);
		the writeGlyph() method will attempt to get the following
		attributes from it:

		width
			the advance width of the glyph
		height
			the advance height of the glyph
		unicodes
			a list of unicode values for this glyph
		note
			a string
		lib
			a dictionary containing custom data
		image
			a dictionary containing image data
		guidelines
			a list of guideline data dictionaries
		anchors
			a list of anchor data dictionaries

		All attributes are optional: if 'glyphObject' doesn't
		have the attribute, it will simply be skipped.

		To write outline data to the .glif file, writeGlyph() needs
		a function (any callable object actually) that will take one
		argument: an object that conforms to the PointPen protocol.
		The function will be called by writeGlyph(); it has to call the
		proper PointPen methods to transfer the outline to the .glif file.

		The GLIF format version will be chosen based on the ufoFormatVersion
		passed during the creation of this object. If a particular format
		version is desired, it can be passed with the formatVersion argument.
		The formatVersion argument accepts either a tuple of integers for
		(major, minor), or a single integer for the major digit only (with
		minor digit implied as 0).

		An UnsupportedGLIFFormat exception is raised if the requested GLIF
		formatVersion is not supported.

		``validate`` will validate the data, by default it is set to the
		class's ``validateWrite`` value, can be overridden.
		"""
		if formatVersion is None:
			formatVersion = GLIFFormatVersion.default(self.ufoFormatVersionTuple)
		else:
			try:
				formatVersion = GLIFFormatVersion(formatVersion)
			except ValueError as e:
				from fontTools.ufoLib.errors import UnsupportedGLIFFormat

				raise UnsupportedGLIFFormat(
					f"Unsupported GLIF format version: {formatVersion!r}"
				) from e
		if formatVersion not in GLIFFormatVersion.supported_versions(
			self.ufoFormatVersionTuple
		):
			from fontTools.ufoLib.errors import UnsupportedGLIFFormat

			raise UnsupportedGLIFFormat(
				f"Unsupported GLIF format version ({formatVersion!s}) "
				f"for UFO format version {self.ufoFormatVersionTuple!s}."
			)
		if validate is None:
			validate = self._validateWrite
		fileName = self.contents.get(glyphName)
		if fileName is None:
			if self._existingFileNames is None:
				self._existingFileNames = {
					fileName.lower() for fileName in self.contents.values()
				}
			fileName = self.glyphNameToFileName(glyphName, self._existingFileNames)
			self.contents[glyphName] = fileName
			self._existingFileNames.add(fileName.lower())
			if self._reverseContents is not None:
				self._reverseContents[fileName.lower()] = glyphName
		data = _writeGlyphToBytes(
			glyphName,
			glyphObject,
			drawPointsFunc,
			formatVersion=formatVersion,
			validate=validate,
		)
		if (
			self._havePreviousFile
			and self.fs.exists(fileName)
			and data == self.fs.readbytes(fileName)
		):
			return
		self.fs.writebytes(fileName, data)

	def deleteGlyph(self, glyphName):
		"""Permanently delete the glyph from the glyph set on disk. Will
		raise KeyError if the glyph is not present in the glyph set.
		"""
		fileName = self.contents[glyphName]
		self.fs.remove(fileName)
		if self._existingFileNames is not None:
			self._existingFileNames.remove(fileName.lower())
		if self._reverseContents is not None:
			del self._reverseContents[fileName.lower()]
		del self.contents[glyphName]

	# dict-like support

	def keys(self):
		return list(self.contents.keys())

	def has_key(self, glyphName):
		return glyphName in self.contents

	__contains__ = has_key

	def __len__(self):
		return len(self.contents)

	def __getitem__(self, glyphName):
		if glyphName not in self.contents:
			raise KeyError(glyphName)
		return self.glyphClass(glyphName, self)

	# quickly fetch unicode values

	def getUnicodes(self, glyphNames=None):
		"""
		Return a dictionary that maps glyph names to lists containing
		the unicode value[s] for that glyph, if any. This parses the .glif
		files partially, so it is a lot faster than parsing all files completely.
		By default this checks all glyphs, but a subset can be passed with glyphNames.
		"""
		unicodes = {}
		if glyphNames is None:
			glyphNames = self.contents.keys()
		for glyphName in glyphNames:
			text = self.getGLIF(glyphName)
			unicodes[glyphName] = _fetchUnicodes(text)
		return unicodes

	def getComponentReferences(self, glyphNames=None):
		"""
		Return a dictionary that maps glyph names to lists containing the
		base glyph name of components in the glyph. This parses the .glif
		files partially, so it is a lot faster than parsing all files completely.
		By default this checks all glyphs, but a subset can be passed with glyphNames.
		"""
		components = {}
		if glyphNames is None:
			glyphNames = self.contents.keys()
		for glyphName in glyphNames:
			text = self.getGLIF(glyphName)
			components[glyphName] = _fetchComponentBases(text)
		return components

	def getImageReferences(self, glyphNames=None):
		"""
		Return a dictionary that maps glyph names to the file name of the image
		referenced by the glyph. This parses the .glif files partially, so it is a
		lot faster than parsing all files completely.
		By default this checks all glyphs, but a subset can be passed with glyphNames.
		"""
		images = {}
		if glyphNames is None:
			glyphNames = self.contents.keys()
		for glyphName in glyphNames:
			text = self.getGLIF(glyphName)
			images[glyphName] = _fetchImageFileName(text)
		return images

	def close(self):
		if self._shouldClose:
			self.fs.close()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, exc_tb):
		self.close()


# -----------------------
# Glyph Name to File Name
# -----------------------

def glyphNameToFileName(glyphName, existingFileNames):
	"""
	Wrapper around the userNameToFileName function in filenames.py

	Note that existingFileNames should be a set for large glyphsets
	or performance will suffer.
	"""
	if existingFileNames is None:
		existingFileNames = set()
	return userNameToFileName(glyphName, existing=existingFileNames, suffix=".glif")

# -----------------------
# GLIF To and From String
# -----------------------

def readGlyphFromString(
	aString,
	glyphObject=None,
	pointPen=None,
	formatVersions=None,
	validate=True,
):
	"""
	Read .glif data from a string into a glyph object.

	The 'glyphObject' argument can be any kind of object (even None);
	the readGlyphFromString() method will attempt to set the following
	attributes on it:

	width
		the advance width of the glyph
	height
		the advance height of the glyph
	unicodes
		a list of unicode values for this glyph
	note
		a string
	lib
		a dictionary containing custom data
	image
		a dictionary containing image data
	guidelines
		a list of guideline data dictionaries
	anchors
		a list of anchor data dictionaries

	All attributes are optional, in two ways:

	1) An attribute *won't* be set if the .glif file doesn't
	   contain data for it. 'glyphObject' will have to deal
	   with default values itself.
	2) If setting the attribute fails with an AttributeError
	   (for example if the 'glyphObject' attribute is read-
	   only), readGlyphFromString() will not propagate that
	   exception, but ignore that attribute.

	To retrieve outline information, you need to pass an object
	conforming to the PointPen protocol as the 'pointPen' argument.
	This argument may be None if you don't need the outline data.

	The formatVersions optional argument define the GLIF format versions
	that are allowed to be read.
	The type is Optional[Iterable[Tuple[int, int], int]]. It can contain
	either integers (for the major versions to be allowed, with minor
	digits defaulting to 0), or tuples of integers to specify both
	(major, minor) versions.
	By default when formatVersions is None all the GLIF format versions
	currently defined are allowed to be read.

	``validate`` will validate the read data. It is set to ``True`` by default.
	"""
	tree = _glifTreeFromString(aString)

	if formatVersions is None:
		validFormatVersions = GLIFFormatVersion.supported_versions()
	else:
		validFormatVersions, invalidFormatVersions = set(), set()
		for v in formatVersions:
			try:
				formatVersion = GLIFFormatVersion(v)
			except ValueError:
				invalidFormatVersions.add(v)
			else:
				validFormatVersions.add(formatVersion)
		if not validFormatVersions:
			raise ValueError(
				"None of the requested GLIF formatVersions are supported: "
				f"{formatVersions!r}"
			)

	_readGlyphFromTree(
		tree, glyphObject, pointPen, formatVersions=validFormatVersions, validate=validate
	)


def _writeGlyphToBytes(
		glyphName,
		glyphObject=None,
		drawPointsFunc=None,
		writer=None,
		formatVersion=None,
		validate=True,
):
	"""Return .glif data for a glyph as a UTF-8 encoded bytes string."""
	try:
		formatVersion = GLIFFormatVersion(formatVersion)
	except ValueError:
		from fontTools.ufoLib.errors import UnsupportedGLIFFormat

		raise UnsupportedGLIFFormat("Unsupported GLIF format version: {formatVersion!r}")
	# start
	if validate and not isinstance(glyphName, str):
		raise GlifLibError("The glyph name is not properly formatted.")
	if validate and len(glyphName) == 0:
		raise GlifLibError("The glyph name is empty.")
	glyphAttrs = OrderedDict([("name", glyphName), ("format", repr(formatVersion.major))])
	if formatVersion.minor != 0:
		glyphAttrs["formatMinor"] = repr(formatVersion.minor)
	root = etree.Element("glyph", glyphAttrs)
	identifiers = set()
	# advance
	_writeAdvance(glyphObject, root, validate)
	# unicodes
	if getattr(glyphObject, "unicodes", None):
		_writeUnicodes(glyphObject, root, validate)
	# note
	if getattr(glyphObject, "note", None):
		_writeNote(glyphObject, root, validate)
	# image
	if formatVersion.major >= 2 and getattr(glyphObject, "image", None):
		_writeImage(glyphObject, root, validate)
	# guidelines
	if formatVersion.major >= 2 and getattr(glyphObject, "guidelines", None):
		_writeGuidelines(glyphObject, root, identifiers, validate)
	# anchors
	anchors = getattr(glyphObject, "anchors", None)
	if formatVersion.major >= 2 and anchors:
		_writeAnchors(glyphObject, root, identifiers, validate)
	# outline
	if drawPointsFunc is not None:
		outline = etree.SubElement(root, "outline")
		pen = GLIFPointPen(outline, identifiers=identifiers, validate=validate)
		drawPointsFunc(pen)
		if formatVersion.major == 1 and anchors:
			_writeAnchorsFormat1(pen, anchors, validate)
		# prevent lxml from writing self-closing tags
		if not len(outline):
			outline.text = "\n  "
	# lib
	if getattr(glyphObject, "lib", None):
		_writeLib(glyphObject, root, validate)
	# return the text
	data = etree.tostring(
		root, encoding="UTF-8", xml_declaration=True, pretty_print=True
	)
	return data


def writeGlyphToString(
	glyphName,
	glyphObject=None,
	drawPointsFunc=None,
	formatVersion=None,
	validate=True,
):
	"""
	Return .glif data for a glyph as a string. The XML declaration's
	encoding is always set to "UTF-8".
	The 'glyphObject' argument can be any kind of object (even None);
	the writeGlyphToString() method will attempt to get the following
	attributes from it:

	width
		the advance width of the glyph
	height
		the advance height of the glyph
	unicodes
		a list of unicode values for this glyph
	note
		a string
	lib
		a dictionary containing custom data
	image
		a dictionary containing image data
	guidelines
		a list of guideline data dictionaries
	anchors
		a list of anchor data dictionaries

	All attributes are optional: if 'glyphObject' doesn't
	have the attribute, it will simply be skipped.

	To write outline data to the .glif file, writeGlyphToString() needs
	a function (any callable object actually) that will take one
	argument: an object that conforms to the PointPen protocol.
	The function will be called by writeGlyphToString(); it has to call the
	proper PointPen methods to transfer the outline to the .glif file.

	The GLIF format version can be specified with the formatVersion argument.
	This accepts either a tuple of integers for (major, minor), or a single
	integer for the major digit only (with minor digit implied as 0).
	By default when formatVesion is None the latest GLIF format version will
	be used; currently it's 2.0, which is equivalent to formatVersion=(2, 0).

	An UnsupportedGLIFFormat exception is raised if the requested UFO
	formatVersion is not supported.

	``validate`` will validate the written data. It is set to ``True`` by default.
	"""
	data = _writeGlyphToBytes(
		glyphName,
		glyphObject=glyphObject,
		drawPointsFunc=drawPointsFunc,
		formatVersion=formatVersion,
		validate=validate,
	)
	return data.decode("utf-8")


def _writeAdvance(glyphObject, element, validate):
	width = getattr(glyphObject, "width", None)
	if width is not None:
		if validate and not isinstance(width, numberTypes):
			raise GlifLibError("width attribute must be int or float")
		if width == 0:
			width = None
	height = getattr(glyphObject, "height", None)
	if height is not None:
		if validate and not isinstance(height, numberTypes):
			raise GlifLibError("height attribute must be int or float")
		if height == 0:
			height = None
	if width is not None and height is not None:
		etree.SubElement(element, "advance", OrderedDict([("height", repr(height)), ("width", repr(width))]))
	elif width is not None:
		etree.SubElement(element, "advance", dict(width=repr(width)))
	elif height is not None:
		etree.SubElement(element, "advance", dict(height=repr(height)))

def _writeUnicodes(glyphObject, element, validate):
	unicodes = getattr(glyphObject, "unicodes", None)
	if validate and isinstance(unicodes, int):
		unicodes = [unicodes]
	seen = set()
	for code in unicodes:
		if validate and not isinstance(code, int):
			raise GlifLibError("unicode values must be int")
		if code in seen:
			continue
		seen.add(code)
		hexCode = "%04X" % code
		etree.SubElement(element, "unicode", dict(hex=hexCode))

def _writeNote(glyphObject, element, validate):
	note = getattr(glyphObject, "note", None)
	if validate and not isinstance(note, str):
		raise GlifLibError("note attribute must be str")
	note = note.strip()
	note = "\n" + note + "\n"
	etree.SubElement(element, "note").text = note

def _writeImage(glyphObject, element, validate):
	image = getattr(glyphObject, "image", None)
	if validate and not imageValidator(image):
		raise GlifLibError("image attribute must be a dict or dict-like object with the proper structure.")
	attrs = OrderedDict([("fileName", image["fileName"])])
	for attr, default in _transformationInfo:
		value = image.get(attr, default)
		if value != default:
			attrs[attr] = repr(value)
	color = image.get("color")
	if color is not None:
		attrs["color"] = color
	etree.SubElement(element, "image", attrs)

def _writeGuidelines(glyphObject, element, identifiers, validate):
	guidelines = getattr(glyphObject, "guidelines", [])
	if validate and not guidelinesValidator(guidelines):
		raise GlifLibError("guidelines attribute does not have the proper structure.")
	for guideline in guidelines:
		attrs = OrderedDict()
		x = guideline.get("x")
		if x is not None:
			attrs["x"] = repr(x)
		y = guideline.get("y")
		if y is not None:
			attrs["y"] = repr(y)
		angle = guideline.get("angle")
		if angle is not None:
			attrs["angle"] = repr(angle)
		name = guideline.get("name")
		if name is not None:
			attrs["name"] = name
		color = guideline.get("color")
		if color is not None:
			attrs["color"] = color
		identifier = guideline.get("identifier")
		if identifier is not None:
			if validate and identifier in identifiers:
				raise GlifLibError("identifier used more than once: %s" % identifier)
			attrs["identifier"] = identifier
			identifiers.add(identifier)
		etree.SubElement(element, "guideline", attrs)

def _writeAnchorsFormat1(pen, anchors, validate):
	if validate and not anchorsValidator(anchors):
		raise GlifLibError("anchors attribute does not have the proper structure.")
	for anchor in anchors:
		attrs = {}
		x = anchor["x"]
		attrs["x"] = repr(x)
		y = anchor["y"]
		attrs["y"] = repr(y)
		name = anchor.get("name")
		if name is not None:
			attrs["name"] = name
		pen.beginPath()
		pen.addPoint((x, y), segmentType="move", name=name)
		pen.endPath()

def _writeAnchors(glyphObject, element, identifiers, validate):
	anchors = getattr(glyphObject, "anchors", [])
	if validate and not anchorsValidator(anchors):
		raise GlifLibError("anchors attribute does not have the proper structure.")
	for anchor in anchors:
		attrs = OrderedDict()
		x = anchor["x"]
		attrs["x"] = repr(x)
		y = anchor["y"]
		attrs["y"] = repr(y)
		name = anchor.get("name")
		if name is not None:
			attrs["name"] = name
		color = anchor.get("color")
		if color is not None:
			attrs["color"] = color
		identifier = anchor.get("identifier")
		if identifier is not None:
			if validate and identifier in identifiers:
				raise GlifLibError("identifier used more than once: %s" % identifier)
			attrs["identifier"] = identifier
			identifiers.add(identifier)
		etree.SubElement(element, "anchor", attrs)

def _writeLib(glyphObject, element, validate):
	lib = getattr(glyphObject, "lib", None)
	if not lib:
		# don't write empty lib
		return
	if validate:
		valid, message = glyphLibValidator(lib)
		if not valid:
			raise GlifLibError(message)
	if not isinstance(lib, dict):
		lib = dict(lib)
	# plist inside GLIF begins with 2 levels of indentation
	e = plistlib.totree(lib, indent_level=2)
	etree.SubElement(element, "lib").append(e)

# -----------------------
# layerinfo.plist Support
# -----------------------

layerInfoVersion3ValueData = {
	"color"			: dict(type=str, valueValidator=colorValidator),
	"lib"			: dict(type=dict, valueValidator=genericTypeValidator)
}

def validateLayerInfoVersion3ValueForAttribute(attr, value):
	"""
	This performs very basic validation of the value for attribute
	following the UFO 3 fontinfo.plist specification. The results
	of this should not be interpretted as *correct* for the font
	that they are part of. This merely indicates that the value
	is of the proper type and, where the specification defines
	a set range of possible values for an attribute, that the
	value is in the accepted range.
	"""
	if attr not in layerInfoVersion3ValueData:
		return False
	dataValidationDict = layerInfoVersion3ValueData[attr]
	valueType = dataValidationDict.get("type")
	validator = dataValidationDict.get("valueValidator")
	valueOptions = dataValidationDict.get("valueOptions")
	# have specific options for the validator
	if valueOptions is not None:
		isValidValue = validator(value, valueOptions)
	# no specific options
	else:
		if validator == genericTypeValidator:
			isValidValue = validator(value, valueType)
		else:
			isValidValue = validator(value)
	return isValidValue

def validateLayerInfoVersion3Data(infoData):
	"""
	This performs very basic validation of the value for infoData
	following the UFO 3 layerinfo.plist specification. The results
	of this should not be interpretted as *correct* for the font
	that they are part of. This merely indicates that the values
	are of the proper type and, where the specification defines
	a set range of possible values for an attribute, that the
	value is in the accepted range.
	"""
	for attr, value in infoData.items():
		if attr not in layerInfoVersion3ValueData:
			raise GlifLibError("Unknown attribute %s." % attr)
		isValidValue = validateLayerInfoVersion3ValueForAttribute(attr, value)
		if not isValidValue:
			raise GlifLibError(f"Invalid value for attribute {attr} ({value!r}).")
	return infoData

# -----------------
# GLIF Tree Support
# -----------------

def _glifTreeFromFile(aFile):
	if etree._have_lxml:
		tree = etree.parse(aFile, parser=etree.XMLParser(remove_comments=True))
	else:
		tree = etree.parse(aFile)
	root = tree.getroot()
	if root.tag != "glyph":
		raise GlifLibError("The GLIF is not properly formatted.")
	if root.text and root.text.strip() != '':
		raise GlifLibError("Invalid GLIF structure.")
	return root


def _glifTreeFromString(aString):
	data = tobytes(aString, encoding="utf-8")
	if etree._have_lxml:
		root = etree.fromstring(data, parser=etree.XMLParser(remove_comments=True))
	else:
		root = etree.fromstring(data)
	if root.tag != "glyph":
		raise GlifLibError("The GLIF is not properly formatted.")
	if root.text and root.text.strip() != '':
		raise GlifLibError("Invalid GLIF structure.")
	return root


def _readGlyphFromTree(
	tree,
	glyphObject=None,
	pointPen=None,
	formatVersions=GLIFFormatVersion.supported_versions(),
	validate=True,
):
	# check the format version
	formatVersionMajor = tree.get("format")
	if validate and formatVersionMajor is None:
		raise GlifLibError("Unspecified format version in GLIF.")
	formatVersionMinor = tree.get("formatMinor", 0)
	try:
		formatVersion = GLIFFormatVersion((int(formatVersionMajor), int(formatVersionMinor)))
	except ValueError as e:
		msg = "Unsupported GLIF format: %s.%s" % (formatVersionMajor, formatVersionMinor)
		if validate:
			from fontTools.ufoLib.errors import UnsupportedGLIFFormat

			raise UnsupportedGLIFFormat(msg) from e
		# warn but continue using the latest supported format
		formatVersion = GLIFFormatVersion.default()
		logger.warning(
			"%s. Assuming the latest supported version (%s). "
			"Some data may be skipped or parsed incorrectly.",
			msg,
			formatVersion,
		)

	if validate and formatVersion not in formatVersions:
		raise GlifLibError(f"Forbidden GLIF format version: {formatVersion!s}")

	try:
		readGlyphFromTree = _READ_GLYPH_FROM_TREE_FUNCS[formatVersion]
	except KeyError:
		raise NotImplementedError(formatVersion)

	readGlyphFromTree(
		tree=tree,
		glyphObject=glyphObject,
		pointPen=pointPen,
		validate=validate,
		formatMinor=formatVersion.minor,
	)


def _readGlyphFromTreeFormat1(tree, glyphObject=None, pointPen=None, validate=None, **kwargs):
	# get the name
	_readName(glyphObject, tree, validate)
	# populate the sub elements
	unicodes = []
	haveSeenAdvance = haveSeenOutline = haveSeenLib = haveSeenNote = False
	for element in tree:
		if element.tag == "outline":
			if validate:
				if haveSeenOutline:
					raise GlifLibError("The outline element occurs more than once.")
				if element.attrib:
					raise GlifLibError("The outline element contains unknown attributes.")
				if element.text and element.text.strip() != '':
					raise GlifLibError("Invalid outline structure.")
			haveSeenOutline = True
			buildOutlineFormat1(glyphObject, pointPen, element, validate)
		elif glyphObject is None:
			continue
		elif element.tag == "advance":
			if validate and haveSeenAdvance:
				raise GlifLibError("The advance element occurs more than once.")
			haveSeenAdvance = True
			_readAdvance(glyphObject, element)
		elif element.tag == "unicode":
			try:
				v = element.get("hex")
				v = int(v, 16)
				if v not in unicodes:
					unicodes.append(v)
			except ValueError:
				raise GlifLibError("Illegal value for hex attribute of unicode element.")
		elif element.tag == "note":
			if validate and haveSeenNote:
				raise GlifLibError("The note element occurs more than once.")
			haveSeenNote = True
			_readNote(glyphObject, element)
		elif element.tag == "lib":
			if validate and haveSeenLib:
				raise GlifLibError("The lib element occurs more than once.")
			haveSeenLib = True
			_readLib(glyphObject, element, validate)
		else:
			raise GlifLibError("Unknown element in GLIF: %s" % element)
	# set the collected unicodes
	if unicodes:
		_relaxedSetattr(glyphObject, "unicodes", unicodes)

def _readGlyphFromTreeFormat2(
	tree, glyphObject=None, pointPen=None, validate=None, formatMinor=0
):
	# get the name
	_readName(glyphObject, tree, validate)
	# populate the sub elements
	unicodes = []
	guidelines = []
	anchors = []
	haveSeenAdvance = haveSeenImage = haveSeenOutline = haveSeenLib = haveSeenNote = False
	identifiers = set()
	for element in tree:
		if element.tag == "outline":
			if validate:
				if haveSeenOutline:
					raise GlifLibError("The outline element occurs more than once.")
				if element.attrib:
					raise GlifLibError("The outline element contains unknown attributes.")
				if element.text and element.text.strip() != '':
					raise GlifLibError("Invalid outline structure.")
			haveSeenOutline = True
			if pointPen is not None:
				buildOutlineFormat2(glyphObject, pointPen, element, identifiers, validate)
		elif glyphObject is None:
			continue
		elif element.tag == "advance":
			if validate and haveSeenAdvance:
				raise GlifLibError("The advance element occurs more than once.")
			haveSeenAdvance = True
			_readAdvance(glyphObject, element)
		elif element.tag == "unicode":
			try:
				v = element.get("hex")
				v = int(v, 16)
				if v not in unicodes:
					unicodes.append(v)
			except ValueError:
				raise GlifLibError("Illegal value for hex attribute of unicode element.")
		elif element.tag == "guideline":
			if validate and len(element):
				raise GlifLibError("Unknown children in guideline element.")
			attrib = dict(element.attrib)
			for attr in ("x", "y", "angle"):
				if attr in attrib:
					attrib[attr] = _number(attrib[attr])
			guidelines.append(attrib)
		elif element.tag == "anchor":
			if validate and len(element):
				raise GlifLibError("Unknown children in anchor element.")
			attrib = dict(element.attrib)
			for attr in ("x", "y"):
				if attr in element.attrib:
					attrib[attr] = _number(attrib[attr])
			anchors.append(attrib)
		elif element.tag == "image":
			if validate:
				if haveSeenImage:
					raise GlifLibError("The image element occurs more than once.")
				if len(element):
					raise GlifLibError("Unknown children in image element.")
			haveSeenImage = True
			_readImage(glyphObject, element, validate)
		elif element.tag == "note":
			if validate and haveSeenNote:
				raise GlifLibError("The note element occurs more than once.")
			haveSeenNote = True
			_readNote(glyphObject, element)
		elif element.tag == "lib":
			if validate and haveSeenLib:
				raise GlifLibError("The lib element occurs more than once.")
			haveSeenLib = True
			_readLib(glyphObject, element, validate)
		else:
			raise GlifLibError("Unknown element in GLIF: %s" % element)
	# set the collected unicodes
	if unicodes:
		_relaxedSetattr(glyphObject, "unicodes", unicodes)
	# set the collected guidelines
	if guidelines:
		if validate and not guidelinesValidator(guidelines, identifiers):
			raise GlifLibError("The guidelines are improperly formatted.")
		_relaxedSetattr(glyphObject, "guidelines", guidelines)
	# set the collected anchors
	if anchors:
		if validate and not anchorsValidator(anchors, identifiers):
			raise GlifLibError("The anchors are improperly formatted.")
		_relaxedSetattr(glyphObject, "anchors", anchors)


_READ_GLYPH_FROM_TREE_FUNCS = {
	GLIFFormatVersion.FORMAT_1_0: _readGlyphFromTreeFormat1,
	GLIFFormatVersion.FORMAT_2_0: _readGlyphFromTreeFormat2,
}


def _readName(glyphObject, root, validate):
	glyphName = root.get("name")
	if validate and not glyphName:
		raise GlifLibError("Empty glyph name in GLIF.")
	if glyphName and glyphObject is not None:
		_relaxedSetattr(glyphObject, "name", glyphName)

def _readAdvance(glyphObject, advance):
	width = _number(advance.get("width", 0))
	_relaxedSetattr(glyphObject, "width", width)
	height = _number(advance.get("height", 0))
	_relaxedSetattr(glyphObject, "height", height)

def _readNote(glyphObject, note):
	lines = note.text.split("\n")
	note = "\n".join(line.strip() for line in lines if line.strip())
	_relaxedSetattr(glyphObject, "note", note)

def _readLib(glyphObject, lib, validate):
	assert len(lib) == 1
	child = lib[0]
	plist = plistlib.fromtree(child)
	if validate:
		valid, message = glyphLibValidator(plist)
		if not valid:
			raise GlifLibError(message)
	_relaxedSetattr(glyphObject, "lib", plist)

def _readImage(glyphObject, image, validate):
	imageData = dict(image.attrib)
	for attr, default in _transformationInfo:
		value = imageData.get(attr, default)
		imageData[attr] = _number(value)
	if validate and not imageValidator(imageData):
		raise GlifLibError("The image element is not properly formatted.")
	_relaxedSetattr(glyphObject, "image", imageData)

# ----------------
# GLIF to PointPen
# ----------------

contourAttributesFormat2 = {"identifier"}
componentAttributesFormat1 = {"base", "xScale", "xyScale", "yxScale", "yScale", "xOffset", "yOffset"}
componentAttributesFormat2 = componentAttributesFormat1 | {"identifier"}
pointAttributesFormat1 = {"x", "y", "type", "smooth", "name"}
pointAttributesFormat2 = pointAttributesFormat1 | {"identifier"}
pointSmoothOptions = {"no", "yes"}
pointTypeOptions = {"move", "line", "offcurve", "curve", "qcurve"}

# format 1

def buildOutlineFormat1(glyphObject, pen, outline, validate):
	anchors = []
	for element in outline:
		if element.tag == "contour":
			if len(element) == 1:
				point = element[0]
				if point.tag == "point":
					anchor = _buildAnchorFormat1(point, validate)
					if anchor is not None:
						anchors.append(anchor)
						continue
			if pen is not None:
				_buildOutlineContourFormat1(pen, element, validate)
		elif element.tag == "component":
			if pen is not None:
				_buildOutlineComponentFormat1(pen, element, validate)
		else:
			raise GlifLibError("Unknown element in outline element: %s" % element)
	if glyphObject is not None and anchors:
		if validate and not anchorsValidator(anchors):
			raise GlifLibError("GLIF 1 anchors are not properly formatted.")
		_relaxedSetattr(glyphObject, "anchors", anchors)

def _buildAnchorFormat1(point, validate):
	if point.get("type") != "move":
		return None
	name = point.get("name")
	if name is None:
		return None
	x = point.get("x")
	y = point.get("y")
	if validate and x is None:
		raise GlifLibError("Required x attribute is missing in point element.")
	if validate and y is None:
		raise GlifLibError("Required y attribute is missing in point element.")
	x = _number(x)
	y = _number(y)
	anchor = dict(x=x, y=y, name=name)
	return anchor

def _buildOutlineContourFormat1(pen, contour, validate):
	if validate and contour.attrib:
		raise GlifLibError("Unknown attributes in contour element.")
	pen.beginPath()
	if len(contour):
		massaged = _validateAndMassagePointStructures(contour, pointAttributesFormat1, openContourOffCurveLeniency=True, validate=validate)
		_buildOutlinePointsFormat1(pen, massaged)
	pen.endPath()

def _buildOutlinePointsFormat1(pen, contour):
	for point in contour:
		x = point["x"]
		y = point["y"]
		segmentType = point["segmentType"]
		smooth = point["smooth"]
		name = point["name"]
		pen.addPoint((x, y), segmentType=segmentType, smooth=smooth, name=name)

def _buildOutlineComponentFormat1(pen, component, validate):
	if validate:
		if len(component):
			raise GlifLibError("Unknown child elements of component element.")
		for attr in component.attrib.keys():
			if attr not in componentAttributesFormat1:
				raise GlifLibError("Unknown attribute in component element: %s" % attr)
	baseGlyphName = component.get("base")
	if validate and baseGlyphName is None:
		raise GlifLibError("The base attribute is not defined in the component.")
	transformation = []
	for attr, default in _transformationInfo:
		value = component.get(attr)
		if value is None:
			value = default
		else:
			value = _number(value)
		transformation.append(value)
	pen.addComponent(baseGlyphName, tuple(transformation))

# format 2

def buildOutlineFormat2(glyphObject, pen, outline, identifiers, validate):
	for element in outline:
		if element.tag == "contour":
			_buildOutlineContourFormat2(pen, element, identifiers, validate)
		elif element.tag == "component":
			_buildOutlineComponentFormat2(pen, element, identifiers, validate)
		else:
			raise GlifLibError("Unknown element in outline element: %s" % element.tag)

def _buildOutlineContourFormat2(pen, contour, identifiers, validate):
	if validate:
		for attr in contour.attrib.keys():
			if attr not in contourAttributesFormat2:
				raise GlifLibError("Unknown attribute in contour element: %s" % attr)
	identifier = contour.get("identifier")
	if identifier is not None:
		if validate:
			if identifier in identifiers:
				raise GlifLibError("The identifier %s is used more than once." % identifier)
			if not identifierValidator(identifier):
				raise GlifLibError("The contour identifier %s is not valid." % identifier)
		identifiers.add(identifier)
	try:
		pen.beginPath(identifier=identifier)
	except TypeError:
		pen.beginPath()
		warn("The beginPath method needs an identifier kwarg. The contour's identifier value has been discarded.", DeprecationWarning)
	if len(contour):
		massaged = _validateAndMassagePointStructures(contour, pointAttributesFormat2, validate=validate)
		_buildOutlinePointsFormat2(pen, massaged, identifiers, validate)
	pen.endPath()

def _buildOutlinePointsFormat2(pen, contour, identifiers, validate):
	for point in contour:
		x = point["x"]
		y = point["y"]
		segmentType = point["segmentType"]
		smooth = point["smooth"]
		name = point["name"]
		identifier = point.get("identifier")
		if identifier is not None:
			if validate:
				if identifier in identifiers:
					raise GlifLibError("The identifier %s is used more than once." % identifier)
				if not identifierValidator(identifier):
					raise GlifLibError("The identifier %s is not valid." % identifier)
			identifiers.add(identifier)
		try:
			pen.addPoint((x, y), segmentType=segmentType, smooth=smooth, name=name, identifier=identifier)
		except TypeError:
			pen.addPoint((x, y), segmentType=segmentType, smooth=smooth, name=name)
			warn("The addPoint method needs an identifier kwarg. The point's identifier value has been discarded.", DeprecationWarning)

def _buildOutlineComponentFormat2(pen, component, identifiers, validate):
	if validate:
		if len(component):
			raise GlifLibError("Unknown child elements of component element.")
		for attr in component.attrib.keys():
			if attr not in componentAttributesFormat2:
				raise GlifLibError("Unknown attribute in component element: %s" % attr)
	baseGlyphName = component.get("base")
	if validate and baseGlyphName is None:
		raise GlifLibError("The base attribute is not defined in the component.")
	transformation = []
	for attr, default in _transformationInfo:
		value = component.get(attr)
		if value is None:
			value = default
		else:
			value = _number(value)
		transformation.append(value)
	identifier = component.get("identifier")
	if identifier is not None:
		if validate:
			if identifier in identifiers:
				raise GlifLibError("The identifier %s is used more than once." % identifier)
			if validate and not identifierValidator(identifier):
				raise GlifLibError("The identifier %s is not valid." % identifier)
		identifiers.add(identifier)
	try:
		pen.addComponent(baseGlyphName, tuple(transformation), identifier=identifier)
	except TypeError:
		pen.addComponent(baseGlyphName, tuple(transformation))
		warn("The addComponent method needs an identifier kwarg. The component's identifier value has been discarded.", DeprecationWarning)

# all formats

def _validateAndMassagePointStructures(contour, pointAttributes, openContourOffCurveLeniency=False, validate=True):
	if not len(contour):
		return
	# store some data for later validation
	lastOnCurvePoint = None
	haveOffCurvePoint = False
	# validate and massage the individual point elements
	massaged = []
	for index, element in enumerate(contour):
		# not <point>
		if element.tag != "point":
			raise GlifLibError("Unknown child element (%s) of contour element." % element.tag)
		point = dict(element.attrib)
		massaged.append(point)
		if validate:
			# unknown attributes
			for attr in point.keys():
				if attr not in pointAttributes:
					raise GlifLibError("Unknown attribute in point element: %s" % attr)
			# search for unknown children
			if len(element):
				raise GlifLibError("Unknown child elements in point element.")
		# x and y are required
		for attr in ("x", "y"):
			try:
				point[attr] = _number(point[attr])
			except KeyError as e:
				raise GlifLibError(f"Required {attr} attribute is missing in point element.") from e
		# segment type
		pointType = point.pop("type", "offcurve")
		if validate and pointType not in pointTypeOptions:
			raise GlifLibError("Unknown point type: %s" % pointType)
		if pointType == "offcurve":
			pointType = None
		point["segmentType"] = pointType
		if pointType is None:
			haveOffCurvePoint = True
		else:
			lastOnCurvePoint = index
		# move can only occur as the first point
		if validate and pointType == "move" and index != 0:
			raise GlifLibError("A move point occurs after the first point in the contour.")
		# smooth is optional
		smooth = point.get("smooth", "no")
		if validate and smooth is not None:
			if smooth not in pointSmoothOptions:
				raise GlifLibError("Unknown point smooth value: %s" % smooth)
		smooth = smooth == "yes"
		point["smooth"] = smooth
		# smooth can only be applied to curve and qcurve
		if validate and smooth and pointType is None:
			raise GlifLibError("smooth attribute set in an offcurve point.")
		# name is optional
		if "name" not in element.attrib:
			point["name"] = None
	if openContourOffCurveLeniency:
		# remove offcurves that precede a move. this is technically illegal,
		# but we let it slide because there are fonts out there in the wild like this.
		if massaged[0]["segmentType"] == "move":
			count = 0
			for point in reversed(massaged):
				if point["segmentType"] is None:
					count += 1
				else:
					break
			if count:
				massaged = massaged[:-count]
	# validate the off-curves in the segments
	if validate and haveOffCurvePoint and lastOnCurvePoint is not None:
		# we only care about how many offCurves there are before an onCurve
		# filter out the trailing offCurves
		offCurvesCount = len(massaged) - 1 - lastOnCurvePoint
		for point in massaged:
			segmentType = point["segmentType"]
			if segmentType is None:
				offCurvesCount += 1
			else:
				if offCurvesCount:
					# move and line can't be preceded by off-curves
					if segmentType == "move":
						# this will have been filtered out already
						raise GlifLibError("move can not have an offcurve.")
					elif segmentType == "line":
						raise GlifLibError("line can not have an offcurve.")
					elif segmentType == "curve":
						if offCurvesCount > 2:
							raise GlifLibError("Too many offcurves defined for curve.")
					elif segmentType == "qcurve":
						pass
					else:
						# unknown segment type. it'll be caught later.
						pass
				offCurvesCount = 0
	return massaged

# ---------------------
# Misc Helper Functions
# ---------------------

def _relaxedSetattr(object, attr, value):
	try:
		setattr(object, attr, value)
	except AttributeError:
		pass

def _number(s):
	"""
	Given a numeric string, return an integer or a float, whichever
	the string indicates. _number("1") will return the integer 1,
	_number("1.0") will return the float 1.0.

	>>> _number("1")
	1
	>>> _number("1.0")
	1.0
	>>> _number("a")  # doctest: +IGNORE_EXCEPTION_DETAIL
	Traceback (most recent call last):
	    ...
	GlifLibError: Could not convert a to an int or float.
	"""
	try:
		n = int(s)
		return n
	except ValueError:
		pass
	try:
		n = float(s)
		return n
	except ValueError:
		raise GlifLibError("Could not convert %s to an int or float." % s)

# --------------------
# Rapid Value Fetching
# --------------------

# base

class _DoneParsing(Exception): pass

class _BaseParser:

	def __init__(self):
		self._elementStack = []

	def parse(self, text):
		from xml.parsers.expat import ParserCreate
		parser = ParserCreate()
		parser.StartElementHandler = self.startElementHandler
		parser.EndElementHandler = self.endElementHandler
		parser.Parse(text)

	def startElementHandler(self, name, attrs):
		self._elementStack.append(name)

	def endElementHandler(self, name):
		other = self._elementStack.pop(-1)
		assert other == name


# unicodes

def _fetchUnicodes(glif):
	"""
	Get a list of unicodes listed in glif.
	"""
	parser = _FetchUnicodesParser()
	parser.parse(glif)
	return parser.unicodes

class _FetchUnicodesParser(_BaseParser):

	def __init__(self):
		self.unicodes = []
		super().__init__()

	def startElementHandler(self, name, attrs):
		if name == "unicode" and self._elementStack and self._elementStack[-1] == "glyph":
			value = attrs.get("hex")
			if value is not None:
				try:
					value = int(value, 16)
					if value not in self.unicodes:
						self.unicodes.append(value)
				except ValueError:
					pass
		super().startElementHandler(name, attrs)

# image

def _fetchImageFileName(glif):
	"""
	The image file name (if any) from glif.
	"""
	parser = _FetchImageFileNameParser()
	try:
		parser.parse(glif)
	except _DoneParsing:
		pass
	return parser.fileName

class _FetchImageFileNameParser(_BaseParser):

	def __init__(self):
		self.fileName = None
		super().__init__()

	def startElementHandler(self, name, attrs):
		if name == "image" and self._elementStack and self._elementStack[-1] == "glyph":
			self.fileName = attrs.get("fileName")
			raise _DoneParsing
		super().startElementHandler(name, attrs)

# component references

def _fetchComponentBases(glif):
	"""
	Get a list of component base glyphs listed in glif.
	"""
	parser = _FetchComponentBasesParser()
	try:
		parser.parse(glif)
	except _DoneParsing:
		pass
	return list(parser.bases)

class _FetchComponentBasesParser(_BaseParser):

	def __init__(self):
		self.bases = []
		super().__init__()

	def startElementHandler(self, name, attrs):
		if name == "component" and self._elementStack and self._elementStack[-1] == "outline":
			base = attrs.get("base")
			if base is not None:
				self.bases.append(base)
		super().startElementHandler(name, attrs)

	def endElementHandler(self, name):
		if name == "outline":
			raise _DoneParsing
		super().endElementHandler(name)

# --------------
# GLIF Point Pen
# --------------

_transformationInfo = [
	# field name, default value
	("xScale",    1),
	("xyScale",   0),
	("yxScale",   0),
	("yScale",    1),
	("xOffset",   0),
	("yOffset",   0),
]

class GLIFPointPen(AbstractPointPen):

	"""
	Helper class using the PointPen protocol to write the <outline>
	part of .glif files.
	"""

	def __init__(self, element, formatVersion=None, identifiers=None, validate=True):
		if identifiers is None:
			identifiers = set()
		self.formatVersion = GLIFFormatVersion(formatVersion)
		self.identifiers = identifiers
		self.outline = element
		self.contour = None
		self.prevOffCurveCount = 0
		self.prevPointTypes = []
		self.validate = validate

	def beginPath(self, identifier=None, **kwargs):
		attrs = OrderedDict()
		if identifier is not None and self.formatVersion.major >= 2:
			if self.validate:
				if identifier in self.identifiers:
					raise GlifLibError("identifier used more than once: %s" % identifier)
				if not identifierValidator(identifier):
					raise GlifLibError("identifier not formatted properly: %s" % identifier)
			attrs["identifier"] = identifier
			self.identifiers.add(identifier)
		self.contour = etree.SubElement(self.outline, "contour", attrs)
		self.prevOffCurveCount = 0

	def endPath(self):
		if self.prevPointTypes and self.prevPointTypes[0] == "move":
			if self.validate and self.prevPointTypes[-1] == "offcurve":
				raise GlifLibError("open contour has loose offcurve point")
		# prevent lxml from writing self-closing tags
		if not len(self.contour):
			self.contour.text = "\n  "
		self.contour = None
		self.prevPointType = None
		self.prevOffCurveCount = 0
		self.prevPointTypes = []

	def addPoint(self, pt, segmentType=None, smooth=None, name=None, identifier=None, **kwargs):
		attrs = OrderedDict()
		# coordinates
		if pt is not None:
			if self.validate:
				for coord in pt:
					if not isinstance(coord, numberTypes):
						raise GlifLibError("coordinates must be int or float")
			attrs["x"] = repr(pt[0])
			attrs["y"] = repr(pt[1])
		# segment type
		if segmentType == "offcurve":
			segmentType = None
		if self.validate:
			if segmentType == "move" and self.prevPointTypes:
				raise GlifLibError("move occurs after a point has already been added to the contour.")
			if segmentType in ("move", "line") and self.prevPointTypes and self.prevPointTypes[-1] == "offcurve":
				raise GlifLibError("offcurve occurs before %s point." % segmentType)
			if segmentType == "curve" and self.prevOffCurveCount > 2:
				raise GlifLibError("too many offcurve points before curve point.")
		if segmentType is not None:
			attrs["type"] = segmentType
		else:
			segmentType = "offcurve"
		if segmentType == "offcurve":
			self.prevOffCurveCount += 1
		else:
			self.prevOffCurveCount = 0
		self.prevPointTypes.append(segmentType)
		# smooth
		if smooth:
			if self.validate and segmentType == "offcurve":
				raise GlifLibError("can't set smooth in an offcurve point.")
			attrs["smooth"] = "yes"
		# name
		if name is not None:
			attrs["name"] = name
		# identifier
		if identifier is not None and self.formatVersion.major >= 2:
			if self.validate:
				if identifier in self.identifiers:
					raise GlifLibError("identifier used more than once: %s" % identifier)
				if not identifierValidator(identifier):
					raise GlifLibError("identifier not formatted properly: %s" % identifier)
			attrs["identifier"] = identifier
			self.identifiers.add(identifier)
		etree.SubElement(self.contour, "point", attrs)

	def addComponent(self, glyphName, transformation, identifier=None, **kwargs):
		attrs = OrderedDict([("base", glyphName)])
		for (attr, default), value in zip(_transformationInfo, transformation):
			if self.validate and not isinstance(value, numberTypes):
				raise GlifLibError("transformation values must be int or float")
			if value != default:
				attrs[attr] = repr(value)
		if identifier is not None and self.formatVersion.major >= 2:
			if self.validate:
				if identifier in self.identifiers:
					raise GlifLibError("identifier used more than once: %s" % identifier)
				if self.validate and not identifierValidator(identifier):
					raise GlifLibError("identifier not formatted properly: %s" % identifier)
			attrs["identifier"] = identifier
			self.identifiers.add(identifier)
		etree.SubElement(self.outline, "component", attrs)

if __name__ == "__main__":
	import doctest
	doctest.testmod()
