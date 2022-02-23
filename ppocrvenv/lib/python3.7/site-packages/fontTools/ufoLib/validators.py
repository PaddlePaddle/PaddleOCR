"""Various low level data validators."""

import calendar
from io import open
import fs.base
import fs.osfs

from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes


# -------
# Generic
# -------

def isDictEnough(value):
    """
    Some objects will likely come in that aren't
    dicts but are dict-ish enough.
    """
    if isinstance(value, Mapping):
        return True
    for attr in ("keys", "values", "items"):
        if not hasattr(value, attr):
            return False
    return True

def genericTypeValidator(value, typ):
	"""
	Generic. (Added at version 2.)
	"""
	return isinstance(value, typ)

def genericIntListValidator(values, validValues):
	"""
	Generic. (Added at version 2.)
	"""
	if not isinstance(values, (list, tuple)):
		return False
	valuesSet = set(values)
	validValuesSet = set(validValues)
	if valuesSet - validValuesSet:
		return False
	for value in values:
		if not isinstance(value, int):
			return False
	return True

def genericNonNegativeIntValidator(value):
	"""
	Generic. (Added at version 3.)
	"""
	if not isinstance(value, int):
		return False
	if value < 0:
		return False
	return True

def genericNonNegativeNumberValidator(value):
	"""
	Generic. (Added at version 3.)
	"""
	if not isinstance(value, numberTypes):
		return False
	if value < 0:
		return False
	return True

def genericDictValidator(value, prototype):
	"""
	Generic. (Added at version 3.)
	"""
	# not a dict
	if not isinstance(value, Mapping):
		return False
	# missing required keys
	for key, (typ, required) in prototype.items():
		if not required:
			continue
		if key not in value:
			return False
	# unknown keys
	for key in value.keys():
		if key not in prototype:
			return False
	# incorrect types
	for key, v in value.items():
		prototypeType, required = prototype[key]
		if v is None and not required:
			continue
		if not isinstance(v, prototypeType):
			return False
	return True

# --------------
# fontinfo.plist
# --------------

# Data Validators

def fontInfoStyleMapStyleNameValidator(value):
	"""
	Version 2+.
	"""
	options = ["regular", "italic", "bold", "bold italic"]
	return value in options

def fontInfoOpenTypeGaspRangeRecordsValidator(value):
	"""
	Version 3+.
	"""
	if not isinstance(value, list):
		return False
	if len(value) == 0:
		return True
	validBehaviors = [0, 1, 2, 3]
	dictPrototype = dict(rangeMaxPPEM=(int, True), rangeGaspBehavior=(list, True))
	ppemOrder = []
	for rangeRecord in value:
		if not genericDictValidator(rangeRecord, dictPrototype):
			return False
		ppem = rangeRecord["rangeMaxPPEM"]
		behavior = rangeRecord["rangeGaspBehavior"]
		ppemValidity = genericNonNegativeIntValidator(ppem)
		if not ppemValidity:
			return False
		behaviorValidity = genericIntListValidator(behavior, validBehaviors)
		if not behaviorValidity:
			return False
		ppemOrder.append(ppem)
	if ppemOrder != sorted(ppemOrder):
		return False
	return True

def fontInfoOpenTypeHeadCreatedValidator(value):
	"""
	Version 2+.
	"""
	# format: 0000/00/00 00:00:00
	if not isinstance(value, str):
		return False
	# basic formatting
	if not len(value) == 19:
		return False
	if value.count(" ") != 1:
		return False
	date, time = value.split(" ")
	if date.count("/") != 2:
		return False
	if time.count(":") != 2:
		return False
	# date
	year, month, day = date.split("/")
	if len(year) != 4:
		return False
	if len(month) != 2:
		return False
	if len(day) != 2:
		return False
	try:
		year = int(year)
		month = int(month)
		day = int(day)
	except ValueError:
		return False
	if month < 1 or month > 12:
		return False
	monthMaxDay = calendar.monthrange(year, month)[1]
	if day < 1 or day > monthMaxDay:
		return False
	# time
	hour, minute, second = time.split(":")
	if len(hour) != 2:
		return False
	if len(minute) != 2:
		return False
	if len(second) != 2:
		return False
	try:
		hour = int(hour)
		minute = int(minute)
		second = int(second)
	except ValueError:
		return False
	if hour < 0 or hour > 23:
		return False
	if minute < 0 or minute > 59:
		return False
	if second < 0 or second > 59:
		return False
	# fallback
	return True

def fontInfoOpenTypeNameRecordsValidator(value):
	"""
	Version 3+.
	"""
	if not isinstance(value, list):
		return False
	dictPrototype = dict(nameID=(int, True), platformID=(int, True), encodingID=(int, True), languageID=(int, True), string=(str, True))
	for nameRecord in value:
		if not genericDictValidator(nameRecord, dictPrototype):
			return False
	return True

def fontInfoOpenTypeOS2WeightClassValidator(value):
	"""
	Version 2+.
	"""
	if not isinstance(value, int):
		return False
	if value < 0:
		return False
	return True

def fontInfoOpenTypeOS2WidthClassValidator(value):
	"""
	Version 2+.
	"""
	if not isinstance(value, int):
		return False
	if value < 1:
		return False
	if value > 9:
		return False
	return True

def fontInfoVersion2OpenTypeOS2PanoseValidator(values):
	"""
	Version 2.
	"""
	if not isinstance(values, (list, tuple)):
		return False
	if len(values) != 10:
		return False
	for value in values:
		if not isinstance(value, int):
			return False
	# XXX further validation?
	return True

def fontInfoVersion3OpenTypeOS2PanoseValidator(values):
	"""
	Version 3+.
	"""
	if not isinstance(values, (list, tuple)):
		return False
	if len(values) != 10:
		return False
	for value in values:
		if not isinstance(value, int):
			return False
		if value < 0:
			return False
	# XXX further validation?
	return True

def fontInfoOpenTypeOS2FamilyClassValidator(values):
	"""
	Version 2+.
	"""
	if not isinstance(values, (list, tuple)):
		return False
	if len(values) != 2:
		return False
	for value in values:
		if not isinstance(value, int):
			return False
	classID, subclassID = values
	if classID < 0 or classID > 14:
		return False
	if subclassID < 0 or subclassID > 15:
		return False
	return True

def fontInfoPostscriptBluesValidator(values):
	"""
	Version 2+.
	"""
	if not isinstance(values, (list, tuple)):
		return False
	if len(values) > 14:
		return False
	if len(values) % 2:
		return False
	for value in values:
		if not isinstance(value, numberTypes):
			return False
	return True

def fontInfoPostscriptOtherBluesValidator(values):
	"""
	Version 2+.
	"""
	if not isinstance(values, (list, tuple)):
		return False
	if len(values) > 10:
		return False
	if len(values) % 2:
		return False
	for value in values:
		if not isinstance(value, numberTypes):
			return False
	return True

def fontInfoPostscriptStemsValidator(values):
	"""
	Version 2+.
	"""
	if not isinstance(values, (list, tuple)):
		return False
	if len(values) > 12:
		return False
	for value in values:
		if not isinstance(value, numberTypes):
			return False
	return True

def fontInfoPostscriptWindowsCharacterSetValidator(value):
	"""
	Version 2+.
	"""
	validValues = list(range(1, 21))
	if value not in validValues:
		return False
	return True

def fontInfoWOFFMetadataUniqueIDValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = dict(id=(str, True))
	if not genericDictValidator(value, dictPrototype):
		return False
	return True

def fontInfoWOFFMetadataVendorValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = {"name" : (str, True), "url" : (str, False), "dir" : (str, False), "class" : (str, False)}
	if not genericDictValidator(value, dictPrototype):
		return False
	if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
		return False
	return True

def fontInfoWOFFMetadataCreditsValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = dict(credits=(list, True))
	if not genericDictValidator(value, dictPrototype):
		return False
	if not len(value["credits"]):
		return False
	dictPrototype = {"name" : (str, True), "url" : (str, False), "role" : (str, False), "dir" : (str, False), "class" : (str, False)}
	for credit in value["credits"]:
		if not genericDictValidator(credit, dictPrototype):
			return False
		if "dir" in credit and credit.get("dir") not in ("ltr", "rtl"):
			return False
	return True

def fontInfoWOFFMetadataDescriptionValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = dict(url=(str, False), text=(list, True))
	if not genericDictValidator(value, dictPrototype):
		return False
	for text in value["text"]:
		if not fontInfoWOFFMetadataTextValue(text):
			return False
	return True

def fontInfoWOFFMetadataLicenseValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = dict(url=(str, False), text=(list, False), id=(str, False))
	if not genericDictValidator(value, dictPrototype):
		return False
	if "text" in value:
		for text in value["text"]:
			if not fontInfoWOFFMetadataTextValue(text):
				return False
	return True

def fontInfoWOFFMetadataTrademarkValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = dict(text=(list, True))
	if not genericDictValidator(value, dictPrototype):
		return False
	for text in value["text"]:
		if not fontInfoWOFFMetadataTextValue(text):
			return False
	return True

def fontInfoWOFFMetadataCopyrightValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = dict(text=(list, True))
	if not genericDictValidator(value, dictPrototype):
		return False
	for text in value["text"]:
		if not fontInfoWOFFMetadataTextValue(text):
			return False
	return True

def fontInfoWOFFMetadataLicenseeValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = {"name" : (str, True), "dir" : (str, False), "class" : (str, False)}
	if not genericDictValidator(value, dictPrototype):
		return False
	if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
		return False
	return True

def fontInfoWOFFMetadataTextValue(value):
	"""
	Version 3+.
	"""
	dictPrototype = {"text" : (str, True), "language" : (str, False), "dir" : (str, False), "class" : (str, False)}
	if not genericDictValidator(value, dictPrototype):
		return False
	if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
		return False
	return True

def fontInfoWOFFMetadataExtensionsValidator(value):
	"""
	Version 3+.
	"""
	if not isinstance(value, list):
		return False
	if not value:
		return False
	for extension in value:
		if not fontInfoWOFFMetadataExtensionValidator(extension):
			return False
	return True

def fontInfoWOFFMetadataExtensionValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = dict(names=(list, False), items=(list, True), id=(str, False))
	if not genericDictValidator(value, dictPrototype):
		return False
	if "names" in value:
		for name in value["names"]:
			if not fontInfoWOFFMetadataExtensionNameValidator(name):
				return False
	for item in value["items"]:
		if not fontInfoWOFFMetadataExtensionItemValidator(item):
			return False
	return True

def fontInfoWOFFMetadataExtensionItemValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = dict(id=(str, False), names=(list, True), values=(list, True))
	if not genericDictValidator(value, dictPrototype):
		return False
	for name in value["names"]:
		if not fontInfoWOFFMetadataExtensionNameValidator(name):
			return False
	for val in value["values"]:
		if not fontInfoWOFFMetadataExtensionValueValidator(val):
			return False
	return True

def fontInfoWOFFMetadataExtensionNameValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = {"text" : (str, True), "language" : (str, False), "dir" : (str, False), "class" : (str, False)}
	if not genericDictValidator(value, dictPrototype):
		return False
	if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
		return False
	return True

def fontInfoWOFFMetadataExtensionValueValidator(value):
	"""
	Version 3+.
	"""
	dictPrototype = {"text" : (str, True), "language" : (str, False), "dir" : (str, False), "class" : (str, False)}
	if not genericDictValidator(value, dictPrototype):
		return False
	if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
		return False
	return True

# ----------
# Guidelines
# ----------

def guidelinesValidator(value, identifiers=None):
	"""
	Version 3+.
	"""
	if not isinstance(value, list):
		return False
	if identifiers is None:
		identifiers = set()
	for guide in value:
		if not guidelineValidator(guide):
			return False
		identifier = guide.get("identifier")
		if identifier is not None:
			if identifier in identifiers:
				return False
			identifiers.add(identifier)
	return True

_guidelineDictPrototype = dict(
	x=((int, float), False), y=((int, float), False), angle=((int, float), False),
	name=(str, False), color=(str, False), identifier=(str, False)
)

def guidelineValidator(value):
	"""
	Version 3+.
	"""
	if not genericDictValidator(value, _guidelineDictPrototype):
		return False
	x = value.get("x")
	y = value.get("y")
	angle = value.get("angle")
	# x or y must be present
	if x is None and y is None:
		return False
	# if x or y are None, angle must not be present
	if x is None or y is None:
		if angle is not None:
			return False
	# if x and y are defined, angle must be defined
	if x is not None and y is not None and angle is None:
		return False
	# angle must be between 0 and 360
	if angle is not None:
		if angle < 0:
			return False
		if angle > 360:
			return False
	# identifier must be 1 or more characters
	identifier = value.get("identifier")
	if identifier is not None and not identifierValidator(identifier):
		return False
	# color must follow the proper format
	color = value.get("color")
	if color is not None and not colorValidator(color):
		return False
	return True

# -------
# Anchors
# -------

def anchorsValidator(value, identifiers=None):
	"""
	Version 3+.
	"""
	if not isinstance(value, list):
		return False
	if identifiers is None:
		identifiers = set()
	for anchor in value:
		if not anchorValidator(anchor):
			return False
		identifier = anchor.get("identifier")
		if identifier is not None:
			if identifier in identifiers:
				return False
			identifiers.add(identifier)
	return True

_anchorDictPrototype = dict(
	x=((int, float), False), y=((int, float), False),
	name=(str, False), color=(str, False),
	identifier=(str, False)
)

def anchorValidator(value):
	"""
	Version 3+.
	"""
	if not genericDictValidator(value, _anchorDictPrototype):
		return False
	x = value.get("x")
	y = value.get("y")
	# x and y must be present
	if x is None or y is None:
		return False
	# identifier must be 1 or more characters
	identifier = value.get("identifier")
	if identifier is not None and not identifierValidator(identifier):
		return False
	# color must follow the proper format
	color = value.get("color")
	if color is not None and not colorValidator(color):
		return False
	return True

# ----------
# Identifier
# ----------

def identifierValidator(value):
	"""
	Version 3+.

	>>> identifierValidator("a")
	True
	>>> identifierValidator("")
	False
	>>> identifierValidator("a" * 101)
	False
	"""
	validCharactersMin = 0x20
	validCharactersMax = 0x7E
	if not isinstance(value, str):
		return False
	if not value:
		return False
	if len(value) > 100:
		return False
	for c in value:
		c = ord(c)
		if c < validCharactersMin or c > validCharactersMax:
			return False
	return True

# -----
# Color
# -----

def colorValidator(value):
	"""
	Version 3+.

	>>> colorValidator("0,0,0,0")
	True
	>>> colorValidator(".5,.5,.5,.5")
	True
	>>> colorValidator("0.5,0.5,0.5,0.5")
	True
	>>> colorValidator("1,1,1,1")
	True

	>>> colorValidator("2,0,0,0")
	False
	>>> colorValidator("0,2,0,0")
	False
	>>> colorValidator("0,0,2,0")
	False
	>>> colorValidator("0,0,0,2")
	False

	>>> colorValidator("1r,1,1,1")
	False
	>>> colorValidator("1,1g,1,1")
	False
	>>> colorValidator("1,1,1b,1")
	False
	>>> colorValidator("1,1,1,1a")
	False

	>>> colorValidator("1 1 1 1")
	False
	>>> colorValidator("1 1,1,1")
	False
	>>> colorValidator("1,1 1,1")
	False
	>>> colorValidator("1,1,1 1")
	False

	>>> colorValidator("1, 1, 1, 1")
	True
	"""
	if not isinstance(value, str):
		return False
	parts = value.split(",")
	if len(parts) != 4:
		return False
	for part in parts:
		part = part.strip()
		converted = False
		try:
			part = int(part)
			converted = True
		except ValueError:
			pass
		if not converted:
			try:
				part = float(part)
				converted = True
			except ValueError:
				pass
		if not converted:
			return False
		if part < 0:
			return False
		if part > 1:
			return False
	return True

# -----
# image
# -----

pngSignature = b"\x89PNG\r\n\x1a\n"

_imageDictPrototype = dict(
	fileName=(str, True),
	xScale=((int, float), False), xyScale=((int, float), False),
	yxScale=((int, float), False), yScale=((int, float), False),
	xOffset=((int, float), False), yOffset=((int, float), False),
	color=(str, False)
)

def imageValidator(value):
	"""
	Version 3+.
	"""
	if not genericDictValidator(value, _imageDictPrototype):
		return False
	# fileName must be one or more characters
	if not value["fileName"]:
		return False
	# color must follow the proper format
	color = value.get("color")
	if color is not None and not colorValidator(color):
		return False
	return True

def pngValidator(path=None, data=None, fileObj=None):
	"""
	Version 3+.

	This checks the signature of the image data.
	"""
	assert path is not None or data is not None or fileObj is not None
	if path is not None:
		with open(path, "rb") as f:
			signature = f.read(8)
	elif data is not None:
		signature = data[:8]
	elif fileObj is not None:
		pos = fileObj.tell()
		signature = fileObj.read(8)
		fileObj.seek(pos)
	if signature != pngSignature:
		return False, "Image does not begin with the PNG signature."
	return True, None

# -------------------
# layercontents.plist
# -------------------

def layerContentsValidator(value, ufoPathOrFileSystem):
	"""
	Check the validity of layercontents.plist.
	Version 3+.
	"""
	if isinstance(ufoPathOrFileSystem, fs.base.FS):
		fileSystem = ufoPathOrFileSystem
	else:
		fileSystem = fs.osfs.OSFS(ufoPathOrFileSystem)

	bogusFileMessage = "layercontents.plist in not in the correct format."
	# file isn't in the right format
	if not isinstance(value, list):
		return False, bogusFileMessage
	# work through each entry
	usedLayerNames = set()
	usedDirectories = set()
	contents = {}
	for entry in value:
		# layer entry in the incorrect format
		if not isinstance(entry, list):
			return False, bogusFileMessage
		if not len(entry) == 2:
			return False, bogusFileMessage
		for i in entry:
			if not isinstance(i, str):
				return False, bogusFileMessage
		layerName, directoryName = entry
		# check directory naming
		if directoryName != "glyphs":
			if not directoryName.startswith("glyphs."):
				return False, "Invalid directory name (%s) in layercontents.plist." % directoryName
		if len(layerName) == 0:
			return False, "Empty layer name in layercontents.plist."
		# directory doesn't exist
		if not fileSystem.exists(directoryName):
			return False, "A glyphset does not exist at %s." % directoryName
		# default layer name
		if layerName == "public.default" and directoryName != "glyphs":
			return False, "The name public.default is being used by a layer that is not the default."
		# check usage
		if layerName in usedLayerNames:
			return False, "The layer name %s is used by more than one layer." % layerName
		usedLayerNames.add(layerName)
		if directoryName in usedDirectories:
			return False, "The directory %s is used by more than one layer." % directoryName
		usedDirectories.add(directoryName)
		# store
		contents[layerName] = directoryName
	# missing default layer
	foundDefault = "glyphs" in contents.values()
	if not foundDefault:
		return False, "The required default glyph set is not in the UFO."
	return True, None

# ------------
# groups.plist
# ------------

def groupsValidator(value):
	"""
	Check the validity of the groups.
	Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

	>>> groups = {"A" : ["A", "A"], "A2" : ["A"]}
	>>> groupsValidator(groups)
	(True, None)

	>>> groups = {"" : ["A"]}
	>>> valid, msg = groupsValidator(groups)
	>>> valid
	False
	>>> print(msg)
	A group has an empty name.

	>>> groups = {"public.awesome" : ["A"]}
	>>> groupsValidator(groups)
	(True, None)

	>>> groups = {"public.kern1." : ["A"]}
	>>> valid, msg = groupsValidator(groups)
	>>> valid
	False
	>>> print(msg)
	The group data contains a kerning group with an incomplete name.
	>>> groups = {"public.kern2." : ["A"]}
	>>> valid, msg = groupsValidator(groups)
	>>> valid
	False
	>>> print(msg)
	The group data contains a kerning group with an incomplete name.

	>>> groups = {"public.kern1.A" : ["A"], "public.kern2.A" : ["A"]}
	>>> groupsValidator(groups)
	(True, None)

	>>> groups = {"public.kern1.A1" : ["A"], "public.kern1.A2" : ["A"]}
	>>> valid, msg = groupsValidator(groups)
	>>> valid
	False
	>>> print(msg)
	The glyph "A" occurs in too many kerning groups.
	"""
	bogusFormatMessage = "The group data is not in the correct format."
	if not isDictEnough(value):
		return False, bogusFormatMessage
	firstSideMapping = {}
	secondSideMapping = {}
	for groupName, glyphList in value.items():
		if not isinstance(groupName, (str)):
			return False, bogusFormatMessage
		if not isinstance(glyphList, (list, tuple)):
			return False, bogusFormatMessage
		if not groupName:
			return False, "A group has an empty name."
		if groupName.startswith("public."):
			if not groupName.startswith("public.kern1.") and not groupName.startswith("public.kern2."):
				# unknown public.* name. silently skip.
				continue
			else:
				if len("public.kernN.") == len(groupName):
					return False, "The group data contains a kerning group with an incomplete name."
			if groupName.startswith("public.kern1."):
				d = firstSideMapping
			else:
				d = secondSideMapping
			for glyphName in glyphList:
				if not isinstance(glyphName, str):
					return False, "The group data %s contains an invalid member." % groupName
				if glyphName in d:
					return False, "The glyph \"%s\" occurs in too many kerning groups." % glyphName
				d[glyphName] = groupName
	return True, None

# -------------
# kerning.plist
# -------------

def kerningValidator(data):
	"""
	Check the validity of the kerning data structure.
	Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

	>>> kerning = {"A" : {"B" : 100}}
	>>> kerningValidator(kerning)
	(True, None)

	>>> kerning = {"A" : ["B"]}
	>>> valid, msg = kerningValidator(kerning)
	>>> valid
	False
	>>> print(msg)
	The kerning data is not in the correct format.

	>>> kerning = {"A" : {"B" : "100"}}
	>>> valid, msg = kerningValidator(kerning)
	>>> valid
	False
	>>> print(msg)
	The kerning data is not in the correct format.
	"""
	bogusFormatMessage = "The kerning data is not in the correct format."
	if not isinstance(data, Mapping):
		return False, bogusFormatMessage
	for first, secondDict in data.items():
		if not isinstance(first, str):
			return False, bogusFormatMessage
		elif not isinstance(secondDict, Mapping):
			return False, bogusFormatMessage
		for second, value in secondDict.items():
			if not isinstance(second, str):
				return False, bogusFormatMessage
			elif not isinstance(value, numberTypes):
				return False, bogusFormatMessage
	return True, None

# -------------
# lib.plist/lib
# -------------

_bogusLibFormatMessage = "The lib data is not in the correct format: %s"

def fontLibValidator(value):
	"""
	Check the validity of the lib.
	Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

	>>> lib = {"foo" : "bar"}
	>>> fontLibValidator(lib)
	(True, None)

	>>> lib = {"public.awesome" : "hello"}
	>>> fontLibValidator(lib)
	(True, None)

	>>> lib = {"public.glyphOrder" : ["A", "C", "B"]}
	>>> fontLibValidator(lib)
	(True, None)

	>>> lib = "hello"
	>>> valid, msg = fontLibValidator(lib)
	>>> valid
	False
	>>> print(msg)  # doctest: +ELLIPSIS
	The lib data is not in the correct format: expected a dictionary, ...

	>>> lib = {1: "hello"}
	>>> valid, msg = fontLibValidator(lib)
	>>> valid
	False
	>>> print(msg)
	The lib key is not properly formatted: expected str, found int: 1

	>>> lib = {"public.glyphOrder" : "hello"}
	>>> valid, msg = fontLibValidator(lib)
	>>> valid
	False
	>>> print(msg)  # doctest: +ELLIPSIS
	public.glyphOrder is not properly formatted: expected list or tuple,...

	>>> lib = {"public.glyphOrder" : ["A", 1, "B"]}
	>>> valid, msg = fontLibValidator(lib)
	>>> valid
	False
	>>> print(msg)  # doctest: +ELLIPSIS
	public.glyphOrder is not properly formatted: expected str,...
	"""
	if not isDictEnough(value):
		reason = "expected a dictionary, found %s" % type(value).__name__
		return False, _bogusLibFormatMessage % reason
	for key, value in value.items():
		if not isinstance(key, str):
			return False, (
				"The lib key is not properly formatted: expected str, found %s: %r" %
				(type(key).__name__, key))
		# public.glyphOrder
		if key == "public.glyphOrder":
			bogusGlyphOrderMessage = "public.glyphOrder is not properly formatted: %s"
			if not isinstance(value, (list, tuple)):
				reason = "expected list or tuple, found %s" % type(value).__name__
				return False, bogusGlyphOrderMessage % reason
			for glyphName in value:
				if not isinstance(glyphName, str):
					reason = "expected str, found %s" % type(glyphName).__name__
					return False, bogusGlyphOrderMessage % reason
	return True, None

# --------
# GLIF lib
# --------

def glyphLibValidator(value):
	"""
	Check the validity of the lib.
	Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

	>>> lib = {"foo" : "bar"}
	>>> glyphLibValidator(lib)
	(True, None)

	>>> lib = {"public.awesome" : "hello"}
	>>> glyphLibValidator(lib)
	(True, None)

	>>> lib = {"public.markColor" : "1,0,0,0.5"}
	>>> glyphLibValidator(lib)
	(True, None)

	>>> lib = {"public.markColor" : 1}
	>>> valid, msg = glyphLibValidator(lib)
	>>> valid
	False
	>>> print(msg)
	public.markColor is not properly formatted.
	"""
	if not isDictEnough(value):
		reason = "expected a dictionary, found %s" % type(value).__name__
		return False, _bogusLibFormatMessage % reason
	for key, value in value.items():
		if not isinstance(key, str):
			reason = "key (%s) should be a string" % key
			return False, _bogusLibFormatMessage % reason
		# public.markColor
		if key == "public.markColor":
			if not colorValidator(value):
				return False, "public.markColor is not properly formatted."
	return True, None


if __name__ == "__main__":
	import doctest
	doctest.testmod()
