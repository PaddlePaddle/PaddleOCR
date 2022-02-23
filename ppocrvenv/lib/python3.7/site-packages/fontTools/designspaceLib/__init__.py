# -*- coding: utf-8 -*-

from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
import collections
from io import BytesIO, StringIO
import os
import posixpath
from fontTools.misc import etree as ET
from fontTools.misc import plistlib

"""
    designSpaceDocument

    - read and write designspace files
"""

__all__ = [
    'DesignSpaceDocumentError', 'DesignSpaceDocument', 'SourceDescriptor',
    'InstanceDescriptor', 'AxisDescriptor', 'RuleDescriptor', 'BaseDocReader',
    'BaseDocWriter'
]

# ElementTree allows to find namespace-prefixed elements, but not attributes
# so we have to do it ourselves for 'xml:lang'
XML_NS = "{http://www.w3.org/XML/1998/namespace}"
XML_LANG = XML_NS + "lang"


def posix(path):
    """Normalize paths using forward slash to work also on Windows."""
    new_path = posixpath.join(*path.split(os.path.sep))
    if path.startswith('/'):
        # The above transformation loses absolute paths
        new_path = '/' + new_path
    elif path.startswith(r'\\'):
        # The above transformation loses leading slashes of UNC path mounts
        new_path = '//' + new_path
    return new_path


def posixpath_property(private_name):
    def getter(self):
        # Normal getter
        return getattr(self, private_name)

    def setter(self, value):
        # The setter rewrites paths using forward slashes
        if value is not None:
            value = posix(value)
        setattr(self, private_name, value)

    return property(getter, setter)


class DesignSpaceDocumentError(Exception):
    def __init__(self, msg, obj=None):
        self.msg = msg
        self.obj = obj

    def __str__(self):
        return str(self.msg) + (
            ": %r" % self.obj if self.obj is not None else "")


class AsDictMixin(object):

    def asdict(self):
        d = {}
        for attr, value in self.__dict__.items():
            if attr.startswith("_"):
                continue
            if hasattr(value, "asdict"):
                value = value.asdict()
            elif isinstance(value, list):
                value = [
                    v.asdict() if hasattr(v, "asdict") else v for v in value
                ]
            d[attr] = value
        return d


class SimpleDescriptor(AsDictMixin):
    """ Containers for a bunch of attributes"""

    # XXX this is ugly. The 'print' is inappropriate here, and instead of
    # assert, it should simply return True/False
    def compare(self, other):
        # test if this object contains the same data as the other
        for attr in self._attrs:
            try:
                assert(getattr(self, attr) == getattr(other, attr))
            except AssertionError:
                print("failed attribute", attr, getattr(self, attr), "!=", getattr(other, attr))


class SourceDescriptor(SimpleDescriptor):
    """Simple container for data related to the source"""
    flavor = "source"
    _attrs = ['filename', 'path', 'name', 'layerName',
              'location', 'copyLib',
              'copyGroups', 'copyFeatures',
              'muteKerning', 'muteInfo',
              'mutedGlyphNames',
              'familyName', 'styleName']

    def __init__(
        self,
        *,
        filename=None,
        path=None,
        font=None,
        name=None,
        location=None,
        layerName=None,
        familyName=None,
        styleName=None,
        copyLib=False,
        copyInfo=False,
        copyGroups=False,
        copyFeatures=False,
        muteKerning=False,
        muteInfo=False,
        mutedGlyphNames=None,
    ):
        self.filename = filename
        """The original path as found in the document."""

        self.path = path
        """The absolute path, calculated from filename."""

        self.font = font
        """Any Python object. Optional. Points to a representation of this
        source font that is loaded in memory, as a Python object (e.g. a
        ``defcon.Font`` or a ``fontTools.ttFont.TTFont``).

        The default document reader will not fill-in this attribute, and the
        default writer will not use this attribute. It is up to the user of
        ``designspaceLib`` to either load the resource identified by
        ``filename`` and store it in this field, or write the contents of
        this field to the disk and make ```filename`` point to that.
        """

        self.name = name
        self.location = location
        self.layerName = layerName
        self.familyName = familyName
        self.styleName = styleName

        self.copyLib = copyLib
        self.copyInfo = copyInfo
        self.copyGroups = copyGroups
        self.copyFeatures = copyFeatures
        self.muteKerning = muteKerning
        self.muteInfo = muteInfo
        self.mutedGlyphNames = mutedGlyphNames or []

    path = posixpath_property("_path")
    filename = posixpath_property("_filename")


class RuleDescriptor(SimpleDescriptor):
    """Represents the rule descriptor element

    .. code-block:: xml

        <!-- optional: list of substitution rules -->
        <rules>
            <rule name="vertical.bars">
                <conditionset>
                    <condition minimum="250.000000" maximum="750.000000" name="weight"/>
                    <condition minimum="100" name="width"/>
                    <condition minimum="10" maximum="40" name="optical"/>
                </conditionset>
                <sub name="cent" with="cent.alt"/>
                <sub name="dollar" with="dollar.alt"/>
            </rule>
        </rules>
    """
    _attrs = ['name', 'conditionSets', 'subs']   # what do we need here

    def __init__(self, *, name=None, conditionSets=None, subs=None):
        self.name = name
        # list of lists of dict(name='aaaa', minimum=0, maximum=1000)
        self.conditionSets = conditionSets or []
        # list of substitutions stored as tuples of glyphnames ("a", "a.alt")
        self.subs = subs or []


def evaluateRule(rule, location):
    """ Return True if any of the rule's conditionsets matches the given location."""
    return any(evaluateConditions(c, location) for c in rule.conditionSets)


def evaluateConditions(conditions, location):
    """ Return True if all the conditions matches the given location.
        If a condition has no minimum, check for < maximum.
        If a condition has no maximum, check for > minimum.
    """
    for cd in conditions:
        value = location[cd['name']]
        if cd.get('minimum') is None:
            if value > cd['maximum']:
                return False
        elif cd.get('maximum') is None:
            if cd['minimum'] > value:
                return False
        elif not cd['minimum'] <= value <= cd['maximum']:
            return False
    return True


def processRules(rules, location, glyphNames):
    """ Apply these rules at this location to these glyphnames
        - rule order matters
    """
    newNames = []
    for rule in rules:
        if evaluateRule(rule, location):
            for name in glyphNames:
                swap = False
                for a, b in rule.subs:
                    if name == a:
                        swap = True
                        break
                if swap:
                    newNames.append(b)
                else:
                    newNames.append(name)
            glyphNames = newNames
            newNames = []
    return glyphNames


class InstanceDescriptor(SimpleDescriptor):
    """Simple container for data related to the instance"""
    flavor = "instance"
    _defaultLanguageCode = "en"
    _attrs = ['path',
              'name',
              'location',
              'familyName',
              'styleName',
              'postScriptFontName',
              'styleMapFamilyName',
              'styleMapStyleName',
              'kerning',
              'info',
              'lib']

    def __init__(
        self,
        *,
        filename=None,
        path=None,
        font=None,
        name=None,
        location=None,
        familyName=None,
        styleName=None,
        postScriptFontName=None,
        styleMapFamilyName=None,
        styleMapStyleName=None,
        localisedFamilyName=None,
        localisedStyleName=None,
        localisedStyleMapFamilyName=None,
        localisedStyleMapStyleName=None,
        glyphs=None,
        kerning=True,
        info=True,
        lib=None,
    ):
        # the original path as found in the document
        self.filename = filename
        # the absolute path, calculated from filename
        self.path = path
        # Same as in SourceDescriptor.
        self.font = font
        self.name = name
        self.location = location
        self.familyName = familyName
        self.styleName = styleName
        self.postScriptFontName = postScriptFontName
        self.styleMapFamilyName = styleMapFamilyName
        self.styleMapStyleName = styleMapStyleName
        self.localisedFamilyName = localisedFamilyName or {}
        self.localisedStyleName = localisedStyleName or {}
        self.localisedStyleMapFamilyName = localisedStyleMapFamilyName or {}
        self.localisedStyleMapStyleName = localisedStyleMapStyleName or {}
        self.glyphs = glyphs or {}
        self.kerning = kerning
        self.info = info

        self.lib = lib or {}
        """Custom data associated with this instance."""

    path = posixpath_property("_path")
    filename = posixpath_property("_filename")

    def setStyleName(self, styleName, languageCode="en"):
        self.localisedStyleName[languageCode] = tostr(styleName)

    def getStyleName(self, languageCode="en"):
        return self.localisedStyleName.get(languageCode)

    def setFamilyName(self, familyName, languageCode="en"):
        self.localisedFamilyName[languageCode] = tostr(familyName)

    def getFamilyName(self, languageCode="en"):
        return self.localisedFamilyName.get(languageCode)

    def setStyleMapStyleName(self, styleMapStyleName, languageCode="en"):
        self.localisedStyleMapStyleName[languageCode] = tostr(styleMapStyleName)

    def getStyleMapStyleName(self, languageCode="en"):
        return self.localisedStyleMapStyleName.get(languageCode)

    def setStyleMapFamilyName(self, styleMapFamilyName, languageCode="en"):
        self.localisedStyleMapFamilyName[languageCode] = tostr(styleMapFamilyName)

    def getStyleMapFamilyName(self, languageCode="en"):
        return self.localisedStyleMapFamilyName.get(languageCode)


def tagForAxisName(name):
    # try to find or make a tag name for this axis name
    names = {
        'weight':   ('wght', dict(en = 'Weight')),
        'width':    ('wdth', dict(en = 'Width')),
        'optical':  ('opsz', dict(en = 'Optical Size')),
        'slant':    ('slnt', dict(en = 'Slant')),
        'italic':   ('ital', dict(en = 'Italic')),
    }
    if name.lower() in names:
        return names[name.lower()]
    if len(name) < 4:
        tag = name + "*" * (4 - len(name))
    else:
        tag = name[:4]
    return tag, dict(en=name)


class AxisDescriptor(SimpleDescriptor):
    """ Simple container for the axis data
        Add more localisations?
    """
    flavor = "axis"
    _attrs = ['tag', 'name', 'maximum', 'minimum', 'default', 'map']

    def __init__(
        self,
        *,
        tag=None,
        name=None,
        labelNames=None,
        minimum=None,
        default=None,
        maximum=None,
        hidden=False,
        map=None,
    ):
        # opentype tag for this axis
        self.tag = tag
        # name of the axis used in locations
        self.name = name
        # names for UI purposes, if this is not a standard axis,
        self.labelNames = labelNames or {}
        self.minimum = minimum
        self.maximum = maximum
        self.default = default
        self.hidden = hidden
        self.map = map or []

    def serialize(self):
        # output to a dict, used in testing
        return dict(
            tag=self.tag,
            name=self.name,
            labelNames=self.labelNames,
            maximum=self.maximum,
            minimum=self.minimum,
            default=self.default,
            hidden=self.hidden,
            map=self.map,
        )

    def map_forward(self, v):
        from fontTools.varLib.models import piecewiseLinearMap

        if not self.map:
            return v
        return piecewiseLinearMap(v, {k: v for k, v in self.map})

    def map_backward(self, v):
        from fontTools.varLib.models import piecewiseLinearMap

        if not self.map:
            return v
        return piecewiseLinearMap(v, {v: k for k, v in self.map})


class BaseDocWriter(object):
    _whiteSpace = "    "
    ruleDescriptorClass = RuleDescriptor
    axisDescriptorClass = AxisDescriptor
    sourceDescriptorClass = SourceDescriptor
    instanceDescriptorClass = InstanceDescriptor

    @classmethod
    def getAxisDecriptor(cls):
        return cls.axisDescriptorClass()

    @classmethod
    def getSourceDescriptor(cls):
        return cls.sourceDescriptorClass()

    @classmethod
    def getInstanceDescriptor(cls):
        return cls.instanceDescriptorClass()

    @classmethod
    def getRuleDescriptor(cls):
        return cls.ruleDescriptorClass()

    def __init__(self, documentPath, documentObject):
        self.path = documentPath
        self.documentObject = documentObject
        self.documentVersion = "4.1"
        self.root = ET.Element("designspace")
        self.root.attrib['format'] = self.documentVersion
        self._axes = []     # for use by the writer only
        self._rules = []    # for use by the writer only

    def write(self, pretty=True, encoding="UTF-8", xml_declaration=True):
        if self.documentObject.axes:
            self.root.append(ET.Element("axes"))
        for axisObject in self.documentObject.axes:
            self._addAxis(axisObject)

        if self.documentObject.rules:
            if getattr(self.documentObject, "rulesProcessingLast", False):
                attributes = {"processing": "last"}
            else:
                attributes = {}
            self.root.append(ET.Element("rules", attributes))
        for ruleObject in self.documentObject.rules:
            self._addRule(ruleObject)

        if self.documentObject.sources:
            self.root.append(ET.Element("sources"))
        for sourceObject in self.documentObject.sources:
            self._addSource(sourceObject)

        if self.documentObject.instances:
            self.root.append(ET.Element("instances"))
        for instanceObject in self.documentObject.instances:
            self._addInstance(instanceObject)

        if self.documentObject.lib:
            self._addLib(self.documentObject.lib)

        tree = ET.ElementTree(self.root)
        tree.write(
            self.path,
            encoding=encoding,
            method='xml',
            xml_declaration=xml_declaration,
            pretty_print=pretty,
        )

    def _makeLocationElement(self, locationObject, name=None):
        """ Convert Location dict to a locationElement."""
        locElement = ET.Element("location")
        if name is not None:
            locElement.attrib['name'] = name
        validatedLocation = self.documentObject.newDefaultLocation()
        for axisName, axisValue in locationObject.items():
            if axisName in validatedLocation:
                # only accept values we know
                validatedLocation[axisName] = axisValue
        for dimensionName, dimensionValue in validatedLocation.items():
            dimElement = ET.Element('dimension')
            dimElement.attrib['name'] = dimensionName
            if type(dimensionValue) == tuple:
                dimElement.attrib['xvalue'] = self.intOrFloat(dimensionValue[0])
                dimElement.attrib['yvalue'] = self.intOrFloat(dimensionValue[1])
            else:
                dimElement.attrib['xvalue'] = self.intOrFloat(dimensionValue)
            locElement.append(dimElement)
        return locElement, validatedLocation

    def intOrFloat(self, num):
        if int(num) == num:
            return "%d" % num
        return "%f" % num

    def _addRule(self, ruleObject):
        # if none of the conditions have minimum or maximum values, do not add the rule.
        self._rules.append(ruleObject)
        ruleElement = ET.Element('rule')
        if ruleObject.name is not None:
            ruleElement.attrib['name'] = ruleObject.name
        for conditions in ruleObject.conditionSets:
            conditionsetElement = ET.Element('conditionset')
            for cond in conditions:
                if cond.get('minimum') is None and cond.get('maximum') is None:
                    # neither is defined, don't add this condition
                    continue
                conditionElement = ET.Element('condition')
                conditionElement.attrib['name'] = cond.get('name')
                if cond.get('minimum') is not None:
                    conditionElement.attrib['minimum'] = self.intOrFloat(cond.get('minimum'))
                if cond.get('maximum') is not None:
                    conditionElement.attrib['maximum'] = self.intOrFloat(cond.get('maximum'))
                conditionsetElement.append(conditionElement)
            if len(conditionsetElement):
                ruleElement.append(conditionsetElement)
        for sub in ruleObject.subs:
            subElement = ET.Element('sub')
            subElement.attrib['name'] = sub[0]
            subElement.attrib['with'] = sub[1]
            ruleElement.append(subElement)
        if len(ruleElement):
            self.root.findall('.rules')[0].append(ruleElement)

    def _addAxis(self, axisObject):
        self._axes.append(axisObject)
        axisElement = ET.Element('axis')
        axisElement.attrib['tag'] = axisObject.tag
        axisElement.attrib['name'] = axisObject.name
        axisElement.attrib['minimum'] = self.intOrFloat(axisObject.minimum)
        axisElement.attrib['maximum'] = self.intOrFloat(axisObject.maximum)
        axisElement.attrib['default'] = self.intOrFloat(axisObject.default)
        if axisObject.hidden:
            axisElement.attrib['hidden'] = "1"
        for languageCode, labelName in sorted(axisObject.labelNames.items()):
            languageElement = ET.Element('labelname')
            languageElement.attrib[XML_LANG] = languageCode
            languageElement.text = labelName
            axisElement.append(languageElement)
        if axisObject.map:
            for inputValue, outputValue in axisObject.map:
                mapElement = ET.Element('map')
                mapElement.attrib['input'] = self.intOrFloat(inputValue)
                mapElement.attrib['output'] = self.intOrFloat(outputValue)
                axisElement.append(mapElement)
        self.root.findall('.axes')[0].append(axisElement)

    def _addInstance(self, instanceObject):
        instanceElement = ET.Element('instance')
        if instanceObject.name is not None:
            instanceElement.attrib['name'] = instanceObject.name
        if instanceObject.familyName is not None:
            instanceElement.attrib['familyname'] = instanceObject.familyName
        if instanceObject.styleName is not None:
            instanceElement.attrib['stylename'] = instanceObject.styleName
        # add localisations
        if instanceObject.localisedStyleName:
            languageCodes = list(instanceObject.localisedStyleName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == "en":
                    continue  # already stored in the element attribute
                localisedStyleNameElement = ET.Element('stylename')
                localisedStyleNameElement.attrib[XML_LANG] = code
                localisedStyleNameElement.text = instanceObject.getStyleName(code)
                instanceElement.append(localisedStyleNameElement)
        if instanceObject.localisedFamilyName:
            languageCodes = list(instanceObject.localisedFamilyName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == "en":
                    continue  # already stored in the element attribute
                localisedFamilyNameElement = ET.Element('familyname')
                localisedFamilyNameElement.attrib[XML_LANG] = code
                localisedFamilyNameElement.text = instanceObject.getFamilyName(code)
                instanceElement.append(localisedFamilyNameElement)
        if instanceObject.localisedStyleMapStyleName:
            languageCodes = list(instanceObject.localisedStyleMapStyleName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == "en":
                    continue
                localisedStyleMapStyleNameElement = ET.Element('stylemapstylename')
                localisedStyleMapStyleNameElement.attrib[XML_LANG] = code
                localisedStyleMapStyleNameElement.text = instanceObject.getStyleMapStyleName(code)
                instanceElement.append(localisedStyleMapStyleNameElement)
        if instanceObject.localisedStyleMapFamilyName:
            languageCodes = list(instanceObject.localisedStyleMapFamilyName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == "en":
                    continue
                localisedStyleMapFamilyNameElement = ET.Element('stylemapfamilyname')
                localisedStyleMapFamilyNameElement.attrib[XML_LANG] = code
                localisedStyleMapFamilyNameElement.text = instanceObject.getStyleMapFamilyName(code)
                instanceElement.append(localisedStyleMapFamilyNameElement)

        if instanceObject.location is not None:
            locationElement, instanceObject.location = self._makeLocationElement(instanceObject.location)
            instanceElement.append(locationElement)
        if instanceObject.filename is not None:
            instanceElement.attrib['filename'] = instanceObject.filename
        if instanceObject.postScriptFontName is not None:
            instanceElement.attrib['postscriptfontname'] = instanceObject.postScriptFontName
        if instanceObject.styleMapFamilyName is not None:
            instanceElement.attrib['stylemapfamilyname'] = instanceObject.styleMapFamilyName
        if instanceObject.styleMapStyleName is not None:
            instanceElement.attrib['stylemapstylename'] = instanceObject.styleMapStyleName
        if instanceObject.glyphs:
            if instanceElement.findall('.glyphs') == []:
                glyphsElement = ET.Element('glyphs')
                instanceElement.append(glyphsElement)
            glyphsElement = instanceElement.findall('.glyphs')[0]
            for glyphName, data in sorted(instanceObject.glyphs.items()):
                glyphElement = self._writeGlyphElement(instanceElement, instanceObject, glyphName, data)
                glyphsElement.append(glyphElement)
        if instanceObject.kerning:
            kerningElement = ET.Element('kerning')
            instanceElement.append(kerningElement)
        if instanceObject.info:
            infoElement = ET.Element('info')
            instanceElement.append(infoElement)
        if instanceObject.lib:
            libElement = ET.Element('lib')
            libElement.append(plistlib.totree(instanceObject.lib, indent_level=4))
            instanceElement.append(libElement)
        self.root.findall('.instances')[0].append(instanceElement)

    def _addSource(self, sourceObject):
        sourceElement = ET.Element("source")
        if sourceObject.filename is not None:
            sourceElement.attrib['filename'] = sourceObject.filename
        if sourceObject.name is not None:
            if sourceObject.name.find("temp_master") != 0:
                # do not save temporary source names
                sourceElement.attrib['name'] = sourceObject.name
        if sourceObject.familyName is not None:
            sourceElement.attrib['familyname'] = sourceObject.familyName
        if sourceObject.styleName is not None:
            sourceElement.attrib['stylename'] = sourceObject.styleName
        if sourceObject.layerName is not None:
            sourceElement.attrib['layer'] = sourceObject.layerName
        if sourceObject.copyLib:
            libElement = ET.Element('lib')
            libElement.attrib['copy'] = "1"
            sourceElement.append(libElement)
        if sourceObject.copyGroups:
            groupsElement = ET.Element('groups')
            groupsElement.attrib['copy'] = "1"
            sourceElement.append(groupsElement)
        if sourceObject.copyFeatures:
            featuresElement = ET.Element('features')
            featuresElement.attrib['copy'] = "1"
            sourceElement.append(featuresElement)
        if sourceObject.copyInfo or sourceObject.muteInfo:
            infoElement = ET.Element('info')
            if sourceObject.copyInfo:
                infoElement.attrib['copy'] = "1"
            if sourceObject.muteInfo:
                infoElement.attrib['mute'] = "1"
            sourceElement.append(infoElement)
        if sourceObject.muteKerning:
            kerningElement = ET.Element("kerning")
            kerningElement.attrib["mute"] = '1'
            sourceElement.append(kerningElement)
        if sourceObject.mutedGlyphNames:
            for name in sourceObject.mutedGlyphNames:
                glyphElement = ET.Element("glyph")
                glyphElement.attrib["name"] = name
                glyphElement.attrib["mute"] = '1'
                sourceElement.append(glyphElement)
        locationElement, sourceObject.location = self._makeLocationElement(sourceObject.location)
        sourceElement.append(locationElement)
        self.root.findall('.sources')[0].append(sourceElement)

    def _addLib(self, dict):
        libElement = ET.Element('lib')
        libElement.append(plistlib.totree(dict, indent_level=2))
        self.root.append(libElement)

    def _writeGlyphElement(self, instanceElement, instanceObject, glyphName, data):
        glyphElement = ET.Element('glyph')
        if data.get('mute'):
            glyphElement.attrib['mute'] = "1"
        if data.get('unicodes') is not None:
            glyphElement.attrib['unicode'] = " ".join([hex(u) for u in data.get('unicodes')])
        if data.get('instanceLocation') is not None:
            locationElement, data['instanceLocation'] = self._makeLocationElement(data.get('instanceLocation'))
            glyphElement.append(locationElement)
        if glyphName is not None:
            glyphElement.attrib['name'] = glyphName
        if data.get('note') is not None:
            noteElement = ET.Element('note')
            noteElement.text = data.get('note')
            glyphElement.append(noteElement)
        if data.get('masters') is not None:
            mastersElement = ET.Element("masters")
            for m in data.get('masters'):
                masterElement = ET.Element("master")
                if m.get('glyphName') is not None:
                    masterElement.attrib['glyphname'] = m.get('glyphName')
                if m.get('font') is not None:
                    masterElement.attrib['source'] = m.get('font')
                if m.get('location') is not None:
                    locationElement, m['location'] = self._makeLocationElement(m.get('location'))
                    masterElement.append(locationElement)
                mastersElement.append(masterElement)
            glyphElement.append(mastersElement)
        return glyphElement


class BaseDocReader(LogMixin):
    ruleDescriptorClass = RuleDescriptor
    axisDescriptorClass = AxisDescriptor
    sourceDescriptorClass = SourceDescriptor
    instanceDescriptorClass = InstanceDescriptor

    def __init__(self, documentPath, documentObject):
        self.path = documentPath
        self.documentObject = documentObject
        tree = ET.parse(self.path)
        self.root = tree.getroot()
        self.documentObject.formatVersion = self.root.attrib.get("format", "3.0")
        self._axes = []
        self.rules = []
        self.sources = []
        self.instances = []
        self.axisDefaults = {}
        self._strictAxisNames = True

    @classmethod
    def fromstring(cls, string, documentObject):
        f = BytesIO(tobytes(string, encoding="utf-8"))
        self = cls(f, documentObject)
        self.path = None
        return self

    def read(self):
        self.readAxes()
        self.readRules()
        self.readSources()
        self.readInstances()
        self.readLib()

    def readRules(self):
        # we also need to read any conditions that are outside of a condition set.
        rules = []
        rulesElement = self.root.find(".rules")
        if rulesElement is not None:
            processingValue = rulesElement.attrib.get("processing", "first")
            if processingValue not in {"first", "last"}:
                raise DesignSpaceDocumentError(
                    "<rules> processing attribute value is not valid: %r, "
                    "expected 'first' or 'last'" % processingValue)
            self.documentObject.rulesProcessingLast = processingValue == "last"
        for ruleElement in self.root.findall(".rules/rule"):
            ruleObject = self.ruleDescriptorClass()
            ruleName = ruleObject.name = ruleElement.attrib.get("name")
            # read any stray conditions outside a condition set
            externalConditions = self._readConditionElements(
                ruleElement,
                ruleName,
            )
            if externalConditions:
                ruleObject.conditionSets.append(externalConditions)
                self.log.info(
                    "Found stray rule conditions outside a conditionset. "
                    "Wrapped them in a new conditionset."
                )
            # read the conditionsets
            for conditionSetElement in ruleElement.findall('.conditionset'):
                conditionSet = self._readConditionElements(
                    conditionSetElement,
                    ruleName,
                )
                if conditionSet is not None:
                    ruleObject.conditionSets.append(conditionSet)
            for subElement in ruleElement.findall('.sub'):
                a = subElement.attrib['name']
                b = subElement.attrib['with']
                ruleObject.subs.append((a, b))
            rules.append(ruleObject)
        self.documentObject.rules = rules

    def _readConditionElements(self, parentElement, ruleName=None):
        cds = []
        for conditionElement in parentElement.findall('.condition'):
            cd = {}
            cdMin = conditionElement.attrib.get("minimum")
            if cdMin is not None:
                cd['minimum'] = float(cdMin)
            else:
                # will allow these to be None, assume axis.minimum
                cd['minimum'] = None
            cdMax = conditionElement.attrib.get("maximum")
            if cdMax is not None:
                cd['maximum'] = float(cdMax)
            else:
                # will allow these to be None, assume axis.maximum
                cd['maximum'] = None
            cd['name'] = conditionElement.attrib.get("name")
            # # test for things
            if cd.get('minimum') is None and cd.get('maximum') is None:
                raise DesignSpaceDocumentError(
                    "condition missing required minimum or maximum in rule" +
                    (" '%s'" % ruleName if ruleName is not None else ""))
            cds.append(cd)
        return cds

    def readAxes(self):
        # read the axes elements, including the warp map.
        axisElements = self.root.findall(".axes/axis")
        if not axisElements:
            return
        for axisElement in axisElements:
            axisObject = self.axisDescriptorClass()
            axisObject.name = axisElement.attrib.get("name")
            axisObject.minimum = float(axisElement.attrib.get("minimum"))
            axisObject.maximum = float(axisElement.attrib.get("maximum"))
            if axisElement.attrib.get('hidden', False):
                axisObject.hidden = True
            axisObject.default = float(axisElement.attrib.get("default"))
            axisObject.tag = axisElement.attrib.get("tag")
            for mapElement in axisElement.findall('map'):
                a = float(mapElement.attrib['input'])
                b = float(mapElement.attrib['output'])
                axisObject.map.append((a, b))
            for labelNameElement in axisElement.findall('labelname'):
                # Note: elementtree reads the "xml:lang" attribute name as
                # '{http://www.w3.org/XML/1998/namespace}lang'
                for key, lang in labelNameElement.items():
                    if key == XML_LANG:
                        axisObject.labelNames[lang] = tostr(labelNameElement.text)
            self.documentObject.axes.append(axisObject)
            self.axisDefaults[axisObject.name] = axisObject.default

    def readSources(self):
        for sourceCount, sourceElement in enumerate(self.root.findall(".sources/source")):
            filename = sourceElement.attrib.get('filename')
            if filename is not None and self.path is not None:
                sourcePath = os.path.abspath(os.path.join(os.path.dirname(self.path), filename))
            else:
                sourcePath = None
            sourceName = sourceElement.attrib.get('name')
            if sourceName is None:
                # add a temporary source name
                sourceName = "temp_master.%d" % (sourceCount)
            sourceObject = self.sourceDescriptorClass()
            sourceObject.path = sourcePath        # absolute path to the ufo source
            sourceObject.filename = filename      # path as it is stored in the document
            sourceObject.name = sourceName
            familyName = sourceElement.attrib.get("familyname")
            if familyName is not None:
                sourceObject.familyName = familyName
            styleName = sourceElement.attrib.get("stylename")
            if styleName is not None:
                sourceObject.styleName = styleName
            sourceObject.location = self.locationFromElement(sourceElement)
            layerName = sourceElement.attrib.get('layer')
            if layerName is not None:
                sourceObject.layerName = layerName
            for libElement in sourceElement.findall('.lib'):
                if libElement.attrib.get('copy') == '1':
                    sourceObject.copyLib = True
            for groupsElement in sourceElement.findall('.groups'):
                if groupsElement.attrib.get('copy') == '1':
                    sourceObject.copyGroups = True
            for infoElement in sourceElement.findall(".info"):
                if infoElement.attrib.get('copy') == '1':
                    sourceObject.copyInfo = True
                if infoElement.attrib.get('mute') == '1':
                    sourceObject.muteInfo = True
            for featuresElement in sourceElement.findall(".features"):
                if featuresElement.attrib.get('copy') == '1':
                    sourceObject.copyFeatures = True
            for glyphElement in sourceElement.findall(".glyph"):
                glyphName = glyphElement.attrib.get('name')
                if glyphName is None:
                    continue
                if glyphElement.attrib.get('mute') == '1':
                    sourceObject.mutedGlyphNames.append(glyphName)
            for kerningElement in sourceElement.findall(".kerning"):
                if kerningElement.attrib.get('mute') == '1':
                    sourceObject.muteKerning = True
            self.documentObject.sources.append(sourceObject)

    def locationFromElement(self, element):
        elementLocation = None
        for locationElement in element.findall('.location'):
            elementLocation = self.readLocationElement(locationElement)
            break
        return elementLocation

    def readLocationElement(self, locationElement):
        """ Format 0 location reader """
        if self._strictAxisNames and not self.documentObject.axes:
            raise DesignSpaceDocumentError("No axes defined")
        loc = {}
        for dimensionElement in locationElement.findall(".dimension"):
            dimName = dimensionElement.attrib.get("name")
            if self._strictAxisNames and dimName not in self.axisDefaults:
                # In case the document contains no axis definitions,
                self.log.warning("Location with undefined axis: \"%s\".", dimName)
                continue
            xValue = yValue = None
            try:
                xValue = dimensionElement.attrib.get('xvalue')
                xValue = float(xValue)
            except ValueError:
                self.log.warning("KeyError in readLocation xValue %3.3f", xValue)
            try:
                yValue = dimensionElement.attrib.get('yvalue')
                if yValue is not None:
                    yValue = float(yValue)
            except ValueError:
                pass
            if yValue is not None:
                loc[dimName] = (xValue, yValue)
            else:
                loc[dimName] = xValue
        return loc

    def readInstances(self, makeGlyphs=True, makeKerning=True, makeInfo=True):
        instanceElements = self.root.findall('.instances/instance')
        for instanceElement in instanceElements:
            self._readSingleInstanceElement(instanceElement, makeGlyphs=makeGlyphs, makeKerning=makeKerning, makeInfo=makeInfo)

    def _readSingleInstanceElement(self, instanceElement, makeGlyphs=True, makeKerning=True, makeInfo=True):
        filename = instanceElement.attrib.get('filename')
        if filename is not None and self.documentObject.path is not None:
            instancePath = os.path.join(os.path.dirname(self.documentObject.path), filename)
        else:
            instancePath = None
        instanceObject = self.instanceDescriptorClass()
        instanceObject.path = instancePath    # absolute path to the instance
        instanceObject.filename = filename    # path as it is stored in the document
        name = instanceElement.attrib.get("name")
        if name is not None:
            instanceObject.name = name
        familyname = instanceElement.attrib.get('familyname')
        if familyname is not None:
            instanceObject.familyName = familyname
        stylename = instanceElement.attrib.get('stylename')
        if stylename is not None:
            instanceObject.styleName = stylename
        postScriptFontName = instanceElement.attrib.get('postscriptfontname')
        if postScriptFontName is not None:
            instanceObject.postScriptFontName = postScriptFontName
        styleMapFamilyName = instanceElement.attrib.get('stylemapfamilyname')
        if styleMapFamilyName is not None:
            instanceObject.styleMapFamilyName = styleMapFamilyName
        styleMapStyleName = instanceElement.attrib.get('stylemapstylename')
        if styleMapStyleName is not None:
            instanceObject.styleMapStyleName = styleMapStyleName
        # read localised names
        for styleNameElement in instanceElement.findall('stylename'):
            for key, lang in styleNameElement.items():
                if key == XML_LANG:
                    styleName = styleNameElement.text
                    instanceObject.setStyleName(styleName, lang)
        for familyNameElement in instanceElement.findall('familyname'):
            for key, lang in familyNameElement.items():
                if key == XML_LANG:
                    familyName = familyNameElement.text
                    instanceObject.setFamilyName(familyName, lang)
        for styleMapStyleNameElement in instanceElement.findall('stylemapstylename'):
            for key, lang in styleMapStyleNameElement.items():
                if key == XML_LANG:
                    styleMapStyleName = styleMapStyleNameElement.text
                    instanceObject.setStyleMapStyleName(styleMapStyleName, lang)
        for styleMapFamilyNameElement in instanceElement.findall('stylemapfamilyname'):
            for key, lang in styleMapFamilyNameElement.items():
                if key == XML_LANG:
                    styleMapFamilyName = styleMapFamilyNameElement.text
                    instanceObject.setStyleMapFamilyName(styleMapFamilyName, lang)
        instanceLocation = self.locationFromElement(instanceElement)
        if instanceLocation is not None:
            instanceObject.location = instanceLocation
        for glyphElement in instanceElement.findall('.glyphs/glyph'):
            self.readGlyphElement(glyphElement, instanceObject)
        for infoElement in instanceElement.findall("info"):
            self.readInfoElement(infoElement, instanceObject)
        for libElement in instanceElement.findall('lib'):
            self.readLibElement(libElement, instanceObject)
        self.documentObject.instances.append(instanceObject)

    def readLibElement(self, libElement, instanceObject):
        """Read the lib element for the given instance."""
        instanceObject.lib = plistlib.fromtree(libElement[0])

    def readInfoElement(self, infoElement, instanceObject):
        """ Read the info element."""
        instanceObject.info = True

    def readKerningElement(self, kerningElement, instanceObject):
        """ Read the kerning element."""
        kerningLocation = self.locationFromElement(kerningElement)
        instanceObject.addKerning(kerningLocation)

    def readGlyphElement(self, glyphElement, instanceObject):
        """
        Read the glyph element:

        .. code-block:: xml

            <glyph name="b" unicode="0x62"/>
            <glyph name="b"/>
            <glyph name="b">
                <master location="location-token-bbb" source="master-token-aaa2"/>
                <master glyphname="b.alt1" location="location-token-ccc" source="master-token-aaa3"/>
                <note>
                    This is an instance from an anisotropic interpolation.
                </note>
            </glyph>
        """
        glyphData = {}
        glyphName = glyphElement.attrib.get('name')
        if glyphName is None:
            raise DesignSpaceDocumentError("Glyph object without name attribute")
        mute = glyphElement.attrib.get("mute")
        if mute == "1":
            glyphData['mute'] = True
        # unicode
        unicodes = glyphElement.attrib.get('unicode')
        if unicodes is not None:
            try:
                unicodes = [int(u, 16) for u in unicodes.split(" ")]
                glyphData['unicodes'] = unicodes
            except ValueError:
                raise DesignSpaceDocumentError("unicode values %s are not integers" % unicodes)

        for noteElement in glyphElement.findall('.note'):
            glyphData['note'] = noteElement.text
            break
        instanceLocation = self.locationFromElement(glyphElement)
        if instanceLocation is not None:
            glyphData['instanceLocation'] = instanceLocation
        glyphSources = None
        for masterElement in glyphElement.findall('.masters/master'):
            fontSourceName = masterElement.attrib.get('source')
            sourceLocation = self.locationFromElement(masterElement)
            masterGlyphName = masterElement.attrib.get('glyphname')
            if masterGlyphName is None:
                # if we don't read a glyphname, use the one we have
                masterGlyphName = glyphName
            d = dict(font=fontSourceName,
                     location=sourceLocation,
                     glyphName=masterGlyphName)
            if glyphSources is None:
                glyphSources = []
            glyphSources.append(d)
        if glyphSources is not None:
            glyphData['masters'] = glyphSources
        instanceObject.glyphs[glyphName] = glyphData

    def readLib(self):
        """Read the lib element for the whole document."""
        for libElement in self.root.findall(".lib"):
            self.documentObject.lib = plistlib.fromtree(libElement[0])


class DesignSpaceDocument(LogMixin, AsDictMixin):
    """ Read, write data from the designspace file"""
    def __init__(self, readerClass=None, writerClass=None):
        self.path = None
        self.filename = None
        """String, optional. When the document is read from the disk, this is
        its original file name, i.e. the last part of its path.

        When the document is produced by a Python script and still only exists
        in memory, the producing script can write here an indication of a
        possible "good" filename, in case one wants to save the file somewhere.
        """

        self.formatVersion = None
        self.sources = []
        self.instances = []
        self.axes = []
        self.rules = []
        self.rulesProcessingLast = False
        self.default = None         # name of the default master

        self.lib = {}
        """Custom data associated with the whole document."""

        #
        if readerClass is not None:
            self.readerClass = readerClass
        else:
            self.readerClass = BaseDocReader
        if writerClass is not None:
            self.writerClass = writerClass
        else:
            self.writerClass = BaseDocWriter

    @classmethod
    def fromfile(cls, path, readerClass=None, writerClass=None):
        self = cls(readerClass=readerClass, writerClass=writerClass)
        self.read(path)
        return self

    @classmethod
    def fromstring(cls, string, readerClass=None, writerClass=None):
        self = cls(readerClass=readerClass, writerClass=writerClass)
        reader = self.readerClass.fromstring(string, self)
        reader.read()
        if self.sources:
            self.findDefault()
        return self

    def tostring(self, encoding=None):
        if encoding is str or (
            encoding is not None and encoding.lower() == "unicode"
        ):
            f = StringIO()
            xml_declaration = False
        elif encoding is None or encoding == "utf-8":
            f = BytesIO()
            encoding = "UTF-8"
            xml_declaration = True
        else:
            raise ValueError("unsupported encoding: '%s'" % encoding)
        writer = self.writerClass(f, self)
        writer.write(encoding=encoding, xml_declaration=xml_declaration)
        return f.getvalue()

    def read(self, path):
        if hasattr(path, "__fspath__"):  # support os.PathLike objects
            path = path.__fspath__()
        self.path = path
        self.filename = os.path.basename(path)
        reader = self.readerClass(path, self)
        reader.read()
        if self.sources:
            self.findDefault()

    def write(self, path):
        if hasattr(path, "__fspath__"):  # support os.PathLike objects
            path = path.__fspath__()
        self.path = path
        self.filename = os.path.basename(path)
        self.updatePaths()
        writer = self.writerClass(path, self)
        writer.write()

    def _posixRelativePath(self, otherPath):
        relative = os.path.relpath(otherPath, os.path.dirname(self.path))
        return posix(relative)

    def updatePaths(self):
        """
            Right before we save we need to identify and respond to the following situations:
            In each descriptor, we have to do the right thing for the filename attribute.

            case 1.
            descriptor.filename == None
            descriptor.path == None

            -- action:
            write as is, descriptors will not have a filename attr.
            useless, but no reason to interfere.


            case 2.
            descriptor.filename == "../something"
            descriptor.path == None

            -- action:
            write as is. The filename attr should not be touched.


            case 3.
            descriptor.filename == None
            descriptor.path == "~/absolute/path/there"

            -- action:
            calculate the relative path for filename.
            We're not overwriting some other value for filename, it should be fine


            case 4.
            descriptor.filename == '../somewhere'
            descriptor.path == "~/absolute/path/there"

            -- action:
            there is a conflict between the given filename, and the path.
            So we know where the file is relative to the document.
            Can't guess why they're different, we just choose for path to be correct and update filename.


        """
        assert self.path is not None
        for descriptor in self.sources + self.instances:
            if descriptor.path is not None:
                # case 3 and 4: filename gets updated and relativized
                descriptor.filename = self._posixRelativePath(descriptor.path)

    def addSource(self, sourceDescriptor):
        self.sources.append(sourceDescriptor)

    def addSourceDescriptor(self, **kwargs):
        source = self.writerClass.sourceDescriptorClass(**kwargs)
        self.addSource(source)
        return source

    def addInstance(self, instanceDescriptor):
        self.instances.append(instanceDescriptor)

    def addInstanceDescriptor(self, **kwargs):
        instance = self.writerClass.instanceDescriptorClass(**kwargs)
        self.addInstance(instance)
        return instance

    def addAxis(self, axisDescriptor):
        self.axes.append(axisDescriptor)

    def addAxisDescriptor(self, **kwargs):
        axis = self.writerClass.axisDescriptorClass(**kwargs)
        self.addAxis(axis)
        return axis

    def addRule(self, ruleDescriptor):
        self.rules.append(ruleDescriptor)

    def addRuleDescriptor(self, **kwargs):
        rule = self.writerClass.ruleDescriptorClass(**kwargs)
        self.addRule(rule)
        return rule

    def newDefaultLocation(self):
        """Return default location in design space."""
        # Without OrderedDict, output XML would be non-deterministic.
        # https://github.com/LettError/designSpaceDocument/issues/10
        loc = collections.OrderedDict()
        for axisDescriptor in self.axes:
            loc[axisDescriptor.name] = axisDescriptor.map_forward(
                axisDescriptor.default
            )
        return loc

    def updateFilenameFromPath(self, masters=True, instances=True, force=False):
        # set a descriptor filename attr from the path and this document path
        # if the filename attribute is not None: skip it.
        if masters:
            for descriptor in self.sources:
                if descriptor.filename is not None and not force:
                    continue
                if self.path is not None:
                    descriptor.filename = self._posixRelativePath(descriptor.path)
        if instances:
            for descriptor in self.instances:
                if descriptor.filename is not None and not force:
                    continue
                if self.path is not None:
                    descriptor.filename = self._posixRelativePath(descriptor.path)

    def newAxisDescriptor(self):
        # Ask the writer class to make us a new axisDescriptor
        return self.writerClass.getAxisDecriptor()

    def newSourceDescriptor(self):
        # Ask the writer class to make us a new sourceDescriptor
        return self.writerClass.getSourceDescriptor()

    def newInstanceDescriptor(self):
        # Ask the writer class to make us a new instanceDescriptor
        return self.writerClass.getInstanceDescriptor()

    def getAxisOrder(self):
        names = []
        for axisDescriptor in self.axes:
            names.append(axisDescriptor.name)
        return names

    def getAxis(self, name):
        for axisDescriptor in self.axes:
            if axisDescriptor.name == name:
                return axisDescriptor
        return None

    def findDefault(self):
        """Set and return SourceDescriptor at the default location or None.

        The default location is the set of all `default` values in user space
        of all axes.
        """
        self.default = None

        # Convert the default location from user space to design space before comparing
        # it against the SourceDescriptor locations (always in design space).
        default_location_design = self.newDefaultLocation()

        for sourceDescriptor in self.sources:
            if sourceDescriptor.location == default_location_design:
                self.default = sourceDescriptor
                return sourceDescriptor

        return None

    def normalizeLocation(self, location):
        from fontTools.varLib.models import normalizeValue

        new = {}
        for axis in self.axes:
            if axis.name not in location:
                # skipping this dimension it seems
                continue
            value = location[axis.name]
            # 'anisotropic' location, take first coord only
            if isinstance(value, tuple):
                value = value[0]
            triple = [
                axis.map_forward(v) for v in (axis.minimum, axis.default, axis.maximum)
            ]
            new[axis.name] = normalizeValue(value, triple)
        return new

    def normalize(self):
        # Normalise the geometry of this designspace:
        #   scale all the locations of all masters and instances to the -1 - 0 - 1 value.
        #   we need the axis data to do the scaling, so we do those last.
        # masters
        for item in self.sources:
            item.location = self.normalizeLocation(item.location)
        # instances
        for item in self.instances:
            # glyph masters for this instance
            for _, glyphData in item.glyphs.items():
                glyphData['instanceLocation'] = self.normalizeLocation(glyphData['instanceLocation'])
                for glyphMaster in glyphData['masters']:
                    glyphMaster['location'] = self.normalizeLocation(glyphMaster['location'])
            item.location = self.normalizeLocation(item.location)
        # the axes
        for axis in self.axes:
            # scale the map first
            newMap = []
            for inputValue, outputValue in axis.map:
                newOutputValue = self.normalizeLocation({axis.name: outputValue}).get(axis.name)
                newMap.append((inputValue, newOutputValue))
            if newMap:
                axis.map = newMap
            # finally the axis values
            minimum = self.normalizeLocation({axis.name: axis.minimum}).get(axis.name)
            maximum = self.normalizeLocation({axis.name: axis.maximum}).get(axis.name)
            default = self.normalizeLocation({axis.name: axis.default}).get(axis.name)
            # and set them in the axis.minimum
            axis.minimum = minimum
            axis.maximum = maximum
            axis.default = default
        # now the rules
        for rule in self.rules:
            newConditionSets = []
            for conditions in rule.conditionSets:
                newConditions = []
                for cond in conditions:
                    if cond.get('minimum') is not None:
                        minimum = self.normalizeLocation({cond['name']: cond['minimum']}).get(cond['name'])
                    else:
                        minimum = None
                    if cond.get('maximum') is not None:
                        maximum = self.normalizeLocation({cond['name']: cond['maximum']}).get(cond['name'])
                    else:
                        maximum = None
                    newConditions.append(dict(name=cond['name'], minimum=minimum, maximum=maximum))
                newConditionSets.append(newConditions)
            rule.conditionSets = newConditionSets

    def loadSourceFonts(self, opener, **kwargs):
        """Ensure SourceDescriptor.font attributes are loaded, and return list of fonts.

        Takes a callable which initializes a new font object (e.g. TTFont, or
        defcon.Font, etc.) from the SourceDescriptor.path, and sets the
        SourceDescriptor.font attribute.
        If the font attribute is already not None, it is not loaded again.
        Fonts with the same path are only loaded once and shared among SourceDescriptors.

        For example, to load UFO sources using defcon:

            designspace = DesignSpaceDocument.fromfile("path/to/my.designspace")
            designspace.loadSourceFonts(defcon.Font)

        Or to load masters as FontTools binary fonts, including extra options:

            designspace.loadSourceFonts(ttLib.TTFont, recalcBBoxes=False)

        Args:
            opener (Callable): takes one required positional argument, the source.path,
                and an optional list of keyword arguments, and returns a new font object
                loaded from the path.
            **kwargs: extra options passed on to the opener function.

        Returns:
            List of font objects in the order they appear in the sources list.
        """
        # we load fonts with the same source.path only once
        loaded = {}
        fonts = []
        for source in self.sources:
            if source.font is not None:  # font already loaded
                fonts.append(source.font)
                continue
            if source.path in loaded:
                source.font = loaded[source.path]
            else:
                if source.path is None:
                    raise DesignSpaceDocumentError(
                        "Designspace source '%s' has no 'path' attribute"
                        % (source.name or "<Unknown>")
                    )
                source.font = opener(source.path, **kwargs)
                loaded[source.path] = source.font
            fonts.append(source.font)
        return fonts
