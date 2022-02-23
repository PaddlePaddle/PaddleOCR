# Copyright (c) 2010-2021 openpyxl


import posixpath

from openpyxl.descriptors import (
    String,
    Set,
    NoneSet,
    Alias,
    Sequence,
)
from openpyxl.descriptors.serialisable import Serialisable

from openpyxl.xml.constants import REL_NS, PKG_REL_NS
from openpyxl.xml.functions import (
    Element,
    fromstring,
    tostring
)


class Relationship(Serialisable):
    """Represents many kinds of relationships."""

    tagname = "Relationship"

    Type = String()
    Target = String()
    target = Alias("Target")
    TargetMode = String(allow_none=True)
    Id = String(allow_none=True)
    id = Alias("Id")


    def __init__(self,
                 Id=None,
                 Type=None,
                 type=None,
                 Target=None,
                 TargetMode=None
                 ):
        """
        `type` can be used as a shorthand with the default relationships namespace
        otherwise the `Type` must be a fully qualified URL
        """
        if type is not None:
            Type = "{0}/{1}".format(REL_NS, type)
        self.Type = Type
        self.Target = Target
        self.TargetMode = TargetMode
        self.Id = Id


class RelationshipList(Serialisable):

    tagname = "Relationships"

    Relationship = Sequence(expected_type=Relationship)


    def __init__(self, Relationship=()):
        self.Relationship = Relationship


    def append(self, value):
        values = self.Relationship[:]
        values.append(value)
        if not value.Id:
            value.Id = "rId{0}".format((len(values)))
        self.Relationship = values


    def __len__(self):
        return len(self.Relationship)


    def __bool__(self):
        return bool(self.Relationship)


    def find(self, content_type):
        """
        Find relationships by content-type
        NB. these content-types namespaced objects and different to the MIME-types
        in the package manifest :-(
        """
        for r in self.Relationship:
            if r.Type == content_type:
                yield r


    def __getitem__(self, key):
        for r in self.Relationship:
            if r.Id == key:
                return r
        raise KeyError("Unknown relationship: {0}".format(key))


    def to_tree(self):
        tree = Element("Relationships", xmlns=PKG_REL_NS)
        for idx, rel in enumerate(self.Relationship, 1):
            if not rel.Id:
                rel.Id = "rId{0}".format(idx)
            tree.append(rel.to_tree())

        return tree


def get_rels_path(path):
    """
    Convert relative path to absolutes that can be loaded from a zip
    archive.
    The path to be passed in is that of containing object (workbook,
    worksheet, etc.)
    """
    folder, obj = posixpath.split(path)
    filename = posixpath.join(folder, '_rels', '{0}.rels'.format(obj))
    return filename


from warnings import warn

def get_dependents(archive, filename):
    """
    Normalise dependency file paths to absolute ones

    Relative paths are relative to parent object
    """
    src = archive.read(filename)
    node = fromstring(src)
    try:
        rels = RelationshipList.from_tree(node)
    except TypeError:
        msg = "{0} contains invalid dependency definitions".format(filename)
        warn(msg)
        rels = RelationshipList()
    folder = posixpath.dirname(filename)
    parent = posixpath.split(folder)[0]
    for r in rels.Relationship:
        if r.TargetMode == "External":
            continue
        elif r.target.startswith("/"):
            r.target = r.target[1:]
        else:
            pth = posixpath.join(parent, r.target)
            r.target = posixpath.normpath(pth)
    return rels


def get_rel(archive, deps, id=None, cls=None):
    """
    Get related object based on id or rel_type
    """
    if not any([id, cls]):
        raise ValueError("Either the id or the content type are required")
    if id is not None:
        rel = deps[id]
    else:
        try:
            rel = next(deps.find(cls.rel_type))
        except StopIteration: # no known dependency
            return

    path = rel.target
    src = archive.read(path)
    tree = fromstring(src)
    obj = cls.from_tree(tree)

    rels_path = get_rels_path(path)
    try:
        obj.deps = get_dependents(archive, rels_path)
    except KeyError:
        obj.deps = []

    return obj
