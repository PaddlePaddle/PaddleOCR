u"""
Base classes for features that are backwards-incompatible.

Usage:
features = Features()
features.add(Feature("py3k_feature", "power< 'py3k' any* >", "2.7"))
PATTERN = features.PATTERN
"""

pattern_unformatted = u"%s=%s" # name=pattern, for dict lookups
message_unformatted = u"""
%s is only supported in Python %s and above."""

class Feature(object):
    u"""
    A feature has a name, a pattern, and a minimum version of Python 2.x
    required to use the feature (or 3.x if there is no backwards-compatible
    version of 2.x)
    """
    def __init__(self, name, PATTERN, version):
        self.name = name
        self._pattern = PATTERN
        self.version = version

    def message_text(self):
        u"""
        Format the above text with the name and minimum version required.
        """
        return message_unformatted % (self.name, self.version)

class Features(set):
    u"""
    A set of features that generates a pattern for the features it contains.
    This set will act like a mapping in that we map names to patterns.
    """
    mapping = {}

    def update_mapping(self):
        u"""
        Called every time we care about the mapping of names to features.
        """
        self.mapping = dict([(f.name, f) for f in iter(self)])

    @property
    def PATTERN(self):
        u"""
        Uses the mapping of names to features to return a PATTERN suitable
        for using the lib2to3 patcomp.
        """
        self.update_mapping()
        return u" |\n".join([pattern_unformatted % (f.name, f._pattern) for f in iter(self)])

    def __getitem__(self, key):
        u"""
        Implement a simple mapping to get patterns from names.
        """
        return self.mapping[key]
