"""Python module which parses and emits TOML.

Released under the MIT license.
"""

from pip._vendor.toml import encoder
from pip._vendor.toml import decoder

__version__ = "0.10.0"
_spec_ = "0.5.0"

load = decoder.load
loads = decoder.loads
TomlDecoder = decoder.TomlDecoder
TomlDecodeError = decoder.TomlDecodeError

dump = encoder.dump
dumps = encoder.dumps
TomlEncoder = encoder.TomlEncoder
TomlArraySeparatorEncoder = encoder.TomlArraySeparatorEncoder
TomlPreserveInlineDictEncoder = encoder.TomlPreserveInlineDictEncoder
