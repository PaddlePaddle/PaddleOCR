# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
# https://developers.google.com/protocol-buffers/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Provides a factory class for generating dynamic messages.

The easiest way to use this class is if you have access to the FileDescriptor
protos containing the messages you want to create you can just do the following:

message_classes = message_factory.GetMessages(iterable_of_file_descriptors)
my_proto_instance = message_classes['some.proto.package.MessageName']()
"""

__author__ = 'matthewtoia@google.com (Matt Toia)'

from google.protobuf.internal import api_implementation
from google.protobuf import descriptor_pool
from google.protobuf import message

if api_implementation.Type() == 'cpp':
  from google.protobuf.pyext import cpp_message as message_impl
else:
  from google.protobuf.internal import python_message as message_impl


# The type of all Message classes.
_GENERATED_PROTOCOL_MESSAGE_TYPE = message_impl.GeneratedProtocolMessageType


class MessageFactory(object):
  """Factory for creating Proto2 messages from descriptors in a pool."""

  def __init__(self, pool=None):
    """Initializes a new factory."""
    self.pool = pool or descriptor_pool.DescriptorPool()

    # local cache of all classes built from protobuf descriptors
    self._classes = {}

  def GetPrototype(self, descriptor):
    """Obtains a proto2 message class based on the passed in descriptor.

    Passing a descriptor with a fully qualified name matching a previous
    invocation will cause the same class to be returned.

    Args:
      descriptor: The descriptor to build from.

    Returns:
      A class describing the passed in descriptor.
    """
    if descriptor not in self._classes:
      result_class = self.CreatePrototype(descriptor)
      # The assignment to _classes is redundant for the base implementation, but
      # might avoid confusion in cases where CreatePrototype gets overridden and
      # does not call the base implementation.
      self._classes[descriptor] = result_class
      return result_class
    return self._classes[descriptor]

  def CreatePrototype(self, descriptor):
    """Builds a proto2 message class based on the passed in descriptor.

    Don't call this function directly, it always creates a new class. Call
    GetPrototype() instead. This method is meant to be overridden in subblasses
    to perform additional operations on the newly constructed class.

    Args:
      descriptor: The descriptor to build from.

    Returns:
      A class describing the passed in descriptor.
    """
    descriptor_name = descriptor.name
    result_class = _GENERATED_PROTOCOL_MESSAGE_TYPE(
        descriptor_name,
        (message.Message,),
        {
            'DESCRIPTOR': descriptor,
            # If module not set, it wrongly points to message_factory module.
            '__module__': None,
        })
    result_class._FACTORY = self  # pylint: disable=protected-access
    # Assign in _classes before doing recursive calls to avoid infinite
    # recursion.
    self._classes[descriptor] = result_class
    for field in descriptor.fields:
      if field.message_type:
        self.GetPrototype(field.message_type)
    for extension in result_class.DESCRIPTOR.extensions:
      if extension.containing_type not in self._classes:
        self.GetPrototype(extension.containing_type)
      extended_class = self._classes[extension.containing_type]
      extended_class.RegisterExtension(extension)
    return result_class

  def GetMessages(self, files):
    """Gets all the messages from a specified file.

    This will find and resolve dependencies, failing if the descriptor
    pool cannot satisfy them.

    Args:
      files: The file names to extract messages from.

    Returns:
      A dictionary mapping proto names to the message classes. This will include
      any dependent messages as well as any messages defined in the same file as
      a specified message.
    """
    result = {}
    for file_name in files:
      file_desc = self.pool.FindFileByName(file_name)
      for desc in file_desc.message_types_by_name.values():
        result[desc.full_name] = self.GetPrototype(desc)

      # While the extension FieldDescriptors are created by the descriptor pool,
      # the python classes created in the factory need them to be registered
      # explicitly, which is done below.
      #
      # The call to RegisterExtension will specifically check if the
      # extension was already registered on the object and either
      # ignore the registration if the original was the same, or raise
      # an error if they were different.

      for extension in file_desc.extensions_by_name.values():
        if extension.containing_type not in self._classes:
          self.GetPrototype(extension.containing_type)
        extended_class = self._classes[extension.containing_type]
        extended_class.RegisterExtension(extension)
    return result


_FACTORY = MessageFactory()


def GetMessages(file_protos):
  """Builds a dictionary of all the messages available in a set of files.

  Args:
    file_protos: Iterable of FileDescriptorProto to build messages out of.

  Returns:
    A dictionary mapping proto names to the message classes. This will include
    any dependent messages as well as any messages defined in the same file as
    a specified message.
  """
  # The cpp implementation of the protocol buffer library requires to add the
  # message in topological order of the dependency graph.
  file_by_name = {file_proto.name: file_proto for file_proto in file_protos}
  def _AddFile(file_proto):
    for dependency in file_proto.dependency:
      if dependency in file_by_name:
        # Remove from elements to be visited, in order to cut cycles.
        _AddFile(file_by_name.pop(dependency))
    _FACTORY.pool.Add(file_proto)
  while file_by_name:
    _AddFile(file_by_name.popitem()[1])
  return _FACTORY.GetMessages([file_proto.name for file_proto in file_protos])
