# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Dict, Optional, TypeVar


try:
    from tmetry.writer import TmetryWriter
    from tmetry.simpleevent import SimpleEventRecord

    b_tmetry_available = True
except ImportError:
    b_tmetry_available = False

VTYPE = TypeVar("T", str, int, bool, float)


class EventLogger:
    """
    Base class for providing event logging in a path handler
    It implements event logging by wrapping the tmetry interface.
    If tmetry packages is not available, it is a no-op.
    """

    DEFAULT_TOPIC = "iopath_tmetry"

    def __init__(self, *args, **kwargs):
        if b_tmetry_available:
            self._writers = []
            self._evt = SimpleEventRecord()

    def add_writer(self, writer):
        if b_tmetry_available:
            if isinstance(writer, TmetryWriter):
                self._writers.append(writer)

    def add_key(self, key: str, val: VTYPE):
        if b_tmetry_available:
            self._evt.set(key, val)

    def add_keys(self, kvs: Dict[str, VTYPE]):
        if b_tmetry_available:
            self._evt.set_keys(kvs)

    def log_event(self, topic: Optional[str] = None):
        if b_tmetry_available:
            if topic is None:
                topic = self.DEFAULT_TOPIC

            for writer in self._writers:
                writer.writeRecord(topic, self._evt)
            del self._evt
            self._evt = SimpleEventRecord()
