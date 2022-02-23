###############################################################################
# compat for UNIX 2.7 and 3.3
# Manager with LokyContext server.
# This avoids having a Manager using fork and breaks the fd.
#
# author: Thomas Moreau and Olivier Grisel
#
# based on multiprocessing/managers.py (17/02/2017)
#  * Overload the start method to use LokyContext and launch a loky subprocess
#

import multiprocessing as mp
from multiprocessing.managers import SyncManager, State
from .process import LokyProcess as Process


class LokyManager(SyncManager):
    def start(self, initializer=None, initargs=()):
        '''Spawn a server process for this manager object'''
        assert self._state.value == State.INITIAL

        if (initializer is not None
                and not hasattr(initializer, '__call__')):
            raise TypeError('initializer must be a callable')

        # pipe over which we will retrieve address of server
        reader, writer = mp.Pipe(duplex=False)

        # spawn process which runs a server
        self._process = Process(
            target=type(self)._run_server,
            args=(self._registry, self._address, bytes(self._authkey),
                  self._serializer, writer, initializer, initargs),
        )
        ident = ':'.join(str(i) for i in self._process._identity)
        self._process.name = type(self).__name__ + '-' + ident
        self._process.start()

        # get address of server
        writer.close()
        self._address = reader.recv()
        reader.close()

        # register a finalizer
        self._state.value = State.STARTED
        self.shutdown = mp.util.Finalize(
            self, type(self)._finalize_manager,
            args=(self._process, self._address, self._authkey,
                  self._state, self._Client),
            exitpriority=0
        )
