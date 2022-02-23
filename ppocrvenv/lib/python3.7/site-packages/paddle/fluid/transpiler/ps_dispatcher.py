#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function


class PSDispatcher(object):
    """
    PSDispatcher is the base class for dispatching vars
    into different pserver instance.
    You need to implement the `dispatch` interface.
    """

    def __init__(self, pserver_endpoints):
        self._eps = pserver_endpoints
        self._step = 0

    @property
    def eps(self):
        return self._eps

    def reset(self):
        """
        reset the step counter, set it zero.
        """
        self._step = 0

    def dispatch(self, varlist):
        """
        Args:
            varlist(list): a list of Variables
        Returns:
            a map of pserver endpoint -> varname
        """
        AssertionError("Interface has not been implemented.")


class HashName(PSDispatcher):
    """
	:api_attr: Static Graph

    Hash variable names to several endpoints using python
    "hash()" function.

    Args:
        pserver_endpoints (list): list of endpoint(ip:port).

    Examples:
        .. code-block:: python

        pserver_endpoints = ["127.0.0.1:6007", "127.0.0.1:6008"]
        vars = ["var1","var2","var3","var4","var5"]

        rr = RoundRobin(pserver_endpoints)
        rr.dispatch(vars)

    """

    def __init__(self, pserver_endpoints):
        super(self.__class__, self).__init__(pserver_endpoints)

    def _hash_block(self, block_str, total):
        return hash(block_str) % total

    def dispatch(self, varlist):
        """
        use `HashName` method to dispatch variables with each parameter server.
        Args:
            varlist (list): a list of Variables

        """
        eplist = []
        for var in varlist:
            server_id = self._hash_block(var.name(), len(self._eps))
            server_for_param = self._eps[server_id]
            eplist.append(server_for_param)
        return eplist


class RoundRobin(PSDispatcher):
    """
	:api_attr: Static Graph

    Distribute variables to several endpoints using
    RondRobin<https://en.wikipedia.org/wiki/Round-robin_scheduling> method.

    Args:
        pserver_endpoints (list): list of endpoint(ip:port).

    Examples:
        .. code-block:: python

        pserver_endpoints = ["127.0.0.1:6007", "127.0.0.1:6008"]
        vars = ["var1","var2","var3","var4","var5"]

        rr = RoundRobin(pserver_endpoints)
        rr.dispatch(vars)

    """

    def __init__(self, pserver_endpoints):
        super(RoundRobin, self).__init__(pserver_endpoints)

    def dispatch(self, varlist):
        """
        use `RoundRobin` method to dispatch variables with each parameter server.
        Args:
            varlist (list): a list of Variables

        """
        eplist = []
        for var in varlist:
            server_for_param = self._eps[self._step]
            eplist.append(server_for_param)
            self._step += 1
            if self._step >= len(self._eps):
                self._step = 0
        return eplist
