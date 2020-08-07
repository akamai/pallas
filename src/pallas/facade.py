# Copyright 2020 Akamai Technologies, Inc
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

from __future__ import annotations

from typing import Optional

from pallas.base import AthenaClient, AthenaWrapper
from pallas.caching import AthenaCachingWrapper
from pallas.interruptions import AthenaKillOnInterruptWrapper
from pallas.normalization import AthenaNormalizationWrapper
from pallas.results import QueryResults
from pallas.sql import quote
from pallas.storage import Storage


class Athena(AthenaWrapper):
    """
    Athena client
    """

    quote = staticmethod(quote)

    def __init__(
        self,
        client: AthenaClient,
        storage_remote: Optional[Storage] = None,
        storage_local: Optional[Storage] = None,
        normalize: bool = False,
        kill_on_interrupt: bool = False,
    ):
        if storage_remote is not None:
            client = AthenaCachingWrapper(
                client, storage=storage_remote, cache_results=False
            )
        if storage_local is not None:
            client = AthenaCachingWrapper(
                client, storage=storage_local, cache_results=True
            )
        if normalize:
            client = AthenaNormalizationWrapper(client)
        if kill_on_interrupt:
            client = AthenaKillOnInterruptWrapper(client)
        super().__init__(client)

    def execute(self, sql: str, *, ignore_cache: bool = False) -> QueryResults:
        """
        Execute a query and wait for results.

        This is a blocking method that waits until query finishes.
        Returns :class:`QueryResults`.

        :param sql: SQL query to be executed
        :param ignore_cache: do not load cached results
        :return: query results
        """
        return self.submit(sql, ignore_cache=ignore_cache).get_results()
