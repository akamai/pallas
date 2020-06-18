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

"""
Interruption handling of Athena queries
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from pallas.base import AthenaWrapper, Query, QueryWrapper
from pallas.results import QueryResults


@contextmanager
def _kill_on_interrupt(query: Query) -> Iterator[None]:
    try:
        yield
    except KeyboardInterrupt:
        query.kill()
        query.join()  # Wait until killed and raise


class QueryKillOnInterruptWrapper(QueryWrapper):
    """
    Query wrapper that kills queries on the KeyboardInterrupt exception.
    """

    def get_results(self) -> QueryResults:
        with _kill_on_interrupt(self.wrapped):
            return self.wrapped.get_results()

    def join(self) -> None:
        with _kill_on_interrupt(self.wrapped):
            return self.wrapped.join()


class AthenaKillOnInterruptWrapper(AthenaWrapper):
    """
    Athena wrapper that kills queries on the KeyboardInterrupt exception.
    """

    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        query = super().submit(sql, ignore_cache=ignore_cache)
        return self._wrap_query(query)

    def get_query(self, execution_id: str) -> Query:
        query = super().get_query(execution_id)
        return self._wrap_query(query)

    def _wrap_query(self, query: Query) -> Query:
        return QueryKillOnInterruptWrapper(query)
