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

from pallas.base import AthenaWrapper
from pallas.results import QueryResults


class AthenaKillOnInterruptWrapper(AthenaWrapper):
    """
    Athena wrapper that kills queries on the KeyboardInterrupt exception.
    """

    def get_query_results(self, execution_id: str) -> QueryResults:
        with self._kill_on_interrupt(execution_id):
            return self.wrapped.get_query_results(execution_id)

    def join_query(self, execution_id: str) -> None:
        with self._kill_on_interrupt(execution_id):
            return self.wrapped.join_query(execution_id)

    @contextmanager
    def _kill_on_interrupt(self, execution_id: str) -> Iterator[None]:
        try:
            yield
        except KeyboardInterrupt:
            self.kill_query(execution_id)
            self.join_query(execution_id)  # Wait until killed and raise
