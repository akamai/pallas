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
Exceptions raised when querying Athena.
"""

from __future__ import annotations

import re
from typing import Optional


class AthenaQueryError(Exception):
    """Athena query failed."""

    #: State of the query execution (FAILED or CANCELLED)
    state: str

    #: Reason of the state of the query execution.
    state_reason: Optional[str]

    def __init__(self, state: str, state_reason: Optional[str]):
        self.state = state
        self.state_reason = state_reason

    def __str__(self) -> str:
        """
        Report query state with its reason.
        """
        if self.state_reason is not None:
            return f"Athena query {self.state.lower()}: {self.state_reason}"
        return f"Athena query {self.state.lower()}"


class DatabaseNotFoundError(AthenaQueryError):
    """
    Athena database does not exist.

    Pallas maps string errors returned by Athena to exception classes.
    """


class TableNotFoundError(AthenaQueryError):
    """
    Athena table does not exist.

    Pallas maps string errors returned by Athena to exception classes.
    """


error_map = [
    (re.compile("Schema (.*) does not exist"), DatabaseNotFoundError),  # Presto
    (re.compile("Database does not exist: (.*)"), DatabaseNotFoundError),  # Hive
    (re.compile("Table (.*) does not exist"), TableNotFoundError),  # Presto
    (re.compile("Table not found (.*)"), TableNotFoundError),  # Hive
]


def get_error(state: str, state_reason: Optional[str]) -> AthenaQueryError:
    error = AthenaQueryError
    if state_reason is not None:
        for pattern, patter_error in error_map:
            if pattern.search(state_reason):
                error = patter_error
                break
    return error(state, state_reason)
