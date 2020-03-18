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

from typing import Optional


class AthenaQueryError(Exception):
    """Athena query failed."""

    def __init__(self, state: str, state_reason: Optional[str]):
        self.state = state
        self.state_reason = state_reason

    def __str__(self) -> str:
        if self.state_reason is not None:
            return f"Athena query {self.state.lower()}: {self.state_reason}"
        return f"Athena query {self.state.lower()}"
