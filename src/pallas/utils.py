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
Assorted helpers.
"""

from __future__ import annotations

from typing import Iterator, Optional


class Fibonacci:

    max_value: Optional[int]

    def __init__(self, *, max_value: Optional[int] = None) -> None:
        self.max_value = max_value

    def __iter__(self) -> Iterator[int]:
        a = b = 1
        while self.max_value is None or a < self.max_value:
            yield a
            a, b = b, a + b
        while True:
            yield self.max_value


def truncate_str(v: str, max_length: int = 80) -> str:
    """Trim the given text if too long."""
    if len(v) <= max_length:
        return v
    head = max_length * 2 // 3
    tail = max_length - head - 3
    return v[:head] + "..." + v[-tail:]
