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
Encapsulation of information returned by Athena GetQueryExecution method.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping, Optional, cast

from pallas.exceptions import AthenaQueryError

unit_prefixes = ["k", "M", "G", "T"]


def format_price(v: float) -> str:
    """
    Format price in dollars.
    """
    if v > 1:
        return "$%.2f" % v
    return "%.2f¢" % (100 * v)


def format_size(v: float) -> str:
    """
    Format size in bytes.

    This function assumes that 1kB = 1000B.
    """
    if v < 1000:
        return f"{v:.0f}B"
    for prefix in unit_prefixes:
        v /= 1000
        if v < 1000:
            break
    return f"{v:.2f}{prefix}B"


def format_time(v: dt.timedelta) -> str:
    """
    Format time.
    """
    if v < dt.timedelta(minutes=1):
        s = v.total_seconds()
        return f"{s:.1f}s"
    m, s = divmod(v.seconds, 60)
    return f"{m}min {s}s"


class QueryInfo:
    """
    Information about query execution.

    Provides access to data returned by Athena GetQueryExecution API method.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {str(self)!r}>"

    def __str__(self) -> str:
        return (
            f"{self.state}, "
            f"scanned {format_size(self.scanned_bytes)} "
            f"in {format_time(self.execution_time)}, "
            f"approx. price {format_price(self.approx_price)}"
        )

    @property
    def execution_id(self) -> str:
        return cast(str, self._data["QueryExecutionId"])

    @property
    def sql(self) -> str:
        rv = self._data["Query"]
        return cast(str, rv)

    @property
    def output_location(self) -> str:
        rv = self._data["ResultConfiguration"].get("OutputLocation")
        return cast(str, rv)

    @property
    def database(self) -> Optional[str]:
        rv = self._data["QueryExecutionContext"].get("Database")
        return cast(Optional[str], rv)

    @property
    def finished(self) -> bool:
        return self.state in ("SUCCEEDED", "FAILED", "CANCELLED")

    @property
    def succeeded(self) -> bool:
        return self.state == "SUCCEEDED"

    @property
    def state(self) -> str:
        rv = self._data["Status"]["State"]
        return cast(str, rv)

    @property
    def state_reason(self) -> Optional[str]:
        rv = self._data["Status"].get("StateChangeReason")
        return cast(Optional[str], rv)

    @property
    def scanned_bytes(self) -> int:
        rv = self._data["Statistics"].get("DataScannedInBytes", 0)
        return cast(int, rv)

    @property
    def execution_time(self) -> dt.timedelta:
        milliseconds = self._data["Statistics"].get("TotalExecutionTimeInMillis", 0)
        return dt.timedelta(milliseconds=milliseconds)

    @property
    def approx_price(self) -> float:
        price_per_tb = 5  # https://aws.amazon.com/athena/pricing/
        return price_per_tb * self.scanned_bytes / 10 ** 12

    def check(self) -> None:
        if self.finished and not self.succeeded:
            raise AthenaQueryError(self.state, self.state_reason)
