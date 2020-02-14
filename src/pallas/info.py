from __future__ import annotations

from typing import cast, Any, Mapping, Optional

from pallas.exceptions import AthenaQueryError


class QueryInfo:
    def __init__(self, data: Mapping[str, Any]):
        self._data = data

    @property
    def execution_id(self) -> str:
        return cast(str, self._data["QueryExecutionId"])

    @property
    def sql(self) -> str:
        return cast(str, self._data["Query"])

    @property
    def database(self) -> Optional[str]:
        return cast(Optional[str], self._data["QueryExecutionContext"].get("Database"))

    @property
    def finished(self) -> bool:
        return self.state in ("SUCCEEDED", "FAILED", "CANCELLED")

    @property
    def succeeded(self) -> bool:
        return self.state == "SUCCEEDED"

    @property
    def state(self) -> str:
        return cast(str, self._data["Status"]["State"])

    @property
    def state_reason(self) -> Optional[str]:
        return cast(Optional[str], self._data["Status"].get("StateChangeReason"))

    def check(self) -> None:
        if self.finished and not self.succeeded:
            raise AthenaQueryError(self.state, self.state_reason)
