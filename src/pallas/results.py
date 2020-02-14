from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union, overload

from pallas.conversions import convert_value

try:
    import pandas as pd
except ImportError:
    pd = None


QueryRecord = Dict[str, Any]


class QueryResults(Sequence[QueryRecord]):
    """
    Collection of Athena query results.

    Implements list-like interface.
    """

    _column_names: Tuple[str, ...]
    _column_types: Tuple[str, ...]
    _data: Sequence[Tuple[str, ...]]

    def __init__(
        self,
        column_names: Sequence[str],
        column_types: Sequence[str],
        data: Sequence[Sequence[str]],
    ) -> None:
        self._column_names = tuple(column_names)
        self._column_types = tuple(column_types)
        self._data = [tuple(row) for row in data]

    def __repr__(self) -> str:
        parts = [
            f"{len(self)} results",
            f"column_names={self.column_names!r}",
            f"column_types={self.column_types!r}",
        ]
        return f"<{type(self).__name__}: {', '.join(parts)}>"

    @overload
    def __getitem__(self, index: int) -> QueryRecord:
        ...

    @overload  # noqa: F811
    def __getitem__(self, index: slice) -> Sequence[QueryRecord]:
        ...

    def __getitem__(  # noqa: F811
        self, index: Union[int, slice]
    ) -> Union[QueryRecord, Sequence[QueryRecord]]:
        if isinstance(index, slice):
            return QueryResults(self.column_names, self.column_types, self.data[index])
        row = self._data[index]
        info = zip(self._column_names, self._column_types, row)
        return {cn: convert_value(ct, v) for cn, ct, v in info}

    def __len__(self) -> int:
        return len(self._data)

    @property
    def column_names(self) -> Tuple[str, ...]:
        return self._column_names

    @property
    def column_types(self) -> Tuple[str, ...]:
        return self._column_types

    @property
    def data(self) -> Sequence[Tuple[str, ...]]:
        return self._data

    def to_df(self) -> pd.DataFrame:
        return results_to_df(self)


def results_to_df(results: QueryResults) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("Pandas cannot be imported.")
    column_info = zip(results.column_names, results.column_types)
    data = results.data
    series = {}
    for i, (column_name, column_type) in enumerate(column_info):
        series[column_name] = pd.Series((row[i] for row in data))
    return pd.DataFrame(series)
