from __future__ import annotations

import textwrap
from typing import Optional

from pallas.base import Athena, Query


def normalize_sql(sql: str) -> str:
    lines = sql.splitlines()
    stripped = (l.rstrip() for l in lines)
    nonempty = (l for l in stripped if l)
    joined = "\n".join(nonempty)
    return textwrap.dedent(joined)


class AthenaNormalizationWrapper(Athena):
    """
    Athena wrapper that normalizes executed queries.

    Query normalization can improve caching.

    Following normalization operations are done:
    - Trailing whitespace is removed from end of lines.
    - Empty lines and lines with whitespace only are removed.
    - Common indentation is removed.
    - Line endings are normalized to LF
    """

    _wrapped: Athena

    def __init__(self, athena: Athena) -> None:
        self._wrapped = athena

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self._wrapped!r}>"

    @property
    def wrapped(self) -> Athena:
        return self._wrapped

    @property
    def database(self) -> Optional[str]:
        return self._wrapped.database

    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        normalized = normalize_sql(sql)
        return self._wrapped.submit(normalized, ignore_cache=ignore_cache)

    def get_query(self, execution_id: str) -> Query:
        return self._wrapped.get_query(execution_id)
