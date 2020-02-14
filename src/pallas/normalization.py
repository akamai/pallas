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


class AthenaNormalizationWrapper:
    """
    Athena wrapper that normalizes executed queries.

    Query normalization can improve caching.

    Following normalization operations are done:
    - Trailing whitespace is removed from end of lines.
    - Empty lines and lines with whitespace only are removed.
    - Common indentation is removed.
    - Line endings are normalized to LF
    """

    _inner_athena: Athena

    def __init__(self, athena: Athena) -> None:
        self._inner_athena = athena

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self._inner_athena!r}>"

    @property
    def database(self) -> Optional[str]:
        return self._inner_athena.database

    def submit(self, sql: str) -> Query:
        normalized = normalize_sql(sql)
        return self._inner_athena.submit(normalized)

    def get_query(self, execution_id: str) -> Query:
        return self._inner_athena.get_query(execution_id)
