"""
Usability helpers for querying Athena from Jupyter Notebook.
"""

from __future__ import annotations

import textwrap

from pallas.base import AthenaWrapper, Query, QueryWrapper


def normalize_sql(sql: str) -> str:
    lines = sql.splitlines()
    stripped = (l.rstrip() for l in lines)
    nonempty = (l for l in stripped if l)
    joined = "\n".join(nonempty)
    return textwrap.dedent(joined)


class AthenaNormalizationWrapper(AthenaWrapper):
    """
    Athena wrapper that normalizes executed queries.

    Query normalization can improve caching.

    Following normalization operations are done:
    - Trailing whitespace is removed from end of lines.
    - Empty lines and lines with whitespace only are removed.
    - Common indentation is removed.
    - Line endings are normalized to LF
    """

    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        normalized = normalize_sql(sql)
        return super().submit(normalized, ignore_cache=ignore_cache)


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


class QueryKillOnInterruptWrapper(QueryWrapper):
    """
    Query wrapper that kills queries on the KeyboardInterrupt exception.
    """

    def join(self) -> None:
        try:
            super().join()
        except KeyboardInterrupt:
            self.kill()
            super().join()
