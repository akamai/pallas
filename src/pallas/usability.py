from __future__ import annotations

import textwrap

from pallas.base import AthenaWrapper, Query


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
