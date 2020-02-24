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
