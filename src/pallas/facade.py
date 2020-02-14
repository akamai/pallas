from __future__ import annotations

import os
from typing import Mapping, Optional

from pallas.base import Athena
from pallas.proxies import AthenaProxy


def from_environ(
    environ: Optional[Mapping[str, str]] = None, *, prefix: str = "PALLAS"
) -> Athena:
    if environ is None:
        environ = os.environ
    return AthenaProxy(
        output_location=environ[f"{prefix}_OUTPUT_LOCATION"],
        database=environ[f"{prefix}_DATABASE"],
        region_name=environ.get(f"{prefix}_REGION_NAME"),
    )
