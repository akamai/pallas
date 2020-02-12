import os

from pallas.proxies import AthenaProxy


def from_environ(environ=None, *, prefix="PALLAS"):
    if environ is None:
        environ = os.environ
    return AthenaProxy(
        output_location=environ[f"{prefix}_OUTPUT_LOCATION"],
        database=environ[f"{prefix}_DATABASE"],
        region_name=environ.get(f"{prefix}_REGION_NAME"),
    )
