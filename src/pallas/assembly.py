from __future__ import annotations

from typing import Optional

from pallas.base import Athena
from pallas.caching import AthenaCachingWrapper
from pallas.proxies import AthenaProxy
from pallas.storage import storage_from_uri
from pallas.usability import AthenaKillOnInterruptWrapper, AthenaNormalizationWrapper


def setup(
    database: Optional[str] = None,
    workgroup: Optional[str] = None,
    output_location: Optional[str] = None,
    region_name: Optional[str] = None,
    cache_remote: Optional[str] = None,
    cache_local: Optional[str] = None,
    normalize: bool = False,
    kill_on_interrupt: bool = False,
) -> Athena:
    """
    Assembly :class:`.Athena` instance.

    Initializes :class:`.AthenaProxy` and decorates it by caching wrappers.

    :param database: a name of Athena database.
        If omitted, database should be specified in SQL.
    :param workgroup: a name of Athena workgroup.
        If omitted, default workgroup will be used.
    :param output_location: an output location at S3 for query results.
        Optional if a default location is specified for the *workgroup*.
    :param region_name: an AWS region.
        By default, region from AWS config is used.
    :param cache_remote: an URI of a remote cache.
        Query execution IDs without results are stored to the remote cache.
    :param cache_local: an URI of a local cache.
        Both results and query execution IDs are stored to the local cache.
    :param normalize: whether to normalize SQL
        Normalizes whitespace to improve caching.
    :param kill_on_interrupt: whether to kill queries on KeyboardInterrupt
        Kills query when interrupted during waiting.
    :return: an Athena instance
        A :class:`.AthenaProxy` instance wrapped necessary in decorators.
    """
    athena: Athena
    athena = AthenaProxy(
        database=database,
        workgroup=workgroup,
        output_location=output_location,
        region_name=region_name,
    )
    if cache_remote is not None:
        storage = storage_from_uri(cache_remote)
        athena = AthenaCachingWrapper(athena, storage=storage, cache_results=False)
    if cache_local is not None:
        storage = storage_from_uri(cache_local)
        athena = AthenaCachingWrapper(athena, storage=storage, cache_results=True)
    if normalize:
        athena = AthenaNormalizationWrapper(athena)
    if kill_on_interrupt:
        athena = AthenaKillOnInterruptWrapper(athena)
    return athena
