from typing import Optional

from pallas.base import Athena
from pallas.caching import AthenaCachingWrapper
from pallas.normalization import AthenaNormalizationWrapper
from pallas.proxies import AthenaProxy
from pallas.storage import storage_from_uri


def setup(
    output_location: str,
    database: Optional[str] = None,
    region_name: Optional[str] = None,
    cache_remote: Optional[str] = None,
    cache_local: Optional[str] = None,
    normalize: bool = False,
) -> Athena:
    """
    Assembly :class:`.Athena` instance.

    Initializes :class:`.AthenaProxy` and decorates it by caching wrappers.

    :param output_location: Athena output location
    :param database: Athena database to query
    :param region_name: AWS region
    :param cache_remote: set to cache query IDs
    :param cache_local: set to cache query results
    :param normalize: set to true to normalize executed SQL
    :return:
    """
    athena: Athena
    athena = AthenaProxy(
        output_location=output_location, database=database, region_name=region_name
    )
    if cache_remote is not None:
        remote_storage = storage_from_uri(cache_remote)
        athena = AthenaCachingWrapper(
            athena, storage=remote_storage, cache_results=False
        )
    if cache_local is not None:
        local_storage = storage_from_uri(cache_local)
        athena = AthenaCachingWrapper(athena, storage=local_storage, cache_results=True)
    if normalize:
        athena = AthenaNormalizationWrapper(athena)
    return athena
