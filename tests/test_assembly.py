from pallas.assembly import setup
from pallas.caching import AthenaCachingWrapper
from pallas.normalization import AthenaNormalizationWrapper
from pallas.proxies import AthenaProxy


class TestSetup:
    def test_default(self):
        athena = setup(output_location="s3://example-output/")
        assert isinstance(athena, AthenaProxy)
        assert athena.output_location == "s3://example-output/"

    def test_cache_remote(self):
        athena = setup(
            output_location="s3://example-output/", cache_remote="s3://bucket/path/",
        )
        assert isinstance(athena, AthenaCachingWrapper)
        assert athena.storage.uri == "s3://bucket/path/"
        assert not athena.cache_results
        assert isinstance(athena.wrapped, AthenaProxy)

    def test_cache_local(self):
        athena = setup(output_location="s3://example-output/", cache_local="/path")
        assert isinstance(athena, AthenaCachingWrapper)
        assert athena.storage.uri == "file:/path"
        assert athena.cache_results
        assert isinstance(athena.wrapped, AthenaProxy)

    def test_cache_remote_and_local(self):
        athena = setup(
            output_location="s3://example-output/",
            cache_remote="s3://bucket/path/",
            cache_local="/path",
        )
        assert isinstance(athena, AthenaCachingWrapper)
        assert athena.storage.uri == "file:/path"
        assert athena.cache_results
        assert isinstance(athena.wrapped, AthenaCachingWrapper)
        assert athena.wrapped.storage.uri == "s3://bucket/path/"
        assert not athena.wrapped.cache_results
        assert isinstance(athena.wrapped.wrapped, AthenaProxy)

    def test_normalize(self):
        athena = setup(output_location="s3://example-output/", normalize=True)
        assert isinstance(athena, AthenaNormalizationWrapper)
        assert isinstance(athena.wrapped, AthenaProxy)

    def test_normalize_and_cache_remote(self):
        athena = setup(
            output_location="s3://example-output/",
            cache_remote="s3://bucket/path/",
            normalize=True,
        )
        assert isinstance(athena, AthenaNormalizationWrapper)
        assert isinstance(athena.wrapped, AthenaCachingWrapper)

    def test_normalize_and_cache_local(self):
        athena = setup(
            output_location="s3://example-output/", cache_remote="/path", normalize=True
        )
        assert isinstance(athena, AthenaNormalizationWrapper)
        assert isinstance(athena.wrapped, AthenaCachingWrapper)
