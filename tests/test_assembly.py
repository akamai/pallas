from pallas.assembly import setup
from pallas.caching import AthenaCachingWrapper
from pallas.proxies import AthenaProxy
from pallas.usability import AthenaKillOnInterruptWrapper, AthenaNormalizationWrapper


class TestSetup:
    def test_default(self):
        athena = setup(environ={}, output_location="s3://example-output/")
        self.assert_default(athena)

    def test_default_from_env(self):
        athena = setup(environ={"PALLAS_OUTPUT_LOCATION": "s3://example-output/"})
        self.assert_default(athena)

    def assert_default(self, athena):
        assert isinstance(athena, AthenaKillOnInterruptWrapper)
        assert isinstance(athena.wrapped, AthenaNormalizationWrapper)
        assert isinstance(athena.wrapped.wrapped, AthenaProxy)
        assert athena.wrapped.wrapped.output_location == "s3://example-output/"

    def test_do_not_normalize(self):
        athena = setup(environ={}, normalize=False)
        self.assert_do_not_normalize(athena)

    def test_do_not_normalize_from_env(self):
        athena = setup(environ={"PALLAS_NORMALIZE": "0"})
        self.assert_do_not_normalize(athena)

    def assert_do_not_normalize(self, athena):
        assert isinstance(athena, AthenaKillOnInterruptWrapper)
        assert isinstance(athena.wrapped, AthenaProxy)

    def test_do_not_kill_on_interrupt(self):
        athena = setup(environ={}, kill_on_interrupt=False)
        self.asseet_do_not_kill_on_interrupt(athena)

    def test_do_not_kill_on_interrupt_from_env(self):
        athena = setup(environ={"PALLAS_KILL_ON_INTERRUPT": "0"})
        self.asseet_do_not_kill_on_interrupt(athena)

    def asseet_do_not_kill_on_interrupt(self, athena):
        assert isinstance(athena, AthenaNormalizationWrapper)
        assert isinstance(athena.wrapped, AthenaProxy)

    def test_cache_remote(self):
        athena = setup(environ={}, cache_remote="s3://bucket/path/")
        self.assert_cache_remote(athena)

    def test_cache_remote_from_env(self):
        athena = setup(environ={"PALLAS_CACHE_REMOTE": "s3://bucket/path/"})
        self.assert_cache_remote(athena)

    def assert_cache_remote(self, athena):
        caching_wrapper = athena.wrapped.wrapped
        assert isinstance(caching_wrapper, AthenaCachingWrapper)
        assert caching_wrapper.storage.uri == "s3://bucket/path/"
        assert not caching_wrapper.cache_results
        assert isinstance(caching_wrapper.wrapped, AthenaProxy)

    def test_cache_local(self):
        athena = setup(environ={}, cache_local="/path")
        self.assert_cache_local(athena)

    def test_cache_local_from_env(self):
        athena = setup(environ={"PALLAS_CACHE_LOCAL": "/path"})
        self.assert_cache_local(athena)

    def assert_cache_local(self, athena):
        caching_wrapper = athena.wrapped.wrapped
        assert isinstance(caching_wrapper, AthenaCachingWrapper)
        assert caching_wrapper.storage.uri == "file:/path"
        assert caching_wrapper.cache_results
        assert isinstance(caching_wrapper.wrapped, AthenaProxy)

    def test_cache_remote_and_local(self):
        athena = setup(cache_remote="s3://bucket/path/", cache_local="/path",)
        self.assert_cache_remote_and_local(athena)

    def test_cache_remote_and_local_from_env(self):
        athena = setup(
            environ={
                "PALLAS_CACHE_REMOTE": "s3://bucket/path/",
                "PALLAS_CACHE_LOCAL": "/path",
            }
        )
        self.assert_cache_remote_and_local(athena)

    def assert_cache_remote_and_local(self, athena):
        caching_wrapper = athena.wrapped.wrapped
        assert isinstance(caching_wrapper, AthenaCachingWrapper)
        assert caching_wrapper.storage.uri == "file:/path"
        assert caching_wrapper.cache_results
        assert isinstance(caching_wrapper.wrapped, AthenaCachingWrapper)
        assert caching_wrapper.wrapped.storage.uri == "s3://bucket/path/"
        assert not caching_wrapper.wrapped.cache_results
        assert isinstance(caching_wrapper.wrapped.wrapped, AthenaProxy)
