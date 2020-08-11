# Copyright 2020 Akamai Technologies, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pallas.assembly import environ_setup, setup


class TestSetup:
    def test_default(self):
        athena = setup(output_location="s3://example-output/")
        assert athena.output_location == "s3://example-output/"
        assert athena.normalize
        assert athena.kill_on_interrupt

    def test_default_from_env(self):
        athena = environ_setup(
            environ={"PALLAS_OUTPUT_LOCATION": "s3://example-output/"}
        )
        assert athena.output_location == "s3://example-output/"
        assert athena.normalize
        assert athena.kill_on_interrupt

    def test_do_not_normalize(self):
        athena = setup(normalize=False)
        assert not athena.normalize

    def test_do_not_normalize_from_env(self):
        athena = environ_setup(environ={"PALLAS_NORMALIZE": "0"})
        assert not athena.normalize

    def test_do_not_kill_on_interrupt(self):
        athena = setup(kill_on_interrupt=False)
        assert not athena.kill_on_interrupt

    def test_do_not_kill_on_interrupt_from_env(self):
        athena = environ_setup(environ={"PALLAS_KILL_ON_INTERRUPT": "0"})
        assert not athena.kill_on_interrupt

    def test_cache_remote(self):
        athena = setup(cache_remote="s3://bucket/path/")
        self.assert_cache_remote(athena)

    def test_cache_remote_from_env(self):
        athena = environ_setup(environ={"PALLAS_CACHE_REMOTE": "s3://bucket/path/"})
        self.assert_cache_remote(athena)

    def assert_cache_remote(self, athena):
        assert athena.cache.remote.uri == "s3://bucket/path/"
        assert athena.cache.local is None

    def test_cache_local(self):
        athena = setup(cache_local="/path")
        self.assert_cache_local(athena)

    def test_cache_local_from_env(self):
        athena = environ_setup(environ={"PALLAS_CACHE_LOCAL": "/path"})
        self.assert_cache_local(athena)

    def assert_cache_local(self, athena):
        assert athena.cache.local.uri == "file:/path/"
        assert athena.cache.remote is None

    def test_cache_remote_and_local(self):
        athena = setup(cache_remote="s3://bucket/path/", cache_local="/path")
        self.assert_cache_remote_and_local(athena)

    def test_cache_remote_and_local_from_env(self):
        athena = environ_setup(
            environ={
                "PALLAS_CACHE_REMOTE": "s3://bucket/path/",
                "PALLAS_CACHE_LOCAL": "/path",
            }
        )
        self.assert_cache_remote_and_local(athena)

    def assert_cache_remote_and_local(self, athena):
        assert athena.cache.local.uri == "file:/path/"
        assert athena.cache.remote.uri == "s3://bucket/path/"
