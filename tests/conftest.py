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

import secrets
from urllib.parse import urlsplit

import boto3
import pytest

from pallas.assembly import EnvironConfig

config = EnvironConfig(prefix="TEST_PALLAS")


@pytest.fixture(name="region_name", scope="session")
def region_name_fixture():
    # Do not raise or skip tests if region is not defined.
    # Region can be defined in ~/.aws/config.
    return config.get_str("REGION")


@pytest.fixture(name="athena_database", scope="session")
def athena_database_fixture():
    """
    Athena database.

    Tests depending on this fixture are skipped
    if the PALLAS_TEST_ATHENA_DATABASE environment variable is not defined.
    """
    database = config.get_str("DATABASE")
    if not database:
        pytest.skip(f"{config.key('DATABASE')} not defined.")
    return database


@pytest.fixture(name="athena_workgroup", scope="session")
def athena_workgroup_fixture():
    """
    Athena workgroup.
    """
    return config.get_str("WORKGROUP")


def _s3_recursive_delete(uri):
    scheme, netloc, path, query, fragment = urlsplit(uri)
    assert scheme == "s3"
    assert query == fragment == ""
    if path and not path.endswith("/"):
        path += "/"
    bucket = boto3.resource("s3").Bucket(netloc)
    for item in bucket.objects.filter(Prefix=path):
        item.delete()


@pytest.fixture(name="s3_session_tmp_uri", scope="session")
def s3_session_tmp_uri_fixture():
    """
    Base URI of a temporary S3 locations.

    Performs cleanup at the end of the test session.

    Tests depending on this fixture are skipped
    if the PALLAS_TEST_S3_TMP environment variable is not defined.
    """
    base_uri = config.get_str("OUTPUT_LOCATION")
    if not base_uri:
        pytest.skip(f"{config.key('OUTPUT_LOCATION')} not defined.")
    if base_uri and not base_uri.endswith("/"):
        base_uri += "/"
    token = secrets.token_hex(4)  # Unique path allows parallel test runs.
    uri = base_uri + f"test-pallas-{token}"
    yield uri
    _s3_recursive_delete(uri)


@pytest.fixture(name="s3_tmp_uri")
def s3_tmp_uri_fixture(s3_session_tmp_uri):
    """
    URI of a temporary S3 location than can be used for testing.

    A unique URI is generated for each test.
    Cleanup is performed at once at the end of the test session.

    Tests depending on this fixture are skipped
    if the PALLAS_TEST_S3_TMP environment variable is not defined.
    """
    if s3_session_tmp_uri and not s3_session_tmp_uri.endswith("/"):
        s3_session_tmp_uri += "/"
    return s3_session_tmp_uri + secrets.token_hex(8)
