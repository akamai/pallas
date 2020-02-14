import os
import secrets
from urllib.parse import urlsplit

import boto3
import pytest


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
    base_uri = os.environ.get("TEST_PALLAS_OUTPUT_LOCATION")
    if base_uri is None:
        pytest.skip("Skipping S3 integration tests.")
        return  # Mypy does not recognize that pytest.skip never returns.
    if base_uri and not base_uri.endswith("/"):
        base_uri += "/"
    token = secrets.token_hex(4)  # Unique path allows parallel test runs.
    uri = base_uri + f"test-pallas-{token}"
    yield uri
    _s3_recursive_delete(uri)


@pytest.fixture(name="s3_tmp_uri")
def s3_tmp_uri_fixture(s3_session_tmp_uri):
    if s3_session_tmp_uri and not s3_session_tmp_uri.endswith("/"):
        s3_session_tmp_uri += "/"
    return s3_session_tmp_uri + secrets.token_hex(8)
