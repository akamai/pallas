import pytest

from pallas.testing import AthenaFake


@pytest.fixture
def fake_athena():
    return AthenaFake()
