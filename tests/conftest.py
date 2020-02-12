import pytest

import pallas


@pytest.fixture
def athena():
    return pallas.from_environ(prefix="TEST_PALLAS")
