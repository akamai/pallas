import pytest

import pallas


@pytest.fixture
def athena():
    return pallas.Athena.from_environ(prefix="TEST_PALLAS")
