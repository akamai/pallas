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

from contextlib import contextmanager

import pytest

from pallas import Athena
from pallas.exceptions import AthenaQueryError
from pallas.testing import AthenaFake, QueryFake


class InterruptQueryFake(QueryFake):
    _interrupted: bool = False

    def join(self) -> None:
        if not self._interrupted:
            self._interrupted = True
            raise KeyboardInterrupt
        super().join()


class InterruptAthenaFake(AthenaFake):

    state = "CANCELLED"
    query_cls = InterruptQueryFake


@pytest.fixture(name="fake")
def fake_fixture():
    return InterruptAthenaFake()


@pytest.fixture(name="athena")
def athena_fixture(fake):
    return Athena(fake, kill_on_interrupt=True)


@contextmanager
def no_keyboard_interrupt():
    """
    Convert KeyboardInterrupt to test failures.

    When KeyboardInterrupt is not caught by AthenaKillOnInterruptWrapper,
    do not want to terminate the pytest process.
    """
    try:
        yield
    except KeyboardInterrupt:
        raise AssertionError("Unexpected KeyboardInterrupt")


class TestAthenaKillOnInterruptWrapper:
    def test_execute(self, athena, fake):
        with no_keyboard_interrupt():
            with pytest.raises(AthenaQueryError):
                athena.execute("SELECT 1")
            assert fake.request_log == [
                "StartQueryExecution",
                "StopQueryExecution",
                "GetQueryExecution",
            ]

    def test_get_results(self, athena, fake):
        with no_keyboard_interrupt():
            query = athena.submit("SELECT 1")
            fake.request_log.clear()
            with pytest.raises(AthenaQueryError):
                query.get_results()
            assert fake.request_log == [
                "StopQueryExecution",
                "GetQueryExecution",
            ]

    def test_join(self, athena, fake):
        with no_keyboard_interrupt():
            query = athena.submit("SELECT 1")
            fake.request_log.clear()
            with pytest.raises(AthenaQueryError):
                query.join()
            assert fake.request_log == [
                "StopQueryExecution",
                "GetQueryExecution",
            ]
