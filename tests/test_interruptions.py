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

import time
from contextlib import contextmanager

import pytest

from pallas import Athena
from pallas.exceptions import AthenaQueryError
from pallas.testing import FakeProxy


def fake_sleep(seconds):
    raise KeyboardInterrupt


@pytest.fixture(name="proxy")
def proxy_fixture():
    proxy = FakeProxy()
    proxy.state = "RUNNING"
    return proxy


@pytest.fixture(name="athena")
def athena_fixture(proxy):
    orig_sleep = time.sleep
    time.sleep = fake_sleep
    athena = Athena(proxy)
    athena.kill_on_interrupt = True
    yield athena
    time.sleep = orig_sleep


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


class TestAthenaKillOnInterrupt:
    def test_execute_kill_on_interrupt(self, athena, proxy):
        with no_keyboard_interrupt():
            with pytest.raises(AthenaQueryError):
                athena.execute("SELECT 1")
            assert proxy.request_log == [
                "StartQueryExecution",
                "GetQueryExecution",
                "StopQueryExecution",
                "GetQueryExecution",
            ]

    def test_execute_do_not_kill_on_interrupt(self, athena, proxy):
        athena.kill_on_interrupt = False
        with pytest.raises(KeyboardInterrupt):
            athena.execute("SELECT 1")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
        ]

    def test_get_results_kill_on_interrupt(self, athena, proxy):
        with no_keyboard_interrupt():
            query = athena.submit("SELECT 1")
            proxy.request_log.clear()
            with pytest.raises(AthenaQueryError):
                query.get_results()
            assert proxy.request_log == [
                "GetQueryExecution",
                "StopQueryExecution",
                "GetQueryExecution",
            ]

    def test_get_results_do_not_kill_on_interrupt(self, athena, proxy):
        athena.kill_on_interrupt = False
        query = athena.submit("SELECT 1")
        proxy.request_log.clear()
        with pytest.raises(KeyboardInterrupt):
            query.get_results()
        assert proxy.request_log == [
            "GetQueryExecution",
        ]

    def test_join_kill_on_interrupt(self, athena, proxy):
        with no_keyboard_interrupt():
            query = athena.submit("SELECT 1")
            proxy.request_log.clear()
            with pytest.raises(AthenaQueryError):
                query.join()
            assert proxy.request_log == [
                "GetQueryExecution",
                "StopQueryExecution",
                "GetQueryExecution",
            ]

    def test_join_do_not_kill_on_interrupt(self, athena, proxy):
        athena.kill_on_interrupt = False
        query = athena.submit("SELECT 1")
        proxy.request_log.clear()
        with pytest.raises(KeyboardInterrupt):
            query.join()
            assert proxy.request_log == [
                "GetQueryExecution",
            ]
