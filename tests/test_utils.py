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

import itertools

from pallas.utils import Fibonacci, truncate_str


def test_fibonacci_wo_max_value():
    generator = Fibonacci()
    items = itertools.islice(generator, 12)
    assert list(items) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


def test_fibonacci_w_max_value():
    generator = Fibonacci(max_value=60)
    items = itertools.islice(generator, 12)
    assert list(items) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 60, 60]


def test_truncate_str_short():
    assert truncate_str("Hello world!") == "Hello world!"


def test_truncate_str_long():
    s = "Hello, " + 20 * "hello, " + "world!"
    assert truncate_str(s) == (
        "Hello, hello, hello, hello, hello, hello, hello,"
        " hell...lo, hello, hello, world!"
    )
