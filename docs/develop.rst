
Development
===========

Installation
------------

Pallas can be installed with development dependencies using pip:

.. code-block:: shell

    pip install -e .[dev]


Configuration
-------------

For integration test to run, access to AWS resources has to be configured.

.. code-block:: shell

    export TEST_PALLAS_REGION=            # AWS region, can be also specified in ~/.aws/config
    export TEST_PALLAS_DATABASE=          # Name of Athena database
    export TEST_PALLAS_WORKGROUP=         # Optional
    export TEST_PALLAS_OUTPUT_LOCATION=   # s3:// URI

If the above environment variables are not defined, integration tests will be skipped.


Tools
-----

* Code is checked with flake8_ and Mypy_.
* Tests are run using pytest_.
* Code is formatted using Black_ and isort_.
* Documentation is built using Sphinx_.

Tox_ can run the above tools:

.. code-block:: shell

    tox -e format
    tox --parallel


.. _Black: https://black.readthedocs.io
.. _flake8: https://flake8.pycqa.org
.. _isort: https://pycqa.github.io/isort/
.. _Mypy: http://mypy-lang.org
.. _pytest: https://docs.pytest.org/
.. _Sphinx: https://www.sphinx-doc.org/
.. _Tox: https://tox.readthedocs.io/
