
Pallas – Convenient Facade to AWS Athena
========================================

Development
-----------

The following environment variables has to be defined: ::

    export PALLAS_TEST_REGION=...           # AWS region
    export PALLAS_TEST_ATHENA_DATABASE=...  # Name of Athena database
    export PALLAS_TEST_S3_TMP=...           # s3:// URI



Tests can be run using tox_: ::

    tox

.. _tox: https://tox.readthedocs.io/en/latest/
