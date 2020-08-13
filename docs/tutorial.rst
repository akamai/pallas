
Tutorial
========

AWS credentials
---------------

Pallas uses boto3_ internally, so it reads `AWS credentials`_ from the standard locations.
This includes:

 * Shared credential file (``~/.aws/credentials``)
 * Environment variables (``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``)
 * Instance metadata service when run on an Amazon EC2 instance

The ``~/.aws/credentials`` file can be generated using the AWS CLI.

.. code-block:: shell

    aws configure


We recommend to use the AWS CLI to check the configuration.
If the AWS CLI is able to authenticate then Pallas should work too.

.. code-block:: shell

    aws sts get-caller-identity
    aws athena list-databases --catalog-name AwsDataCatalog


Initialization
--------------

Athena client can be obtained using the :func:`pallas.setup` function.
All arguments are optional.

.. code-block:: python

    import pallas
    athena = pallas.setup(
        # Athena (AWS Glue) database. Can be overridden in queries.
        database=None,
        # Athena workgroup. Will use default workgroup if omitted.
        workgroup=None,
        # Athena output location, will use workgroup default location if omitted.
        output_location="s3://...",
        # AWS region, read from ~/.aws/config if not specified.
        region=None,
        # Optional query execution cache.
        cache_remote="s3://...",
        # Optional query result cache.
        cache_local="~/Notebooks/.cache/",
        # Normalize white whitespace for better caching. Enabled by default.
        normalize=True,
        # Kill queries on KeybordInterrupt. Enabled by default.
        kill_on_interrupt=True
    )


To avoid hardcoded configuration values,
the :func:`pallas.environ_setup` function can be setup from environment variables,
corresponding to arguments in the previous example:

.. code-block:: shell

    export PALLAS_DATABASE=
    export PALLAS_WORKGROUP=
    export PALLAS_OUTPUT_LOCATION=
    export PALLAS_REGION=
    export PALLAS_NORMALIZE=true
    export PALLAS_KILL_ON_INTERRUPT=true
    export PALLAS_CACHE_REMOTE=$PALLAS_OUTPUT_LOCATION
    export PALLAS_CACHE_LOCAL=~/Notebooks/.cache/


.. code-block:: python

    athena = pallas.environ_setup()


Python standard logging can be configured to monitor query status,
including estimated query price:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.INFO)


Executing queries
-----------------

Use the :meth:`.Athena.execute` method to execute queries:

.. code-block:: python

    sql = "SELECT %s id, %s name, %s value"
    results = athena.execute(sql, (1, "foo", 3.14))

If you rerun same query, results should be read from cache.

Pallas also support non-blocking query execution:

.. code-block:: python

    query = athena.submit(sql)  # Submit a query and return
    query.join()  # Wait for query completion.
    results = query.get_results()  # Retrieve results. Calls query.join() internally.

The result objects provides a list-like interface
and can be converted to a Pandas DataFrame:

.. code-block:: python

    df = results.to_df()


.. _AWS credentials: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
.. _boto3: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
