
Tutorial
========

AWS credentials
---------------

Pallas uses boto3_ internally, so it reads `AWS credentials`_ from the standard locations:

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

.. _AWS credentials: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
.. _boto3: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html


Initialization
--------------

An :class:`.Athena` client can be obtained using the :func:`.setup` function.
All arguments are optional.

.. code-block:: python

    import pallas
    athena = pallas.setup(
        # AWS region, read from ~/.aws/config if not specified.
        region=None,
        # Athena (AWS Glue) database.
        database=None,
        # Athena workgroup. Will use default workgroup if omitted.
        workgroup=None,
        # Athena output location, will use workgroup default location if omitted.
        output_location="s3://...",
        # Optional query execution cache.
        cache_remote="s3://...",
        # Optional query result cache.
        cache_local="~/Notebooks/.cache/",
        # Whether to return failed queries from cache. Defaults to False.
        cache_failed=False,
        # Normalize white whitespace for better caching. Enabled by default.
        normalize=True,
        # Kill queries on KeybordInterrupt. Enabled by default.
        kill_on_interrupt=True
    )


To avoid hardcoded configuration values, the :func:`.environ_setup` function
can initialize :class:`.Athena` from environment variables,
corresponding to arguments in the previous example:

.. code-block:: shell

    export PALLAS_REGION=
    export PALLAS_DATABASE=
    export PALLAS_WORKGROUP=
    export PALLAS_OUTPUT_LOCATION=
    export PALLAS_NORMALIZE=true
    export PALLAS_KILL_ON_INTERRUPT=true
    export PALLAS_CACHE_REMOTE=$PALLAS_OUTPUT_LOCATION
    export PALLAS_CACHE_LOCAL=~/Notebooks/.cache/
    export PALLAS_CACHE_FAILED=false


.. code-block:: python

    athena = pallas.environ_setup()

Pallas uses Python standard logging. You can use
:func:`.configure_logging` instead of :func:`logging.basicConfig`
to enable logging for Pallas only. At the DEBUG level, Pallas emits
logs with query status including an estimated price:

.. code-block:: python

    pallas.configure_logging(level="DEBUG")


Executing queries
-----------------

Use the :meth:`.Athena.execute` method to execute queries:

.. code-block:: python

    sql = "SELECT %s id, %s name, %s value"
    results = athena.execute(sql, (1, "foo", 3.14))

Pallas also support non-blocking query execution:

.. code-block:: python

    query = athena.submit(sql)  # Submit a query and return
    query.join()  # Wait for query completion.
    results = query.get_results()  # Retrieve results. Joins the query internally.

The result objects provides a list-like interface
and can be converted to a Pandas DataFrame:

.. code-block:: python

    df = results.to_df()


Caching
-------

AWS Athena stores query results in S3 and does not delete them, so all past results are cached implicitly.
To retrieve results of a past query, an ID of the query execution is needed.

Pallas can cache in two modes - remote and local:

* In the remote mode, Pallas stores IDs of query executions.
  Using that, it can download previous results from S3 when they are available.
* In the local mode, it copies query results. Thanks to that,
  locally cached queries can be executed without an internet connection.

.. note::

    Pallas is designed to promote reproducible analyses and data pipelines:

    * Using the local caching, it is possible to regularly restart Jupyter
      notebooks without waiting for or paying for additional Athena queries.
    * Thanks to the remote caching, results can be reproduced at a different
      machine by a different person.

    Reproducible queries should be deterministic.
    For example, if you query data that are ingested regularly,
    you should always filter on the date column.

    Pallas assumes that your queries are deterministic
    and does not invalidate its cache.


Caching configuration can be passed to :func:`.setup` or :func:`.environ_setup`,
as shown in the `Initialization`_ section.

After the initialization, caching can be customized later using the :attr:`.Athena.cache` property:

.. code-block:: python

    athena.cache.enabled = True  # Default
    athena.cache.read = True  # Can be set to False to write but not read the cache
    athena.cache.write = True  # Can be set to False to read but not write the cache
    athena.cache.local = "~/Notebooks/.cache/"
    athena.cache.remote = "s3://..."
    athena.cache.failed = True

Alternatively, the :meth:`.Athena.using` method can override a configuration
for selected queries only:

.. code-block:: python

    athena.using(cache_enabled=False).execute(...)


Only SELECT queries are cached.
