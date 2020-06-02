
# Pallas – AWS Athena client

Pallas makes querying AWS Athena easy.

We found it valuable for analyses in Jupyter Notebook,
but it is designed to be generic and usable in any application.

Features:

 * Friendly interface to AWS Athena.
 * Performance – Large results are downloaded directly from S3,
   which is much faster than using Athena API.
 * Pandas integration - Results can be converted to Pandas DataFrame
   with correct data types mapped automatically.
 * Local caching – Query results can be cached locally,
   so no data have to be downloaded when a Jupyter notebook is restarted.
 * Remote caching – Query IDs can be cached in S3,
   so team mates can reproduce results without incurring additional costs.
 * Fixes malformed results returned by Athena to DCL
   (for example DESCRIBE) queries.
 * Optional white space normalization for better caching.
 * Kills queries on KeyboardInterrupt.

## Installation

Pallas requires Python 3.7 or newer. It can be installed using pip:

```shell script
pip install --upgrade pallas
```

## Quick start

Athena client can be obtained using the ``pallas.setup()`` method.
All arguments are optional.

```python
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
    # Query execution cache.
    cache_remote="s3://...",
    # Query result cache.
    cache_local="~/Notebooks/.cache/",
    # Normalize white whitespace for better caching. Enabled by default.
    normalize=True,
    # Kill queries on KeybordInterrupt. Enabled by default.
    kill_on_interrupt=True
)
```

To avoid hardcoded configuration values,
Pallas can be setup using environment variables,
corresponding to arguments in the previous example:

```shell script
export PALLAS_DATABASE=
export PALLAS_WORKGROUP=
export PALLAS_OUTPUT_LOCATION=
export PALLAS_REGION=
export PALLAS_NORMALIZE=true
export PALLAS_KILL_ON_INTERRUPT=true
export PALLAS_CACHE_REMOTE=$PALLAS_OUTPUT_LOCATION
export PALLAS_CACHE_LOCAL=~/Notebooks/.cache/
```

```python
athena = pallas.environ_setup()
```
Python standard logging is available for monitoring:

```python
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
```

Use the `Athena.execute()` method to execute queries:

```python
sql = """
    SELECT * FROM (
        VALUES (1, 'foo', 3.14), (2, 'bar', NULL)
    ) AS t (id, name, value)
"""
results = athena.execute(sql)
```
If you rerun same query, results should be read from cache.

Pallas also support non-blocking query execution:

```python
query = athena.submit(sql)  # Submit a query and return
query.join()  # Wait for query completion.
results = query.get_results()  # Retrieve results. Calls query.join() internally.
```

The result objects provides a list-like interface
and can be converted to a Pandas DataFrame:

```python
df = results.to_df()
```

## Alternatives

### PyAthena

[PyAthena] is a Python DB API 2.0 (PEP 249) compliant client for Amazon Athena.
It is integrated with Pandas and SQLAlchemy.

The main difference between Pallas and PyAthena are the interfaces of the libraries.
Pallas does not implement the Python DB API. Instead, it adheres to the Athena REST API.

Pallas exposes an object representing a query execution.
Thanks to that, it can get back to queries executed in the past and retrieve their results.
One client natively supports both blocking and non-blocking execution.

PyAthena advantages:

 * PyAthena is older and more popular.
 * SQLAlchemy integration.
 * Standard Python DB API.
 * More configuration options.


Pallas advantages:

 * Pallas offers more powerful caching. It can cache results locally,
   and the cache is not limited to last N queries.
 * For better performance, Pallas downloads results directly from S3.
   PyAthena can also download results from S3, but it reads them using Pandas,
   failing to convert some data types.
 * Small helpers: smarter polling, query normalization,
   estimated price in logs, or kill on KeyboardInterrupt.
 * Nicer interface (from Pallas's author point of view).


### boto3

[boto3] is the official AWS SDK for Python. Pallas uses boto3 internally.

Querying Athena using boto3 directly is complicated and requires a lot of boilerplate code.


## Development

Pallas can be installed with development dependencies using pip:

```shell script
$ pip install -e .[dev]
```

Code is checked with [flake8] and [Mypy]. Tests are run using [pytest].

For integration test to run, access to AWS resources has to be configured:

```shell script
export PALLAS_TEST_REGION=            # AWS region, can be also specified in ~/.aws/config
export PALLAS_TEST_ATHENA_DATABASE=   # Name of Athena database
export PALLAS_TEST_ATHENA_WORKGROUP=  # Optional
export PALLAS_TEST_S3_TMP=            # s3:// URI
```
Code checks and testing are automated using tox:

```shell script
$ tox
```

[PyAthena]: https://github.com/laughingman7743/PyAthena
[boto3]: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
[flake8]: https://flake8.pycqa.org/en/latest/
[Mypy]: http://mypy-lang.org
[pytest]: https://docs.pytest.org/en/latest/


## Changelog

### v0.2

* Cache SELECT statements only (starting with SELECT or WITH)
* Preserve empty lines in the middle of normalized queries.

### v0.1

* Initial release.
