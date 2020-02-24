
# Pallas – AWS Athena client

Pallas makes querying AWS Athena easy.
It was designed for research and analysis in Jupyter Notebook,
but it can be useful for many other applications.

Features:

 * Friendly interface to AWS Athena.
 * Local caching – Query results can be cached locally,
   so no data have to be downloaded when a Jupyter notebook is restarted.
 * Remote caching – Query IDs can be cached in S3,
   so team mates can reproduce exact results without incurring additional costs.
 * Speed – Results are downloaded directly from S3,
   which is much faster than using Athena API.
 * Pandas integration - Results can be converted to Pandas DataFrame
   with correct data types mapped automatically.
 * Optional white space normalization for better caching.
 * Kills queries on KeyboardInterrupt.


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
Pallas can be setup using environment variables:

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
and can be converted to Pandas DataFrame:

```python
df = results.to_df()
```


## Development

Pallas can be installed with development dependecies using pip:

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
Code checks and testing is automated using tox:

```shell script
$ tox
```

[flake8]: https://flake8.pycqa.org/en/latest/
[Mypy]: http://mypy-lang.org
[pytest]: https://docs.pytest.org/en/latest/
