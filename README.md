
# Pallas – AWS Athena client

Pallas makes querying AWS Athena easy.
It was designed for analytical work in Jupyter Notebook,
but it can be useful for many other applications. 

Features:

 * Friendly interface.
 * Local caching – Query results can be cached locally, 
   so no data have to be downloaded when a Jupyter notebook is restarted.
 * Remote caching – Query IDs can be cached in S3,
   so team mates can run notebooks without incurring additional costs.
 * Speed – Results are downloaded directly from S3, 
   which is much faster than using Athena API.
 * Pandas integration - Results can be converted to Pandas DataFrame
   with correct data types mapped automatically.
 * Optional white space normalization for better caching.
 * Kills queries on KeyboardInterrupt.  


## Quick start

Pallas can be configured using environment variables

```shell script
export PALLAS_DATABASE=               # Has to be specified in queries if empty.
export PALLAS_WORKGROUP=              # Will use default workgroup if empty.
export PALLAS_OUTPUT_LOCATION=        # s3:// URI, will use default workgroup output location if empty.
export PALLAS_REGION=                 # AWS region, can be also specified in ~/.aws/config
export PALLAS_NORMALIZE=true          # Enabled by defafault
export PALLAS_KILL_ON_INTERRUPT=true  # Enabled by default
export PALLAS_CACHE_REMOTE=$PALLAS_OUTPUT_LOCATION
export PALLAS_CACHE_LOCAL=~/Notebooks/.cache/
```

Logging can be configured to monitor query status: 

```python
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
```

To construct a new Athena client, call `pallas.setup()`:

```python
import pallas
athena = pallas.setup()
```

By default, configuration is read from environment variables,
but it is possible to pass all options to the `pallas.setup()` function.  

Use `Athena.execute()` method to execute queries: 

```python
sql = """
    SELECT * FROM (
        VALUES (1, 'foo', 3.14), (2, 'bar', NULL)
    ) AS t (id, name, value)
"""
results = athena.execute(sql)
```
If you rerun same query, results should be read from cache.
Note that common indentation is removed for better cache utilization.    

Results can be converted to Pandas DataFrame:

```python
df = results.to_df()
```

Pallas support non-blocking query execution:

```python
query = athena.submit(sql)  # Submit a query and return
query.get_info()  # Check query status
query.join()  # Wait for query completion.
results = query.get_results()  # Retrieve results. Calls query.join() internally.
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
