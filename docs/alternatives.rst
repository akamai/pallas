
Alternatives
============

PyAthena_ and `AWS Data Wrangler`_ are good alternatives to Pallas.
They are more widespread and presumably more mature than Pallas.


Intro
-----

The main benefit of Pallas is the powerful caching designed
for workflows in Jypyter Notebook. Thanks to the local cache,
it is possible to restart notebooks often without waiting for data.
The cache in S3 allows to reproduce results from teammates
without incurring additional costs.

Pallas offers small but useful helpers.
Query normalization allows to write nicer (indented) code without impact on caching.
Estimated price in logs and kill on KeyboardInterrupt can help you to control costs.

Pallas has an opinionated API,
which does not implement Python DB API nor copies boto3_.

* Unlike Python DB API, Pallas interface embraces asynchronous execution.
  It allows to retrieve past queries by their ID and download old results.
* Pallas does not follow procedural style of boto3.
  A client object holds all necessary configuration,
  and query objects encapsulates everything related to query executions.


PyAthena
--------

PyAthena_ is a Python DB API 2.0 (PEP 249) compliant client for Amazon Athena.
It is integrated with Pandas and SQLAlchemy.


Pallas vs PyAthena
..................

* PyAthena is older and more popular.
* Pallas does not offer Python DB API or SQLAlchemy integration.
* PyAthena uses a distinct cursor type for execution in a background thread.
  Pallas can submit a query without waiting for results and
  offers a Query class for monitoring or joining the query.
* PyAthena can list last N queries when looking for cached results.
  Pallas can cache queries locally and to S3,
  so the cache is unlimited and can work offline.
* PyAthena downloads results directly from S3 only if PandasCursor is used.
* PyAthena uses Pandas for reading CSV files.
  Pallas implements own CSV parser with explicit mapping
  from Athena types to Pandas types.
* Pallas does not have helpers for creating new tables.


AWS Data Wrangler
-----------------

`AWS Data Wrangler`_ integrates Pandas with many AWS services, including Athena.

Interface of its Athena client is very similar to the boto3 API.
Function names copy function methods from boto3,
but invocation is simplified thanks to flattened arguments.

AWS Data Wrangler uses an interesting trick to obtain results in Parquet format.
Its CTAS approach rewrites ``SELECT`` queries to ``CREATE TABLE`` statements,
and then reads Parquet output from S3.
Advantages of the CTAS approach are performance
and handling of complex types that cannot be read from CSV.


Pallas vs AWS Data Wrangler
...........................

* AWS Data Wrangler is a part of AWS Labs
  and is managed by AWS Professional Services.
* Pallas does not offer the CTAS approach, but it downloads CSV files from S3.
  The main performance improvement comes from bypassing Athena API.
  CSV parsing can be slower than reading Parquet, but this difference
  should be negligible compared to the time spent downloading data.
* AWS Data Wrangler lists last N queries when looking for cached results.
  Pallas can cache queries locally and to S3,
  so the cache is unlimited and can work offline.
* AWS Data Wrangler uses on pyarrow for reading Parquet files
  and Pandas for reading CSV files.
  Pallas implements own CSV parser with explicit mapping
  from Athena types to Pandas types.
* Pallas does not mimic boto3 API, it provides object interface instead.
* Pallas misses helpers to call ``MSCK REPAIR TABLE`` or
  create an S3 bucket for AWS results.


boto3
-----

boto3_ is the official AWS SDK for Python. Pallas uses boto3 internally.

Querying Athena using boto3 directly is complicated and requires a lot of boilerplate code.


.. _boto3: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
.. _AWS Data Wrangler: https://github.com/awslabs/aws-data-wrangler
.. _PyAthena: https://github.com/laughingman7743/PyAthena
