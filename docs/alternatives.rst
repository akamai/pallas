
Alternatives
============

PyAthena
--------

PyAthena_ is a Python DB API 2.0 (PEP 249) compliant client for Amazon Athena.
It is integrated with Pandas and SQLAlchemy.

The main difference between Pallas and PyAthena are the interfaces of the libraries.
Pallas does not implement the Python DB API. Instead, it adheres to the Athena REST API.

Pallas exposes an object representing a query execution.
Thanks to that, it can get back to queries executed in the past and retrieve their results.
One client natively supports both blocking and non-blocking execution.

PyAthena advantages:

* PyAthena is older and more popular.
* SQLAlchemy integration.
* More configuration options.
* Standard Python DB API.


Pallas advantages:

* Pallas offers more powerful caching. It can cache results locally,
  and the cache is not limited to last N queries.
* For better performance, Pallas downloads results directly from S3.
  PyAthena can also download results from S3, but it reads them using Pandas,
  failing to convert some data types.
* Small helpers: smarter polling, query normalization,
  estimated price in logs, or kill on KeyboardInterrupt.
* API designed for analyses in Jupyter notebooks and ETL pipelines.


boto3
-----

boto3_ is the official AWS SDK for Python. Pallas uses boto3 internally.

Querying Athena using boto3 directly is complicated and requires a lot of boilerplate code.


.. _boto3: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
.. _PyAthena: https://github.com/laughingman7743/PyAthena
