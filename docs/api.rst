
API
===

This page describes the public API of the Pallas library.

All public functions and classes are imported to the top level :mod:`pallas` module.
Imports from internals of the package are not recommended and can break in future.


Assembly
--------

To construct an :class:`.Athena` client, use :func:`.setup` or :func:`.environ_setup` functions.

.. module:: pallas.assembly

.. autofunction:: setup

.. autofunction:: environ_setup

.. autofunction:: configure_logging


Client
------

The :class:`.Athena` class is a facade to all functionality offered by the library.

In the most common scenario, you may need only its :meth:`~.Athena.execute` method.
If you need to submit queries in a non-blocking fashion, you can use the
:meth:`~.Athena.submit` method, which returns a :class:`.Query` instance.
The same class is also returned by :meth:`~.Athena.get_query` method,
which can be useful if you want to get back to queries executed in the past.

.. module:: pallas.client

.. autoclass:: Athena
    :members:

.. autoclass:: Query
    :members:


Query information
-----------------

Information about query execution are returned as :class:`.QueryInfo` instances.
If you call :meth:`.Query.get_info` multiple times,
it can return different information as the query execution proceeds.

.. module:: pallas.info

.. autoclass:: QueryInfo
    :special-members: __str__
    :members:


Query results
-------------

Results of query executions are encapsulated by the :class:`.QueryResults` class.

.. module:: pallas.results

.. autoclass:: QueryResults
    :special-members: __getitem__, __len__
    :members:


Caching
-------

.. module:: pallas.caching

.. autoclass:: AthenaCache
    :members:


Exceptions
----------

Pallas can raise :class:`.AthenaQueryError` when a query fails.
For transport errors (typically connectivity problems or authorization failures),
:mod:`boto3` exceptions bubble unmodified.

.. module:: pallas.exceptions

.. autoclass:: AthenaQueryError
    :special-members: __str__
    :members:

.. autoclass:: DatabaseNotFoundError
    :show-inheritance:

.. autoclass:: TableNotFoundError
    :show-inheritance:
