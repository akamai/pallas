
Changelog
=========

v0.10 (2022-01-06)
------------------

* Failed or cancelled queries found in cached are ignored.
  Export ``PALLAS_CACHE_FAILED=true`` to use failed queries from the cache.
* A small optimization to avoid duplicate check of local cache.
* Use PEP 585 type annotations.
* Test with Python 3.9 and Python 3.10
* Test environment variables consistent with runtime environment variables.
* Refactor and cleanup of tests.


v0.9 (2021-03-03)
-----------------

* Better logging. Log summary at INFO level and details at DEBUG level.
  Add a helper for logging configuration.
* Include QueryExecutionId in exception messages.
* Fix conversion because Athena sometimes returns "real" instead of "float".


v0.8 (2020-10-06)
-----------------

* Remove deprecated ignore_cache parameter.
* Fix query execution ID not cached locally when cached remotely.


v0.7 (2020-08-31)
-----------------

* Export new exceptions introduced v0.6 to the top level module.


v0.6 (2020-08-31)
-----------------

* Raise :class:`.AthenaQueryError` subclasses when a database or a table is not found.
* Add more configuration options to the :meth:`.Athena.using` method.


v0.5 (2020-08-19)
-----------------

* Do not substitute parameters (require quoted percent signs) when no parameters are given.


v0.4 (2020-08-18)
-----------------

* Add support for parametrized queries.
* More options for cache configuration.
* Allow to override configuration of the Athena class after it is initialized.
* Refactored implementation from layered decorators to one class using specialized  helpers.
* New documentation.
* All public (documented) functions and classes are available the top-level module.


v0.3 (2020-06-18)
-----------------

* Athena and Query classes available from the top-level module (useful for type hints).
* AthenaQueryError from the top-level module.
* Fix: SELECT queries cached only when uppercase.
* Fix: Queries not killed on KeyboardInterrupt.


v0.2 (2020-06-02)
-----------------

* Cache SELECT statements only (starting with SELECT or WITH).
* Preserve empty lines in the middle of normalized queries.


v0.1 (2020-03-24)
-----------------

* Initial release.
