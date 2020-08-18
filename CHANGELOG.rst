
Changelog
=========

v0.4.dev
--------

* Add support for parametrized queries.
* More options for cache configuration.
* Allow to override configuration of the Athena class after it is initialized.
* Refactored implementation from layered decorators to one class using specialized  helpers.


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
