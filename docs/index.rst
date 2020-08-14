
Pallas documentation
==================================

Pallas makes querying AWS Athena easy.

It is especially valuable for analyses in Jupyter Notebook,
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


Table of Contents
-----------------

.. toctree::

    install
    tutorial
    api
    develop
    alternatives
    license


Indices and tables
..................

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
