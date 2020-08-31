
Pallas – AWS Athena client
==========================

Pallas makes querying AWS Athena easy.

It is especially valuable for analyses in Jupyter Notebook,
but it is designed to be generic and usable in any application.


Main features:

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


.. code-block:: python

    import pallas
    athena = pallas.environ_setup()
    df = athena.execute("SELECT 'Hello world!").to_df()


Pallas is hosted at `GitHub <http://github.com/akamai/pallas>`_ and
it can be installed from `PyPI <https://pypi.org/project/pallas/>`_.


Documentation
-------------

Documentation in the ``docs/`` directory can be read online
at `Read the Docs <https://pallas.readthedocs.io/>`_.


Changelog
---------

Changelog is the ``CHANGELOG.rst`` file. It is also available
`online in docs <https://pallas.readthedocs.io/en/latest/changelog.html>`_.


License
-------

::

    Copyright 2020 Akamai Technologies, Inc

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


Contributing
------------

::

    By submitting a contribution (the “Contribution”) to this project,
    and for good and valuable consideration, the receipt and sufficiency of which
    are hereby acknowledged, you (the “Assignor”) irrevocably convey, transfer,
    and assign the Contribution to the owner of the repository (the “Assignee”),
    and the Assignee hereby accepts, all of your right, title, and interest in and
    to the Contribution along with all associated copyrights, copyright
    registrations, and/or applications for registration and all issuances,
    extensions and renewals thereof (collectively, the “Assigned Copyrights”).
    You also assign all of your rights of any kind whatsoever accruing under
    the Assigned Copyrights provided by applicable law of any jurisdiction,
    by international treaties and conventions and otherwise throughout the world.
