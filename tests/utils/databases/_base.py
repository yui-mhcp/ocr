# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Shared helpers for the generic ``utils.databases`` test-suite.

This module is **not** collected by pytest (it does not match ``test_*.py``). It
declares the list of database backends exercised by ``test_database.py`` and small
builders, so that every ``backend x primary-key`` combination is driven through the
very same generic tests. Adding a new implementation is a one-line entry in
``BACKENDS`` (plus its own file under ``utils/databases/``).

Kept dependency-light : only ``numpy`` at import time — no keras / tensorflow.
"""

from dataclasses import dataclass, field

from utils.databases import (
    JSONDatabase, JSONDir, JSONLDatabase, DatabaseWrapper, OrderedDatabaseWrapper
)


@dataclass
class Backend:
    """ Description of a concrete ``Database`` implementation to test generically.

        - ``kwargs``  : extra constructor kwargs (e.g. the wrapped ``database`` for wrappers)
        - ``ordered`` : whether positional (int / slice) indexing is supported
        - ``append_only`` : whether ``update`` / ``pop`` / ``filter`` raise ``TypeError``
                            (mutating tests are skipped, see the ``mutable`` fixture)
    """
    id      : str
    cls     : type
    kwargs  : dict  = field(default_factory = dict)
    ordered : bool  = False
    append_only : bool  = False


BACKENDS = [
    Backend('json',     JSONDatabase),
    Backend('json_dir', JSONDir),
    Backend('wrapper',  DatabaseWrapper,        {'database' : 'JSONDatabase'}),
    Backend('ordered',  OrderedDatabaseWrapper, {'database' : 'JSONDatabase'}, ordered = True),
    Backend('jsonl',    JSONLDatabase, append_only = True),
    Backend('jsonl_buffered', JSONLDatabase, {'autoflush' : False}, append_only = True),
]

# primary-key arities exercised by the suite
SINGLE_KEY = 'id'
MULTI_KEY  = ('source', 'idx')

KEYS = ['single', 'multi']


def primary_key_for(kind):
    """ The ``primary_key`` argument for the given arity (``'single'`` / ``'multi'``). """
    return SINGLE_KEY if kind == 'single' else MULTI_KEY

def entry_id(kind, i):
    """ Expected identifier (as returned by ``insert`` / accepted by ``__contains__``) for row ``i``.

        Note : all identifiers are stringified by ``Database._get_entry`` (hence ``str(i)``).
    """
    return 'id-{}'.format(i) if kind == 'single' else ('src', str(i))

def make_entry(kind, i, ** extra):
    """ Build one entry (``dict``) for row ``i`` with a deterministic ``text`` column.

        Primary-key components are strings on purpose : the database stringifies every
        identifier (``Database._get_entry``), so a retrieved value carries string keys.
    """
    data = {'text' : 'text {}'.format(i), ** extra}
    if kind == 'single':
        data['id'] = 'id-{}'.format(i)
    else:
        data['source'], data['idx'] = 'src', str(i)
    return data

def make_entries(kind, n, ** extra):
    """ Build ``n`` entries (rows ``0 .. n-1``). """
    return [make_entry(kind, i, ** extra) for i in range(n)]
