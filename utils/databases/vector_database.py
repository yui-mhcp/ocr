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

import os
import numpy as np

from loggers import timer
from ..keras import ops
from ..embeddings.index import VectorIndex, init_index
from ..embeddings.index.vector_index import mask_to_indices
from .ordered_database_wrapper import OrderedDatabaseWrapper

class VectorDatabase(OrderedDatabaseWrapper):
    """
        Generic vector store : a regular `Database` (entries = `dict` with a primary key)
        associated to a `VectorIndex` (1 vector per entry, in insertion order : the **row** of
        an entry, see `rows`).

        The index configuration (backend + metric) is stored in the database config, and is
        therefore automatically restored when re-loading the database from its `path`.

        Search restrictions (`search`) :
            - `filters` on the entries' fields are evaluated **by column** : the values of a
              filtered field are cached as an array aligned with the rows (`column`), built at
              its first use and maintained by the insertions — an equality is vectorized, and a
              callable is only applied to the column values (never to copies of the entries)
            - `mask` : candidate rows computed by the caller (e.g. from its own state, see
              `rows`), combined with the filters
        `version` is incremented by every mutation : a caller caching rows (`rows`) or columns
        knows when they are stale.
    """
    def __init__(self,
                 path,
                 primary_key,
                 *,

                 index  = 'NumpyIndex',
                 metric = 'cosine',
                 database   = 'JSONDatabase',

                 vector_key = 'embedding',

                 ** kwargs
                ):
        super().__init__(path, primary_key, database = database, ** kwargs)

        self.vector_key = vector_key

        if not isinstance(index, VectorIndex):
            index = init_index(index, filename = self.index_path, metric = metric, ** kwargs)

        self._index = index

        self.version    = 0
        # `{field : list of values}` aligned with the rows (see `column`), for the filtered fields
        self._columns   = {}

    @property
    def index_path(self):
        if not self.path or not os.path.exists(self.path):
            return None
        elif os.path.isdir(self.path):
            return os.path.join(self.path, 'vectors.index')
        else:
            return os.path.splitext(self.path)[0] + '.index'

    @property
    def vectors(self):
        return self._index

    @property
    def embedding_dim(self):
        return self._index.embedding_dim

    ##############################
    #         Mutations          #
    ##############################

    def _changed(self, entries = None, *, reset = False):
        """ Bookkeeping of a mutation : new `version`, and the cached columns either extended by
            the inserted `entries` or dropped (`reset`, rebuilt at their next use) """
        self.version += 1
        if reset or not self._columns:
            self._columns = {}
        elif entries:
            values = self._db.multi_get(entries)
            for name, column in self._columns.items():
                column.extend(value.get(name, None) for value in values)

    def insert(self, data, ** kwargs):
        data    = data.copy()
        vector = data.pop(self.vector_key)
        entry  = super().insert(data, ** kwargs)
        self.vectors.add(vector)
        self._changed([entry])
        return entry

    def update(self, data):
        if self.vector_key in data:
            data = data.copy()
            data.pop(self.vector_key)

        result = super().update(data)
        if self._columns:
            # patches the updated fields in place (the row does not move)
            row, value = self.index(data), self._db.get(data)
            for name, column in self._columns.items():
                if name in data: column[row] = value.get(name, None)
        self.version += 1
        return result

    def pop(self, key, ** kwargs):
        """ Remove and return the given entry from the database """
        entry = self._get_entry(key)
        index = self.index(entry)
        item  = super().pop(entry, ** kwargs)
        self.vectors.remove(index)
        self._changed(reset = True)

        return item

    def multi_insert(self, iterable, /, vectors = None, ** kwargs):
        if vectors is None:
            iterable = [data.copy() for data in iterable]
            vectors  = np.array([data.pop(self.vector_key) for data in iterable])

        entries = super().multi_insert(iterable, ** kwargs)
        self.vectors.add(vectors)
        self._changed(entries)
        return entries

    def multi_update(self, iterable, /, ** kwargs):
        return [self.update(data) for data in iterable]

    def multi_pop(self, iterable, /, ** kwargs):
        entries = [self._get_entry(data) for data in iterable]
        indexes = [self.index(entry) for entry in entries]

        items   = super().multi_pop(entries)
        self.vectors.remove(indexes)
        self._changed(reset = True)

        return items

    ##############################
    #      Rows and columns      #
    ##############################

    def rows(self, keys):
        """ The rows (vector positions) of `keys` (entries or primary keys), `-1` if absent """
        keys = list(keys)
        get  = self._entry_to_idx.get
        return np.fromiter(
            (get(self._get_entry(key), -1) for key in keys), dtype = 'int64', count = len(keys)
        )

    def column(self, name):
        """ The values of the field `name` of every entry, as a (read-only) `np.ndarray` of
            objects aligned with the rows (`None` for an entry without this field) """
        if name not in self._columns:
            self._columns[name] = self._build_column(name)
        return _as_object_array(self._columns[name])

    def _build_column(self, name):
        if name == self.primary_key: return list(self._idx_to_entry)

        # fast path : the inner database iterates its values in the row order
        if list(self._db.keys()) == self._idx_to_entry:
            return list(self._db.get_column(name))
        return [value.get(name, None) for value in self._db.multi_get(self._idx_to_entry)]

    def filter_mask(self, filters):
        """
            Boolean mask over the rows of the entries matching every filter of `filters`
            (`{field : filter}`) :
                - a `callable`  : `filter(value)` is truthy
                - a `list` / `tuple` / `set` / `frozenset` / `np.ndarray` : the value is **one of**
                  the given values
                - any other value   : the value is equal to it (`None` matches a missing field)
        """
        mask = np.ones((len(self), ), dtype = bool)
        for name, filt in filters.items():
            mask &= _match_column(self.column(name), filt)
            if not mask.any(): break
        return mask

    ##############################
    #           Search           #
    ##############################

    @timer
    def search(self, query, k = 10, *, filters = None, mask = None, reverse = False, ** kwargs):
        """
            Returns the `k` entries most similar to (each of) the `query` vector(s)

            Arguments :
                - query : a 1-D or 2-D array of query embedding(s)
                - k     : the number of entries to retrieve per query
                - filters   : `dict` of field filters restricting the candidate entries (see
                              `filter_mask`), e.g. `{'source' : lambda s: s in sources}`
                - mask  : candidate rows (indices, or a boolean mask over the rows), combined
                          with `filters` ; negative indices are ignored (the `-1` of `rows`
                          for an absent key)
                - reverse   : whether to return the results in ascending score order
                              (i.e., most relevant last)
            Return :
                - results   : a list (1 item per query) of lists of entries (`dict`),
                              each enriched with a `score` entry
        """
        candidates = None
        if filters:
            candidates = self.filter_mask(filters)
        if mask is not None:
            restricted = np.zeros((len(self), ), dtype = bool)
            rows = mask_to_indices(mask)
            restricted[rows[rows >= 0]] = True
            candidates = restricted if candidates is None else candidates & restricted

        if not len(self) or (candidates is not None and not candidates.any()):
            query = self.vectors.prepare_embeddings(query)
            return [[] for _ in range(int(query.shape[0]))]

        indexes, scores = self.vectors.top_k(query, k = k, mask = candidates, ** kwargs)
        indexes = ops.convert_to_numpy(indexes)
        scores  = ops.convert_to_numpy(scores)
        if reverse: indexes, scores = indexes[:, ::-1], scores[:, ::-1]

        results = [self[idx] for idx in indexes.tolist()]
        for res_list, score_list in zip(results, scores):
            for res, score in zip(res_list, score_list): res['score'] = float(score)

        return results

    def save_data(self, ** kwargs):
        """ Save the database """
        super().save_data(** kwargs)
        if self.index_path: self._index.save(self.index_path, ** kwargs)

    def get_config(self):
        return {
            ** super().get_config(),

            'vector_key'    : self.vector_key,
            'index' : self._index.get_config()
        }

def _as_object_array(values):
    array = np.empty((len(values), ), dtype = object)
    array[:] = values
    return array

def _match_column(values, filt):
    """ Boolean mask of the `values` (object array) matching `filt` (see `filter_mask`) """
    if callable(filt):
        return np.fromiter((bool(filt(v)) for v in values), dtype = bool, count = len(values))

    if isinstance(filt, (list, tuple, set, frozenset, np.ndarray)):
        try:
            accepted = set(filt)
            return np.fromiter((v in accepted for v in values), dtype = bool, count = len(values))
        except TypeError:   # unhashable filter or column values
            accepted = list(filt)
            return np.fromiter((v in accepted for v in values), dtype = bool, count = len(values))

    if filt is None:
        return np.fromiter((v is None for v in values), dtype = bool, count = len(values))

    mask = values == filt
    if not isinstance(mask, np.ndarray) or mask.shape != values.shape:
        mask = np.fromiter((v == filt for v in values), dtype = bool, count = len(values))
    return mask.astype(bool, copy = False)
