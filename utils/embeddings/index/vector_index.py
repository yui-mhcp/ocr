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

from abc import ABC, abstractmethod

from ...keras import ops
from ...distances import distance

class VectorIndex(ABC):
    """
        A flat vector index : one vector per row, rows identified by their position (insertion
        order, shifted by `remove`).

        Search (`top_k`) : the similarity of the query to the (candidate) rows, then the `k` best
        ones. A `mask` restricts the candidates, with two strategies (see `dense_mask_ratio`) :
            - a **sparse** mask (few candidates) gathers the candidate rows, then scores them ;
            - a **dense** mask scores every row and masks the others to `-inf` : gathering most
              of the rows would copy most of the matrix, which costs more than the scoring itself
    """
    # ratio of candidate rows (over the index size) from which a masked search scores every row
    # (measured crossover, numpy, 50k x 1024 : ~10-20 %)
    dense_mask_ratio    = 0.15

    @abstractmethod
    def concat_embeddings(self, embeddings):
        """ Concatenate a list of embeddings """

    @abstractmethod
    def prepare_embeddings(self, embeddings):
        """ Converts `embeddings` to a 2D array of the correct format """

    def __init__(self, *, metric = 'cosine', embeddings = None, ** _):
        self.metric = metric

        self._embeddings    = []
        self.effective_metric   = 'dp' if metric == 'cosine' else metric

        if embeddings is not None: self.embeddings = self.prepare_embeddings(embeddings)

    @property
    def shape(self):
        return (len(self), self.embedding_dim)

    @property
    def dtype(self):
        return self.embeddings.dtype

    @property
    def embedding_dim(self):
        return self.embeddings.shape[-1]

    @property
    def embeddings(self):
        if isinstance(self._embeddings, list):
            if len(self._embeddings) == 1:
                self._embeddings = self._embeddings[0]
            else:
                self._embeddings = self.concat_embeddings(self._embeddings)
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value

    def __len__(self):
        if isinstance(self._embeddings, list):
            return sum(len(emb) for emb in self._embeddings)
        else:
            return len(self._embeddings)

    def __repr__(self):
        return '<{} shape={}>'.format(self.__class__.__name__, self.shape if len(self) else (0, None))

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, index):
        """ Return the embeddings at the given `index`(es) """
        if isinstance(index, list): index = np.array(index)
        return self.embeddings[index]

    def _normalize(self, embeddings):
        """ L2-normalization of the rows (stored / queried), for the `cosine` metric """
        return ops.normalize(embeddings)

    def add(self, embeddings, ** kwargs):
        """ Add `vectors` to the index """
        embeddings = self.prepare_embeddings(embeddings)
        if self.metric == 'cosine': embeddings = self._normalize(embeddings)

        if len(self) == 0:
            self.embeddings = embeddings
        elif embeddings.shape[-1] != self.embedding_dim:
            raise ValueError('Expected dim {}, got {}'.format(
                self.embedding_dim, embeddings.shape[-1]
            ))
        else:
            self._append(embeddings)

    def _append(self, embeddings):
        """ Appends (prepared) `embeddings` to a non-empty index """
        if isinstance(self._embeddings, list):
            self._embeddings.append(embeddings)
        else:
            self._embeddings = [self._embeddings, embeddings]

    def remove(self, index):
        """ Remove the embeddings at the given `index`(es) """
        if isinstance(index, int):      index = np.array([index], np.int32)
        elif isinstance(index, list):   index = np.array(index, np.int32)

        if np.any(index >= len(self)):
            raise IndexError('Some indexes are higher or equal than {}: {}'.format(
                len(self), index
            ))

        mask = np.ones((len(self), ), dtype = bool)
        mask[index] = False

        self.embeddings = self[mask]

    ##############################
    #           Search           #
    ##############################

    def prepare_query(self, query):
        """ The query as a 2-D array of the index format (normalized for the `cosine` metric) """
        query = self.prepare_embeddings(query)
        if self.metric == 'cosine': query = self._normalize(query)
        return query

    def similarity(self, query, embeddings):
        """ The `(n_queries, n_rows)` similarity matrix of a **prepared** `query` to `embeddings` """
        return distance(query, embeddings, self.effective_metric, as_matrix = True, mode = 'similarity')

    def _top_k(self, scores, k):
        """ `(indices, scores)` of the `k` best `scores` (per row) """
        dists, indices = ops.top_k(scores, k)
        return indices, dists

    def _mask_scores(self, scores, candidates):
        """ `scores` with the rows outside of `candidates` (indices) set to `-inf` """
        penalty = np.full((scores.shape[-1], ), - np.inf, dtype = 'float32')
        penalty[candidates] = 0.
        if isinstance(scores, np.ndarray): return scores + penalty.astype(scores.dtype)
        return ops.add(scores, ops.cast(penalty, scores.dtype))

    def top_k(self, query, k = 10, *, mask = None, ** kwargs):
        """
            Returns a tuple `(top_k_indices, top_k_scores)`

            Arguments :
                - query : a 2-D `Tensor` with shape `(n_queries, embedding_dim)`
                - k     : the number of best items to retrieve
                - mask  : subset of candidate rows, either a list/array of indices, or a
                          boolean mask over the whole index (see the class docstring)
            Return :
                - indices   : a 2-D `Tensor` of shape `(n_queries, k)`, the indexes of nearest data
                              When `mask` is provided, this is a `np.ndarray` of **global** indices
                - scores    : a 2-D `Tensor` of shape `(n_queries, k)`, the scores of the nearest data
        """
        query = self.prepare_query(query)

        if mask is None:
            return self._top_k(self.similarity(query, self.embeddings), min(k, len(self)))

        mask = mask_to_indices(mask)
        k    = min(k, len(mask))
        if k == 0:
            n_queries = int(query.shape[0])
            return np.zeros((n_queries, 0), dtype = 'int64'), np.zeros((n_queries, 0), dtype = 'float32')

        if len(mask) >= self.dense_mask_ratio * len(self):
            scores = self._mask_scores(self.similarity(query, self.embeddings), mask)
            indices, dists = self._top_k(scores, k)
            return ops.convert_to_numpy(indices), dists

        indices, dists = self._top_k(self.similarity(query, self[mask]), k)
        return mask[ops.convert_to_numpy(indices)], dists

    def score(self, query, rows):
        """
            The `(n_queries, len(rows))` similarities of `query` to the given `rows` (indices) :
            the rescoring of candidates found otherwise (lexical, graph structure, ...)
        """
        rows = mask_to_indices(rows)
        if len(rows) == 0:
            query = self.prepare_embeddings(query)
            return np.zeros((int(query.shape[0]), 0), dtype = 'float32')
        return self.similarity(self.prepare_query(query), self[rows])

    ##############################
    #         Persistence        #
    ##############################

    def save(self, filename):
        """ Save the index to `filename` """
        if not filename.endswith('.npy'): filename += '.npy'
        np.save(filename, ops.convert_to_numpy(self.embeddings))

    def get_config(self):
        return {'index' : self.__class__.__name__, 'metric' : self.metric}

    @classmethod
    def load(cls, filename, ** kwargs):
        if not filename.endswith('.npy'): filename += '.npy'
        return cls(embeddings = np.load(filename) if os.path.exists(filename) else None, ** kwargs)

def mask_to_indices(mask):
    """ Normalizes a (boolean or index) `mask` to a 1-D `np.ndarray` of indices """
    mask = ops.convert_to_numpy(mask)
    if mask.ndim == 0: mask = mask.reshape(1)
    if mask.dtype == bool:  mask = np.where(mask)[0]
    elif not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int32)
    return mask
