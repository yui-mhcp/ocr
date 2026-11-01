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

import numpy as np

from ...keras import ops
from .vector_index import VectorIndex

class NumpyIndex(VectorIndex):
    """
        `VectorIndex` over a `np.ndarray`, stored in a **growable buffer** : its capacity is
        doubled when full, so that `add` only copies the new rows (amortized) and `embeddings`
        is a view of the filled part. Without it, every `add` re-concatenated the whole matrix
        at the next search (~25 ms for 50k x 1024 rows), which an index updated by small batches
        (a memory store) paid after every write
    """
    def __init__(self, ** kwargs):
        self._buffer    = None
        self._size      = 0
        super().__init__(** kwargs)

    def concat_embeddings(self, embeddings):
        """ Concatenate a list of embeddings """
        return np.concatenate(embeddings, axis = 0)

    def prepare_embeddings(self, embeddings):
        """ Converts `embeddings` to a 2D array of the correct format """
        embeddings = ops.convert_to_numpy(embeddings)
        if embeddings.ndim == 1: embeddings = np.expand_dims(embeddings, 0)
        return embeddings

    @property
    def embeddings(self):
        if self._buffer is None: return []
        return self._buffer[: self._size]

    @embeddings.setter
    def embeddings(self, value):
        if isinstance(value, list):
            value = self.concat_embeddings(value) if value else None
        if value is None or len(value) == 0:
            self._buffer, self._size = None, 0
        else:
            self._buffer, self._size = np.asarray(value), len(value)

    @property
    def capacity(self):
        return 0 if self._buffer is None else len(self._buffer)

    def __len__(self):
        return self._size

    def _append(self, embeddings):
        embeddings = embeddings.astype(self._buffer.dtype, copy = False)

        size = self._size + len(embeddings)
        if size > self.capacity:
            buffer = np.empty((max(size, 2 * self.capacity), self._buffer.shape[1]), dtype = self._buffer.dtype)
            buffer[: self._size] = self._buffer[: self._size]
            self._buffer = buffer

        self._buffer[self._size : size] = embeddings
        self._size = size
