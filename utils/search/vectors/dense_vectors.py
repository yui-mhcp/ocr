# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .base_vectors_db import BaseVectorsDB
from utils.keras_utils import ops

class DenseVectors(BaseVectorsDB):
    @property
    def vectors_dim(self):
        return self.vectors.shape[1]
    
    def append_vectors(self, vectors):
        if self.vectors is None: self.vectors = vectors
        else:   self.vectors = ops.concat([self.vectors, vectors], axis = 0)

    def update_vectors(self, indices, vectors):
        self.vectors = ops.scatter_update(self.vectors, np.array(indices).reshape(-1, 1), vectors)
    
    def get_vectors(self, indices):
        return ops.take(self.vectors, indices, axis = 0)
    
    def top_k(self, inputs, k = 10, ** kwargs):
        scores  = self.compute_scores(inputs, ** kwargs)
        
        scores, indices = ops.top_k(scores, k)
        return indices, scores
    
    
    