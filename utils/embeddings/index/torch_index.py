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

from ...keras import ops
from .vector_index import VectorIndex

class TorchIndex(VectorIndex):
    """ `VectorIndex` over a `torch.Tensor` (the search strategies are those of `VectorIndex`) """
    def concat_embeddings(self, embeddings):
        """ Concatenate a list of embeddings """
        import torch
        return torch.concat(embeddings, 0)

    def prepare_embeddings(self, embeddings):
        """ Converts `embeddings` to a 2D array of the correct format """
        embeddings = ops.convert_to_torch_tensor(embeddings)
        if len(embeddings.shape) == 1: embeddings = embeddings.unsqueeze(0)
        return embeddings

    def _normalize(self, embeddings):
        import torch
        return embeddings / torch.norm(embeddings, dim = 1, keepdim = True)

    def similarity(self, query, embeddings):
        import torch

        if self.effective_metric == 'dp':
            return torch.einsum('ik,jk->ij', query, embeddings)
        elif self.effective_metric == 'euclidian':
            xx = torch.einsum('...i, ...i -> ...', query, query)[:, None]
            yy = torch.einsum('...i, ...i -> ...', embeddings, embeddings)[None, :]
            xy = torch.einsum('ik,jk->ij', query, embeddings)
            return - torch.sqrt(xx - 2 * xy + yy)
        return super().similarity(query, embeddings)

    def _top_k(self, scores, k):
        import torch
        dists, indices = torch.topk(scores, k)
        return indices, dists

    def _mask_scores(self, scores, candidates):
        import torch

        penalty = torch.full((scores.shape[-1], ), float('-inf'), dtype = scores.dtype, device = scores.device)
        penalty[torch.as_tensor(candidates, device = scores.device)] = 0.
        return scores + penalty
