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

""" Pure-``pytest`` tests for the :class:`VectorIndex` implementations.

The original ``test_index.py`` used an abstract ``TestVectorIndex`` base shared by
three ``unittest`` subclasses (numpy / keras / torch). That hierarchy is now a
single parametrized ``index_cls`` fixture :
  - ``NumpyIndex``  : keras-free -> unmarked.
  - ``KerasIndex``  : builds keras tensors -> marked ``keras``.
  - ``TorchIndex``  : builds torch tensors -> marked ``torch``.

All raw-value comparisons use ``metric = 'euclidian'`` so the stored embeddings
are kept as-is (``cosine`` would L2-normalize them on ``add``).
"""

import os
import pytest
import numpy as np

from utils.keras import ops

# --- backend (index class) fixture -------------------------------------------------

_INDEXES = [
    pytest.param('NumpyIndex', id = 'numpy'),
    pytest.param('KerasIndex', id = 'keras', marks = pytest.mark.keras),
    pytest.param('TorchIndex', id = 'torch', marks = pytest.mark.torch),
]

@pytest.fixture(params = _INDEXES)
def index_cls(request):
    from utils.embeddings import index as index_pkg
    return getattr(index_pkg, request.param)

def _make(index_cls, metric = 'euclidian', ** kwargs):
    return index_cls(metric = metric, ** kwargs)

def _random_vectors(n_vectors = 100, embedding_dim = 128, seed = 0):
    return np.random.default_rng(seed).random((n_vectors, embedding_dim)).astype(np.float32)


# --- add / shape -------------------------------------------------------------------

def test_add(index_cls, asserts):
    dim   = 64
    index = _make(index_cls)

    emb_1 = _random_vectors(50, dim, seed = 1)
    emb_2 = _random_vectors(25, dim, seed = 2)

    index.add(emb_1)
    asserts.assert_equal(50, len(index))
    asserts.assert_equal((50, dim), index.shape)

    index.add(emb_2)
    asserts.assert_equal(75, len(index))
    asserts.assert_equal((75, dim), index.shape)

    asserts.assert_equal(np.concatenate([emb_1, emb_2], axis = 0), index.embeddings)


# --- __getitem__ -------------------------------------------------------------------

def test_get(index_cls, asserts):
    dim   = 64
    index = _make(index_cls)
    emb_1 = _random_vectors(50, dim, seed = 1)

    index.add(emb_1)

    asserts.assert_equal(emb_1,      index.embeddings)
    asserts.assert_equal(emb_1,      index[:])
    asserts.assert_equal(emb_1[0],   index[0])
    asserts.assert_equal(emb_1[:5],  index[np.arange(5)])


# --- remove ------------------------------------------------------------------------

def test_remove(index_cls, asserts):
    dim   = 64
    index = _make(index_cls)
    emb_1 = _random_vectors(50, dim, seed = 1)

    index.add(emb_1)
    asserts.assert_equal(emb_1, index.embeddings)

    index.remove(0)
    asserts.assert_equal(49, len(index))
    asserts.assert_equal(emb_1[1:], index.embeddings)

    index.remove(np.arange(45, 49))
    asserts.assert_equal(45, len(index))
    asserts.assert_equal(emb_1[1:46], index.embeddings)


# --- top_k -------------------------------------------------------------------------

def test_top_k(index_cls, asserts):
    dim   = 8
    index = _make(index_cls)

    emb = np.arange(dim * 64).reshape((64, dim)).astype('float32')
    emb_1, emb_2 = emb[:48], emb[48:]

    index.add(emb_1)
    asserts.assert_equal(48, len(index))

    indices, scores = index.top_k(emb_2[0], k = 5)
    asserts.assert_equal((1, 5), tuple(scores.shape))
    asserts.assert_equal((1, 5), tuple(indices.shape))
    asserts.assert_equal([47, 46, 45, 44, 43], indices[0])

    indices, scores = index.top_k(emb_2[:2], k = 3)
    asserts.assert_equal((2, 3), tuple(scores.shape))
    asserts.assert_equal((2, 3), tuple(indices.shape))
    asserts.assert_equal([[47, 46, 45], [47, 46, 45]], indices)

    index.add(emb_2)
    indices, scores = index.top_k(emb_2[0], k = 5)
    asserts.assert_equal(48, indices[0, 0])
    asserts.assert_equal(0,  scores[0, 0])

def test_top_k_clamps_k(index_cls, asserts):
    dim   = 8
    index = _make(index_cls)
    index.add(np.arange(dim * 4).reshape((4, dim)).astype('float32'))

    indices, scores = index.top_k(_random_vectors(1, dim, seed = 1), k = 10)
    asserts.assert_equal((1, 4), tuple(ops.convert_to_numpy(indices).shape))
    asserts.assert_equal((1, 4), tuple(ops.convert_to_numpy(scores).shape))

def test_top_k_masked(index_cls, asserts):
    dim   = 8
    index = _make(index_cls)
    index.add(np.arange(dim * 10).reshape((10, dim)).astype('float32'))

    query = np.full((dim, ), dim * 10., dtype = 'float32')

    # indices must be **global** (relative to the whole index, not to the masked subset)
    indices, scores = index.top_k(query, k = 3, mask = [0, 2, 4, 6])
    asserts.assert_equal((1, 3), tuple(ops.convert_to_numpy(indices).shape))
    asserts.assert_equal([6, 4, 2], ops.convert_to_numpy(indices)[0])

    # boolean masks are supported as well
    bool_mask = np.arange(10) % 2 == 0
    indices, scores = index.top_k(query, k = 3, mask = bool_mask)
    asserts.assert_equal([8, 6, 4], ops.convert_to_numpy(indices)[0])

    # `k` larger than the masked subset is clamped
    indices, scores = index.top_k(query, k = 10, mask = [1, 3])
    asserts.assert_equal([3, 1], ops.convert_to_numpy(indices)[0])


# --- error handling ----------------------------------------------------------------

def test_error_handling(index_cls):
    index = _make(index_cls)
    index.add(_random_vectors(64, 128, seed = 1))

    with pytest.raises(Exception):
        _ = index[64]

    with pytest.raises(IndexError):
        index.remove(64)

    with pytest.raises(ValueError):
        index.add(_random_vectors(64, 64, seed = 2))


# --- save / load -------------------------------------------------------------------

@pytest.fixture
def index_path(temp_dir):
    path = os.path.join(temp_dir, 'test_vector_index.index')
    yield path
    for p in (path, path + '.npy'):
        if os.path.exists(p): os.remove(p)

def test_io(index_cls, index_path, asserts):
    dim   = 64
    index = _make(index_cls)
    emb_1 = _random_vectors(50, dim, seed = 1)

    index.add(emb_1)
    asserts.assert_equal(50, len(index))

    index.save(index_path)
    assert os.path.exists(index_path) or os.path.exists(index_path + '.npy')

    loaded = index_cls.load(index_path)
    asserts.assert_equal(50, len(loaded))
    asserts.assert_equal((50, dim), loaded.shape)
    asserts.assert_equal(emb_1, loaded.embeddings)


# --- search strategies (dense / sparse mask) ---------------------------------------

def _naive_top_k(embeddings, query, k, rows = None, metric = 'cosine'):
    """ Reference search : full scoring, then a stable sort (ties : lowest index first) """
    embeddings, query = np.asarray(embeddings, 'float64'), np.asarray(query, 'float64')
    if metric == 'cosine':
        embeddings = embeddings / np.linalg.norm(embeddings, axis = -1, keepdims = True)
        query      = query / np.linalg.norm(query, axis = -1, keepdims = True)
    rows   = np.arange(len(embeddings)) if rows is None else np.asarray(rows)
    scores = query @ embeddings[rows].T
    order  = np.argsort(- scores, axis = -1, kind = 'stable')[:, :k]
    return rows[order], np.take_along_axis(scores, order, -1)

@pytest.mark.parametrize('ratio', [0.02, 0.1, 0.5, 1.0])
def test_top_k_masked_matches_naive(index_cls, ratio, asserts):
    rng   = np.random.default_rng(0)
    emb   = rng.standard_normal((400, 32)).astype('float32')
    query = rng.standard_normal((3, 32)).astype('float32')
    rows  = np.sort(rng.choice(400, max(1, int(400 * ratio)), replace = False))

    index = _make(index_cls, metric = 'cosine')
    index.add(emb)

    for mask in (rows, np.isin(np.arange(400), rows)):
        indices, scores = index.top_k(query, k = 7, mask = mask)
        ref_indices, ref_scores = _naive_top_k(emb, query, 7, rows)
        asserts.assert_equal(ref_indices, ops.convert_to_numpy(indices))
        np.testing.assert_allclose(ops.convert_to_numpy(scores), ref_scores, atol = 1e-3)

def test_top_k_masked_strategies_agree(asserts):
    from utils.embeddings.index import NumpyIndex

    rng   = np.random.default_rng(1)
    emb   = rng.standard_normal((300, 16)).astype('float32')
    query = rng.standard_normal((2, 16)).astype('float32')
    rows  = rng.choice(300, 60, replace = False)

    dense, sparse = NumpyIndex(), NumpyIndex()
    dense.dense_mask_ratio, sparse.dense_mask_ratio = 0., 2.
    for index in (dense, sparse): index.add(emb)

    for k in (1, 5, 60, 100):
        d_idx, d_scores = dense.top_k(query, k = k, mask = rows)
        s_idx, s_scores = sparse.top_k(query, k = k, mask = rows)
        assert d_idx.shape == (2, min(k, 60))
        asserts.assert_equal(s_idx, d_idx)
        np.testing.assert_allclose(d_scores, s_scores, atol = 1e-6)

def test_top_k_empty_mask(index_cls):
    index = _make(index_cls)
    index.add(_random_vectors(10, 8))
    indices, scores = index.top_k(_random_vectors(2, 8, seed = 1), k = 3, mask = [])
    assert tuple(ops.convert_to_numpy(indices).shape) == (2, 0)
    assert tuple(ops.convert_to_numpy(scores).shape) == (2, 0)

def test_score_rows(index_cls):
    rng   = np.random.default_rng(2)
    emb   = rng.standard_normal((50, 16)).astype('float32')
    query = rng.standard_normal((2, 16)).astype('float32')
    index = _make(index_cls, metric = 'cosine')
    index.add(emb)

    rows = [3, 41, 0, 7]
    _, ref = _naive_top_k(emb, query, 50)
    full   = np.asarray(_naive_top_k(emb, query, 50)[0])
    scores = ops.convert_to_numpy(index.score(query, rows))
    assert scores.shape == (2, 4)
    for q in range(2):
        expected = {int(i) : s for i, s in zip(full[q], ref[q])}
        # `1e-3` : the keras backend may run the matmul in TF32 on GPU
        np.testing.assert_allclose(scores[q], [expected[r] for r in rows], atol = 1e-3)

    assert ops.convert_to_numpy(index.score(query, [])).shape == (2, 0)


# --- numpy growable buffer ---------------------------------------------------------

def test_numpy_buffer_grows_by_doubling(asserts):
    from utils.embeddings.index import NumpyIndex

    index, batches = NumpyIndex(metric = 'euclidian'), []
    for i in range(20):
        batch = _random_vectors(3, 8, seed = i)
        batches.append(batch)
        index.add(batch)
        assert len(index) == 3 * (i + 1)
        assert index.capacity >= len(index)
        asserts.assert_equal(np.concatenate(batches), index.embeddings)
    # amortized : far fewer re-allocations than additions
    assert index.capacity < 4 * len(index)

def test_numpy_buffer_never_writes_into_the_caller_array(asserts):
    from utils.embeddings.index import NumpyIndex

    first = _random_vectors(4, 8, seed = 1)
    copy  = first.copy()
    index = NumpyIndex(metric = 'euclidian')
    index.add(first)
    index.add(_random_vectors(2, 8, seed = 2))
    index.add(_random_vectors(2, 8, seed = 3))
    asserts.assert_equal(copy, first)

def test_numpy_buffer_remove_then_add(asserts):
    from utils.embeddings.index import NumpyIndex

    index = NumpyIndex(metric = 'euclidian')
    emb   = _random_vectors(10, 8)
    index.add(emb[:6])
    index.add(emb[6:])
    index.remove([0, 5])
    index.add(emb[:1])
    asserts.assert_equal(np.concatenate([emb[1:5], emb[6:], emb[:1]]), index.embeddings)

def test_numpy_buffer_io(index_path, asserts):
    from utils.embeddings.index import NumpyIndex

    index = NumpyIndex(metric = 'euclidian')
    for i in range(5): index.add(_random_vectors(3, 8, seed = i))
    index.save(index_path)

    loaded = NumpyIndex.load(index_path, metric = 'euclidian')
    asserts.assert_equal(index.embeddings, loaded.embeddings)
    assert len(np.load(index_path + '.npy')) == 15
