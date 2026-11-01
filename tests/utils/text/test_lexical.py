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

""" Tests for ``utils.text.lexical`` : ``NgramIndex`` against the brute-force ``lexical_search`` """

import random

import numpy as np
import pytest

from utils.text.lexical import NgramIndex, lexical_search, lexical_scores

_WORDS = [
    'habite', 'Bruxelles', 'Paris', 'travaille', 'projet', 'Alice', 'Alex', 'directeur',
    'aime', 'thé', 'le', 'la', 'de', 'sœur', 'Rust', 'ownership', 'siège', '42B', 'train'
]

def _texts(n, seed = 0):
    rng = random.Random(seed)
    return [' '.join(rng.choices(_WORDS, k = rng.randint(0, 8))) for _ in range(n)]

_QUERIES = ['siège 42B du train', 'Alice habite à Bruxelles', 'SIEGE', 'xyz', '']

@pytest.mark.parametrize('query', _QUERIES)
def test_search_matches_lexical_search(query):
    texts = _texts(300)
    index = NgramIndex()
    index.add(texts)

    for k in (1, 5, 50):
        ref = lexical_search(query, texts, k)
        out = index.search(query, k)
        assert [s for _, s in out] == pytest.approx([s for _, s in ref], abs = 1e-6)
        # same rows when the scores are distinct (the tie order may differ)
        ref_scores = {r : s for r, s in ref}
        for row, score in out:
            assert score == pytest.approx(float(lexical_scores(query, [texts[row]])[0]), abs = 1e-6)
            if row in ref_scores: assert ref_scores[row] == pytest.approx(score, abs = 1e-6)

def test_scores_and_rows():
    texts = _texts(100, seed = 1)
    index = NgramIndex()
    assert index.add(texts[:60]) == list(range(60))
    assert index.add(texts[60:]) == list(range(60, 100))

    query = 'Alex directeur'
    np.testing.assert_allclose(index.scores(query), lexical_scores(query, texts), atol = 1e-6)
    np.testing.assert_allclose(index.scores(query, [5, 2]), lexical_scores(query, [texts[5], texts[2]]), atol = 1e-6)

def test_search_mask_and_threshold():
    texts = ['Paris', 'Bruxelles', 'Paris centre', 'Pariser']
    index = NgramIndex()
    index.add(texts)

    assert [r for r, _ in index.search('Paris', k = 4)][0] == 0
    assert sorted(r for r, _ in index.search('Paris', k = 4, mask = [2, 3])) == [2, 3]
    assert all(r in (2, 3) for r, _ in index.search('Paris', k = 4, mask = np.array([False, False, True, True])))
    assert all(s > 0.5 for _, s in index.search('Paris', k = 4, threshold = 0.5))
    assert index.search('Paris', k = 4, mask = []) == []

def test_remove_compacts_the_rows():
    texts = _texts(50, seed = 2)
    index = NgramIndex()
    index.add(texts)
    index.remove([0, 10, 10, 49])

    kept = [t for i, t in enumerate(texts) if i not in (0, 10, 49)]
    assert len(index) == len(kept)
    for query in _QUERIES[:3]:
        np.testing.assert_allclose(index.scores(query), lexical_scores(query, kept), atol = 1e-6)

    index.add(['siège 42B'])
    assert index.search('siège 42B', k = 1)[0][0] == len(kept)

    with pytest.raises(IndexError):
        index.remove([len(index)])

def test_empty_index():
    index = NgramIndex()
    assert index.search('Paris') == []
    assert index.scores('Paris').shape == (0, )
