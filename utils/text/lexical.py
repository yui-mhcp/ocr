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

from .cleaners import lowercase, remove_accents, collapse_whitespace

def normalize_lexical(text):
    """ Cheap normalization for lexical matching : lowercase, no accents, single spaces """
    return collapse_whitespace(remove_accents(lowercase(text))).strip()

def char_ngrams(text, n = 3, *, normalize = True, pad = True):
    """
        Return the `set` of character n-grams of `text`.
        With `pad`, the text is wrapped in spaces so that word boundaries produce n-grams
        (makes short names more robust, e.g. `" al"` / `"al "` for `"al"`)
    """
    if normalize: text = normalize_lexical(text)
    if pad: text = ' {} '.format(text)
    if len(text) < n: return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}

def ngram_similarity(a, b, n = 3, ** kwargs):
    """ Jaccard similarity between the character n-gram sets of `a` and `b` (in [0, 1]) """
    ga, gb = char_ngrams(a, n, ** kwargs), char_ngrams(b, n, ** kwargs)
    if not ga or not gb: return 0.
    return len(ga & gb) / len(ga | gb)

def lexical_scores(query, texts, n = 3, ** kwargs):
    """ Return a 1-D `np.ndarray` with the n-gram similarity between `query` and each text """
    gq = char_ngrams(query, n, ** kwargs)
    if not gq: return np.zeros((len(texts), ), dtype = 'float32')

    scores = np.zeros((len(texts), ), dtype = 'float32')
    for i, text in enumerate(texts):
        gt = char_ngrams(text, n, ** kwargs)
        if gt: scores[i] = len(gq & gt) / len(gq | gt)
    return scores

def lexical_search(query, texts, k = 10, *, n = 3, mask = None, threshold = 0., ** kwargs):
    """
        Return the `k` texts lexically closest to `query` as a `list` of `(index, score)`,
        sorted by decreasing score, restricted to `mask` (list / array of candidate
        indices, or a boolean mask) and to scores strictly above `threshold`
    """
    if mask is None:
        indices = np.arange(len(texts))
    else:
        mask    = np.asarray(mask)
        indices = np.where(mask)[0] if mask.dtype == bool else mask.astype('int32')

    if len(indices) == 0: return []

    scores  = lexical_scores(query, [texts[i] for i in indices], n, ** kwargs)
    order   = np.argsort(- scores)[:k]
    return [(int(indices[i]), float(scores[i])) for i in order if scores[i] > threshold]

class NgramIndex:
    """
        Inverted index of character n-grams : the indexed counterpart of `lexical_search`
        (same Jaccard similarity between n-gram sets, same `(row, score)` results), whose cost
        no longer grows with the number of indexed texts — only with the size of the posting
        lists of the query n-grams (~1 ms for 50k short texts, vs ~600 ms for `lexical_search`).

        Rows are positions in insertion order (like a `VectorIndex`) : `remove` compacts them,
        so that a caller keeping a vector index and a lexical index aligned removes the same
        rows from both. A `mask` (indices or boolean, over the rows) restricts a search.

        ```python
        index = NgramIndex()
        index.add(['Alice habite à Bruxelles', 'Le siège 42B du train'])
        index.search('siege 42b', k = 1)   # [(1, 0.4...)]
        ```
    """
    def __init__(self, n = 3, *, normalize = True, pad = True):
        self.n  = n
        self.normalize  = normalize
        self.pad    = pad

        self._vocab     = {}    # n-gram -> id
        self._postings  = []    # id -> rows containing the n-gram (`list`, in row order)
        self._arrays    = {}    # id -> `np.ndarray` of the postings (cache, invalidated by `add`)
        self._lengths   = []    # row -> number of distinct n-grams of its text

    def __len__(self):
        return len(self._lengths)

    def __repr__(self):
        return '<NgramIndex n={} rows={} vocab={}>'.format(self.n, len(self), len(self._vocab))

    def ngrams(self, text):
        return char_ngrams(text or '', self.n, normalize = self.normalize, pad = self.pad)

    def add(self, texts):
        """ Indexes `texts` (a `str` or a list of them) as new rows, returning these rows """
        if isinstance(texts, str): texts = [texts]

        start = len(self._lengths)
        for row, text in enumerate(texts, start = start):
            grams = self.ngrams(text)
            self._lengths.append(len(grams))
            for gram in grams:
                gid = self._vocab.get(gram, None)
                if gid is None:
                    gid = self._vocab[gram] = len(self._postings)
                    self._postings.append([])
                self._postings[gid].append(row)
                self._arrays.pop(gid, None)
        return list(range(start, len(self._lengths)))

    def remove(self, rows):
        """ Removes `rows` (indices) : the following rows are shifted down (compaction) """
        rows = np.unique(np.asarray(rows, dtype = 'int64').reshape(-1))
        if len(rows) == 0: return
        if rows[0] < 0 or rows[-1] >= len(self):
            raise IndexError('Rows out of range [0, {}) : {}'.format(len(self), rows))

        keep = np.ones((len(self), ), dtype = bool)
        keep[rows] = False
        new_rows = np.cumsum(keep) - 1

        for gid, posting in enumerate(self._postings):
            posting = np.asarray(posting, dtype = 'int64')
            self._postings[gid] = new_rows[posting[keep[posting]]].tolist()
        self._arrays    = {}
        self._lengths   = [length for length, kept in zip(self._lengths, keep) if kept]

    def _posting(self, gid):
        array = self._arrays.get(gid, None)
        if array is None:
            array = self._arrays[gid] = np.asarray(self._postings[gid], dtype = 'int64')
        return array

    def _intersections(self, grams):
        """ Number of n-grams shared by `grams` and every row """
        postings = [self._posting(self._vocab[g]) for g in grams if g in self._vocab]
        if not postings: return np.zeros((len(self), ), dtype = 'int64')
        return np.bincount(np.concatenate(postings), minlength = len(self))

    def scores(self, query, rows = None):
        """ The similarity of `query` to every row (or to the given `rows`), as a 1-D array """
        grams   = self.ngrams(query)
        n_rows  = len(self)
        if not grams or n_rows == 0:
            return np.zeros((n_rows if rows is None else len(rows), ), dtype = 'float32')

        inter   = self._intersections(grams)
        lengths = np.asarray(self._lengths, dtype = 'int64')
        if rows is not None:
            rows    = np.asarray(rows, dtype = 'int64')
            inter, lengths = inter[rows], lengths[rows]
        return (inter / np.maximum(len(grams) + lengths - inter, 1)).astype('float32')

    def search(self, query, k = 10, *, mask = None, threshold = 0.):
        """
            The `k` rows lexically closest to `query`, as a `list` of `(row, score)` sorted by
            decreasing score (ties : lowest row first), restricted to `mask` and to the scores
            strictly above `threshold` — the contract of `lexical_search`
        """
        grams = self.ngrams(query)
        if not grams or len(self) == 0 or k <= 0: return []

        inter = self._intersections(grams)
        rows  = np.flatnonzero(inter)
        if mask is not None:
            mask = np.asarray(mask)
            allowed = np.zeros((len(self), ), dtype = bool)
            if mask.dtype == bool: allowed[: len(mask)] = mask
            else:                  allowed[mask.astype('int64')] = True
            rows = rows[allowed[rows]]
        if len(rows) == 0: return []

        lengths = np.asarray(self._lengths, dtype = 'int64')[rows]
        scores  = inter[rows] / np.maximum(len(grams) + lengths - inter[rows], 1)

        order = np.lexsort((rows, - scores))[:k]
        return [(int(rows[i]), float(scores[i])) for i in order if scores[i] > threshold]
