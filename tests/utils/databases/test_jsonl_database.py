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

""" Tests specific to ``utils.databases.jsonl.JSONLDatabase`` (append-only log).

The ``Database`` contract (insert / get / keys / save-reload / append-only rejections) is
covered by the generic suite (``test_database.py``, backends ``jsonl`` / ``jsonl_buffered``).
This file covers the log-specific API (``seq`` / ``since`` / ``append``), the ``autoflush``
persistence modes, crash recovery, and the ``VectorDatabase`` integration (an embedding
index over a buffered log). Keras-free (``NumpyIndex``) -> unmarked.

NB : ``Database`` instances are cached per path ; re-opening a log within a test therefore
uses ``reload = True`` (the cache itself is reset between tests by ``_isolate_db_cache``).
"""

import os

import numpy as np
import pytest

from utils.databases import JSONLDatabase, VectorDatabase
from ._base import make_entry, make_entries


def _entry(i, ** extra):
    return make_entry('single', i, ** extra)

def _entries(n, ** extra):
    return make_entries('single', n, ** extra)

def _log_path(tmp_path, name = 'log'):
    return os.path.join(str(tmp_path), name)

def _open(path, primary_key = 'id', ** kwargs):
    """ (Re-)open the log stored at `path`, bypassing the per-path instance cache """
    return JSONLDatabase(path, primary_key, reload = True, ** kwargs)

def _read(path):
    with open(os.path.join(path, 'data.jsonl'), 'r', encoding = 'utf-8') as file:
        return file.read()

def _lines(path):
    return [line for line in _read(path).splitlines() if line.strip()]

def _write(path, content):
    os.makedirs(path, exist_ok = True)
    with open(os.path.join(path, 'data.jsonl'), 'w', encoding = 'utf-8', newline = '') as file:
        file.write(content)


# --- log API : seq / since / iteration ---------------------------------------------

def test_seq(asserts):
    db = JSONLDatabase(None, 'id')
    db.multi_insert(_entries(3))

    asserts.assert_equal([0, 1, 2], [db.seq('id-{}'.format(i)) for i in range(3)])
    asserts.assert_equal(1, db.seq(_entry(1)))

    with pytest.raises(KeyError):
        db.seq('id-99')

def test_since(asserts):
    db = JSONLDatabase(None, 'id')
    db.multi_insert(_entries(4))

    asserts.assert_equal(['text 0', 'text 1', 'text 2', 'text 3'], [r['text'] for r in db.since()])
    asserts.assert_equal(['text 2', 'text 3'], [r['text'] for r in db.since(2)])
    asserts.assert_equal([], db.since(4))

def test_insertion_order(asserts):
    db = JSONLDatabase(None, 'id')
    ids = ['id-3', 'id-0', 'id-2', 'id-1']
    for i in (3, 0, 2, 1): db.insert(_entry(i))

    asserts.assert_equal(ids, [r['id'] for r in db])
    asserts.assert_equal(ids, db.get_column('id'))
    asserts.assert_equal(ids, list(db.keys()))

def test_returned_records_are_copies(asserts):
    db = JSONLDatabase(None, 'id')
    db.insert(_entry(0))

    db.get('id-0')['text'] = 'mutated'
    next(iter(db))['text'] = 'mutated'
    db.since()[0]['text'] = 'mutated'

    asserts.assert_equal('text 0', db.get('id-0')['text'])

def test_insert_does_not_alias_input(asserts):
    db = JSONLDatabase(None, 'id')
    data = _entry(0)
    db.insert(data)

    data['text'] = 'mutated'
    asserts.assert_equal('text 0', db.get('id-0')['text'])


# --- stringified primary keys ------------------------------------------------------

def test_single_key_is_stringified(tmp_path, asserts):
    path = _log_path(tmp_path)
    db = JSONLDatabase(path, 'id')
    db.insert({'id' : 1, 'text' : 'a'})

    asserts.assert_equal('1', db.get(1)['id'])
    asserts.assert_equal('1', db.get('1')['id'])
    asserts.assert_equal(['1'], db.get_column('id'))

    # persisted as a string as well
    asserts.assert_equal('1', _open(path).get(1)['id'])

def test_multi_key_is_stringified(asserts):
    db = JSONLDatabase(None, ('source', 'idx'))
    db.append([{'source' : 'src', 'idx' : 0, 'text' : 'a'}])

    record = db.get(('src', 0))
    asserts.assert_equal('0', record['idx'])
    asserts.assert_equal('src', record['source'])

def test_legacy_non_string_key_is_stringified_on_load(tmp_path, asserts):
    path = _log_path(tmp_path)
    _write(path, '{"id": 1, "text": "a"}\n')

    db = _open(path)
    asserts.assert_equal('1', db.get(1)['id'])


# --- append ------------------------------------------------------------------------

def test_append_returns_seqs(tmp_path, asserts):
    path = _log_path(tmp_path)
    db = JSONLDatabase(path, 'id')

    asserts.assert_equal([0, 1], db.append(_entries(2)))
    asserts.assert_equal(2, len(db))
    asserts.assert_equal(2, len(_lines(path)))

def test_append_skip_existing(tmp_path, asserts):
    path = _log_path(tmp_path)
    db = JSONLDatabase(path, 'id')
    db.append(_entries(2))

    # idempotent replay : already-logged records return their existing seq
    seqs = db.append([_entry(1, text = 'replayed'), _entry(2), _entry(0)])

    asserts.assert_equal([1, 2, 0], seqs)
    asserts.assert_equal(3, len(db))
    asserts.assert_equal('text 1', db.get('id-1')['text'])
    asserts.assert_equal(3, len(_lines(path)))

def test_append_skip_existing_after_reload(tmp_path, asserts):
    path = _log_path(tmp_path)
    JSONLDatabase(path, 'id').append(_entries(2))

    db = _open(path)
    asserts.assert_equal([0, 1, 2], db.append(_entries(3)))
    asserts.assert_equal(3, len(_lines(path)))

def test_append_duplicate_within_batch(asserts):
    db = JSONLDatabase(None, 'id')

    seqs = db.append([_entry(0), _entry(1), _entry(1, text = 'dup'), _entry(2)])

    asserts.assert_equal([0, 1, 1, 2], seqs)
    asserts.assert_equal(3, len(db))
    asserts.assert_equal('text 1', db.get('id-1')['text'])

@pytest.mark.parametrize('batch', [
    pytest.param([_entry(1), _entry(0)], id = 'existing'),
    pytest.param([_entry(1), _entry(1)], id = 'within_batch'),
])
def test_append_no_skip_is_atomic(tmp_path, batch, asserts):
    path = _log_path(tmp_path)
    db = JSONLDatabase(path, 'id')
    db.insert(_entry(0))

    with pytest.raises(ValueError):
        db.append(batch, skip_existing = False)

    # nothing from the rejected batch is kept (in memory nor on disk)
    asserts.assert_equal(1, len(db))
    asserts.assert_equal(0, db.pending)
    assert 'id-1' not in db
    asserts.assert_equal(1, len(_lines(path)))

    # ... nor leaked by the next flush
    db.insert(_entry(2))
    asserts.assert_equal(['id-0', 'id-2'], db.get_column('id'))
    asserts.assert_equal(2, len(_lines(path)))

def test_append_missing_key_is_atomic(tmp_path, asserts):
    path = _log_path(tmp_path)
    db = JSONLDatabase(path, 'id')
    db.insert(_entry(0))

    with pytest.raises(KeyError):
        db.append([_entry(1), {'text' : 'no key'}])

    asserts.assert_equal(1, len(db))
    assert 'id-1' not in db
    asserts.assert_equal(1, len(_lines(path)))


# --- persistence -------------------------------------------------------------------

def test_autoflush_writes_through(tmp_path, asserts):
    path = _log_path(tmp_path)
    db = JSONLDatabase(path, 'id')

    db.insert(_entry(0))
    db.multi_insert([_entry(1), _entry(2)])

    # written without any `save`
    asserts.assert_equal(0, db.pending)
    asserts.assert_equal(3, len(_lines(path)))

    reloaded = _open(path)
    asserts.assert_equal(['text 0', 'text 1', 'text 2'], [r['text'] for r in reloaded.since()])

def test_buffered_writes_on_save(tmp_path, asserts):
    path = _log_path(tmp_path)
    db = JSONLDatabase(path, 'id', autoflush = False)

    db.extend(_entries(2))
    asserts.assert_equal(2, db.pending)
    assert not os.path.exists(os.path.join(path, 'data.jsonl'))

    # `close` deliberately does not flush
    db.close()
    assert not os.path.exists(os.path.join(path, 'data.jsonl'))

    db.save_data()
    asserts.assert_equal(0, db.pending)
    asserts.assert_equal(2, len(_lines(path)))

    # a later save only appends the new records (no duplicated lines)
    db.insert(_entry(2))
    asserts.assert_equal(1, db.pending)
    db.save()
    asserts.assert_equal(3, len(_lines(path)))

def test_config_restores_autoflush(tmp_path, asserts):
    path = _log_path(tmp_path)
    db = JSONLDatabase(path, ('source', 'idx'), autoflush = False)
    db.insert(make_entry('multi', 0))
    db.save()

    reloaded = JSONLDatabase(path, reload = True)

    assert reloaded.autoflush is False
    asserts.assert_equal(('source', 'idx'), reloaded.primary_key)
    asserts.assert_equal(1, len(reloaded))

def test_save_empty_log(tmp_path, asserts):
    path = os.path.join(str(tmp_path), 'nested', 'log')
    db = JSONLDatabase(path, 'id')

    db.save()   # nothing flushed : the directory does not exist yet

    assert os.path.exists(os.path.join(path, 'config.json'))
    assert not os.path.exists(os.path.join(path, 'data.jsonl'))

    reloaded = JSONLDatabase(path, reload = True)
    asserts.assert_equal(0, len(reloaded))
    asserts.assert_equal('id', reloaded.primary_key)


# --- crash recovery ----------------------------------------------------------------

def test_load_deduplicates_first_wins(tmp_path, asserts):
    path = _log_path(tmp_path)
    _write(path, '{"id": "a", "text": "first"}\n{"id": "b", "text": "b"}\n{"id": "a", "text": "second"}\n')

    db = _open(path)
    asserts.assert_equal(2, len(db))
    asserts.assert_equal('first', db.get('a')['text'])
    asserts.assert_equal(['a', 'b'], db.get_column('id'))
    asserts.assert_equal(1, db.seq('b'))

def test_load_skips_invalid_records(tmp_path, asserts):
    path = _log_path(tmp_path)
    _write(path, '\n'.join([
        '{"id": "a", "text": "a"}',
        '{"text": "no key"}',       # missing primary key
        '[1, 2, 3]',                # not a record
        '{"id": "b", "te',          # corrupt line in the middle
        '{"id": "c", "text": "c"}',
    ]) + '\n')

    db = _open(path)
    asserts.assert_equal(['a', 'c'], db.get_column('id'))
    asserts.assert_equal(0, db.pending)

def test_partial_trailing_line_does_not_swallow_next_record(tmp_path, asserts):
    path = _log_path(tmp_path)
    # crash mid-write : the last line is truncated and has no trailing newline
    _write(path, '{"id": "a", "text": "a"}\n{"id": "b", "te')

    db = _open(path)
    asserts.assert_equal(['a'], db.get_column('id'))

    db.insert({'id' : 'c', 'text' : 'c'})

    reloaded = _open(path)
    asserts.assert_equal(['a', 'c'], reloaded.get_column('id'))
    asserts.assert_equal('c', reloaded.get('c')['text'])

    # later appends keep producing well-formed lines
    reloaded.insert({'id' : 'd', 'text' : 'd'})
    asserts.assert_equal(['a', 'c', 'd'], _open(path).get_column('id'))

def test_well_formed_file_gets_no_blank_line(tmp_path, asserts):
    path = _log_path(tmp_path)
    JSONLDatabase(path, 'id').insert(_entry(0))

    _open(path).insert(_entry(1))

    content = _read(path)
    assert '\n\n' not in content.replace('\r\n', '\n')
    asserts.assert_equal(2, len(_lines(path)))


# --- instance cache ----------------------------------------------------------------

def test_cache_false_is_a_private_instance(tmp_path, asserts):
    """ `cache = False` re-reads the disk and is never retained : a reader of a log written by
        someone else (another process) is never stale, and nothing accumulates in the cache """
    path    = _log_path(tmp_path)
    writer  = JSONLDatabase(path, 'id', autoflush = False, cache = False)
    writer.append(_entries(2))
    writer.save_data()

    reader  = JSONLDatabase(path, 'id', cache = False)
    assert reader is not writer and path not in JSONLDatabase._instances
    asserts.assert_equal(2, len(reader))

    writer.insert(_entry(2))
    writer.save_data()
    # the first reader is a snapshot, a new one sees the new record
    asserts.assert_equal(2, len(reader))
    asserts.assert_equal(3, len(JSONLDatabase(path, 'id', cache = False)))

def test_cache_false_does_not_touch_the_cached_instance(tmp_path, asserts):
    path    = _log_path(tmp_path)
    cached  = JSONLDatabase(path, 'id')
    cached.insert(_entry(0))

    private = JSONLDatabase(path, 'id', cache = False)
    assert private is not cached and JSONLDatabase(path, 'id') is cached
    asserts.assert_equal(1, len(private))


# --- in-memory ---------------------------------------------------------------------

def test_in_memory_log(asserts):
    db = JSONLDatabase(None, 'id')
    db.append(_entries(2))

    assert db.data_file is None
    asserts.assert_equal(2, len(db))
    asserts.assert_equal(1, db.seq('id-1'))


# --- integration : VectorDatabase over a buffered log ------------------------------

def _linear_vectors(values, dim = 4):
    return np.array(values, dtype = 'float32')[:, None] * np.ones((dim, ), dtype = 'float32')

def _make_vector_db(path):
    return VectorDatabase(
        path, 'hash',
        database = {'class_name' : 'JSONLDatabase', 'autoflush' : False},
        index = 'numpy', metric = 'euclidian'
    )

def test_vector_database_over_jsonl(tmp_path, asserts):
    path = _log_path(tmp_path, 'embeddings')
    db = _make_vector_db(path)

    entries = [{'hash' : 'chunk-{}'.format(i), 'text' : 'text {}'.format(i)} for i in range(3)]
    db.multi_insert(entries, vectors = _linear_vectors([0., 1., 2.]))

    # buffered : the log is only written at `save`, together with the vectors
    assert not os.path.exists(os.path.join(path, 'data.jsonl'))

    # removal is rejected, and neither the log nor the vectors are touched
    with pytest.raises(TypeError):
        db.pop('chunk-1')
    asserts.assert_equal(3, len(db))
    asserts.assert_equal(3, len(db.vectors))

    db.save()
    asserts.assert_equal(3, len(_lines(path)))

    reloaded = VectorDatabase(path, reload = True)

    asserts.assert_equal(3, len(reloaded))
    asserts.assert_equal(3, len(reloaded.vectors))
    asserts.assert_equal('text 2', reloaded['chunk-2']['text'])

    # records and vectors stay aligned after reload
    results = reloaded.search(_linear_vectors([1.9]), k = 1)
    asserts.assert_equal('chunk-2', results[0][0]['hash'])
