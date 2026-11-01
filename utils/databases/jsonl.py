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
import logging

from .database import Database
from ..generic_utils import to_json
from ..file_utils import load_jsonl, dump_jsonl

logger = logging.getLogger(__name__)

class JSONLDatabase(Database):
    """
        **Append-only** log of records, persisted as JSON-lines (`{path}/data.jsonl`, one
        record per line, in insertion order).

        This is the durable-log building block (mutation journal, LLM-decision log, ...) :
        records are identified by their primary key (single or composite, see `Database`),
        kept in memory in insertion order, and **never modified nor removed** once written.
        The mutating methods of the `Database` contract, `update`, `pop` and `filter`,
        therefore raise `TypeError` — this is the only (documented) deviation from the
        contract. As a consequence, `insert_or_update` / `__setitem__` only work for new keys.

        Additional API (log-specific) :
            - `seq(key)`    : the position (0-based sequence number) of a record in the log
            - `since(seq)`  : every record from position `seq` onwards (in order)
            - `append(records, skip_existing = True)`   : idempotent (and atomic) multi-insert
            - iteration yields the records in insertion order

        Persistence :
            - `autoflush = True` (default) : every `insert` is appended to the file right away
              (write-through), which is what a durable log wants — a crash never loses an
              acknowledged record
            - `autoflush = False` : records are buffered and appended by `save_data` (never
              by `close`). Useful when the log is the inner database of a wrapper persisting
              other artefacts at `save` time (e.g. `VectorDatabase`), so both stay in sync
            - `path = None` : in-memory log (never persisted), e.g. for evaluation fixtures

        The file is de-duplicated by key at load time (a record logged twice is loaded once),
        `load_jsonl` skips a corrupt line, and a partial trailing line (crash mid-write) is
        terminated before the next append : the log is self-healing.

        Records are stored **with** their key fields, stringified like every `Database`
        identifier (the line is self-describing) ; `get` returns a shallow copy, to be treated
        as read-only.
    """
    def __init__(self, path, primary_key, *, autoflush = True):
        super().__init__(path, primary_key)

        self.autoflush  = autoflush

        self._records   = []
        self._key_to_seq    = {}
        # whether the file ends with a partial line (no trailing `\n`) : the next flush must
        # terminate it first, otherwise the new record would be glued to the corrupt line
        self._needs_newline = False

        if path and os.path.exists(self.data_file):
            for record in load_jsonl(self.data_file):
                try:
                    if not isinstance(record, dict): raise ValueError()
                    entry = self._get_entry(record)
                except (KeyError, ValueError):
                    logger.warning('Skipping record without primary key in {} : {}'.format(
                        self.data_file, record
                    ))
                    continue

                if entry in self._key_to_seq: continue

                self._key_to_seq[entry] = len(self._records)
                self._records.append(self._add_entry_to_value(entry, record))

            self._needs_newline = _ends_without_newline(self.data_file)

        self._n_flushed = len(self._records)

    @property
    def data_file(self):
        return os.path.join(self.path, 'data.jsonl') if self.path else None

    @property
    def pending(self):
        """ Number of records not yet written to the file """
        return len(self._records) - self._n_flushed

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        for record in self._records:
            yield dict(record)

    def __contains__(self, key):
        try:
            return self._get_entry(key) in self._key_to_seq
        except (KeyError, ValueError):
            return False

    def get(self, key):
        """ Return (a shallow copy of) the record for the given key """
        return dict(self._records[self.seq(key)])

    def seq(self, key):
        """ Return the sequence number (position in the log) of the given key """
        entry = self._get_entry(key)
        if entry not in self._key_to_seq:
            raise KeyError('The entry `{}` is not in the database'.format(entry))
        return self._key_to_seq[entry]

    def since(self, seq = 0):
        """ Return the records from position `seq` onwards (shallow copies, in order) """
        return [dict(record) for record in self._records[seq:]]

    def insert(self, data):
        """
            Append a new record to the log and return its entry (primary key)
            Raise a `ValueError` if the key is already in the log
        """
        entry = self._get_entry(data)
        if entry in self._key_to_seq:
            raise ValueError('The entry `{}` is already in the database'.format(entry))

        # convert *before* registering the key : a failing conversion leaves the log untouched
        record = self._add_entry_to_value(entry, to_json(data))

        self._key_to_seq[entry] = len(self._records)
        self._records.append(record)

        if self.autoflush: self._flush()
        return entry

    def append(self, records, *, skip_existing = True):
        """
            Append multiple records and return their sequence numbers.
            With `skip_existing`, a record whose key is already logged (or appears earlier in
            `records`) is silently skipped (idempotent replay) and its *existing* sequence
            number is returned.

            The operation is atomic : the whole batch is validated before anything is appended
        """
        new_records, new_seqs, seqs = [], {}, []
        for record in records:
            entry = self._get_entry(record)
            if entry in self._key_to_seq or entry in new_seqs:
                if not skip_existing:
                    raise ValueError('The entry `{}` is already in the database'.format(entry))
                seqs.append(self._key_to_seq.get(entry, new_seqs.get(entry)))
                continue

            new_seqs[entry] = len(self._records) + len(new_records)
            new_records.append(self._add_entry_to_value(entry, to_json(record)))
            seqs.append(new_seqs[entry])

        self._records.extend(new_records)
        self._key_to_seq.update(new_seqs)

        if self.autoflush: self._flush()
        return seqs

    def update(self, data):
        raise TypeError('{} is append-only : records cannot be updated'.format(
            self.__class__.__name__
        ))

    def pop(self, key):
        raise TypeError('{} is append-only : records cannot be removed'.format(
            self.__class__.__name__
        ))

    def filter(self, ** filters):
        raise TypeError('{} is append-only : records cannot be filtered out'.format(
            self.__class__.__name__
        ))

    def get_column(self, column):
        """ Return the values stored in `column` for each record (in order) """
        return [record.get(column, None) for record in self._records]

    def _flush(self):
        """
            Append the not-yet-written records to the file (no-op for in-memory logs).
            Only the data file is written : the config is the business of `save` (like every
            `Database`), so that a wrapper sharing the directory keeps its own config
        """
        if not self.path or self._n_flushed == len(self._records): return

        os.makedirs(self.path, exist_ok = True)
        if self._needs_newline:
            with open(self.data_file, 'ab') as file:
                file.write(b'\n')
            self._needs_newline = False

        dump_jsonl(self.data_file, self._records[self._n_flushed:], mode = 'a')
        self._n_flushed = len(self._records)

    def save_data(self, ** kwargs):
        """ Write the pending records (only relevant when `autoflush = False`) """
        self._flush()

    def save_config(self, ** kwargs):
        # the directory may not exist yet (empty log : nothing flushed)
        os.makedirs(self.path, exist_ok = True)
        super().save_config()

    # NB : `close` is deliberately *not* overridden (no flush) : with `autoflush = False`, the
    # records are meant to be written by `save`, together with whatever the owner persists
    # alongside (e.g. the vectors of a `VectorDatabase`) — an implicit flush at `__del__`
    # time would let the two drift apart

    def get_config(self):
        return {** super().get_config(), 'autoflush' : self.autoflush}

def _ends_without_newline(filename):
    """ Return whether `filename` is non-empty and does not end with `\\n` """
    with open(filename, 'rb') as file:
        file.seek(0, os.SEEK_END)
        if file.tell() == 0: return False
        file.seek(-1, os.SEEK_END)
        return file.read(1) != b'\n'
