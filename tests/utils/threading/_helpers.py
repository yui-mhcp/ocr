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

""" Shared helpers for the ``utils.threading`` test-suite.

The ``_`` prefix keeps pytest from collecting this module as a test file (same
convention as the ``_base.py`` shared base classes) — it is only imported by the
sibling ``test_*.py`` modules and by the local ``conftest.py``.

Two things live here :
  - :func:`run_with_timeout` : run a callable in a worker thread and fail with a
    clear "possible deadlock" message if it does not return in time. This is a
    complement to the global ``pytest-timeout`` safety-net : it gives a readable
    assertion (and keeps the session alive on Windows, where ``pytest-timeout``
    falls back to the process-killing ``thread`` method).
  - :class:`Recorder` : a small thread-safe "spy" replacing the per-``TestCase``
    counter attributes of the old ``unittest`` suite, so the tests can stay plain
    functions. It is exposed as the ``recorder`` fixture (see ``conftest.py``).
"""

import threading


class _DEADLOCK:
    """ Sentinel marking that the worker thread never finished. """


def run_with_timeout(fn, *args, timeout = 5, **kwargs):
    """ Run ``fn(*args, **kwargs)`` in a daemon thread and return its result.

        Raises :class:`AssertionError` (``possible deadlock``) if the call does not
        finish within ``timeout`` seconds, and re-raises any exception raised by
        ``fn`` in the caller thread.
    """
    box = {'result' : _DEADLOCK, 'exc' : None}

    def _target():
        try:
            box['result'] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - propagated to the caller below
            box['exc'] = e

    thread = threading.Thread(target = _target, daemon = True)
    thread.start()
    thread.join(timeout)

    assert not thread.is_alive(), \
        'Possible deadlock : `{}` did not finish within {}s'.format(
            getattr(fn, '__name__', fn), timeout
        )

    if box['exc'] is not None:
        raise box['exc']
    return box['result']


def drain_queue(q):
    """ Non-blocking snapshot of a ``queue.Queue`` as a list (in FIFO order). """
    return [q.get() for _ in range(q.qsize())]


class Recorder:
    """ Thread-safe callback/​function spy used across the ``Stream`` tests.

        It replaces the ``self._counter`` / ``self._start_called`` / ... attributes
        of the former ``unittest.TestCase``. The bound methods are passed directly
        to a ``Stream`` (as ``fn``, ``callback``, ``start_callback`` ...) :

          - :meth:`counter_check`  : used as the stream ``fn`` (returns its input) ;
          - :meth:`callback_check` : item ``callback`` ;
          - :meth:`start_check` / :meth:`stop_check` : lifecycle callbacks.

        Authoritative verification is done **after** consumption via
        :meth:`check_counters` (assertions raised inside ``Stream`` callbacks are
        swallowed by ``Stream._run_callbacks``, so the lifecycle checks below are
        only best-effort — the final ``check_counters`` is the real guarantee).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        """ Zero every counter — used to reuse a single spy across two rounds. """
        self.counter         = 0
        self.start_called    = 0
        self.stop_called     = 0
        self.callback_called = 0
        self.callback_items  = []

    # --- spies ----------------------------------------------------------------
    def counter_check(self, i, max_workers = 0):
        """ Stream ``fn`` : in (single-threaded) ordered mode the i-th processed
            item must equal the running counter + 1. """
        with self._lock:
            if max_workers <= 1:
                assert i == self.counter + 1, \
                    'Out-of-order item : got {}, expected {}'.format(i, self.counter + 1)
            self.counter += 1
        return i

    def start_check(self):
        assert self.start_called == 0, 'start callback fired more than once'
        assert self.counter == 0, 'start callback fired after items were processed'
        self.start_called += 1

    def stop_check(self):
        assert self.start_called == 1, 'stop callback fired before start'
        assert self.stop_called == 0, 'stop callback fired more than once'
        self.stop_called += 1

    def callback_check(self, i):
        with self._lock:
            self.callback_called += 1
            self.callback_items.append(i)

    # --- assertions -----------------------------------------------------------
    def check_counters(self, *, start, stop, callback, counter = None):
        assert self.start_called == start, \
            'start_called : expected {}, got {}'.format(start, self.start_called)
        assert self.stop_called == stop, \
            'stop_called : expected {}, got {}'.format(stop, self.stop_called)
        assert self.callback_called == callback, \
            'callback_called : expected {}, got {} ({})'.format(
                callback, self.callback_called, self.callback_items
            )
        if counter is not None:
            assert self.counter == counter, \
                'counter : expected {}, got {}'.format(counter, self.counter)
