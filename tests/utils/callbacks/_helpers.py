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

""" Shared test doubles for the inference-callbacks test-suite.

These stay pure-Python (no keras / tensorflow) so the whole `tests/utils/callbacks`
package runs without a DL backend. Importing `utils.callbacks` still pulls
`matplotlib` (via `displayer` -> `plot_utils`), which is a core project dependency.
"""

from threading import Lock

from utils.callbacks import Callback


class RecordingCallback(Callback):
    """ A `Callback` that records every `apply` invocation and counts `build` calls. """
    def __init__(self, return_value = 'applied', ** kwargs):
        super().__init__(** kwargs)
        self.return_value   = return_value
        self.calls          = []
        self.build_count    = 0

    def build(self):
        super().build()
        self.build_count += 1

    def apply(self, infos, output, ** kwargs):
        self.calls.append((infos, output, kwargs))
        return self.return_value


class RaisingCallback(Callback):
    """ A `Callback` whose `apply` always raises (to test error isolation). """
    def apply(self, infos, output, ** kwargs):
        raise RuntimeError('boom')


class RecordingSaveFn:
    """ A thread-safe `save_fn` double recording `(filename, data, kwargs)` tuples. """
    def __init__(self):
        self._lock  = Lock()
        self.calls  = []

    def __call__(self, filename, data, ** kwargs):
        with self._lock:
            self.calls.append((filename, data, kwargs))

    @property
    def filenames(self):
        with self._lock:
            return [c[0] for c in self.calls]


class FakeQueue:
    """ A minimal queue whose `put(** kwargs)` matches `QueueCallback`'s call convention. """
    def __init__(self):
        self.items = []

    def put(self, ** kwargs):
        self.items.append(kwargs)
