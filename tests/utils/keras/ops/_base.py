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

""" Shared helpers for the ``utils.keras.ops`` test-suite (plain module, not a
``unittest.TestCase`` — it is never collected since it does not match
``test_*.py``).

The single :func:`check_ops` helper factorizes the value + output-type assertions
that every op test needs, so the ``numpy`` / ``tensor`` backends share one code
path (the comparison is always done against an **independent numpy reference**,
keeping the ``numpy`` backend keras-free). Graph / XLA compatibility is checked
separately through ``asserts.assert_graph_compatible`` in ``tensorflow``-marked
tests.
"""


def to_tensor(x, dtype = None):
    """ Convert ``x`` to a keras tensor (lazy import so ``numpy`` cases stay keras-free). """
    import keras.ops as K
    return K.convert_to_tensor(x, dtype = dtype)

def prepare(backend, * arrays):
    """ Return the (numpy) reference ``arrays`` converted to the inputs of ``backend``.

        ``numpy``  -> arrays are returned untouched ; ``tensor`` -> each is converted
        to a keras tensor (preserving its dtype). A single array is returned unwrapped.
    """
    if backend == 'tensor':
        arrays = tuple(to_tensor(a) for a in arrays)
    return arrays[0] if len(arrays) == 1 else arrays

def _check_type(value, check_fn):
    """ Apply ``check_fn`` to ``value``, or to each element if it is a list / tuple. """
    if isinstance(value, (list, tuple)):
        for v in value: check_fn(v)
    else:
        check_fn(value)

def check_ops(fn, * args, ref, backend, asserts, target_type = None, max_err = 1e-6, ** kwargs):
    """ Run ``fn(*args, **kwargs)``, compare it to the numpy reference ``ref`` and check
        its output type matches the backend (np in -> np out, tensor in -> tensor out).

        ``target_type`` defaults to ``'array'`` for the ``numpy`` backend and ``'tensor'``
        for the ``tensor`` backend ; pass it explicitly (or ``None`` to skip the type
        check) for backend-independent cases (e.g. ``mixed`` inputs).
    """
    value = fn(* args, ** kwargs)
    asserts.assert_equal(ref, value, max_err = max_err)

    if target_type is None and backend in ('numpy', 'tensor'):
        target_type = 'array' if backend == 'numpy' else 'tensor'
    if target_type == 'array':
        _check_type(value, asserts.assert_array)
    elif target_type == 'tensor':
        _check_type(value, asserts.assert_tensor)

    return value
