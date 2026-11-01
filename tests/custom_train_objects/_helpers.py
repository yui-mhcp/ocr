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

""" Shared helpers for the ``custom_train_objects`` test-suite.

Kept dependency-light (no ``keras`` import) so the keras-free test modules
(``test_history`` / ``test_checkpoint_manager``) can import it too. The keras
objects are only ever *passed in* by the keras-marked modules.
"""

import pytest


def optional_object(module, name):
    """ ``module.<name>``, or ``None`` when this project does not ship it.

        Each published sub-project only keeps the custom objects it uses (e.g. ``TextLoss``
        lives in the text projects), while this test-suite is shared by all of them.
    """
    return getattr(module, name, None)

def requires(obj, name):
    """ ``skipif`` mark for the tests of an object returned by ``optional_object`` """
    return pytest.mark.skipif(obj is None, reason = '`{}` is not part of this project'.format(name))

def optional_param(name, cls, * args, id, ** kwargs):
    """ ``pytest.param(name, cls(* args, ** kwargs))``, skipped when ``cls`` is not available """
    if cls is None:
        return pytest.param(name, None, id = id, marks = requires(cls, name))
    return pytest.param(name, cls(* args, ** kwargs), id = id)


def normalize_object(obj):
    """ Reduce a keras object (loss / metric / optimizer / scheduler / callback)
        to a plain, comparable structure :
          - anything exposing ``get_config`` -> its (recursively normalized) config ;
          - a bare callable (e.g. a loss function) -> its ``__name__`` ;
          - dicts / lists / tuples -> normalized element-wise.

        This mirrors the recursive ``get_config`` comparison the old
        ``TestCustomObject.assertEqual`` performed, but without importing keras.
    """
    if hasattr(obj, 'get_config'):
        return normalize_object(obj.get_config())
    if isinstance(obj, dict):
        return {k : normalize_object(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(normalize_object(v) for v in obj)
    if callable(obj):
        return getattr(obj, '__name__', repr(obj))
    return obj


def assert_config_roundtrip(load, obj, asserts):
    """ ``load(**obj.get_config())`` must rebuild an equivalent object.

        The ``fn`` key (present on keras ``*FunctionWrapper`` configs) is dropped :
        it holds a serialized callable that the factories re-resolve themselves.
    """
    config = {k : v for k, v in obj.get_config().items() if k != 'fn'}
    asserts.assert_equal(normalize_object(obj), normalize_object(load(** config)))


def assert_serialize_roundtrip(load, serialize, obj, asserts):
    """ ``load(serialize(obj))`` must rebuild an equivalent object. """
    asserts.assert_equal(normalize_object(obj), normalize_object(load(serialize(obj))))
