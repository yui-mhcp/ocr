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

""" Keras-free test doubles for the `models.core` abstraction.

The whole point of these tests is to exercise the **wrapper logic** of `BaseModel`
(construction branching, config round-trip, processing pipeline wiring, prediction /
callback orchestration) **without loading a real model nor running any inference**.

Two doubles make that possible :

  - :class:`FakeRuntime` : a minimal concrete `Runtime` (the two abstract methods are
    stubbed to *raise*, since no test should ever run the forward pass). It carries a
    dummy ``engine`` so `BaseModel`'s model-setter accepts it as-is (no re-wrapping via
    `build_runtime`). ``supports_training`` stays `False` so the setter does not touch a
    `checkpoint_manager` engine.

  - :class:`DummyModel` : a leaf `BaseModel` whose ``build`` simply assigns a
    `FakeRuntime` (instead of a `keras.Model`) and which defines the mandatory
    ``input_signature`` / ``output_signature``. It never imports keras.

`FakeRuntime` is also registered under the ``'fake'`` runtime key (see the package
``conftest``), so the *inference-only* code path of `BaseModel.__init__`
(``runtime != 'keras'`` -> ``build_runtime(runtime, ...)``) and the keras-free
``save`` (only ``config.json`` is written) can be covered too.
"""

from utils.keras import TensorSpec
from utils.keras.runtimes import Runtime

from models.core import BaseModel


class FakeRuntime(Runtime):
    """ Minimal, inference-free `Runtime` double (see module docstring). """

    supports_training = False

    def __init__(self, path = None, *, engine = None, base_dtype = 'float32',
                 supports_training = None, ** kwargs):
        # NB: we bypass `Runtime.__init__` (which would call `load_engine`) on purpose.
        self.path   = path
        self.engine = engine if engine is not None else object()
        self._base_dtype = base_dtype
        if supports_training is not None:
            self.supports_training = supports_training

    @property
    def base_dtype(self):
        return self._base_dtype

    def __call__(self, * args, ** kwargs):
        raise AssertionError(
            'FakeRuntime.__call__ must never run : these tests do not perform inference'
        )

    @staticmethod
    def load_engine(path, ** kwargs):
        raise AssertionError('FakeRuntime.load_engine must never be called')


class DummyModel(BaseModel):
    """ Leaf `BaseModel` whose `build` assigns a `FakeRuntime` instead of a keras model.

        Signatures default to trivial fixed specs but can be overridden per-instance
        (constructor kwargs `input_signature` / `output_signature`) to test the derived
        shape / dtype / unbatched properties.
    """

    def __init__(self,
                 *,
                 input_signature  = None,
                 output_signature = None,
                 fake_runtime     = None,
                 ** kwargs
                ):
        # set *before* `super().__init__` : `build` (called from it) reads `_fake_runtime`,
        # and the signature properties read `_input_signature` / `_output_signature`.
        self._input_signature  = input_signature if input_signature is not None else \
            TensorSpec(shape = (None, 8), dtype = 'float32')
        self._output_signature = output_signature if output_signature is not None else \
            TensorSpec(shape = (None, 2), dtype = 'float32')
        self._fake_runtime = fake_runtime
        super().__init__(** kwargs)

    def build(self, ** kwargs):
        self.model = self._fake_runtime if self._fake_runtime is not None else FakeRuntime()

    @property
    def input_signature(self):
        return self._input_signature

    @property
    def output_signature(self):
        return self._output_signature


class NoSignatureModel(BaseModel):
    """ Leaf model that leaves `input_signature` / `output_signature` unimplemented, to test
        that the base class enforces the signature contract (`NotImplementedError`). """

    def build(self, ** kwargs):
        self.model = FakeRuntime()


class OtherDummyModel(DummyModel):
    """ A distinct class re-using `DummyModel`'s body, to test the `ModelInstances` guard that
        rejects reloading a persisted config under a mismatching `class_name`. """
