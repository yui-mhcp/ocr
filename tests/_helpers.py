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

""" Low-level test helpers : paths, side-effects and `tensorflow` utilities.

This module is intentionally dependency-light (only `os` / `sys` / `numpy`) and
performs the package import-time side-effects (silencing warnings, creating the
temporary directory). It is imported by both `tests.asserts` and `tests.__init__`
so neither has to import the other (avoids a circular import).
"""

import os
import sys
import warnings
import numpy as np

from functools import cache

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir    = os.path.join(os.path.dirname(__file__), 'data')
temp_dir    = os.path.join(os.path.dirname(__file__), 'tmp')
reproductibility_dir    = os.path.join(os.path.dirname(__file__), '__reproduction')

os.makedirs(temp_dir, exist_ok = True)

# stores the functions that already failed graph-compatibility so the (expensive)
# check is only performed once per session
_graph_failed = {}

@cache
def is_tensorflow_available():
    """ Return whether `tensorflow` is available without importing it ! """
    return _is_module_available('tensorflow')

@cache
def is_keras_available():
    """ Return whether `keras` is available without importing it ! """
    return _is_module_available('keras')

@cache
def is_torch_available():
    """ Return whether `torch` is available without importing it ! """
    return _is_module_available('torch')

@cache
def is_cv2_available():
    """ Return whether `cv2` (opencv) is available without importing it ! """
    return _is_module_available('cv2')

def _is_module_available(name):
    """ Return whether `name` can be imported without actually importing it. """
    if name in sys.modules: return True

    for finder in sys.meta_path:
        spec = finder.find_spec(name, None)
        if spec is not None: return True
    return False

@cache
def get_graph_function(fn, reduce_retracing = True, jit_compile = False):
    import tensorflow as tf

    @tf.function(reduce_retracing = reduce_retracing, jit_compile = jit_compile)
    def foo(* args, ** kwargs):
        return fn(* args, ** kwargs)
    
    if hasattr(fn, 'name'):
        fn.__name__ = fn.name
    elif hasattr(fn, '__name__'):
        fn.__name__ = fn.__name__
    else:
        fn.__name__ = fn.__class__.__name__
    return foo

def get_xla_function(fn):
    return get_graph_function(fn, jit_compile = True)

def convert_to_tf_tensor(x):
    import keras
    import keras.ops as K
    import tensorflow as tf

    if K.is_tensor(x):
        if keras.backend.backend() == 'tensorflow': return x
        return tf.convert_to_tensor(K.convert_to_numpy(x))
    elif isinstance(x, np.ndarray):
        return tf.convert_to_tensor(x)
    return x
