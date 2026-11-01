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

""" Pure-``pytest`` tests for :mod:`utils.image.image_augmentation`.

The transformations wrap ``tf.image.random_*`` / ``keras.ops`` functions -> marked
``tensorflow``. Each method is checked on its contract (shape / dtype / value range, and
whether it changes the image), never on exact random values.

Coverage :
  - ``augment_image`` dispatch : unknown method, ``prct = 0`` identity, ``prct = 1`` applies ;
  - every registered method : eager execution and ``tf.data`` (graph) compatibility ;
  - the box-safe (pixel-only) methods keep the shape and the [0, 1] range ;
  - ``rotate`` geometry (0° identity, 90° = counter-clockwise ``np.rot90``) and ``resize`` bounds.
"""

import numpy as np
import pytest

from utils.image.image_augmentation import (
    BOX_SAFE_METHODS, augment_image, rotate, resize, _image_augmentations_fn
)

pytestmark = pytest.mark.tensorflow

ALL_METHODS = sorted(_image_augmentations_fn)


@pytest.fixture
def image():
    return np.random.default_rng(0).uniform(size = (64, 48, 3)).astype('float32')


# --- dispatch ----------------------------------------------------------------------

def test_unknown_method_raises(image):
    with pytest.raises(ValueError):
        augment_image(image, 'unknown')

def test_prct_zero_is_identity(image):
    out = augment_image(image, ALL_METHODS, prct = 0.)
    np.testing.assert_allclose(np.asarray(out), image)

def test_callable_method(image):
    out = augment_image(image, lambda img, ** _: img * 0., prct = 1.)
    np.testing.assert_allclose(np.asarray(out), 0.)

def test_box_safe_methods_are_registered():
    assert set(BOX_SAFE_METHODS) <= set(ALL_METHODS)


# --- every method ------------------------------------------------------------------

@pytest.mark.parametrize('method', ALL_METHODS)
def test_method_eager(image, method):
    # `rot90` draws `n` in [0, 4) : force a non-identity rotation
    kwargs = {'n' : 1} if method == 'rot90' else {}
    out = np.asarray(augment_image(image, method, prct = 1., ** kwargs))
    assert out.dtype == np.float32
    assert out.ndim == 3 and out.shape[-1] == 3
    assert out.shape != image.shape or not np.allclose(out, image)

@pytest.mark.parametrize('method', ALL_METHODS)
def test_method_in_tf_data(image, method):
    import tensorflow as tf

    ds  = tf.data.Dataset.from_tensor_slices(image[None]).map(
        lambda img: augment_image(img, method, prct = 1.)
    )
    out = next(iter(ds)).numpy()
    assert out.ndim == 3 and out.shape[-1] == 3

@pytest.mark.parametrize('method', BOX_SAFE_METHODS)
def test_box_safe_method_keeps_shape_and_range(image, method):
    out = np.asarray(augment_image(image, method, prct = 1.))
    assert out.shape == image.shape
    assert out.min() >= 0. and out.max() <= 1.


# --- geometry ----------------------------------------------------------------------

def test_rotate_zero_is_identity(image):
    np.testing.assert_allclose(np.asarray(rotate(image, 0., 0.)), image, atol = 1e-5)

def test_rotate_90_is_counter_clockwise(image):
    square = image[:48]
    out = np.asarray(rotate(square, 90., 90., interpolation = 'nearest'))
    np.testing.assert_allclose(out, np.rot90(square, 1), atol = 1e-5)

def test_resize_within_scale_bounds(image):
    out = np.asarray(resize(image, min_scale = 0.5, max_scale = 1.))
    assert 32 <= out.shape[0] <= 64 and 24 <= out.shape[1] <= 48
