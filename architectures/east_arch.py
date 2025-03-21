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

import keras
import keras.ops as K
import tensorflow as tf

from keras import layers

from .layers import get_activation
from .current_blocks import (
    _get_var, get_merging_layer, Conv2DBN, Conv3DBN, Conv2DTransposeBN, Conv3DTransposeBN
)

_default_vgg_layers =  [
    64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'
]

@keras.saving.register_keras_serializable('east')
class UpSampling2DWithAlignedCorners(layers.Layer):
    def __init__(self, scale_factor = 2, interpolation = 'bilinear', ** kwargs):
        super().__init__(** kwargs)
        self.scale_factor   = scale_factor
        self.interpolation  = interpolation
        
        if not isinstance(self.scale_factor, (list, tuple)):
            self.scale_factor = (self.scale_factor, self.scale_factor)
    
    def call(self, x):
        new_shape = (K.shape(x)[1] * self.scale_factor[0], K.shape(x)[2] * self.scale_factor[1])
        if keras.backend.backend() == 'tensorflow':
            import tensorflow as tf
            return tf.compat.v1.image.resize(
                x, size = new_shape, method = self.interpolation, align_corners = True
            )
        elif keras.backend.backend() == 'torch':
            import torch
            return torch.nn.functional.interpolate(
                x.permute(0, 3, 1, 2), new_shape, mode = self.interpolation, align_corners = True
            ).permute(0, 2, 3, 1)
        elif keras.backend.backend() == 'jax':
            return self.jax_resize_with_aligned_corners(x, new_shape, self.interpolation)
    
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if input_shape[1]: input_shape[1] *= self.scale_factor[0]
        if input_shape[2]: input_shape[2] *= self.scale_factor[1]
        
        return tuple(input_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'scale_factor'  : self.scale_factor,
            'interpolation' : self.interpolation
        })
        return config

    @staticmethod
    def jax_resize_with_aligned_corners(image, shape, method, antialias = True):
        """
            Method adapted from https://github.com/google/jax/issues/11206
            
            This is an alternative to `jax.image.resize`, which emulates `align_corners = True`
        """
        import jax
        import jax.numpy as jnp
        
        shape = (K.shape(image)[0], ) + shape + (K.shape(image)[3], )
        spatial_dims = tuple(
            i for i in range(4)
            if not jax.core.symbolic_equal_dim(K.shape(image)[i], shape[i])
        )
        scale = jnp.array([(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims])
        translation = -(scale / 2.0 - 0.5)
        return jax.image.scale_and_translate(
            image,
            shape,
            method  = method,
            scale   = scale,
            spatial_dims    = spatial_dims,
            translation = translation,
            antialias   = antialias
        )
    
    
def EASTVGG(input_shape   = (None, None, 3),
            output_dim    = None,
            final_name    = None,
            final_activation  = None,

            _layers       = _default_vgg_layers,
            batch_norm    = True,
            epsilon   = 1e-5,
            pretrained    = '{}/east_vgg16.pth',
            name  = 'features'
           ):
    if not isinstance(input_shape, (list, tuple)): input_shape = (input_shape, input_shape, 3)
    inputs = layers.Input(shape = input_shape, name = 'input_image')
    
    prefix = 'extractor_features'
    
    i = 0
    x = inputs
    residuals = []
    for l in _layers:
        if l == 'M':
            x = layers.MaxPooling2D(2, name = '{}_pool{}'.format(prefix, i))(x)
            residuals.append(x)
            i += 1
            continue
        
        x = layers.ZeroPadding2D(((1, 1), (1, 1)), name = '{}_pad{}'.format(prefix, i))(x)
        x = layers.Conv2D(
            l, kernel_size = 3, name = '{}_conv{}'.format(prefix, i)
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(
                epsilon = epsilon, name = '{}_norm{}'.format(prefix, i + 1)
            )(x)
        x = layers.ReLU(name = '{}_relu{}'.format(prefix, i + 2))(x)
        
        i += 2 if not batch_norm else 3
    
    residuals = residuals[1:-1]
    
    for i, res in enumerate(residuals[::-1]):
        x = UpSampling2DWithAlignedCorners(2, interpolation = 'bilinear')(x)
        x = layers.Concatenate()([x, res])
        for j in range(2 if i < len(residuals) - 1 else 3):
            idx = i * 2 + j + 1
            if j > 0:
                x = layers.ZeroPadding2D(
                    ((1, 1), (1, 1)), name = 'merge_pad{}'.format(idx)
                )(x)
            x = layers.Conv2D(
                128 // 2 ** i, kernel_size = 1 if j == 0 else 3, name = 'merge_conv{}'.format(idx)
            )(x)
            if batch_norm:
                x = layers.BatchNormalization(
                    epsilon = epsilon, name = 'merge_bn{}'.format(idx)
                )(x)
            x = layers.ReLU(name = 'merge_relu{}'.format(idx))(x)
    
    score   = layers.Conv2D(1, kernel_size = 1, name = 'output_conv1')(x)
    score   = layers.Activation('sigmoid', dtype = 'float32')(score)
    
    pos = layers.Conv2D(4, kernel_size = 1, name = 'output_conv2')(x)
    pos = layers.Activation('sigmoid', dtype = 'float32')(pos)

    angle   = layers.Conv2D(1, kernel_size = 1, name = 'output_conv3')(x)
    angle   = layers.Activation('sigmoid', dtype = 'float32')(angle)
    
    out     = layers.Concatenate()([score, pos, angle])
    model   = keras.Model(inputs, out, name = name)
    
    if pretrained:
        import torch
        
        from models.weights_converter import name_based_partial_transfer_learning
        
        if '{}' in pretrained:
            pretrained = pretrained.format('pretrained_models/pretrained_weights')
        
        name_based_partial_transfer_learning(model, torch.load(pretrained, map_location = 'cpu'))
    
    return model


custom_functions    = {
    'EASTVGG'   : EASTVGG
}


custom_objects  = {
    'UpSampling2DV1'    : UpSampling2DWithAlignedCorners,
    'UpSampling2DWithAlignedCorners'    : UpSampling2DWithAlignedCorners
}