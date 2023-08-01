# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D

from custom_layers import get_activation
from custom_architectures.current_blocks import (
    _get_var, _get_concat_layer, Conv2DBN, Conv3DBN, Conv2DTransposeBN, Conv3DTransposeBN, SeparableConv3DBN
)

_default_vgg_layers =  [
    64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'
]

class UpSampling2DV1(tf.keras.layers.Layer):
    def __init__(self, scale_factor = 2, interpolation = 'bilinear', align_corners = False, ** kwargs):
        super().__init__(** kwargs)
        self.scale_factor   = scale_factor
        self.interpolation  = interpolation
        self.align_corners  = align_corners
        
        if not isinstance(self.scale_factor, (list, tuple)):
            self.scale_factor = (self.scale_factor, self.scale_factor)
        
    def call(self, x):
        return tf.compat.v1.image.resize(
            x,
            size    = (tf.shape(x)[1] * self.scale_factor[0], tf.shape(x)[2] * self.scale_factor[1]),
            method  = self.interpolation,
            align_corners = self.align_corners
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'scale_factor'  : self.scale_factor,
            'interpolation' : self.interpolation,
            'align_corners' : self.align_corners
        })
        return config
    
    
def VGGBNUNet(input_shape   = (None, None, 3),
              output_dim    = None,
              final_name    = None,
              final_activation  = None,
              
              layers        = _default_vgg_layers,
              batch_norm    = True,
              epsilon   = 1e-5,
              pretrained    = '{}/east_vgg16.pth',
              name  = 'features'
             ):
    if not isinstance(input_shape, (list, tuple)): input_shape = (input_shape, input_shape, 3)
    inputs = tf.keras.layers.Input(shape = input_shape, name = 'input_image')
    
    prefix = 'extractor/features'
    
    i = 0
    x = inputs
    residuals = []
    for l in layers:
        if l == 'M':
            x = tf.keras.layers.MaxPooling2D(2, name = '{}/pool{}'.format(prefix, i))(x)
            residuals.append(x)
            i += 1
            continue
        
        x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name = '{}/pad{}'.format(prefix, i))(x)
        x = tf.keras.layers.Conv2D(
            l, kernel_size = 3, name = '{}/conv{}'.format(prefix, i)
        )(x)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization(
                epsilon = epsilon, name = '{}/norm{}'.format(prefix, i + 1)
            )(x)
        x = tf.keras.layers.ReLU(name = '{}/relu{}'.format(prefix, i + 2))(x)
        
        i += 2 if not batch_norm else 3
    
    residuals = residuals[1:-1]
    
    for i, res in enumerate(residuals[::-1]):
        x = UpSampling2DV1(2, interpolation = 'bilinear', align_corners = True)(x)
        x = tf.keras.layers.Concatenate()([x, res])
        for j in range(2 if i < len(residuals) - 1 else 3):
            idx = i * 2 + j + 1
            if j > 0:
                x = tf.keras.layers.ZeroPadding2D(
                    ((1, 1), (1, 1)), name = 'merge/pad{}'.format(idx)
                )(x)
            x = tf.keras.layers.Conv2D(
                128 // 2 ** i, kernel_size = 1 if j == 0 else 3, name = 'merge/conv{}'.format(idx)
            )(x)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization(
                    epsilon = epsilon, name = 'merge/bn{}'.format(idx)
                )(x)
            x = tf.keras.layers.ReLU(name = 'merge/relu{}'.format(idx))(x)
    
    score   = tf.keras.layers.Conv2D(1, kernel_size = 1, name = 'output/conv1')(x)
    score   = tf.keras.layers.Activation('sigmoid', dtype = 'float32')(score)
    
    pos = tf.keras.layers.Conv2D(4, kernel_size = 1, name = 'output/conv2')(x)
    pos = tf.keras.layers.Activation('sigmoid', dtype = 'float32')(pos)

    angle   = tf.keras.layers.Conv2D(1, kernel_size = 1, name = 'output/conv3')(x)
    angle   = tf.keras.layers.Activation('sigmoid', dtype = 'float32')(angle)
    
    out     = [score, pos, angle]
    model   = tf.keras.Model(inputs, out, name = name)
    
    if pretrained:
        import torch
        
        from models.weights_converter import name_based_partial_transfer_learning
        
        if '{}' in pretrained: pretrained = pretrained.format('pretrained_models/pretrained_weights')
        
        name_based_partial_transfer_learning(model, torch.load(pretrained, map_location = 'cpu'))
    
    return model

def VGGUNet(input_shape    = 512,
            output_dim     = 1,
         
            n_conv_per_stage   = lambda i: 1 if i == 0 else 2,
         
            filters        = [16, 32, 64, 128, 256],
            kernel_size    = 3,
            strides        = 1,
            use_bias       = True,
         
            activation     = 'relu',
            pool_type      = 'max',
            pool_strides   = 2,
            bnorm          = 'after',
            drop_rate      = 0.25,
         
            n_middle_stages = 1,
            n_middle_conv   = 2,
            middle_filters  = 256,
            middle_kernel_size = 3,
            middle_use_bias    = True,
            middle_activation  = 'relu',
            middle_bnorm       = 'never',
            middle_drop_rate   = 0.25,
         
            upsampling_activation  = None,
         
            concat_mode    = 'concat',
             
            final_name     = 'segmentation_layer',
            final_activation   = 'sigmoid',
            
            freeze  = True,
         
            name   = None,
            ** kwargs
           ):
    if not isinstance(input_shape, tuple): input_shape = (input_shape, input_shape, 3)

    assert len(input_shape) == 3

    conv_fn     = Conv2DBN
    pool_fn     = MaxPooling2D if pool_type == 'max' else AveragePooling2D
    upsample_fn = Conv2DTransposeBN
    out_layer   = tf.keras.layers.Conv2D

    vgg = tf.keras.applications.VGG16(
        input_shape = input_shape, include_top = False, weights = 'imagenet'
    )
    
    x = vgg.get_layer('block5_pool').output
    residuals = [None] + [
        vgg.get_layer('block{}_pool'.format(i)).output
        for i in range(1, 5)
    ]

    ####################
    #    Middle part   #
    ####################
    
    for i in range(n_middle_stages):
        n_conv = _get_var(n_middle_conv, i)
        for j in range(n_conv):
            x = conv_fn(
                x,
                filters     = _get_var(_get_var(middle_filters, i), j),
                kernel_size = _get_var(_get_var(middle_kernel_size, i), j),
                use_bias    = _get_var(_get_var(middle_use_bias, i), j),
                strides     = 1,
                padding     = 'same',

                activation  = _get_var(_get_var(middle_activation, i), j),

                bnorm       = _get_var(_get_var(middle_bnorm, i), j),
                drop_rate   = _get_var(_get_var(middle_drop_rate, i), j),

                bn_name = 'middle_bn{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1)),
                name    = 'middle_conv{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1))
            )

    ####################
    # Upsampling part  #
    ####################
    
    for i in reversed(range(5)):
        x = upsample_fn(
            x,
            filters     = _get_var(filters, i),
            kernel_size = 1 + (_get_var(pool_strides, i) if pool_type else _get_var(strides, i)),
            strides     = _get_var(pool_strides, i) if pool_type else _get_var(strides, i),
            padding     = 'same',
            activation  = _get_var(upsampling_activation, i),
            bnorm       = 'never',
            drop_rate   = 0.,
            name        = 'upsampling_{}'.format(i + 1)
        )
        if i > 0:
            concat_mode_i = _get_var(concat_mode, i)
            if concat_mode_i is not None:
                x = _get_concat_layer(concat_mode_i)([x, residuals[i]])
        
        n_conv = _get_var(kwargs.get('up_n_conv_per_stage', n_conv_per_stage), i)
        for j in range(n_conv):
            x = conv_fn(
                x,
                filters     = _get_var(_get_var(kwargs.get('up_filters', filters), i), j),
                kernel_size = _get_var(_get_var(kwargs.get('up_kernel_size', kernel_size), i), j),
                strides     = 1,
                use_bias    = _get_var(_get_var(kwargs.get('up_use_bias', use_bias), i), j),
                padding     = 'same',

                activation  = _get_var(_get_var(kwargs.get('up_activation', activation), i), j),

                bnorm       = _get_var(_get_var(kwargs.get('up_bnorm', bnorm), i), j),
                drop_rate   = 0. if j < n_conv - 1 else _get_var(kwargs.get('up_drop_rate', drop_rate), i),

                bn_name = 'up_bn{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1)),
                name    = 'up_conv{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1))
            )

    ####################
    #   Output part    #
    ####################
    
    if isinstance(output_dim, (list, tuple)):
        out = [out_layer(
            filters     = out_dim_i,
            kernel_size = 1,
            strides     = 1,
            name    = '{}_{}'.format(final_name, i + 1) if isinstance(final_name, str) else final_name[i]
        )(x) for i, out_dim_i in enumerate(output_dim)]
        out = [
            get_activation(_get_var(final_activation, i), dtype = 'float32')(out_i) if _get_var(final_activation, i) else out_i
            for out_i in out
        ]
    else:
        out = out_layer(
            output_dim, kernel_size = 1, strides = 1, name = final_name
        )(x)
        out = get_activation(final_activation, dtype = 'float32')(out)
    
    model = tf.keras.models.Model(inputs = vgg.input, outputs =  out, name = name)
    
    if freeze:
        for layer in vgg.layers: layer.trainable = False
    
    return model

def UNet(input_shape    = 512,
         output_dim     = 1,
         
         n_stages       = 4,
         n_conv_per_stage   = lambda i: 1 if i == 0 else 2,
         
         filters        = [32, 64, 128, 256],
         kernel_size    = 3,
         strides        = 1,
         use_bias       = True,
         
         activation     = 'relu',
         pool_type      = 'max',
         pool_strides   = 2,
         bnorm          = 'never',
         drop_rate      = 0.25,
         
         n_middle_stages = 1,
         n_middle_conv   = 2,
         middle_filters  = 512,
         middle_kernel_size = 3,
         middle_use_bias    = True,
         middle_activation  = 'relu',
         middle_bnorm       = 'never',
         middle_drop_rate   = 0.25,
         
         upsampling_activation  = None,
         
         concat_mode    = 'concat',
         
         final_name     = 'segmentation_layer',
         final_activation   = 'sigmoid',
         
         mixed_precision     = False,
         
         name   = None,
         ** kwargs
        ):
    if not isinstance(input_shape, tuple): input_shape = (input_shape, input_shape, 3)

    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    if len(input_shape) == 3:
        conv_fn     = Conv2DBN
        pool_fn     = MaxPooling2D if pool_type == 'max' else AveragePooling2D
        upsample_fn = Conv2DTransposeBN
        out_layer   = tf.keras.layers.Conv2D
    else:
        conv_fn     = Conv3DBN
        pool_fn     = MaxPooling3D if pool_type == 'max' else AveragePooling3D
        upsample_fn = Conv3DTransposeBN
        out_layer   = tf.keras.layers.Conv3D

    inputs  = tf.keras.layers.Input(shape = input_shape, name = 'input_image')

    x = inputs
    residuals = []
    ##############################
    #     Downsampling part      #
    ##############################
    
    for i in range(n_stages):
        n_conv = _get_var(n_conv_per_stage, i)
        for j in range(n_conv):
            if j == n_conv - 1 and not pool_type: residuals.append(x)
            
            x = conv_fn(
                x,
                filters     = _get_var(_get_var(filters, i), j),
                kernel_size = _get_var(_get_var(kernel_size, i), j),
                strides     = 1 if j < n_conv - 1 else _get_var(strides, i),
                use_bias    = _get_var(_get_var(use_bias, i), j),
                padding     = 'same',

                activation  = _get_var(_get_var(activation, i), j),

                bnorm       = _get_var(_get_var(bnorm, i), j),
                drop_rate   = 0. if pool_type else _get_var(drop_rate, i),

                bn_name = 'down_bn{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1)),
                name    = 'down_conv{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1))
            )

        if pool_type:
            residuals.append(x)
            x = pool_fn(
                pool_size = _get_var(pool_strides, i), strides = _get_var(pool_strides, i)
            )(x)

            if _get_var(drop_rate, i) > 0:
                x = tf.keras.layers.Dropout(_get_var(drop_rate, i))(x)
    
    ####################
    #    Middle part   #
    ####################
    
    for i in range(n_middle_stages):
        n_conv = _get_var(n_middle_conv, i)
        for j in range(n_conv):
            x = conv_fn(
                x,
                filters     = _get_var(_get_var(middle_filters, i), j),
                kernel_size = _get_var(_get_var(middle_kernel_size, i), j),
                use_bias    = _get_var(_get_var(middle_use_bias, i), j),
                strides     = 1,
                padding     = 'same',

                activation  = _get_var(_get_var(middle_activation, i), j),

                bnorm       = _get_var(_get_var(middle_bnorm, i), j),
                drop_rate   = _get_var(_get_var(middle_drop_rate, i), j),

                bn_name = 'middle_bn{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1)),
                name    = 'middle_conv{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1))
            )

    ####################
    # Upsampling part  #
    ####################
    
    for i in reversed(range(n_stages)):
        x = upsample_fn(
            x,
            filters     = _get_var(filters, i),
            kernel_size = 1 + (_get_var(pool_strides, i) if pool_type else _get_var(strides, i)),
            strides     = _get_var(pool_strides, i) if pool_type else _get_var(strides, i),
            padding     = 'same',
            activation  = _get_var(upsampling_activation, i),
            bnorm       = 'never',
            drop_rate   = 0.,
            name        = 'upsampling_{}'.format(i + 1)
        )
        concat_mode_i = _get_var(concat_mode, i)
        if concat_mode_i is not None:
            x = _get_concat_layer(concat_mode_i)([x, residuals[i]])
        
        n_conv = _get_var(kwargs.get('up_n_conv_per_stage', n_conv_per_stage), i)
        for j in range(n_conv):
            x = conv_fn(
                x,
                filters     = _get_var(_get_var(kwargs.get('up_filters', filters), i), j),
                kernel_size = _get_var(_get_var(kwargs.get('up_kernel_size', kernel_size), i), j),
                strides     = 1,
                use_bias    = _get_var(_get_var(kwargs.get('up_use_bias', use_bias), i), j),
                padding     = 'same',

                activation  = _get_var(_get_var(kwargs.get('up_activation', activation), i), j),

                bnorm       = _get_var(_get_var(kwargs.get('up_bnorm', bnorm), i), j),
                drop_rate   = 0. if j < n_conv - 1 else _get_var(kwargs.get('up_drop_rate', drop_rate), i),

                bn_name = 'up_bn{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1)),
                name    = 'up_conv{}{}'.format(i + 1, '' if n_conv == 1 else '-{}'.format(j + 1))
            )

    ####################
    #   Output part    #
    ####################
    
    if isinstance(output_dim, (list, tuple)):
        out = [out_layer(
            filters     = out_dim_i,
            kernel_size = 1,
            strides     = 1,
            name    = '{}_{}'.format(final_name, i + 1) if isinstance(final_name, str) else final_name[i]
        )(x) for i, out_dim_i in enumerate(output_dim)]
        out = [
            get_activation(_get_var(final_activation, i), dtype = 'float32')(out_i) if _get_var(final_activation, i) else out_i
            for out_i in out
        ]
    else:
        out = out_layer(
            output_dim, kernel_size = 1, strides = 1, name = final_name
        )(x)
        out = get_activation(final_activation, dtype = 'float32')(out)
    
    return tf.keras.Model(inputs = inputs, outputs = out, name = name)

custom_functions    = {
    'UNet'    : UNet,
    'VGGUNet'   : VGGUNet,
    'VGGBNUNet' : VGGBNUNet
}


custom_objects  = {
    'UpSampling2DV1'    : UpSampling2DV1
}