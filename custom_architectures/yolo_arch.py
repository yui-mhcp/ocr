# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import keras
import logging
import numpy as np
import keras.ops as K

from keras import layers

from .current_blocks import Conv2DBN

logger = logging.getLogger(__name__)

FULL_YOLO_BACKEND_PATH  = "pretrained_models/pretrained_weights/yolo_backend/full_yolo_backend.h5"
TINY_YOLO_BACKEND_PATH  = "pretrained_models/pretrained_weights/yolo_backend/tiny_yolo_backend.h5"

@keras.saving.register_keras_serializable('yolo')
class SpaseToDepth(layers.Layer):
    def __init__(self, ** kwargs):
        super().__init__(** kwargs)
        
        self.call = self.tf_call if keras.backend.backend() == 'tensorflow' else self.k_call
    
    def tf_call(self, inputs):
        import tensorflow as tf
        return tf.nn.space_to_depth(inputs, 2)
    
    def k_call(self, inputs):
        shape   = K.shape(inputs)
        batch_size  = shape[0]
        new_h       = shape[1] // 2
        new_w       = shape[2] // 2
        d           = shape[3]
        return K.reshape(K.transpose(K.reshape(
            inputs, [batch_size, new_h, 2, new_w, 2, d]
        ), [0, 1, 3, 2, 4, 5]), [batch_size, new_h, new_w, d * 2 * 2])

@keras.saving.register_keras_serializable('yolo')
class YOLOLayer(layers.Layer):
    def __init__(self, anchors, ** kwargs):
        super().__init__(** kwargs)
        assert len(anchors) % 2 == 0
        
        self.anchors    = K.reshape(K.convert_to_tensor(anchors, 'float32'), [1, 1, 1, -1, 2])
        self.nb_box     = len(anchors) // 2
    
    def build(self, input_shape):
        self.grid_h = input_shape[1]
        self.grid_w = input_shape[2]
        self.last_dim   = input_shape[3] // self.nb_box
        
        cell_x = K.tile(
            K.arange(self.grid_w, dtype = 'float32')[None], [self.grid_h, 1]
        )[None, :, :, None, None]
        cell_y  = K.transpose(cell_x, (0, 2, 1, 3, 4))

        self.grid_cell  = K.concatenate([cell_x, cell_y], axis = -1)
    
    def call(self, inputs):
        out = K.reshape(
            inputs, [K.shape(inputs)[0], self.grid_h, self.grid_w, self.nb_box, self.last_dim]
        )
        return K.concatenate([
            K.sigmoid(out[..., :2]) + self.grid_cell,
            K.exp(out[..., 2:4]) * self.anchors,
            K.sigmoid(out[..., 4:5]),
            K.softmax(out[..., 5:], axis = -1)
        ], axis = -1)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'anchors' : [float(v) for v in K.convert_to_numpy(self.anchors).reshape(-1)]
        })
        return config
    
def FullYoloBackend(input_shape = (416, 416, 3),
                    weight_path = FULL_YOLO_BACKEND_PATH,
                   
                    name = 'feature_extractor',
                    ** kwargs
                   ):
    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    
    input_image = input_shape
    if isinstance(input_shape, int): input_shape = (input_shape, input_shape, 3)
    if isinstance(input_shape, tuple):
        input_image = layers.Input(shape = input_shape, name = 'input_image')
    
    _config = {
        'padding'   : 'same',
        'use_bias'  : False,
        
        'bnorm'     : 'after',
        'activation'    : 'leaky',
        'activation_kwargs' : {'negative_slope' : 0.1},
        'drop_rate' : 0.
    }
    x = input_image
    # Layers 1 and 2
    
    for i, (filters, kernel, pooling) in enumerate([
        (32, 3, 'max'),
        (64, 3, 'max'),
        (128, 3, None),
        (64, 1, None),
        (128, 3, 'max'),
        (256, 3, None),
        (128, 1, None),
        (256, 3, 'max'),
        (512, 3, None),
        (256, 1, None),
        (512, 3, None),
        (256, 1, None),
        (512, 3, None)
    ], start = 1):
        x = Conv2DBN(
            x,
            filters     = filters,
            kernel_size = kernel,
            pooling     = pooling,
            name    = 'conv_{}'.format(i),
            ** _config
        )

    skip_connection = x

    x = layers.MaxPooling2D(2)(x)

    for i, (filters, kernel) in enumerate([
        (1024, 3),
        (512, 1),
        (1024, 3),
        (512, 1),
        (1024, 3),
        (1024, 3),
        (1024, 3)
    ], start = 14):
        x = Conv2DBN(
            x, filters = filters, kernel_size = kernel, name = f'conv_{i}', ** _config
        )

    # Layer 21
    skip_connection = Conv2DBN(
        skip_connection, filters = 64, kernel_size = 1, name = 'conv_21', ** _config
    )
    skip_connection = SpaseToDepth()(skip_connection)

    x = layers.Concatenate()([skip_connection, x])

    # Layer 22
    output = Conv2DBN(
        x, filters = 1024, kernel_size = 3, name = 'conv_22', ** _config
    )
    
    model = keras.Model(inputs = input_image, outputs = output, name = name)

    if weight_path and os.path.exists(weight_path):
        logger.info("Loading weights from {}".format(weight_path))
        model.load_weights(weight_path)
    elif weight_path:
        logger.warning('Weight file {} does not exist !'.format(weight_path))

    return model

def TinyYoloBackend(input_shape,
                    weight_path = TINY_YOLO_BACKEND_PATH,
                   
                    name = 'feature_extractor',
                    ** kwargs
                   ):
    input_image = input_shape
    if isinstance(input_shape, int): input_shape = (input_shape, input_shape, 3)
    if isinstance(input_shape, tuple):
        input_image = tf.keras.layers.Input(shape = input_shape, name = 'input_image')

    # Layer 1
    x = Conv2DBN(
        input_image, filters = 16, kernel_size = (3,3),
        padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = 'max', drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_1'
    )

    # Layer 2 - 5
    for i in range(4):
        x = Conv2DBN(
            x, filters = 32 * (2**i), kernel_size = (3,3),
            padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = 'max', drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(i+2)
        )

    # Layer 6
    x = Conv2DBN(
        x, filters = 512, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', drop_rate = 0.,
        pooling = 'max', pool_strides = (1,1), pool_padding = 'same',
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_6'
    )

    # Layer 7 - 8
    for i in range(0,2):
        x = Conv2DBN(
            x, filters = 1024, kernel_size = (3,3), padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = None, drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(i+7)
        )
    
    model = tf.keras.Model(inputs = input_image, outputs = x, name = name)

    if weight_path is not None and os.path.exists(weight_path):
        logger.info('Loading backend weights from {}'.format(weight_path))
        model.load_weights(weight_path)
    elif weight_path is not None:
        logger.warning('Weight file {} does not exist !'.format(weight_path))

    return model

def YOLO(feature_extractor,
         nb_class,
         anchors,
         input_shape    = None,
         flatten    = True,
         randomize  = False,
         name       = 'yolo',
         ** kwargs
        ):
    assert len(anchors) % 2 == 0
    assert isinstance(feature_extractor, keras.Model), 'Unhandled feature extractor (type {}) : {} '.format(type(feature_extractor), feature_extractor)
    
    if isinstance(input_shape, int): input_shape = (input_shape, input_shape, 3)
    
    if (input_shape is None or input_shape == feature_extractor.input.shape[1:]) and flatten:
        input_image = feature_extractor.input
        features    = feature_extractor.output
    else:
        input_image = layers.Input(shape = input_shape, name = 'input_image')
        features    = feature_extractor(input_image)
    
    nb_box = len(anchors) // 2
    # make the object detection layer
    output = layers.Conv2D(
        filters     = nb_box * (4 + 1 + nb_class),
        kernel_size = 1,
        padding     = 'same', 
        kernel_initializer = 'lecun_normal',
        name        = 'detection_layer'
    )(features)
    output  = YOLOLayer(anchors)(output)
    
    model   = keras.Model(inputs = input_image, outputs = output, name = name)

    if randomize:
        # initialize the weights of the detection layer
        layer = model.layers[-2]
        kernel, bias = layer.get_weights()

        new_kernel = np.random.normal(size = kernel.shape)  / (grid_h * grid_w)
        new_bias   = np.random.normal(size = bias.shape)    / (grid_h * grid_w)

        layer.set_weights([new_kernel, new_bias])

    return model


# aliases

full_yolo = FullYOLO = FullYoloBackend
tiny_yolo = TinyYOLO = TinyYoloBackend