
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

import os
import logging
import numpy as np
import tensorflow as tf

from custom_architectures.current_blocks import Conv2DBN

logger = logging.getLogger(__name__)

FULL_YOLO_BACKEND_PATH  = "pretrained_models/yolo_backend/full_yolo_backend.h5"
TINY_YOLO_BACKEND_PATH  = "pretrained_models/yolo_backend/tiny_yolo_backend.h5"

class SpaseToDepth(tf.keras.layers.Layer):
    def call(self, inputs):
        return space_to_depth_x2(inputs)

def space_to_depth_x2(x):
    import tensorflow as tf
    return tf.nn.space_to_depth(x, block_size = 2)

def FullYoloBackend(input_shape,
                    weight_path = FULL_YOLO_BACKEND_PATH,
                   
                    name = 'feature_extractor',
                    ** kwargs
                   ):
    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    
    input_image = input_shape
    if isinstance(input_shape, int): input_shape = (input_shape, input_shape, 3)
    if isinstance(input_shape, tuple):
        input_image = tf.keras.layers.Input(shape = input_shape, name = 'input_image')
    

    x = input_image
    # Layers 1 and 2
    for i, filters in enumerate([32, 64]):
        x = Conv2DBN(
            x, filters = filters, kernel_size = (3,3), padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = 'max', drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(i+1)
        )
        
    # Layer 3
    x = Conv2DBN(
        x, filters = 128, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_3'
    )


    # Layer 4
    x = Conv2DBN(
        x, filters = 64, kernel_size = (1,1), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_4'
    )

    # Layer 5
    x = Conv2DBN(
        x, filters = 128, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = 'max', drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_5'
    )

    # Layer 6
    x = Conv2DBN(
        x, filters = 256, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_6'
    )

    # Layer 7
    x = Conv2DBN(
        x, filters = 128, kernel_size = (1,1), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_7'
    )

    # Layer 8
    x = Conv2DBN(
        x, filters = 256, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = 'max', drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_8'
    )

    # Layers 9 to 13
    for i, (filters, kernel) in enumerate([(512, (3,3)), (256, (1,1)), (512, (3,3)), (256, (1,1)), (512, (3,3))]):
        x = Conv2DBN(
            x, filters = filters, kernel_size = kernel,
            padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = None, drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(9 + i)
        )


    skip_connection = x

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    for i, (filters, kernel) in enumerate([(1024, (3,3)), (512, (1,1)), (1024, (3,3)), (512, (1,1)), (1024, (3,3)), (1024, (3,3)), (1024, (3,3))]):
        x = Conv2DBN(
            x, filters = filters, kernel_size = kernel,
            padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = None, drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(14 + i)
        )

    # Layer 21
    skip_connection = Conv2DBN(
        skip_connection, filters = 64, kernel_size = (1,1),
        padding = 'same', use_bias = False,

        bnorm = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_21'
    )
    skip_connection = SpaseToDepth()(skip_connection)

    x = tf.keras.layers.Concatenate()([skip_connection, x])

    # Layer 22
    output = Conv2DBN(
        x, filters = 1024, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_22'
    )
    
    model = tf.keras.Model(inputs = input_image, outputs = output, name = name)

    if weight_path is not None and os.path.exists(weight_path):
        logger.info("Loading weights from {}".format(weight_path))
        model.load_weights(weight_path)
    elif weight_path is not None:
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
         nb_box     = 5,
         input_shape    = None,
         flatten    = True,
         randomize  = True,
         name       = 'yolo',
         ** kwargs
        ):
    assert isinstance(feature_extractor, tf.keras.Model), 'Unhandled feature extractor (type {}) : {} '.format(type(feature_extractor), feature_extractor)
    
    if isinstance(input_shape, int): input_shape = (input_shape, input_shape, 3)
    
    if input_shape is None or input_shape == feature_extractor.input.shape[1:] and flatten:
        input_image = feature_extractor.input
        features    = feature_extractor.output
    else:
        input_image = tf.keras.layers.Input(shape = input_shape, name = 'input_image')
        features    = feature_extractor(input_image)
    
    grid_h, grid_w = features.shape[1:3]     
    
    # make the object detection layer
    output = tf.keras.layers.Conv2D(
        filters = nb_box * (4 + 1 + nb_class), kernel_size = (1,1), padding = 'same', 
        kernel_initializer = 'lecun_normal', name = 'detection_layer'
    )(features)
    
    output = tf.keras.layers.Reshape((grid_h, grid_w, nb_box, 4 + 1 + nb_class))(output)

    model = tf.keras.Model(inputs = input_image, outputs = output, name = name)

    if randomize:
        # initialize the weights of the detection layer
        layer = model.layers[-2]
        kernel, bias = layer.get_weights()

        new_kernel = np.random.normal(size = kernel.shape)  / (grid_h * grid_w)
        new_bias   = np.random.normal(size = bias.shape)    / (grid_h * grid_w)

        layer.set_weights([new_kernel, new_bias])

    return model

custom_objects  = {
    'SpaseToDepth'  : SpaseToDepth
}

custom_functions    = {
    'full_yolo' : FullYoloBackend,
    'FullYolo'  : FullYoloBackend,
    'tiny_yolo' : TinyYoloBackend,
    'TinyYolo'  : TinyYoloBackend,
    'YOLO'  : YOLO
}
