
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

import numpy as np
import tensorflow as tf

from loggers import timer
from utils.image.box_utils.geo_utils import *
from models.detection.base_detector import BaseDetector

class EAST(BaseDetector):
    def __init__(self,
                 * args,
                 image_normalization    = 'east',
                 resize_kwargs  = {'antialias' : True},
                 ** kwargs
                ):
        super().__init__(
            * args,
            resize_kwargs   = resize_kwargs,
            image_normalization = image_normalization,
            ** kwargs
        )
    
    def _build_model(self, architecture = 'VGGBNUNet', ** kwargs):
        super()._build_model(model = {
            'architecture_name' : architecture,
            'input_shape'   : self.input_size,
            'output_dim'    : [1, 4, 1] + ([self.nb_class] if self.use_labels else []),
            'final_activation'  : ['sigmoid', 'sigmoid', 'sigmoid', 'softmax'],
            'final_name'    : ['score_map', 'geo_map', 'theta_map', 'class_map'],
            ** kwargs
        })
    
    @property
    def output_signature(self):
        sign = (
            tf.TensorSpec(shape = (None, ) + self.input_size[:-1], dtype = tf.float32),
            tf.TensorSpec(shape = (None, ) + self.input_size[:-1] + (5, ), dtype = tf.float32),
            tf.TensorSpec(shape = (None, ) + self.input_size[:-1], dtype = tf.bool),
        )
        if self.use_labels:
            sign += (
                tf.TensorSpec(shape = (None, ) + self.input_size[:-1] + (1, ), dtype = tf.int32), 
            )
        return sign
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            min_poly_size   = 6,
            max_wh_factor   = 5,
            shrink_ratio    = 0.1
        )
    
    def compile(self, loss = 'EASTLoss', ** kwargs):
        super().compile(loss = loss, ** kwargs)

    @timer
    def decode_output(self, model_output, nms_method = 'lanms', normalize = True, ** kwargs):
        kwargs.setdefault('threshold',      self.obj_threshold)
        kwargs.setdefault('nms_threshold',  self.nms_threshold)
        kwargs.setdefault('method',         nms_method)
        
        score_map, geo_map, theta_map = [out.numpy() for out in model_output[:3]]
        
        return restore_polys_from_map(
            score_map   = score_map,
            geo_map     = geo_map * 512,
            theta_map   = (theta_map - 0.5) * np.pi,
            normalize   = normalize,
            scale   = self.downscale_factor,
            ** kwargs
        )

    def get_rbox(self,
                 polys,
                 labels,
                 img_size,
                 
                 min_poly_size  = -1,
                 shrink_ratio   = -1,
                 max_wh_factor  = -1
                ):
        """
            Generates score_map and geo_map

            Arguments :
                - polys : np.ndarray of shape [N, 4, 2] of `(y, x)` coordinates
                - tags  : np.ndarray of labels
                - im_size   : (height, width) of the image
                - min_poly_size : the minimal area for a polygon to be valid
            Returns :
                - score_map : np.ndarray of shape `im_size` with value of 1 for pixels within a box
                - geo_map   : np.ndarray of shape `im_size + (5, )` where the last axis represents
                    the distance between the top / right / bottom / left sides of the rectangle
                    and the rotation angle
        """
        if min_poly_size == -1: min_poly_size = self.min_poly_size
        if shrink_ratio == -1:  shrink_ratio = self.shrink_ratio
        if max_wh_factor == -1: max_wh_factor = self.max_wh_factor
        
        return get_rbox_map(
            polys,
            img_shape   = img_size,
            out_shape   = self.input_size[:2],
            labels      = labels,
            mapping     = self.label_to_idx if self.use_labels else None,
            
            min_poly_size   = min_poly_size,
            shrink_ratio    = shrink_ratio,
            max_wh_factor   = max_wh_factor
        )

    def filter_data(self, inputs, outputs):
        if tf.reduce_any(tf.math.is_nan(outputs[1])): return False
        return tf.reduce_any(tf.logical_and(
            tf.cast(outputs[0], tf.bool), outputs[-1]
        ))
    
    def get_output(self, data):
        outputs = tf.numpy_function(
            self.get_rbox, [data['mask'], data['label'], (data['height'], data['width'])],
            Tout = [tf.float32, tf.float32, tf.bool, tf.int32]
        )
        score_map, geo_map, valid_mask = outputs[:3]
        
        score_map   .set_shape(self.input_size[:2])
        geo_map     .set_shape([self.input_size[0], self.input_size[1], 5])
        valid_mask  .set_shape(self.input_size[:2])
        
        return score_map, geo_map, valid_mask
