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

import numpy as np

from loggers import timer
from .base_detector import BaseDetector
from utils.keras_utils import TensorSpec, ops
from utils.image.bounding_box import nms, restore_polys_from_map

class EAST(BaseDetector):
    _default_loss   = 'EASTLoss'
    
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
    
    def build(self, architecture = 'EASTVGG', model = None, ** kwargs):
        if model is None:
            model = {
                'architecture'  : architecture,
                'input_shape'   : self.input_size,
                'output_dim'    : [1, 4, 1] + ([self.nb_class] if self.use_labels else []),
                'final_activation'  : ['sigmoid', 'sigmoid', 'sigmoid', 'softmax'],
                'final_name'    : ['score_map', 'geo_map', 'theta_map', 'class_map'],
                ** kwargs
            }
        return super().build(model = model)
    
    @property
    def output_signature(self):
        sign = (
            TensorSpec(shape = (None, ) + self.input_size[:-1], dtype = 'float32'),
            TensorSpec(shape = (None, ) + self.input_size[:-1] + (5, ), dtype = 'float32'),
            TensorSpec(shape = (None, ) + self.input_size[:-1], dtype = 'bool'),
        )
        if self.use_labels:
            sign += (
                TensorSpec(shape = (None, ) + self.input_size[:-1] + (1, ), dtype = 'int32'), 
            )
        return sign
    
    @property
    def use_labels(self):
        return self.nb_class > 1
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            min_poly_size   = 6,
            max_wh_factor   = 5,
            shrink_ratio    = 0.1
        )
    
    @timer
    def decode_output(self,
                      output,
                      inputs    = None,
                      normalize = True,
                      nms_method    = 'lanms',
                      nms_threshold = None,
                      obj_threshold = None,
                      ** kwargs
                     ):
        if obj_threshold is None: obj_threshold = self.obj_threshold
        if nms_threshold is None: nms_threshold = self.nms_threshold
        
        output  = ops.convert_to_numpy(output)
        
        boxes = restore_polys_from_map(
            score_map   = output[..., :1],
            geo_map     = output[..., 1:5] * 512,
            theta_map   = (output[..., 5:6] - 0.5) * np.pi,
            normalize   = normalize,
            threshold   = obj_threshold,
            scale   = np.array(inputs.shape[1:-1]) // np.array(output.shape[1:-1]),
            ** kwargs
        )
        if nms_threshold < 1.:
            results = [
                nms(b, method = nms_method, ** kwargs) for b in boxes
            ]
            boxes = [
                nms_boxes[0][mask[0]] if len(nms_boxes) > 0 else nms_boxes
                for nms_boxes, _, mask in results
            ]
            boxes = [{'boxes' : b, 'format' : 'xyxy'} for b in boxes]
        
        return boxes

    def filter_input(self, inputs):
        return ops.any(ops.logical_and(
            ops.cast(inputs[0], 'bool'), outputs[-1]
        ))
    
    def filter_output(self, outputs):
        return ops.logical_not(ops.any(ops.is_nan(outputs[1])))
    
    def get_output(self, data, ** kwargs):
        raise NotImplementedError()
