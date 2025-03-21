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

import numpy as np

from loggers import timer
from utils.image import nms
from utils.keras import TensorSpec, ops
from .base_detector import BaseDetector

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
    
    @timer
    def decode_output(self,
                      output,
                      inputs,
                      normalize = True,
                      nms_method    = 'lanms',
                      nms_threshold = None,
                      obj_threshold = None,
                      ** kwargs
                     ):
        from utils import plot
        
        if obj_threshold is None: obj_threshold = self.obj_threshold
        if nms_threshold is None: nms_threshold = self.nms_threshold
        
        output  = ops.convert_to_numpy(output)
        
        boxes = restore_polys_from_map(
            score_map   = output[..., :1],
            geo_map     = output[..., 1:5] * 512,
            theta_map   = (output[..., 5:6] - 0.5) * np.pi,
            
            normalize   = normalize,
            threshold   = obj_threshold,
            input_shape = np.array(inputs.shape[1:-1]),
            output_shape    = np.array(output.shape[:-1])
        )
        if nms_threshold < 1. and len(boxes['boxes']):
            boxes, _, mask = nms(boxes, method = nms_method, nms_threshold = nms_threshold, ** kwargs)
            boxes = {'boxes' : boxes[mask], 'format' : 'xyxy'}
        
        return boxes
    
    def get_output(self, data, ** kwargs):
        raise NotImplementedError()

""" These functions are inspired from https://github.com/SakuraRiven/EAST """

@timer
def restore_polys_from_map(score_map,
                           geo_map,
                           theta_map,
                           input_shape,
                           output_shape,
                           *,
                           
                           normalize    = True,
                           threshold    = 0.5
                          ):
    if len(score_map.shape) == 4:
        return [restore_polys_from_map(
            score_map   = s_map,
            geo_map     = g_map,
            theta_map   = t_map,
            input_shape = input_shape,
            output_shape    = output_shape,
            
            normalize   = normalize,
            threshold   = threshold
        ) for s_map, g_map, t_map in zip(score_map, geo_map, theta_map)]
    
    if len(score_map.shape) == 3:
        score_map   = score_map[:, :, 0]
        theta_map   = theta_map[:, :, 0]
    
    # filter the score map
    points = np.argwhere(score_map > threshold)

    # sort the text boxes via the y axis
    points  = points[np.argsort(points[:, 0])]
    scores  = score_map[points[:, 0], points[:, 1]]
    # restore
    valid_polys, valid_indices = restore_polys(
        points[:, ::-1],
        geo_map[points[:, 0], points[:, 1]],
        theta_map[points[:, 0], points[:, 1]],
        input_shape,
        output_shape
    )
    scores  = scores[valid_indices]
    
    if normalize:
        input_shape_wh  = input_shape[::-1].reshape(1, 1, 2)
        valid_polys     = (valid_polys / input_shape_wh).astype(np.float32)

    return {
        'boxes' : valid_polys, 'scores' : scores, 'format' : 'poly'
    }

@timer
def restore_polys(pos, d, angle, input_shape, output_shape):
    scale   = input_shape // output_shape
    pos     = pos * scale[None]

    x, y    = pos[:, 0], pos[:, 1]
    
    y_min, y_max    = y - d[:, 0], y + d[:, 1]
    x_min, x_max    = x - d[:, 2], x + d[:, 3]

    rotate_mat  = get_rotation_matrix(- angle)

    temp_x      = np.array([[x_min, x_max, x_max, x_min]]) - x
    temp_y      = np.array([[y_min, y_min, y_max, y_max]]) - y
    coordinates = np.concatenate((temp_x, temp_y), axis = 0)

    res = np.matmul(
        np.transpose(coordinates, [2, 1, 0]),
        np.transpose(rotate_mat, [2, 1, 0])
    )
    res[:, :, 0] += x[:, np.newaxis]
    res[:, :, 1] += y[:, np.newaxis]

    mask = filter_polys(res, input_shape)

    return res[mask], np.argwhere(mask)[:, 0]

def get_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def filter_polys(res, input_shape):
    input_shape = input_shape[::-1][None, None, :]
    return np.count_nonzero(
        np.any(res < 0, axis = -1) | np.any(res >= input_shape, axis = -1), axis = -1
    ) <= 1


