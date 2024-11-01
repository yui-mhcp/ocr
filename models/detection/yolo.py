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
import logging
import numpy as np

from utils import download_file
from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, execute_eagerly
from utils.image.bounding_box import *
from .base_detector import BaseDetector

logger = logging.getLogger(__name__)

PRETRAINED_COCO_URL = 'https://pjreddie.com/media/files/yolov2.weights'

DEFAULT_ANCHORS = [
    0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
]

COCO_CONFIG = {
    'labels' : ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'brocoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
    'anchors' : DEFAULT_ANCHORS
}
VOC_CONFIG  = {
    'labels' : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    'anchors'   : DEFAULT_ANCHORS #[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
}

class YOLO(BaseDetector):
    _default_loss   = 'YoloLoss'
    
    def __init__(self,
                 * args,
                 max_box_per_image  = 100,
                 
                 input_size = 416,
                 nb_class   = 2,
                 backend    = "FullYolo",
                 
                 anchors    = DEFAULT_ANCHORS,

                 **kwargs
                ):
        assert len(anchors) % 2 == 0
        
        self.backend    = backend
        self.anchors    = anchors
        self.max_box_per_image = max_box_per_image

        self.np_anchors = np.array(anchors).reshape(-1, 2)
        
        super().__init__(* args, input_size = input_size, nb_class = nb_class, ** kwargs)
            
    def build(self, flatten = True, randomize = False, model = None, ** kwargs):
        if model is None:
            from custom_architectures import get_architecture
            
            feature_extractor = get_architecture(
                architecture    = self.backend,
                input_shape     = self.input_size,
                include_top     = False,
                ** kwargs
            )

            model = {
                'architecture'  : 'yolo',
                'feature_extractor' : feature_extractor,
                'input_shape'   : self.input_size,
                'anchors'   : self.anchors,
                'nb_class'  : self.nb_class,
                'flatten'   : flatten,
                'randomize' : randomize
            }
        
        return super().build(model = model)
    
    @property
    def grid_h(self):
        return self.output_shape[1]
    
    @property
    def grid_w(self):
        return self.output_shape[2]

    @property
    def nb_box(self):
        return len(self.anchors) // 2

    @property
    def output_signature(self):
        return (
            TensorSpec(
                shape = (None, self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class),
                dtype = 'float32'
            ),
            TensorSpec(
                shape = (None, 1, 1, 1, self.max_box_per_image, 4), dtype = 'float32'
            )
        )
    
    def __str__(self):
        des = super().__str__()
        des += "- Feature extractor : {}\n".format(self.backend)
        return des
    
    def decode_output(self, output, obj_threshold = None, nms_threshold = None, ** kwargs):
        if obj_threshold is None: obj_threshold = self.obj_threshold
        if nms_threshold is None: nms_threshold = self.nms_threshold
        kwargs.setdefault('labels', self.labels)
        
        return decode_output(
            output, obj_threshold = obj_threshold, nms_threshold = nms_threshold, ** kwargs
        )

    @execute_eagerly(Tout = ('float32', 'float32'), numpy = True)
    def get_output(self, boxes, labels, nb_box, image_h, image_w, ** kwargs):
        output      = np.zeros(
            (self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class), dtype = np.float32
        )
        true_boxes  = np.zeros((self.max_box_per_image, 4), dtype = np.float32)
        
        logger.debug("Image with shape ({}, {}) and {} boxes :".format(image_h, image_w, len(boxes)))
        
        for i in range(nb_box):
            x, y, w, h = boxes[i]
            label_idx = self.labels.index(labels[i]) if labels[i] in self.labels else 0

            center_y = ((y + 0.5 * h) / image_h) * self.grid_h
            center_x = ((x + 0.5 * w) / image_w) * self.grid_w
            
            w = (w / image_w) * self.grid_w  # unit: grid cell
            h = (h / image_h) * self.grid_h  # unit: grid cell
            
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            
            logger.debug("Boxes {} ({}) go to grid ({}, {})".format(i, boxes[i], grid_y, grid_x))
            
            if w > 0. and h > 0. and grid_x < self.grid_w and grid_y < self.grid_h:
                box = np.array([center_x, center_y, w, h])
                yolo_box = np.array([center_x, center_y, w, h, 1.])
                
                true_boxes[i % self.max_box_per_image, :] = box
                
                # find the anchor that best predicts this box
                
                box_wh = np.repeat(np.array([[w, h]]), self.nb_box, axis = 0)
                
                intersect = np.minimum(box_wh, self.np_anchors)
                intersect = intersect[:,0] * intersect[:,1]
                union = (self.np_anchors[:,0] * self.np_anchors[:,1]) +  (box_wh[:,0] * box_wh[:,1]) - intersect
                
                iou = intersect / union
                
                best_anchor = np.argmax(iou)
                
                logger.debug("Normalized box {} with label {} to anchor idx {} with score {}".format(box, label_idx, best_anchor, iou[best_anchor]))
                
                if iou[best_anchor] > 0.:
                    output[grid_y, grid_x, best_anchor, :5] = yolo_box
                    output[grid_y, grid_x, best_anchor, 5 + label_idx] = 1
        
        return output, true_boxes
    
    def prepare_output(self, infos, ** _):
        return self.get_output(
            infos['boxes'],
            infos['label'],
            infos['nb_box'],
            infos['height'],
            infos['width'],
            shape = [
                (self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class),
                (self.max_box_per_image, 4)
            ]
        )
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_image(),
            'anchors'   : self.anchors,
            'backend'   : self.backend,
            'max_box_per_image' : self.max_box_per_image
        })
        
        return config

    @classmethod
    def from_darknet_pretrained(cls,
                                weight_path = 'yolov2.weights',
                                name    = 'coco_pretrained',
                                labels  = COCO_CONFIG['labels'],
                                ** kwargs
                               ):
        if not os.path.exists(weight_path) and weight_path.endswith('yolov2.weights'):
            weight_path = download_file(PRETRAINED_COCO_URL, filename = weight_path)
            
        instance = cls(
            name = name, labels = labels, max_to_keep = 1, pretrained_name = weight_path, ** kwargs
        )
        
        decode_darknet_weights(instance.model, weight_path)
        
        instance.save()
        
        return instance

@timer(name = 'output decoding')
def decode_output(output, *, obj_threshold = 0.35, nms_threshold = 0.2, labels = None, ** kwargs):
    output = ops.convert_to_numpy(output)
    
    if len(output.shape) == 5:
        kwargs.update({
            'obj_threshold' : obj_threshold, 'nms_threshold' : nms_threshold, 'labels' : labels
        })
        return [decode_output(out, ** kwargs) for out in output]
    
    with time_logger.timer('init'):
        grid_h, grid_w, nb_box = output.shape[:3]
        nb_class = output.shape[3] - 5
        
        scores  = output[..., 5:] * output[..., 4:5]

    with time_logger.timer('box selection'):
        candidates = np.where(np.max(scores, axis = -1) > obj_threshold)

        pos    = output[..., :4][candidates]
        pos    = pos / np.array([grid_w, grid_h, grid_w, grid_h], dtype = pos.dtype)
        scores = scores[candidates]

        xy_min = np.maximum(pos[:, :2] - pos[:, 2:] / 2., 0.)
        xy_max = np.minimum(pos[:, :2] + pos[:, 2:] / 2., 1.)
        
        valids  = np.all(xy_max > xy_min, axis = 1)
        boxes   = np.concatenate([xy_min[valids], xy_max[valids]], axis = 1)
        classes  = scores[valids]

    # suppress non-maximal boxes
    with time_logger.timer('NMS'):
        ious = {}
        for c in range(nb_class):
            scores_c = classes[:, c]
            sorted_indices = np.argsort(scores_c)[::-1]
            sorted_indices = sorted_indices[scores_c[sorted_indices] > obj_threshold]

            for i, index_i in enumerate(sorted_indices):
                if classes[index_i, c] < obj_threshold: continue

                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if classes[index_j, c] < obj_threshold: continue

                    if (index_i, index_j) not in ious:
                        ious[(index_i, index_j)] = compute_iou(
                            boxes[index_i], boxes[index_j], source = 'xyxy'
                        )

                    if ious[(index_i, index_j)] >= nms_threshold:
                        classes[index_j, c] = 0

    scores  = np.max(classes, axis = 1)
    mask    = scores > obj_threshold
    classes = np.argmax(classes[mask], axis = 1)
    return {
        'boxes'     : boxes[mask],
        'labels'    : [labels[idx] for idx in classes] if labels else classes,
        'scores'    : scores[mask],
        'format'    : 'xyxy'
    }

def decode_darknet_weights(model, wt_path):
    #Chargement des poids
    weight_reader = WeightReader(wt_path)
    weight_reader.reset()
    nb_conv = 23
    for i in range(1, nb_conv+1):
        name = 'conv_{}'.format(i) if i < 23 else 'detection_layer'
        if i < nb_conv and len(model.layers) < 10:
            conv_layer = model.layers[1].get_layer(name = name)
            norm_layer = model.layers[1].get_layer(
                'batch_normalization_{}'.format(i-1) if i > 1 else 'batch_normalization'
            )
        else:
            conv_layer = model.get_layer(name = name)
            norm_layer = model.get_layer(
                'batch_normalization_{}'.format(i-1) if i > 1 else 'batch_normalization'
            ) if i < nb_conv else None
        
        if (i < nb_conv):
            size = np.prod(norm_layer.get_weights()[0].shape)
            
            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])
        
        if (len(conv_layer.get_weights()) > 1):
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])

class WeightReader:
    def __init__(self, path):
        self.offset = 4
        self.all_weights = np.fromfile(path, dtype = 'float32')
    
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size : self.offset]
    
    def reset(self):
        self.offset = 4
