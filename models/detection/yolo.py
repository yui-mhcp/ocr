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

import os
import logging
import numpy as np

from utils import download_file
from loggers import Timer, timer
from utils.image.bounding_box import compute_iou
from utils.keras import TensorSpec, ops, execute_eagerly
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
                 labels,
                 *,
                 
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
        
        super().__init__(labels = labels, input_size = input_size, nb_class = nb_class, ** kwargs)
            
    def build(self, flatten = True, randomize = False, model = None, ** kwargs):
        if model is None:
            from architectures import get_architecture
            
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
        return self.model.output_shape[1]

    @property
    def grid_w(self):
        return self.model.output_shape[2]

    @property
    def nb_box(self):
        return len(self.anchors) // 2

    @property
    def output_signature(self):
        """ The training target : `(encoded output, true_boxes)` (see `encode_output`) """
        return (
            TensorSpec(shape = (None, ) + self.target_shape, dtype = 'float32'),
            TensorSpec(shape = (None, self.max_box_per_image, 4), dtype = 'float32')
        )

    @property
    def target_shape(self):
        """ Shape of the encoded target : `(x, y, w, h, confidence, class index)` per anchor """
        return (self.grid_h, self.grid_w, self.nb_box, 6)
    
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
    def encode_boxes(self, boxes, labels, nb_box, image_h, image_w):
        """ Encodes the `nb_box` first `boxes` (`xywh`, in pixels) of an image (see `encode_output`) """
        return encode_output(
            boxes,
            labels,
            nb_box,
            image_h,
            image_w,
            
            grid_h  = self.grid_h,
            grid_w  = self.grid_w,
            anchors = self.np_anchors,
            label_to_idx    = self.label_to_idx,
            max_box_per_image   = self.max_box_per_image
        )
    
    def prepare_output(self, infos, ** _):
        return self.encode_boxes(
            infos['boxes'],
            infos['label'],
            infos['nb_box'],
            infos['height'],
            infos['width'],
            shape = [self.target_shape, (self.max_box_per_image, 4)]
        )
    
    def get_config(self):
        return {
            ** super().get_config(),
            'anchors'   : self.anchors,
            'backend'   : self.backend,
            'max_box_per_image' : self.max_box_per_image
        }

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

def encode_output(boxes,
                  labels,
                  nb_box,
                  image_h,
                  image_w,
                  *,
                  
                  grid_h,
                  grid_w,
                  anchors,
                  label_to_idx,
                  max_box_per_image
                 ):
    """
        Encodes the boxes of an image into the YOLO training target
        
        Arguments :
            - boxes     : `np.ndarray` with shape `(n, 4)`, the boxes in the `xywh` format (in pixels)
            - labels    : (list of) `str` / `bytes`, the label of each box
            - nb_box    : the number of valid boxes (i.e. the `nb_box` first ones, the others are padding)
            - image_{h / w} : the image size (in pixels)
            
            - grid_{h / w}  : the output grid size
            - anchors   : `np.ndarray` with shape `(nb_anchors, 2)`, the anchors (in grid cells)
            - label_to_idx  : `dict` mapping each label to its class index
            - max_box_per_image : the number of `true_boxes`
        Return :
            - output    : `np.ndarray` with shape `(grid_h, grid_w, nb_anchors, 6)`, where each box
                          is set at its center cell, on its best anchor, as
                          `(center_x, center_y, w, h, 1, class index)` (in grid cells)
            - true_boxes    : `np.ndarray` with shape `(max_box_per_image, 4)`, the encoded
                              `(center_x, center_y, w, h)` of each box (zero-padded)
        
        Note : a label unknown from `label_to_idx` is encoded as the class 0
    """
    nb_anchors  = len(anchors)
    output      = np.zeros((grid_h, grid_w, nb_anchors, 6), dtype = np.float32)
    true_boxes  = np.zeros((max_box_per_image, 4), dtype = np.float32)
    
    logger.debug("Image with shape ({}, {}) and {} boxes :".format(image_h, image_w, nb_box))
    
    for i in range(nb_box):
        x, y, w, h = boxes[i]
        label = labels[i].decode('utf-8') if isinstance(labels[i], bytes) else str(labels[i])
        label_idx = label_to_idx.get(label, 0)

        center_y = ((y + 0.5 * h) / image_h) * grid_h
        center_x = ((x + 0.5 * w) / image_w) * grid_w
        
        w = (w / image_w) * grid_w  # unit: grid cell
        h = (h / image_h) * grid_h  # unit: grid cell
        
        grid_x = int(np.floor(center_x))
        grid_y = int(np.floor(center_y))
        
        logger.debug("Boxes {} ({}) go to grid ({}, {})".format(i, boxes[i], grid_y, grid_x))
        
        if w > 0. and h > 0. and grid_x < grid_w and grid_y < grid_h:
            box = np.array([center_x, center_y, w, h])
            
            true_boxes[i % max_box_per_image, :] = box
            
            # find the anchor that best predicts this box
            intersect = np.minimum(np.array([[w, h]]), anchors)
            intersect = intersect[:, 0] * intersect[:, 1]
            union = (anchors[:, 0] * anchors[:, 1]) + (w * h) - intersect
            
            iou = intersect / union
            
            best_anchor = np.argmax(iou)
            
            logger.debug("Normalized box {} with label {} to anchor idx {} with score {}".format(box, label_idx, best_anchor, iou[best_anchor]))
            
            if iou[best_anchor] > 0.:
                output[grid_y, grid_x, best_anchor] = [center_x, center_y, w, h, 1., label_idx]
    
    return output, true_boxes

@timer(name = 'output decoding')
def decode_output(output, *, obj_threshold = 0.35, nms_threshold = 0.2, labels = None, ** kwargs):
    output = ops.convert_to_numpy(output)
    
    if len(output.shape) == 5:
        kwargs.update({
            'obj_threshold' : obj_threshold, 'nms_threshold' : nms_threshold, 'labels' : labels
        })
        return [decode_output(out, ** kwargs) for out in output]
    
    with Timer('init'):
        grid_h, grid_w, nb_box = output.shape[:3]
        nb_class = output.shape[3] - 5
        
        scores  = output[..., 5:] * output[..., 4:5]

    with Timer('box selection'):
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
    with Timer('NMS'):
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