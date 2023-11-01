
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

from loggers import timer
from utils import download_file
from utils.image.box_utils import *
from utils.distance.distance_method import iou
from custom_architectures import get_architecture
from models.detection.base_detector import BaseDetector

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

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
        
        self.np_anchors = np.reshape(np.array(anchors), [self.nb_box, 2])

        super().__init__(* args, input_size = input_size, nb_class = nb_class, ** kwargs)

    def init_train_config(self, * args, ** kwargs):
        super().init_train_config(* args, ** kwargs)
        
        if hasattr(self, 'model_loss'):
            self.model_loss.seen = self.current_epoch
            
    def _build_model(self, flatten = True, randomize = True, ** kwargs):
        feature_extractor = get_architecture(
            architecture_name = self.backend,
            input_shape = self.input_size,
            include_top = False,
            ** kwargs
        )
        
        super()._build_model(model = {
            'architecture_name' : 'yolo',
            'feature_extractor' : feature_extractor,
            'input_shape'       : self.input_size,
            'nb_class'      : self.nb_class,
            'nb_box'        : self.nb_box,
            'flatten'       : flatten,
            'randomize'     : randomize
        })
    
    @property
    def grid_h(self):
        return self.output_shape[1]
    
    @property
    def grid_w(self):
        return self.output_shape[2]

    @property
    def output_signature(self):
        return (
            tf.TensorSpec(
                shape = (None, self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class),
                dtype = tf.float32
            ),
            tf.TensorSpec(
                shape = (None, 1, 1, 1, self.max_box_per_image, 4), dtype = tf.float32
            )
        )
    
    @property
    def nb_box(self):
        return len(self.anchors) // 2
    
    def __str__(self):
        des = super().__str__()
        des += "- Feature extractor : {}\n".format(self.backend)
        return des
    
    def compile(self, loss = 'YoloLoss', loss_config = {}, ** kwargs):
        loss_config.update({'anchors' : self.anchors})
        
        super().compile(loss = loss, loss_config = loss_config, ** kwargs)
    
    @timer(name = 'output decoding')
    def decode_output(self, output, obj_threshold = None, nms_threshold = None, ** kwargs):
        if len(output.shape) == 5:
            return [self.decode_output(
                out, obj_threshold = obj_threshold, nms_threshold = nms_threshold, ** kwargs
            ) for out in output]
        
        if obj_threshold is None: obj_threshold = self.obj_threshold
        if nms_threshold is None: nms_threshold = self.nms_threshold
        time_logger.start_timer('init')
        
        grid_h, grid_w, nb_box = output.shape[:3]
        nb_class = output.shape[3] - 5
        
        pos     = output[..., :4].numpy()
        conf    = tf.sigmoid(output[..., 4:5]).numpy()
        classes = tf.nn.softmax(output[..., 5:], axis = -1).numpy()

        time_logger.stop_timer('init')
        time_logger.start_timer('preprocess')

        # decode the output by the network
        
        scores  = conf * classes
        scores[scores <= obj_threshold] = 0.
        
        class_scores = np.sum(scores, axis = -1)

        conf    = conf[..., 0]
        candidates  = np.where(class_scores > 0.)
        
        time_logger.stop_timer('preprocess')
        time_logger.start_timer('box filtering')

        pos     = pos[candidates]
        conf    = conf[candidates]
        classes = classes[candidates]
        
        row, col, box = candidates
        x, y, w, h    = [pos[:, i] for i in range(4)]
        
        np_anchors = np.array(self.anchors)
        
        x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
        y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
        w = np_anchors[2 * box + 0] * np.exp(w) / grid_w # unit: image width
        h = np_anchors[2 * box + 1] * np.exp(h) / grid_h # unit: image height
        
        x1 = np.maximum(0., x - w / 2.)
        y1 = np.maximum(0., y - h / 2.)
        x2 = np.minimum(1., x + w / 2.)
        y2 = np.minimum(1., y + h / 2.)
        
        valids = np.logical_and(x1 < x2, y1 < y2)
        boxes  = [BoundingBox(
            x1 = float(x1i), y1 = float(y1i), x2 = float(x2i), y2 = float(y2i),
            conf = c, classes = cl
        ) for x1i, y1i, x2i, y2i, c, cl in zip(
            x1[valids], y1[valids], x2[valids], y2[valids], conf[valids], classes[valids]
        )]

        time_logger.stop_timer('box filtering')
        time_logger.start_timer('NMS')
        # suppress non-maximal boxes
        ious = {}
        for c in range(nb_class):
            scores = np.array([box.classes[c] for box in boxes])
            sorted_indices = np.argsort(scores)[::-1]
            sorted_indices = sorted_indices[scores[sorted_indices] > 0]

            for i, index_i in enumerate(sorted_indices):
                if boxes[index_i].classes[c] == 0: continue
                
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if boxes[index_j].classes[c] == 0: continue

                    if (index_i, index_j) not in ious:
                        ious[(index_i, index_j)] = iou(
                            boxes[index_i], boxes[index_j], box_mode = BoxFormat.OBJECT
                        )
                    
                    if ious[(index_i, index_j)] >= nms_threshold:
                        boxes[index_j].classes[c] = 0
        
        time_logger.stop_timer('NMS')
        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.score > 0]
        return boxes

    def get_output_fn(self, boxes, labels, nb_box, image_h, image_w, ** kwargs):
        if hasattr(boxes, 'numpy'): boxes = boxes.numpy()
        if hasattr(image_h, 'numpy'): image_h, image_w = image_h.numpy(), image_w.numpy()
        
        output      = np.zeros((self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class))
        true_boxes  = np.zeros((1, 1, 1, self.max_box_per_image, 4))
        
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
                
                true_boxes[0, 0, 0, i % self.max_box_per_image, :] = box
                
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
    
    def get_output(self, infos):
        output, true_boxes = tf.py_function(
            self.get_output_fn,
            [infos['box'], infos['label'], infos['nb_box'], infos['height'], infos['width']],
            Tout = [tf.float32, tf.float32]
        )

        output.set_shape([self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class])
        true_boxes.set_shape([1, 1, 1, self.max_box_per_image, 4])
        
        return output, true_boxes
    
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
                                nom     = 'coco_pretrained',
                                labels  = COCO_CONFIG['labels'],
                                ** kwargs
                               ):
        if not os.path.exists(weight_path) and weight_path.endswith('yolov2.weights'):
            weight_path = download_file(PRETRAINED_COCO_URL, filename = weight_path)
            
        instance = cls(
            nom = nom, labels = labels, max_to_keep = 1, pretrained_name = weight_path, ** kwargs
        )
        
        decode_darknet_weights(instance.get_model(), weight_path)
        
        instance.save()
        
        return instance

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

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
