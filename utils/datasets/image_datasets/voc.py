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
import xml.etree.ElementTree as ET

from ..loader import Task
from .processing import image_dataset_wrapper

@image_dataset_wrapper(
    name = ('VOC', 'Pascal VOC'), task = Task.OBJECT_DETECTION, directory = '{}/VOC2012'
)
def load_voc(directory, ** kwargs):
    return load_voc_annotations(directory, 'JPEGImages', 'Annotations')

@image_dataset_wrapper(
    name = 'kangaroo', task = Task.OBJECT_DETECTION, directory = '{}/kangaroo-master'
)
def load_kangaroo(directory, ** kwargs):
    return load_voc_annotations(directory, 'images', 'annots')

@image_dataset_wrapper(
    name = 'raccoon', task = Task.OBJECT_DETECTION, directory = '{}/raccoon-master'
)
def load_raccoon(directory, ** kwargs):
    return load_voc_annotations(directory, 'images', 'annots')

def load_voc_annotations(directory, images_dir, annotations_dir):
    """
        Parses a dataset annotated in the `Pascal VOC` format (one `xml` file per image)
        
        Arguments :
            - directory : the dataset directory
            - images_dir    : the images sub-directory
            - annotations_dir   : the annotations sub-directory
        Return :
            - dataset   : a `list` of `dict` with keys `(filename, label, height, width, boxes)`,
                          where `boxes` are in the `xywh` format (in pixels)
    """
    from ...image import _image_formats
    
    images_dir  = os.path.join(directory, images_dir)
    annotations_dir = os.path.join(directory, annotations_dir)
    
    dataset = []
    for ann in sorted(os.listdir(annotations_dir)):
        filename, image_h, image_w = None, None, None
        
        boxes, labels = [], []
        for elem in ET.parse(os.path.join(annotations_dir, ann)).iter():
            if 'filename' in elem.tag:
                filename = os.path.join(images_dir, str(elem.text))
                if not filename.endswith(_image_formats): filename += '.jpg'
            if 'width' in elem.tag:     image_w = int(elem.text)
            if 'height' in elem.tag:    image_h = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = str(attr.text)
                    
                    if 'bndbox' in attr.tag:
                        x0, y0, x1, y1 = 0, 0, 0, 0
                        for dim in list(attr):
                            if 'xmin' in dim.tag: x0 = int(round(float(dim.text)))
                            if 'ymin' in dim.tag: y0 = int(round(float(dim.text)))
                            if 'xmax' in dim.tag: x1 = int(round(float(dim.text)))
                            if 'ymax' in dim.tag: y1 = int(round(float(dim.text)))
                        
                        labels.append(label)
                        boxes.append([x0, y0, x1 - x0, y1 - y0])
        
        dataset.append({
            'filename'  : filename,
            'label'     : labels,
            'height'    : image_h,
            'width'     : image_w,
            'boxes'     : boxes
        })
    
    return dataset
