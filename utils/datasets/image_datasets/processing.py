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

from functools import wraps

from loggers import timer

def image_dataset_wrapper(name, task, ** infos):
    def wrapper(dataset_loader):
        """
            Wraps a function that loads an image dataset, and returns it as a `pd.DataFrame`
            
            The function must return a `list` of `dict`, one per image, with :
                - filename  : the image filename
                - (optional) label  : the label of each box
                - (optional) boxes  : the bounding boxes, in the `xywh` format (in pixels)
                - (optional) height / width : the image size
            
            The `boxes` are converted to `np.ndarray` with their number (`nb_box`), and `pad_boxes`
            pads them (with their `label`) to the maximal number of boxes per image.
        """
        @timer(name = '{} loading'.format(name if isinstance(name, str) else name[0]))
        @wraps(dataset_loader)
        def _load_and_process(directory, * args, pad_boxes = False, ** kwargs):
            import pandas as pd
            
            dataset = dataset_loader(directory, * args, ** kwargs)
            
            if any('boxes' in row for row in dataset):
                for row in dataset:
                    row['boxes']    = np.array(row.get('boxes', []), dtype = np.int32).reshape(-1, 4)
                    row['nb_box']   = len(row['boxes'])
                
                if pad_boxes: _pad_boxes(dataset)
            
            return pd.DataFrame(dataset)
        
        _load_and_process.task    = task
        _load_and_process.infos   = infos
        _load_and_process.dataset = name
        
        return _load_and_process
    return wrapper

def _pad_boxes(dataset):
    """ Pads the `boxes` (and `label`) of each image to the maximal number of boxes (inplace) """
    n = max(row['nb_box'] for row in dataset)
    for row in dataset:
        if row['nb_box'] == n: continue
        
        row['boxes'] = np.pad(row['boxes'], [(0, n - row['nb_box']), (0, 0)])
        if 'label' in row: row['label'] = list(row['label']) + [''] * (n - row['nb_box'])
