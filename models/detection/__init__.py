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

import logging

from models.detection.yolo import YOLO
from models.detection.east import EAST
from models.saving import get_models, get_model_config, is_model_name

logger = logging.getLogger(__name__)

def get_model(label = None, model = None, ** kwargs):
    from models import get_pretrained
    
    assert label is not None or model is not None
    
    if model is None:
        if label in _pretrained:
            model = _pretrained[label]
        else:
            logger.info('No default model for label {}, searching for a model with this label...'.format(label))
            for name in get_models(model_class = 'YOLO'):
                if label in get_model_config(name).get('labels', []):
                    model = name
                    break
    
    if model is None:
        raise ValueError('No model found for label {}'.format(model, label))
    elif isinstance(model, str) and not is_model_name(model):
        raise ValueError('Model {} does not exist !'.format(model))
    
    return get_pretrained(model) if isinstance(model, str) else model
    

def stream(label = None, model = None, ** kwargs):
    model = get_model(label = label, model = model)
    model.stream(** kwargs)

def detect(images, label = None, model = None, ** kwargs):
    model = get_model(label = label, model = model)
    return model.predict(images, ** kwargs)

_models = {
    'EAST'  : EAST,
    'YOLO'  : YOLO
}

_pretrained = {
    'faces'     : 'yolo_faces'
}
