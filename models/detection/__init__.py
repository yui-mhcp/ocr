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

from utils import import_objects
from models.utils import get_models, get_model_config, is_model_name

_detection_models = import_objects(
    __package__.replace('.', os.path.sep), allow_functions = False
)
globals().update(_detection_models)

logger = logging.getLogger(__name__)

def get_model(label = None, model = None, ** kwargs):
    from models import get_pretrained
    
    assert label or model is not None, 'You must provide either the model name, either the label to detect'
    
    if model is None:
        if label in _pretrained:
            model = _pretrained[label]
        else:
            logger.info('No default model for label {}, searching for a model with this label...'.format(label))
            for name in get_models(model_class = tuple(_detection_models.keys())):
                if label in get_model_config(name).get('labels', []):
                    model = name
                    break
    
    if model is None:
        raise ValueError('No model found for label {}'.format(model, label))
    elif isinstance(model, str):
        if not is_model_name(model):
            raise ValueError('Model {} does not exist !'.format(model))
        model = get_pretrained(model, ** kwargs)
    
    return model
    

def stream(*, label = None, model = None, ** kwargs):
    return get_model(label = label, model = model, ** kwargs).stream(** kwargs)

def detect(images, *, label = None, model = None, ** kwargs):
    return get_model(label = label, model = model, ** kwargs).predict(images, ** kwargs)

_pretrained = {
    'faces'     : 'yolo_faces'
}
