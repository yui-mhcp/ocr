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
import importlib

from utils import setup_environment
from ..interfaces import BaseModel

_models = {}
for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    module = importlib.import_module(__package__ + '.' + module[:-3])
    
    _models.update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, BaseModel)
    })
globals().update(_models)


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
        model = get_pretrained(model, ** kwargs)
    
    return model
    

def stream(stream, *, label = None, model = None, ** kwargs):
    setup_environment(** kwargs)
    return get_model(label = label, model = model, ** kwargs).stream(stream, ** kwargs)

def detect(images, *, label = None, model = None, ** kwargs):
    return get_model(label = label, model = model, ** kwargs).predict(images, ** kwargs)

_pretrained = {
    'faces'     : 'yolo_faces'
}
