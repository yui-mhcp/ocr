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
import importlib

from utils import setup_environment
from ..interfaces import BaseModel
from ..utils import get_model_dir, get_model_config, is_model_name

_models = {}
for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    module = importlib.import_module(__package__ + '.' + module[:-3])
    
    _models.update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, BaseModel)
    })
globals().update(_models)

def get_model(model = None, lang = None):
    assert model is not None or lang is not None
    
    global _pretrained
    
    if model is None:
        if lang not in _pretrained:
            raise ValueError('No default model for language {}'.format(lang))
        model = _pretrained[lang]

    if isinstance(model, str):
        from models import get_pretrained
        
        model = get_pretrained(model)
    
    return model

def ocr(image, *, model = None, lang = 'en', ** kwargs):
    """ See `help(ClipCap.predict)` for more information """
    return get_model(model = model, lang = lang).predict(image, ** kwargs)

def stream(stream, *, model = None, lang = 'en', ** kwargs):
    setup_environment(** kwargs)
    return get_model(model = model, lang = lang).stream(stream, ** kwargs)

_pretrained = {
    'en'    : 'crnn_en'
}
