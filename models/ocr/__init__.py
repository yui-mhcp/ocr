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

from utils import import_objects, limit_gpu_memory

globals().update(import_objects(
    __package__.replace('.', os.path.sep), allow_functions = False
))

def get_model(model = None, lang = None, ** kwargs):
    assert model is not None or lang is not None
    
    global _pretrained
    
    if model is None:
        if lang not in _pretrained:
            raise ValueError('No default model for language {}'.format(lang))
        model = _pretrained[lang]

    if isinstance(model, str):
        from models import get_pretrained
        
        model = get_pretrained(model, ** kwargs)
    
    return model

def ocr(image, model = None, lang = 'en', ** kwargs):
    """ See `help(ClipCap.predict)` for more information """
    return get_model(model = model, lang = lang, ** kwargs).predict(image, ** kwargs)

def ocr_stream(stream, model = None, lang = 'en', ** kwargs):
    if 'gpu_memory' in kwargs:  limit_gpu_memory(kwargs.pop('gpu_memory'))
    
    return get_model(model = model, lang = lang, ** kwargs).stream(stream, ** kwargs)

_pretrained = {
    'en'    : 'crnn_en'
}
