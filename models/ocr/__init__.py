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

from .base_ocr import BaseOCR
from .crnn import CRNN

def get_model(model = None, lang = None):
    assert model is not None or lang is not None
    
    global _pretrained
    
    if model is None:
        if lang not in _pretrained: raise ValueError('No default model for language {}'.format(lang))
        model = _pretrained[lang]

    if isinstance(model, str):
        from models import get_pretrained
        
        model = get_pretrained(model)
    
    return model

def ocr(image, model = None, lang = 'en', ** kwargs):
    """ See `help(ClipCap.predict)` for more information """
    model = get_model(model = model, lang = lang)
    return model.predict(image, ** kwargs)

def ocr_stream(filename = None, url = None, model = None, lang = 'en', ** kwargs):
    if 'gpu_memory' in kwargs:  limit_gpu_memory(kwargs.pop('gpu_memory'))
    if 'gpu_growth' in kwargs:  set_memory_growth(kwargs.pop('gpu_growth'))
    model = get_model(model = model, lang = lang)
    return model.stream_video(filename, url, ** kwargs)

_pretrained = {
    'en'    : 'crnn_en'
}
