
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
import glob

from utils.generic_utils import to_lower_keys

def __load():
    for module_name in glob.glob(os.path.join('custom_architectures', 'transformers_arch', '*.py')):
        if module_name.endswith('__init__.py'): continue
        module_name = module_name.replace(os.path.sep, '.')[:-3]

        module = __import__(
            module_name,
            fromlist = ['custom_objects', 'custom_functions', '_encoders', '_decoders', '_transformers']
        )
        if hasattr(module, 'custom_objects'):
            custom_objects.update(module.custom_objects)
        if hasattr(module, 'custom_functions'):
            custom_functions.update(module.custom_functions)
        if hasattr(module, '_encoders'):
            _encoders.update(module._encoders)
        if hasattr(module, '_decoders'):
            _decoders.update(module._decoders)
        if hasattr(module, '_transformers'):
            _transformers.update(module._transformers)
        else:
            if hasattr(module, '_encoders'):
                _transformers.update(module._encoders)
            if hasattr(module, '_decoders'):
                _transformers.update(module._decoders)

def _get_pretrained(pretrained_name,
                    _possible_classes,
                    _class_type = 'transformer',
                    class_name  = None,
                    wrapper = None,
                    ** kwargs
                   ):
    if wrapper:
        return wrapper.from_pretrained(
            pretrained_name = pretrained_name, class_name = class_name, ** kwargs
        )
    if not class_name: class_name = pretrained_name
    class_name = class_name.lower()
    _possible_classes   = to_lower_keys(_possible_classes)
    
    if class_name in _possible_classes:
        return _possible_classes[class_name].from_pretrained(
            pretrained_name = pretrained_name, ** kwargs
        )
    
    for model_name, model_class in sorted(_possible_classes.items(), key = lambda p: len(p[0]), reverse = True):
        if model_name in class_name:
            return model_class.from_pretrained(pretrained_name = pretrained_name, ** kwargs)
    
    raise ValueError('Unknown {} class for pretrained model {}'.format(
        _class_type, pretrained_name
    ))

def get_pretrained_transformer_encoder(pretrained_name, ** kwargs):
    return _get_pretrained(pretrained_name, _encoders, 'encoder', ** kwargs)

def get_pretrained_transformer_decoder(pretrained_name, ** kwargs):
    return _get_pretrained(pretrained_name, _decoders, 'decoder', ** kwargs)

def get_pretrained_transformer(pretrained_name, ** kwargs):
    return _get_pretrained(pretrained_name, _transformers, 'transformers', ** kwargs)

        
custom_objects = {}
custom_functions = {}

_encoders   = {}
_decoders   = {}
_transformers   = {}

__load()

