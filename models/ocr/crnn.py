
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

import tensorflow as tf

from models.ocr.base_ocr import BaseOCR

class CRNN(BaseOCR):
    def __init__(self, lang = 'multi', * args, pretrained = None, pretrained_lang = None, ** kwargs):
        if pretrained or pretrained_lang:
            from custom_architectures.crnn_arch import get_easyocr_crnn_infos
            
            lang = pretrained_lang
            pretrained, infos = get_easyocr_crnn_infos(model = pretrained, lang = lang)
            
            kwargs.update({
                'pretrained'    : pretrained,
                'pretrained_name'   : 'easyocr_{}'.format(pretrained)
            })
            kwargs.setdefault('input_size', (64, None, 1))
            kwargs.setdefault('image_normalization', 'easyocr')
            kwargs.setdefault('text_encoder', {})
            if isinstance(kwargs['text_encoder'], dict) and 'vocab' not in kwargs['text_encoder']:
                kwargs['text_encoder'].update({
                    'vocab' : ['<blank>'] + list(infos['characters']),
                    'level' : 'char',
                    'pad_token' : '<blank>'
                })
            kwargs.setdefault('resize_kwargs', {
                'method'    : 'bilinear',
                'antialias' : True,
                'pad_mode'  : 'repeat_last',
                'preserve_aspect_ratio' : True,
                'manually_compute_ratio'    : True,
                'target_multiple_shape' : 64
            })
        
        super().__init__(lang, * args, ** kwargs)
        
    def _build_model(self, ** kwargs):
        return super(BaseOCR, self)._build_model(model = {
            'architecture_name' : 'CRNN',
            'input_shape'   : self.input_size,
            'output_dim'    : self.vocab_size,
            ** kwargs
        })
    
    def decode_output(self, output, ** kwargs):
        return self.text_encoder.ctc_decode(output, ** kwargs)
    
    def compile(self, loss = 'CTCLoss', metrics = ['TextMetric'], ** kwargs):
        kwargs.setdefault('loss_config', {}).update({'pad_value' : self.blank_token_idx})
        kwargs.setdefault('metrics_config', {}).update({'pad_value' : self.blank_token_idx})
        
        super().compile(
            loss = loss, metrics = metrics, ** kwargs
        )
