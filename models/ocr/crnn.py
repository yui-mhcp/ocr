
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
            
            vocab   = ['<blank>'] + list(infos['characters'])
            
            kwargs.update({
                'pretrained'    : pretrained,
                'pretrained_name'   : 'easyocr_{}'.format(pretrained),
                'original_vocab'    : vocab
            })
            kwargs.setdefault('input_size', (64, None, 1))
            kwargs.setdefault('image_normalization', 'easyocr')
            kwargs.setdefault('text_encoder', {})
            if isinstance(kwargs['text_encoder'], dict) and 'vocab' not in kwargs['text_encoder']:
                kwargs['text_encoder'].update({
                    'vocab' : vocab,
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
        
    def _build_model(self, original_vocab = None, architecture = 'CRNN', ** kwargs):
        super(BaseOCR, self)._build_model(model = {
            'architecture_name' : architecture,
            'input_shape'   : self.input_size,
            'output_dim'    : self.vocab_size,
            'vocab_size'    : self.vocab_size,
            ** kwargs
        })
        
        #if hasattr(self.model, 'change_vocabulary') and original_vocab:
        #    self.model.change_vocabulary(self.vocab, old_vocab = original_vocab)
    
    def compile(self, loss = None, metrics = None, ** kwargs):
        kwargs.setdefault('loss_config', {}).update({
            'pad_value' : self.blank_token_idx, 'from_logits' : True
        })
        kwargs.setdefault('metrics_config', {}).update({
            'pad_value' : self.blank_token_idx, 'from_logits' : True
        })
        
        if not loss: loss = 'CTCLoss' if not self.is_encoder_decoder else 'TextLoss'
        if not metrics: metrics = ['TextMetric'] if not self.is_encoder_decoder else ['TextAccuracy']
        
        return super().compile(loss = loss, metrics = metrics, ** kwargs)
