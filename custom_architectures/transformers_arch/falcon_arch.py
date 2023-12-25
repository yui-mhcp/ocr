
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

""" TF 2.x Falcon model, compatible with the `transformers`' checkpoint. """

import os
import logging
import tensorflow as tf

from custom_layers import get_activation, RotaryMultiHeadAttention
from custom_architectures.transformers_arch.text_transformer_arch import (
    TextTransformerEncoder, HParamsTextTransformerEncoder
)

logger = logging.getLogger(__name__)

HParamsFalcon  = HParamsTextTransformerEncoder(
    use_causal_attention    = True,
    normalize_embeddings    = False,
    scale_embeddings    = False,
    max_input_length    = -1,
    mha_ffn_in_parallel = True,
    
    output_dim      = None,
    final_activation    = 'softmax',
    final_bias      = False,
    
    rotary  = True,
    epsilon = 1e-5,
    normalize   = None,
    normalize_output    = True,
    
    mha_multi_query = True,
    mha_normalize_input = True,
    mha_output_bias = False,
    mha_use_bias    = False,
    mha_mask_factor = -1e9,
    mha_normalize   = False,
    mha_residual    = False,
    mha_epsilon = 1e-5,
    
    ffn_dim     = 4.,
    ffn_use_bias    = False,
    ffn_activation  = 'gelu'
)

def _split_multi_query_attn(values, keys, embedding_dim = None, head_dim = None):
    if len(values) > 1: raise NotImplementedError()
    return {
        'query' : [values[0].T[: embedding_dim].T],
        'key'   : [values[0].T[embedding_dim : - head_dim].T],
        'value' : [values[0].T[- head_dim :].T]
    }

class Falcon(TextTransformerEncoder):
    default_params  = HParamsFalcon

    def __init__(self, * args, ** kwargs):
        if kwargs.get('rotary', HParamsFalcon['rotary']):
            kwargs['mha_class'] = RotaryMultiHeadAttention
        super().__init__(* args, ** kwargs)

        self.final_dense    = tf.keras.layers.Dense(
            self.output_dim, use_bias = self.hparams.final_bias, name = 'final_dense'
        )
        self.final_act_layer    = get_activation(self.hparams.final_activation)
    
    @property
    def output_dim(self):
        return self.hparams.output_dim if self.hparams.output_dim else self.hparams.vocab_size
    
    def compute_output(self, output, apply_softmax = True, ** kwargs):
        output = super().compute_output(output, ** kwargs)
        
        output = self.final_dense(output)
        if self.final_act_layer is not None and apply_softmax:
            output = self.final_act_layer(output)
        return output

    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import _transformer_patterns
        
        kwargs.setdefault('skip_root', False)
        kwargs.setdefault('patterns', {
            ** _transformer_patterns, 'h_to_4h' : '1', '4h_to_h' : '2', 'lm_head' : 'dense_final',
            'h/' : 'layer_', 'self_attention' : 'mha', 'ln_f' : 'norm_final',
            'input_layernorm' : 'norm_input', 'mha/dense' : 'mha/output_layer'
        })

        embedding_dim   = self.embedding_dim
        mha_head_dim    = self._layers[0].attention.depth
        kwargs.setdefault('transforms', {
            'query_key_value' : lambda key, values: {
                key.replace('query_key_value', '{}_layer'.format(qkv)) : v
                for qkv, v in _split_multi_query_attn(
                    values,
                    keys            = ['query', 'key', 'value'],
                    embedding_dim   = embedding_dim,
                    head_dim        = mha_head_dim
                ).items()
            }
        })

        return super(Falcon, self).transfer_weights(pretrained, ** kwargs)
    
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'tiiuae/falcon-7b',
                        pretrained  = None,
                        partial     = False,
                        ** kwargs
                       ):
        from models import _pretrained_models_folder
        
        model_dir   = os.path.join(
            _pretrained_models_folder, 'pretrained_weights', pretrained_name.replace('/', '--')
        )
        config_file = os.path.join(model_dir, 'config.json')
        weights_file    = os.path.join(model_dir, 'weights.keras')
        
        if not os.path.exists(config_file) or not os.path.exists(weights_file):
            if pretrained is None: pretrained = _get_pretrained_falcon(pretrained_name, ** kwargs)

            config = cls.default_params(
                sos_token       = pretrained.config.bos_token_id,
                eos_token       = pretrained.config.eos_token_id,
                pad_token       = pretrained.config.eos_token_id,
                vocab_size      = pretrained.config.vocab_size,
                embedding_dim   = pretrained.config.hidden_size,
                rotary          = pretrained.config.rotary,
                mha_ffn_in_parallel = pretrained.config.parallel_attn,

                num_layers  = pretrained.config.num_hidden_layers,
                mha_num_heads   = pretrained.config.num_attention_heads,
                mha_multi_query = pretrained.config.multi_query
            )


            instance = cls(** config(** kwargs))
            instance._build()

            instance.transfer_weights(pretrained, ** kwargs)
            
            os.makedirs(model_dir, exist_ok = True)
            dump_json(config_file, instance.get_config(), indent = 4)
            instance.save_weights(weights_file)
        else:
            from utils import load_json
            
            logger.info('Building model from config file {}'.format(config_file))
            instance = cls.from_config({** load_json(config_file), ** kwargs})
            instance._build()

            logger.info('Loading weights from {}'.format(weights_file))
            try:
                instance.load_weights(weights_file)
            except ValueError as e:
                if partial:
                    from models.weights_converter import name_based_partial_transfer_learning
                    
                    logger.info('Loading official pretrained model for partial transfer')
                    original = cls.from_pretrained(pretrained_name, pretrained)
                    
                    logger.info('Making partial transfer learning')
                    name_based_partial_transfer_learning(instance, original, ** kwargs)
                    del original
                else:
                    logger.warning(str(e))

        return instance

def _get_pretrained_falcon(model_name, torch_dtype = 'float16', device = 'cpu', ** _):
    import torch
    
    from transformers import AutoTokenizer, pipeline
    
    encoder = AutoTokenizer.from_pretrained(model_name)

    return pipeline(
        'text-generation',
        model      = model_name,
        tokenizer  = encoder,
        device_map = device,
        torch_dtype = getattr(torch, torch_dtype)
    ).model

custom_functions    = {
    'Falcon'    : Falcon
}

custom_objects  = custom_functions

_encoders   = {'Falcon' : Falcon}
_transformers   = _encoders