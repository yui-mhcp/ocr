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

""" TF 2.0 CLIP (Text Encoder) model, compatible with the official clip implementation """

import numpy as np
import tensorflow as tf

from custom_architectures.transformers_arch.gpt2_arch import HParamsBaseGPT2, BaseGPT2
from custom_architectures.transformers_arch.embedding_head import HParamsEmbeddingHead, EmbeddingHead
from custom_architectures.transformers_arch.visual_transformer_arch import (
    HParamsVisualTransformer, VisualTransformer
)

HParamsCLIPTextEncoder  = HParamsBaseGPT2(** HParamsEmbeddingHead(token_selector = 'max'))

HParamsCLIPImageEncoder = HParamsVisualTransformer(
    ** HParamsEmbeddingHead(token_selector = 'first'),
    add_class_embedding = True
)

class CLIPTextEncoder(BaseGPT2):
    default_params  = HParamsCLIPTextEncoder
    
    def __init__(self, vocab_size, embedding_dim, ** kwargs):
        super().__init__(vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs)
        
        self.embedding_head = EmbeddingHead(** self.hparams)

    def compute_output(self, output, training = False, mask = None, inputs = None, ** kwargs):
        output  = super().compute_output(output, training = training, mask = mask, ** kwargs)

        return self.embedding_head(output, training = training, mask = mask, text = inputs)
    
    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import _attn_split

        kwargs.setdefault(
            'transforms', {** _attn_split, 'text_layer' : lambda k, v: {k : [vi.T for vi in v]}}
        )

        return super().transfer_weights(pretrained, ** kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_name = 'RN50', pretrained = None,** kwargs):
        from custom_architectures.clip_arch import load_clip

        state_dict = load_clip(pretrained_name, pretrained = pretrained)
        
        vocab_size      = state_dict["token_embedding.weight"].shape[0]
        output_dim      = state_dict["text_projection"].shape[1]
        context_length  = state_dict["positional_embedding"].shape[0]
        embedding_dim   = state_dict["ln_final.weight"].shape[0]
        num_layers      = len(
            set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks"))
        )

        config = HParamsCLIPTextEncoder(
            vocab_size  = vocab_size,
            output_dim  = output_dim,
            output_bias = False,
            embedding_dim   = embedding_dim,
            max_input_length = context_length,
            scale_embedding = False,
            num_layers      = num_layers,
            mha_num_heads  = embedding_dim // 64,
            ffn_dim        = embedding_dim * 4,
            ffn_activation = 'quick_gelu'
        )
        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(state_dict, ** kwargs)

        return instance

class CLIPImageEncoder(VisualTransformer):
    default_params  = HParamsCLIPImageEncoder
    
    def __init__(self, * args, ** kwargs):
        super().__init__(* args, ** kwargs)
        
        self.embedding_head = EmbeddingHead(** self.hparams, name = 'embedding_layer')
    
    def compute_output(self, output, training = False, mask = None, inputs = None, ** kwargs):
        output = super().compute_output(output, training = training, mask = mask)
        
        return self.embedding_head(output, training = training, mask = mask, ** kwargs)

_clip_objects   = {
    'CLIPTextEncoder'   : CLIPTextEncoder,
    'CLIPImageEncoder'  : CLIPImageEncoder
}
custom_functions    = _clip_objects

custom_objects  = {
    ** _clip_objects,
    'EmbeddingHead' : EmbeddingHead
}

_encoders   = _clip_objects
_transformers   = _encoders