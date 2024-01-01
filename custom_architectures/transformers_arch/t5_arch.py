
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

""" TF 2.0 T5 (and Flan-T5) model, compatible with the `transformers`' model. """

import tensorflow as tf

from tqdm import tqdm

from custom_layers import FasterEmbedding, MultiHeadAttention, HParamsMHA, get_activation
from custom_architectures.transformers_arch.embedding_head import EmbeddingHead, HParamsEmbeddingHead
from custom_architectures.transformers_arch.text_transformer_arch import *

_default_t5_mha_config  = {
    'scale' : False,
    'normalize' : False,
    'use_bias'  : False,
    'output_bias'   : False,
    'mask_factor'   : -1e9,
    'normalize_input'   : True
}

HParamsT5Block  = HParamsTextTransformerBlock(
    ** {'mha_' + k : v for k, v in _default_t5_mha_config.items()},
    ** {'enc_mha_' + k : v for k, v in _default_t5_mha_config.items()},
    use_relative_positional_bias    = True,
    max_distance    = 128,
    num_buckets     = 32,
    
    normalize_embeddings    = False,
    scale_embedding     = False,
    normalize   = 'middle',
    
    ffn_use_bias    = False,
    ffn_use_up_proj = True,
    normalize_output    = True,
)

HParamsT5Encoder    = HParamsT5Block
HParamsT5Embedding  = HParamsT5Encoder(** HParamsEmbeddingHead)

HParamsT5Decoder  = HParamsT5Block(
    use_encoder_attention   = True,
    use_causal_attention    = True,
    final_activation    = 'softmax',
    final_bias  = False
)

class T5LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon = 1e-6, ** kwargs):
        super().__init__(** kwargs)
        self.epsilon    = epsilon
    
    def build(self, input_shape):
        self.weight = self.add_weight('weight', shape = (input_shape[-1], ), initializer = 'ones')
        super().build(input_shape)
    
    def call(self, inputs):
        variances   = tf.math.reduce_mean(tf.math.square(inputs), axis = -1, keepdims = True)
        return inputs * tf.math.rsqrt(variances + self.epsilon) * self.weight 
    
    def get_config(self):
        return {
            ** super().get_config(),
            'epsilon'   : self.epsilon
        }

class T5MultiHeadAttention(MultiHeadAttention):
    def compute_attention(self, attn_logits, v, * args, positional_bias = None, ** kwargs):
        if positional_bias is not None:
            attn_logits = attn_logits + positional_bias
        
        return super().compute_attention(attn_logits, v, * args, ** kwargs)
        
class T5Block(TextTransformerBlock):
    default_params  = HParamsT5Block
    _attr_to_set    = TextTransformerBlock._attr_to_set + [
        'use_relative_positional_bias', 'num_buckets', 'max_distance'
    ]

    def __init__(self, vocab_size, embedding_dim, max_input_length = -1, ** kwargs):
        super().__init__(
            vocab_size,
            embedding_dim,
            max_input_length    = max_input_length,
            norm_class  = T5LayerNormalization,
            mha_class   = T5MultiHeadAttention,
            ** kwargs
        )
    
    def _init_input_layers(self, ** kwargs):
        super()._init_input_layers(** kwargs)
        if self.use_relative_positional_bias:
            with tf.name_scope('relative_pos_bias'):
                self.relative_attention_bias = self.add_weight(
                    shape   = (self.hparams.num_buckets, self.hparams.mha_num_heads),
                    name    = 'embeddings'
                )
    
    def compute_bias(self, q_len, k_len):
        q_pos   = tf.range(q_len)[:, tf.newaxis]
        k_pos   = tf.range(k_len)[tf.newaxis, :]
        rel_pos = k_pos - q_pos
        
        rel_pos_bucket  = self._relative_position_bucket(
            rel_pos,
            bidirectional   = not self.use_causal_attention,
            num_buckets     = self.num_buckets,
            max_distance    = self.max_distance
        )
        
        values  = tf.gather(
            self.relative_attention_bias, rel_pos_bucket
        )
        values  = tf.expand_dims(
            tf.transpose(values, [2, 0, 1]), axis = 0
        )
        return values
    
    def call(self, inputs, * args, first_layer_idx = -1, positional_bias = None, ** kwargs):
        if self.use_relative_positional_bias and positional_bias is None and first_layer_idx == -1:
            positional_bias = self.compute_bias(
                tf.shape(inputs)[1], tf.shape(inputs)[1]
            )
        
        return super().call(
            inputs,
            * args,
            first_layer_idx     = first_layer_idx,
            attention_kwargs    = {'positional_bias' : positional_bias},
            ** kwargs
        )
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
        if bidirectional:
            num_buckets         = num_buckets // 2
            relative_buckets    = tf.where(relative_position > 0, num_buckets, 0)
            relative_position   = tf.abs(relative_position)
        else:
            relative_buckets    = 0
            relative_position   = - tf.math.minimum(relative_position, 0)
        
        max_exact   = num_buckets // 2
        is_small    = relative_position < max_exact
        
        relative_position_large = tf.minimum(max_exact + tf.cast(
            tf.math.log(tf.cast(relative_position, tf.float32) / tf.cast(max_exact, tf.float32))
            / tf.math.log(tf.cast(max_distance, tf.float32) / tf.cast(max_exact, tf.float32))
            * tf.cast(num_buckets - max_exact, tf.float32),
            dtype   = relative_position.dtype
        ), num_buckets - 1)
        return relative_buckets + tf.where(is_small, relative_position, relative_position_large)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'google/flan-t5-large',
                        pretrained_task = 'generation',
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            pretrained = transformers_t5(pretrained_name, pretrained_task)

        config = cls.default_params(
            max_input_length    = -1,
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            sos_token   = pretrained.config.decoder_start_token_id,
            eos_token   = pretrained.config.eos_token_id,
            pad_token   = pretrained.config.pad_token_id,
            epsilon = pretrained.config.layer_norm_epsilon,

            num_layers  = pretrained.config.num_layers,
            ffn_dim     = pretrained.config.d_ff,
            ffn_use_up_proj = pretrained.config.is_gated_act,
            ffn_activation  = pretrained.config.dense_act_fn,
            mha_num_heads   = pretrained.config.num_heads
        )
        
        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(pretrained)
        
        return instance


class T5Encoder(T5Block):
    default_params = HParamsT5Encoder
    
    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import get_layers
        
        kwargs.setdefault('patterns', {
            '/q' : '/query_layer', '/k' : '/key_layer', '/v' : '/value_layer',
            '/o' : '/output_layer', 'layer_._0/layer_norm' : 'mha/norm_input', 'block_._' : 'layer_',
            '.*relative_attention_bias' : 'relative_pos_bias',
            'SelfAttention' : 'mha', 'DenseReluDense' : 'ffn',
            'wi_0' : 'dense_1', 'wi_1' : 'up_proj', 'wo' : 'dense_2', r'layer_._\d/' : '',
            'layer_norm' : 'norm'
        })
        return super().transfer_weights(
            {k : v for k, v in get_layers(pretrained).items() if 'decoder' not in k}, ** kwargs
        )

class T5Embedding(T5Encoder):
    default_params = HParamsT5Embedding
    
    def __init__(self, output_dim, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.embedding_head = EmbeddingHead(** self.hparams)

    def compute_output(self, output, training = False, mask = None, ** kwargs):
        return self.embedding_head(output, mask = mask, training = training)

class T5Decoder(T5Block):
    default_params = HParamsT5Decoder
    
    def __init__(self, vocab_size, embedding_dim, token_embedding = None, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim,
            token_embedding = token_embedding, ** kwargs
        )

        self.final_dense = tf.keras.layers.Dense(
            vocab_size, use_bias = self.hparams.final_bias, name = "classification"
        )
        self.final_act_layer    = get_activation(self.hparams.final_activation)
    
    def change_vocabulary(self, vocab, ** kwargs):
        raise NotImplementedError()
        
    @property
    def output_dim(self):
        return self.vocab_size
    
    def compute_output(self, output, apply_softmax = True, ** kwargs):
        output = super().compute_output(output, ** kwargs)
        
        output = self.final_dense(output)
        if self.final_act_layer is not None and apply_softmax:
            output = self.final_act_layer(output)
        return output
    
    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import get_layers
        
        kwargs.setdefault('patterns', {
            '/q' : '/query_layer', '/k' : '/key_layer', '/v' : '/value_layer',
            '/o' : '/output_layer',
            'layer_._0/layer_norm' : 'mha/norm_input',
            'layer_._1/layer_norm' : 'enc_mha/norm_input', 'block_._' : 'layer_',
            '.*relative_attention_bias' : 'relative_pos_bias', 'lm_head' : 'classification',
            'SelfAttention' : 'mha', 'EncDecAttention' : 'enc_mha', 'DenseReluDense' : 'ffn',
            'wi_0' : 'dense_1', 'wi_1' : 'up_proj', 'wo' : 'dense_2',
            r'layer_._\d/' : '', 'layer_norm' : 'norm'
        })
        return super().transfer_weights(
            {k : v for k, v in get_layers(pretrained).items() if 'encoder' not in k}, ** kwargs
        )

class T5(TextTransformer):
    encoder_class   = T5Encoder
    decoder_class   = T5Decoder
    
    def __init__(self, vocab_size, embedding_dim, max_input_length = -1, ** kwargs):
        shared_embedding = FasterEmbedding(
            vocab_size, embedding_dim, name = "token_embedding"
        )
        super().__init__(
            vocab_size      = vocab_size,
            embedding_dim   = embedding_dim,
            max_input_length    = max_input_length,
            shared_layers   = {'token_embedding' : shared_embedding},
            ** kwargs
        )
        with tf.name_scope(self.name):
            self.shared_embedding = shared_embedding

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'google/flan-t5-large',
                        pretrained_task = 'generation', 
                        pretrained      = None,
                        tqdm    = tqdm,
                        ** kwargs
                       ):
        if pretrained is None:
            pretrained = transformers_t5(pretrained_name, pretrained_task)

        config = cls.default_params(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = -1,
            sos_token   = pretrained.config.decoder_start_token_id,
            eos_token   = pretrained.config.eos_token_id,
            pad_token   = pretrained.config.pad_token_id,
            epsilon = pretrained.config.layer_norm_epsilon,

            num_layers  = pretrained.config.num_layers,
            mha_num_heads   = pretrained.config.num_heads,
            ffn_dim     = pretrained.config.d_ff,
            ffn_use_up_proj = pretrained.config.is_gated_act,
            ffn_activation  = pretrained.config.dense_act_fn
        )
        
        instance = cls(** config(** kwargs))
        instance._build()
        
        instance.encoder.transfer_weights(pretrained, tqdm = tqdm)
        instance.decoder.transfer_weights(pretrained, tqdm = tqdm)
        
        return instance

def transformers_t5(name = 'google/flan-t5-large', task = 'generation'):
    import transformers
    if task == 'generation':
        return transformers.TFT5ForConditionalGeneration.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))

_t5_classes   = {
    'T5Encoder'   : T5Encoder,
    'T5Embedding' : T5Embedding,
    'T5Decoder'   : T5Decoder,
    'T5'          : T5
}
        
custom_functions    = {
    ** _t5_classes,
    'transformers_t5'   : transformers_t5
}

custom_objects  = {
    ** _t5_classes,
    'TransformerEncoder'    : TransformerEncoder
}

_encoders   = {'T5' : T5Embedding}
_decoders   = {'T5' : T5Decoder}
_transformers   = {'T5' : T5}