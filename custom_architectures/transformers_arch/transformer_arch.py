
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

""" Tensorflow 2.x implementation of the main Transformers' blocks """

import json
import logging
import collections
import tensorflow as tf

from tensorflow.keras.models import model_from_json

from loggers import timer
from hparams import HParams
from custom_layers import get_activation, MultiHeadAttention, HParamsMHA

time_logger = logging.getLogger('timer')

TransformerOutput = collections.namedtuple(
    "TransformerOutput", [
        "output",
        "state",
        "logits",
        "attention_weights",
        "hidden_states",
        "mask"
    ]
)


_base_enc_dec_kwargs    = {
    'num_layers'    : 4,
    'normalize_output'  : False
}
_shared_config          = [
    'embedding_dim', 'norm_training', 'epsilon', 'ffn_dim', 'ffn_activation', 'drop_rate'
]

HParamsTransformerLayer = HParams(
    ** HParamsMHA.get_config(add_prefix = 'mha'),
    ** HParamsMHA.get_config(add_prefix = 'enc_mha'),
    embedding_dim   = 512,
    normalize   = 'after',
    epsilon     = 1e-12,
    drop_rate   = 0.1,
    use_encoder_attention   = False,
    encoder_embedding_dim   = None,
    use_causal_attention    = False,
    ffn_dim     = 1024,
    ffn_activation  = 'relu',
    norm_training   = True      # whether to allow `training = True` or not
)
HParamsTransformerBlock = HParamsTransformerLayer(** _base_enc_dec_kwargs)

HParamsTransformerEncoder   = HParamsTransformerBlock
HParamsTransformerDecoder   = HParamsTransformerBlock(
    use_encoder_attention = True, use_causal_attention = True
)

def _get_state_length(state):
    """
        Returns the length of state (i.e. the 3rd dimension of any item)
        `state` is a dict of {layer_name : state(s)} meaning that the saved `state(s)` is either :
        - a tuple `(k, v)` with `{k / v}.shape == [batch_size, num_heads, seq_length, mha_depth]`
        - a tuple of tuple `((k, v), (enc_k, enc_v))` where `k` has the same shape as above
        Therefore, taking `state[0][0]` either returns : `k[0]` or `k`.
        In both cases, the dimension `-2` is the expected sequence length, it is the easiest way to get the information without taking care of the possibly nested tuple
    """
    if not state: return 0
    flattened = tf.nest.flatten(state)
    return tf.shape(flattened[0])[-2] if flattened[0] is not None else 0

def build_padding_mask(seq, mask = None, lengths = None, pad_value = 0, maxlen = None, dtype = tf.bool):
    """
        Return padding mask matching attention shape [batch_size, 1, 1, max(lengths)]
        The mask is `False` (or 0) if the value should be masked and `True` (or 1) otherwise
    """
    if mask is not None:
        if mask.dtype != dtype: mask = tf.cast(mask, dtype)
        if len(mask.shape) == 4: return mask
        return tf.reshape(mask, [tf.shape(mask)[0], 1, 1, tf.shape(mask)[1]])
    
    if lengths is None:
        if len(seq.shape) == 2:
            mask = tf.cast(tf.math.not_equal(seq, tf.cast(pad_value, seq.dtype)), dtype = dtype)
        elif len(seq.shape) == 3:
            mask = tf.cast(tf.reduce_any(seq != tf.cast(pad_value, seq.dtype), axis = -1), dtype)
        else:
            raise ValueError('Unsupported sequence shape : {}'.format(tf.shape(seq)))
    else:
        if maxlen is None: maxlen = tf.shape(seq)[1]
        mask = tf.sequence_mask(lengths, maxlen = maxlen, dtype = dtype)
    
    return tf.reshape(mask, [tf.shape(seq)[0], 1, 1, -1])

def build_look_ahead_mask(batch_size, size, dtype = tf.bool):
    """
        Creates a `look ahead` mask with shape [batch_size, 1, size, size]
        The mask is `False` (or 0) if the value should be masked and `True` (or 1) otherwise
    """
    mask = tf.linalg.band_part(tf.ones((size, size), dtype = dtype), -1, 0)
    return tf.tile(tf.reshape(mask, [1, 1, size, size]), [batch_size, 1, 1, 1])

def build_combined_mask(target, lengths = None, pad_value = 0, dtype = tf.bool):
    """
        Returns a mask combining `padding` and `look_ahead` with shape (batch_size, 1, seq_len, seq_len)
        The mask is `False` (or 0) if the value should be masked and `True` (or 1) otherwise
    """
    look_ahead_mask = build_look_ahead_mask(tf.shape(target)[0], tf.shape(target)[1], dtype = dtype)
    padding_mask    = build_padding_mask(
        target, lengths = lengths, pad_value = pad_value, dtype = dtype
    )
    
    return combine_masks(padding_mask, look_ahead_mask)

def combine_masks(padding_mask, look_ahead_mask):
    if padding_mask.dtype == tf.bool:
        return tf.logical_and(look_ahead_mask, padding_mask)
    return tf.minimum(look_ahead_mask, padding_mask)

#@timer
def format_output(output,
                  state     = None,
                  logits    = None,
                  attn_weights  = None,
                  hidden_states = None,
                  mask      = None,
                  types     = None,
                  
                  return_state      = False,
                  return_logits     = False,
                  return_attention  = False,
                  return_hidden_states  = False,
                  return_mask       = False,
                  return_types      = False,
                  
                  as_dict       = False,
                  ** kwargs
                 ):
    def _maybe_add(out, key, value, should_return):
        return out if value is None or not should_return else (out + (value, ))
    
    if as_dict:
        return TransformerOutput(
            output  = output,
            state   = state if return_state else None,
            logits  = logits if return_logits else None,
            attention_weights   = attn_weights if return_attention else None,
            hidden_states   = hidden_states if return_hidden_states else None,
            mask    = mask if return_mask else None
        )
    
    out = (output, )
    
    out = _maybe_add(out, 'state',          state,        should_return = return_state)
    out = _maybe_add(out, 'logits',         logits,       should_return = return_logits)
    out = _maybe_add(out, 'attention',      attn_weights, should_return = return_attention)
    out = _maybe_add(out, 'hidden_states',  hidden_states,  should_return = return_hidden_states)
    out = _maybe_add(out, 'mask',           mask,         should_return = return_mask)
    out = _maybe_add(out, 'types',          types,        should_return = return_types)
    
    return out[0] if not as_dict and len(out) == 1 else out

#@timer
def build_mask(inputs,
               use_causal_attention,
               lengths  = None,
               pad_value    = 0,
               
               mask = None,
               padding_mask = None,
               look_ahead_mask  = None,
               initial_state    = None,
               
               dtype    = tf.bool
              ):
    if mask is not None:
        if len(mask.shape) == 4: return mask
        padding_mask = mask

    offset = _get_state_length(initial_state)
    maxlen = tf.shape(inputs)[1] + offset
    padding_mask = build_padding_mask(
        inputs, mask = padding_mask, lengths = lengths, maxlen = maxlen, pad_value = pad_value,
        dtype = dtype
    )
    
    if not use_causal_attention or tf.shape(padding_mask)[-1] == 1: return padding_mask

    if look_ahead_mask is None:
        look_ahead_mask = build_look_ahead_mask(
            tf.shape(inputs)[0], tf.shape(padding_mask)[-1], dtype = dtype
        )
    
    return combine_masks(padding_mask, look_ahead_mask)

class FeedForwardNetwork(tf.keras.Model):
    def __init__(self, ffn_dim, ffn_activation, embedding_dim, use_bias = True,
                 use_up_proj = False, name = 'ffn'):
        """
            Simple 2-`Dense` sequential network with an activation function between the 2 layers.
            
            Arguments :
                - ffn_dim   : the 1st layer's number of units
                - ffn_activation    : the activation function between the 2 layers
                - embedding_dim     : the Transformers' depth (the number of units for the 2nd layer)
        """
        super().__init__(name = name)
        
        self.ffn_dim    = ffn_dim
        self.use_bias   = use_bias
        self.use_up_proj    = use_up_proj
        self.ffn_activation = ffn_activation
        self.embedding_dim  = embedding_dim
        
        self.dense_1    = tf.keras.layers.Dense(ffn_dim, use_bias = use_bias, name = 'dense_1')
        self.up_proj    = tf.keras.layers.Dense(ffn_dim, use_bias = use_bias, name = 'up_proj') if use_up_proj else None
        self.act        = get_activation(ffn_activation)
        self.dense_2    = tf.keras.layers.Dense(embedding_dim, use_bias = use_bias, name = 'dense_2')
    
    def call(self, inputs, training = False):
        x = self.dense_1(inputs)
        if self.act is not None: x = self.act(x)
        if self.use_up_proj:     x = x * self.up_proj(inputs)
        return self.dense_2(x)

    def get_config(self):
        return {
            'name'  : self.name,
            'ffn_dim'   : self.ffn_dim,
            'use_bias'  : self.use_bias,
            'use_up_proj'   : self.use_up_proj,
            'ffn_activation'    : self.ffn_activation,
            'embedding_dim' : self.embedding_dim
        }
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, name = None, ** kwargs):
        """
            A fully customizable Transformer layer.
            It handles:
                - self-attention    : when Q = K = V
                    The 1st MHA is by default a self-attention layer
                    - In Encoder-only       : there is only 1 self-MHA
                    - In Encoder-Decoder    : there is 1 self-MHA followed by a causal-MHA
                - causal attention  : when using the masking operator
                    Set `use_causal_attention = True` in the constructor
                    The 2nd attention (if `use_encoder_attention = True`) is by default causal
                - Encoder-Decoder mode  : uses 2 MHA (a self-MHA followed by a causal-MHA)
                    Set `use_encoder_attention = True` in the constructor.
                    Note that the 2nd MHA is not a self-MHA as K and V are the `encoder_output` call argument
            
            See the `HParamsTransformerLayer` class for an exhaustive list of configuration. 
                Those starting with `ffn_` are related to the feed-forward network
                Those starting with `mha_` are related to the 1st MHA
                Those starting with `enc_mha_` are related to the 2nd MHA (ignored if `use_encoder_attention = False`)
                
                - normalize : where to apply the `LayerNormalization`
                    - before    : directly on the layer's input
                    - middle    : just before the FFN call but it does not normalize the FFN's residual !
                    `ffn_out = mha_out + norm(ffn(mha_out))` (it is used by `GPT-2` models)
                    - after     : default case where the normalization is applied on the FFN's output
                - use_causal_attention  : whether to use the masking operator or not (on the 1st MHA)
                - use_encoder_attention : whether to use 1 or 2 MHA
            
            Note that the `epsilon` and `norm_training` are propagated to the MHA
        """
        super().__init__(name = name)
        self.supports_masking   = True

        self.hparams    = HParamsTransformerLayer.extract(kwargs)
        self.hparams    = self.hparams(
            embedding_dim   = embedding_dim,
            mha_epsilon     = self.hparams.epsilon,
            mha_attention_dim   = embedding_dim,
            mha_norm_training   = self.hparams.norm_training,
            mha_is_cross_attention  = False,
            enc_mha_epsilon     = self.hparams.epsilon,
            enc_mha_attention_dim   = embedding_dim,
            enc_mha_norm_training   = self.hparams.norm_training,
            enc_mha_is_cross_attention  = True
        )
        if self.hparams.enc_mha_num_heads == -1:
            self.hparams.enc_mha_num_heads = self.hparams.mha_num_heads
        
        self.normalize  = self.hparams.normalize
        self.norm_training  = self.hparams.norm_training
        self.use_causal_attention   = self.hparams.use_causal_attention
        self.use_encoder_attention  = self.hparams.use_encoder_attention
        
        self.attention  = MultiHeadAttention(
            ** self.hparams.get_config(prefix = 'mha'), name = 'mha'
        )
        self.enc_attention  = MultiHeadAttention(
            ** self.hparams.get_config(prefix = 'enc_mha'), name = 'enc_mha'
        ) if self.use_encoder_attention else None
        
        self.ffn = FeedForwardNetwork(
            self.hparams.ffn_dim, self.hparams.ffn_activation, embedding_dim, name = 'ffn'
        )
        
        self.norm   = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm'
        ) if self.hparams.normalize else None
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)
    
    def get_initial_state(self, inputs, batch_size = None, dtype = tf.float32):
        attn_state = self.attention.get_initial_state(inputs, batch_size = batch_size, dtype = dtype)
        if self.enc_attention is None: return attn_state
        return (attn_state, self.enc_attention.get_initial_state(inputs, batch_size, dtype = dtype))
    
    def initialize_cache(self, inputs, encoder_output = None, only_cross_cache = False):
        attn_state = self.attention.initialize_cache(
            inputs, inputs, inputs
        ) if not only_cross_cache else None
        
        if self.enc_attention is not None:
            enc_attn_state = None
            if encoder_output is not None:
                enc_attn_state = self.enc_attention.initialize_cache(
                    inputs, encoder_output, encoder_output
                )
            return (attn_state, enc_attn_state)
        
        return attn_state
    
    def compute_mask(self,
                     inputs,
                     mask   = None,
                     input_length   = None,
                     padding_mask   = None,
                     look_ahead_mask    = None,
                     initial_state  = None,
                     dtype  = tf.bool
                    ):
        return build_mask(
            inputs,
            self.use_causal_attention,
            lengths = input_length,
            
            mask    = mask,
            padding_mask    = padding_mask,
            look_ahead_mask = look_ahead_mask,
            initial_state   = initial_state,
            
            dtype   = dtype
        )

    #@timer(name = 'layer call')
    def call(self,
             inputs,
             input_length   = None,
             encoder_output = None,
             
             initial_state  = None,
             
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             enc_padding_mask   = None,
             
             training   = False,
             return_state       = False,
             return_attention   = True,
             ** kwargs
            ):
        """
            Arguments :
                - inputs    : the layers' input (the query) with shape [B, q_len, embedding_dim]
                - input_length  : the inputs' sequence lengths (to build the padding mask)
                - encoder_output    : encoder output with shape [B, in_seq_len, encoder_embedding_dim]
                
                - initial_state     : state to use (typically the previous iteration state)
                
                - mask  : the mask to use for the 1st MHA
                - padding_mask  : the padding mask for the 1st MHA          [B, 1, seq_len, seq_len]
                - look_ahead_mask   : the causal mask for the 1st MHA       [B, 1, 1, seq_len]
                - enc_padding_mask  : the padding mask used for the 2nd MHA [B, 1, 1, in_seq_len]
                
                - training  : whether it is training / inference phase
                - return_state      : whether to return the internal state or not
                - return_attention  : whether to return attention weights or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : self-attention weights for each head of the MHA
        """
        if mask is None:
            mask = self.compute_mask(
                inputs,
                input_length    = input_length,
                padding_mask    = padding_mask,
                look_ahead_mask = look_ahead_mask,
                initial_state   = initial_state
            )

        if self.normalize == 'before':
            inputs = self.norm(inputs, training = training and self.norm_training)

        time_logger.start_timer('self MHA call')

        attn_state, enc_attn_state = None, None
        if initial_state:
            attn_state, enc_attn_state = initial_state if self.enc_attention is not None else (initial_state, None)
            
        attn_outputs    = self.attention(
            inputs,
            mask    = mask,
            training    = training,
            initial_state   = attn_state,
            return_attention    = return_attention,
            return_state    = return_state,
            normalize_kv    = True
        )
        if not isinstance(attn_outputs, tuple): attn_outputs = (attn_outputs, )
        attn_out = attn_outputs[0]
        
        time_logger.stop_timer('self MHA call')

        if self.enc_attention is not None:
            if encoder_output is None:
                raise RuntimeError("You must provide encoder output when using encoder attention !")
            
            time_logger.start_timer('cross MHA call')

            enc_attn_outputs = self.enc_attention(
                attn_out,
                encoder_output,
                encoder_output,
                mask    = enc_padding_mask,
                training    = training,
                initial_state   = enc_attn_state,
                return_attention    = return_attention,
                return_state    = return_state,
                normalize_kv    = False
            )
            attn_out = enc_attn_outputs
            if return_attention or return_state:
                attn_out    = enc_attn_outputs[0]
                attn_outputs    = tuple((o1, o2) for o1, o2 in zip(attn_outputs, enc_attn_outputs))
        
            time_logger.stop_timer('cross MHA call')

        elif encoder_output is not None:
            raise RuntimeError(
                "You cannot pass `encoder_output` when `self.use_encoder_attention` is False !"
            )
        time_logger.start_timer('MLP call')

        ffn_in = attn_out
        if self.normalize == 'middle':
            ffn_in = self.norm(ffn_in, training = training and self.norm_training)
        
        ffn_output  = self.ffn(ffn_in, training = training)
        ffn_output  = self.dropout(ffn_output, training = training)
        
        output  = ffn_output + attn_out
        
        if self.normalize == 'after':
            output = self.norm(output, training = training and self.norm_training)
        
        time_logger.stop_timer('MLP call')

        return output if len(attn_outputs) == 1 else ((output,) + attn_outputs[1:])
    
    def get_output_shape(self,
                         input_shape,
                         encoder_output = None,
                         return_state   = False,
                         return_attention   = True,
                        ):
        attn_out_shape    = self.attention.get_output_shape(
            input_shape, input_shape, input_shape,
            return_attention = return_attention, return_state = return_state
        )
        
        if self.enc_attention is not None:
            if encoder_output is None:
                raise ValueError("You must provide encoder output when using encoder attention !")
            
            enc_attn_out_shape = self.enc_attention.get_output_shape(
                input_shape, encoder_output, encoder_output,
                return_attention = return_attention, return_state = return_state
            )
            if return_attention or return_state:
                attn_out_shape  = (enc_attn_out_shape[0], ) + tuple(
                    (o1, o2) for o1, o2 in zip(attn_out_shape, enc_attn_out_shape)
                )[1:]
        elif encoder_output is not None:
            raise ValueError(
                "You cannot pass `encoder_output` when `self.use_encoder_attention` is False !"
            )
        
        return attn_out_shape
    
    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()

class TransformerBlock(tf.keras.Model):
    default_params  = HParamsTransformerBlock
    _attr_to_set    = [
        'embedding_dim', 'norm_training', 'use_causal_attention'
    ]
    
    def __init__(self, embedding_dim, num_layers, name = None, ** kwargs):
        """ Simply a list of `num_layers` TransformerLayer applied sequentially """
        super().__init__(name = name)
        
        kwargs.update({'embedding_dim' : embedding_dim, 'num_layers' : num_layers})
        self.hparams    = self.default_params.extract(kwargs)
        
        for attr_name in self._attr_to_set:
            setattr(self, attr_name, self.hparams[attr_name])
        
        self._init_input_layers(** kwargs)
        
        self._layers = [
            TransformerLayer(name = 'layer_{}'.format(i), ** self.hparams)
            for i in range(self.hparams.num_layers)
        ]
        
        self.norm       = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_final'
        ) if self.hparams.normalize_output else None
    
    @property
    def input_signature(self):
        return tf.TensorSpec(shape = (None, None, self.embedding_dim), dtype = tf.float32)
    
    @property
    def output_dim(self):
        return self.embedding_dim
    
    @property
    def dummy_inputs(self):
        return tf.nest.map_structure(
            lambda sign: tf.random.uniform(
                tuple(s if s is not None else 1 for s in sign.shape), 1, 2, dtype = sign.dtype
            ), self.input_signature
        )
    
    @property
    def dummy_encoder_output(self):
        if not self.hparams.use_encoder_attention: return None
        emb_dim = self.hparams.encoder_embedding_dim if self.hparams.encoder_embedding_dim else self.embedding_dim
        return tf.random.normal((1, 16, emb_dim), dtype = tf.float32)
    
    def _init_input_layers(self, ** kwargs):
        pass
    
    def _build(self, ** kwargs):
        return self(self.dummy_inputs, encoder_output = self.dummy_encoder_output, ** kwargs)

    def __len__(self):
        return len(self._layers)
    
    def __getitem__(self, idx):
        return self._layers[idx]
    
    def freeze(self, trainable = False):
        """ Set all `self._layers.trainable` to `trainable` """
        for layer in self._layers: layer.trainable = trainable

    def get_initial_state(self, inputs, batch_size = None, dtype = tf.float32):
        return {
            layer.name : layer.get_initial_state(inputs, batch_size, dtype) for layer in self._layers
        }
    
    def initialize_cache(self, * args, ** kwargs):
        return {
            layer.name : layer.initialize_cache(* args, ** kwargs) for layer in self._layers
        }

    def prepare_input(self,
                      inputs,
                      input_length  = None,
                      additional_inputs = [],
                      
                      mask  = None,
                      training  = False,
                      
                      ** kwargs
                     ):
        return inputs
    
    def compute_output(self, output, training = False, mask = None, ** kwargs):
        if self.norm is not None:
            output = self.norm(output, training = training and self.norm_training)
        
        return output
    
    #@timer(name = 'Transformer block call')
    def call(self,
             inputs,
             input_length   = None,
             encoder_output = None,
             initial_state  = None,
             
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             enc_padding_mask   = None,
             
             training   = False,
             
             first_layer_idx    = -1,
             last_layer_idx     = -1,
             
             return_state       = False,
             return_attention   = False,
             return_last_attention  = False,
             return_hidden_states   = False,
             return_mask        = False,
             as_dict    = False,
             
             ** kwargs
            ):
        """
            See the TransformerLayer for more information
            
            Arguments :
                - inputs    : block inputs with shape [batch_size, seq_len, embedding_dim], embedded inputs
                - mask      : attention mask (padding mask based in inputs)
                - training  : whether it is training / inference phase
                - return_attention  : whether to return attention weights or not
                - return_states     : whether to return intermediate representation or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : dict self-attention weights for each head of the MHA of each layer
        """
        if last_layer_idx == -1:    last_layer_idx = len(self._layers)
        
        states              = {} if return_state else None
        attention_weights   = {} if return_attention or return_last_attention else None
        hidden_states       = {} if return_hidden_states else None

        additional_inputs   = []
        if isinstance(inputs, (list, tuple)):
            (inputs, input_length), additional_inputs = inputs[:2], inputs[2:]
        
        output = inputs
        if first_layer_idx == -1:
            first_layer_idx = 0
            output = self.prepare_input(
                output,
                input_length    = input_length,
                additional_inputs   = additional_inputs,
                initial_state   = initial_state,
                
                mask    = mask,
                training    = training,
                ** kwargs
            )
            if hasattr(output, '_keras_mask'): padding_mask = output._keras_mask

        mask = self._layers[0].compute_mask(
            inputs,
            input_length    = input_length,
            padding_mask    = padding_mask,
            look_ahead_mask = look_ahead_mask,
            initial_state   = initial_state
        )

        for i, layer in enumerate(self._layers[first_layer_idx : last_layer_idx], start = first_layer_idx):
            output, state, attn_weights = layer(
                output,
                input_length    = input_length,
                encoder_output  = encoder_output,
                initial_state   = None if initial_state is None else initial_state.get(layer.name, None),
                
                training    = training,
                
                mask    = mask,
                padding_mask    = padding_mask,
                look_ahead_mask = look_ahead_mask,
                enc_padding_mask    = enc_padding_mask,
                
                return_attention    = True,
                return_state        = True,
                
                ** kwargs
            )
            if return_state:
                states[layer.name] = state
            
            if return_attention or (return_last_attention and i == len(self._layers) - 1):
                if layer.enc_attention is None:
                    attention_weights['attn_{}'.format(layer.name)] = attn_weights
                else:
                    attention_weights['attn_{}'.format(layer.name)] = attn_weights[0]
                    attention_weights['enc_attn_{}'.format(layer.name)] = attn_weights[1]
            
            if return_hidden_states:
                hidden_states['state_{}'.format(layer.name)] = output
        
        if last_layer_idx >= len(self._layers):
            output = self.compute_output(
                output,
                mask    = mask,
                training    = training,
                inputs  = inputs,
                ** kwargs
            )
        
        return format_output(
            output,
            state   = states,
            attn_weights    = attention_weights,
            hidden_states   = hidden_states,
            mask    = mask,
            
            return_state        = return_state,
            return_attention    = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask         = return_mask,
            as_dict = as_dict
        )
    
    def transfer_weights(self, pretrained, patterns = None, ** kwargs):
        from models.weights_converter import (
            _transformer_patterns, name_based_partial_transfer_learning
        )
        kwargs.setdefault('patterns', _transformer_patterns)

        return name_based_partial_transfer_learning(self, pretrained, ** kwargs)

    def get_output_shape(self,
                         inputs,
                         encoder_output = None,
                         return_state   = None,
                         return_attention   = None,
                         return_last_attention  = None,
                         return_hidden_states   = None,
                         return_mask        = None,
                         as_dict    = False
                        ):
        output_shape    = inputs[:-1] + (self.output_dim, )
        
        mask_shape  = None
        
        states_shape              = {} if return_state else None
        attention_weights_shape   = {} if return_attention or return_last_attention else None
        hidden_states_shape       = {} if return_hidden_states else None
        
        output = inputs
        for i, layer in enumerate(self._layers):
            output, state, attn_weights = layer.get_output_shape(
                output,
                encoder_output  = encoder_output,
                return_attention    = True,
                return_state        = True
            )
            if return_state:
                states_shape[layer.name] = state
            
            if return_attention or (return_last_attention == True and i == len(self._layers) - 1):
                if layer.enc_attention is None:
                    attention_weights_shape['attn_{}'.format(layer.name)] = attn_weights
                else:
                    attention_weights_shape['attn_{}'.format(layer.name)] = attn_weights[0]
                    attention_weights_shape['enc_attn_{}'.format(layer.name)] = attn_weights[1]
            
            if return_hidden_states:
                hidden_states_shape['state_{}'.format(layer.name)] = output
        
        return format_output(
            output_shape,
            state   = states_shape,
            attn_weights    = attention_weights_shape,
            hidden_states   = hidden_states_shape,
            mask    = mask_shape,
            
            return_state        = return_state,
            return_attention    = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask         = return_mask,
            as_dict = as_dict
        )
    
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class TransformerEncoder(TransformerBlock):
    default_params = HParamsTransformerEncoder

class TransformerDecoder(TransformerBlock):
    default_params = HParamsTransformerDecoder

class Transformer(tf.keras.Model):
    encoder_class   = TransformerEncoder
    decoder_class   = TransformerDecoder
    
    _shared_keys    = _shared_config
    _attr_to_set    = []
    
    @classmethod
    @property
    def default_params(cls):
        return HParams(
            ** cls.encoder_class.default_params.get_config(add_prefix = 'encoder'),
            ** cls.decoder_class.default_params.get_config(add_prefix = 'decoder'),
            ** {k : None for k in cls._shared_keys}
        )
    
    def __init__(self,
                 name = None,
                 shared_layers = {},
                 
                 encoder    = None,
                 encoder_wrapper = None,
                 encoder_wrapper_params = None,
                 decoder    = None,
                 decoder_wrapper = None,
                 decoder_wrapper_params = None,
                 
                 ** kwargs
                ):
        super().__init__(name = name)
        
        if encoder is not None: self.encoder_class = encoder.__class__
        if decoder is not None: self.decoder_class = decoder.__class__
            
        # Init the default parameters`
        default_params  = self.default_params
        # Maybe adds parameters for wrappers (if any)
        if encoder_wrapper is None: encoder_wrapper = lambda x, ** kwargs: x
        elif encoder_wrapper_params is not None:
            default_params = default_params(** encoder_wrapper_params.get_config(add_prefix = 'encoder'))
        
        if decoder_wrapper is None: decoder_wrapper = lambda x, ** kwargs: x
        elif decoder_wrapper_params is not None:
            default_params = default_params(** decoder_wrapper_params.get_config(add_prefix = 'decoder'))
        
        self.hparams = default_params.extract(kwargs)
        # Allow to have different embedding dim for encoder and decoder
        _shared = {}
        for k in self._shared_keys:
            if self.hparams[k] is not None:
                _shared.update({
                    'encoder_{}'.format(k) : self.hparams[k],
                    'decoder_{}'.format(k) : self.hparams[k]
                })
        self.hparams.update(_shared)
        
        for attr_name in self._attr_to_set:
            setattr(self, attr_name, self.hparams[attr_name])
        
        # Effectively builds the encoder and decoder classes (with possible wrappers)
        if encoder is None:
            encoder = self.encoder_class(
                ** self.hparams.get_config(prefix = 'encoder'), ** shared_layers, name = 'encoder'
            )
        self.encoder    = encoder_wrapper(encoder, ** self.hparams.get_config(prefix = 'encoder'))
        
        if decoder is None:
            decoder = self.decoder_class(
                ** self.hparams.get_config(prefix = 'decoder'), ** shared_layers, name = 'decoder'
            )
        self.decoder    = decoder_wrapper(decoder, ** self.hparams.get_config(prefix = 'decoder'))
    
    def _build(self, ** kwargs):
        if hasattr(self.encoder, 'dummy_inputs') and hasattr(self.decoder, 'dummy_inputs'):
            return self(self.dummy_inputs, ** kwargs)

    @property
    def dummy_inputs(self):
        return [self.encoder.dummy_inputs, self.decoder.dummy_inputs]
    
    @tf.function(reduce_retracing = True)
    def encode(self,
               inputs,
               input_length,
               
               mask = None,
               training = False,
               
               return_state    = False,
               return_attention    = False,
               return_hidden_states    = False,
               return_mask     = False,
               as_dict     = True,
               
               ** kwargs
              ):
        return self.encoder(
            inputs,
            input_length    = input_length,
            
            mask    = mask,
            training    = training,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            as_dict     = as_dict,
            
            ** kwargs
        )

    #@timer(name = 'Transformer call')
    def call(self,
             inputs,
             input_length   = None,
             decoder_input  = None,
             decoder_input_length   = None,
             initial_state  = None,
             
             training   = False,
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             enc_padding_mask   = None,
             
             return_state       = False,
             return_attention   = False,
             return_last_attention  = False,
             return_hidden_states   = False,
             return_mask        = False,
             as_dict    = False,
             
             ** kwargs
            ):
        encoder_input = inputs
        if isinstance(inputs, (list, tuple)) and decoder_input is None:
            encoder_input, decoder_input = inputs
        
        time_logger.start_timer('Encoder')
        
        encoder_outputs = self.encoder(
            encoder_input,
            input_length    = input_length,
            
            mask    = enc_padding_mask,
            training    = training,
            
            return_state    = False,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = True,
            as_dict = True,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')}
        )
        
        time_logger.stop_timer('Encoder')
        time_logger.start_timer('Decoder')
        
        decoder_outputs = self.decoder(
            decoder_input,
            input_length    = decoder_input_length,
            initial_state   = initial_state,
            
            encoder_output  = encoder_outputs.output,
            enc_padding_mask    = encoder_outputs.mask,
            
            mask    = mask,
            training    = training,
            padding_mask    = padding_mask,
            look_ahead_mask = look_ahead_mask,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            as_dict     = True,
            
            ** {k : v for k, v in kwargs.items() if not k.startswith('encoder_')}
        )
        
        time_logger.stop_timer('Decoder')

        return format_output(
            decoder_outputs.output,
            state   = decoder_outputs.state,
            attn_weights    = (encoder_outputs.attention_weights, decoder_outputs.attention_weights),
            hidden_states   = (encoder_outputs.hidden_states, decoder_outputs.hidden_states),
            mask    = (encoder_outputs.mask, decoder_outputs.mask),
            
            return_state            = return_state,
            return_attention        = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            as_dict = as_dict
        )
    
    #@timer(name = 'Transformer inference')
    def infer(self,
              inputs,
              input_length  = None,
              initial_state = None,

              enc_padding_mask  = None,
              padding_mask  = None,
              training  = False,
              
              return_state      = False,
              return_attention  = False,
              return_last_attention = False,
              return_hidden_states  = False,
              return_mask   = False,
              as_dict       = True,

              ** kwargs
             ):
        encoder_outputs = self.encode(
            inputs,
            input_length    = input_length,
            
            mask    = enc_padding_mask,
            training    = training,
            
            return_state    = False,
            return_attention    = False,
            return_hidden_states    = False,
            return_mask     = True,
            as_dict     = True,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')}
        )
        
        return self.decoder.infer(
            encoder_output  = encoder_outputs.output,
            enc_padding_mask    = encoder_outputs.mask,
            
            training    = training,
            initial_state   = initial_state,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            
            ** {k : v for k, v in kwargs.items() if not k.startswith('encoder_')}
        )
    
    def get_config(self):
        if type(self) is Transformer:
            return {
                'encoder'   : json.loads(self.encoder.to_json()),
                'decoder'   : json.loads(self.decoder.to_json())
            }
        return self.hparams.get_config()

    @classmethod
    def from_config(cls, config, custom_objects = None):
        if 'encoder' in config and 'decoder' in config:
            config.update({
                'encoder'   : model_from_json(
                    json.dumps(config['encoder']), custom_objects = custom_objects
                ),
                'decoder'   : model_from_json(
                    json.dumps(config['decoder']), custom_objects = custom_objects
                )
            })
        return cls(** config)

custom_functions    = {
    'FeedForwardNetwork'    : FeedForwardNetwork,
    'TransformerEncoder'    : TransformerEncoder,
    'TransformerDecoder'    : TransformerDecoder,
    'Transformer'       : Transformer
}

custom_objects  = {
    'MultiHeadAttention'        : MultiHeadAttention,
    
    'FeedForwardNetwork'    : FeedForwardNetwork,
    'TransformerLayer'      : TransformerLayer,
    'TransformerBlock'      : TransformerBlock,
    'TransformerEncoder'    : TransformerEncoder,
    'TransformerDecoder'    : TransformerDecoder,
    'Transformer'       : Transformer
}
