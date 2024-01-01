# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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

from loggers import timer
from hparams import HParams
from custom_layers import FasterEmbedding
from custom_architectures.generation_utils import infer as infer_method
from custom_architectures.transformers_arch.transformer_arch import *
from custom_architectures.transformers_arch.transformer_arch import _get_state_length

HParamsTransformerTokenEmbedding = HParams(
    vocab_size  = None,
    embedding_dim   = None,
    max_input_length    = None,
    max_token_types     = 0,
    
    scale_embedding     = False,
    normalize_embeddings    = True,
    
    repeat_position     = -1,
    positional_offset   = 0,

    norm_training   = True,
    epsilon     = 1e-6,
    drop_rate   = 0.1
)

_shared_config = [
    'vocab_size', 'sos_token', 'eos_token', 'pad_token', 'max_input_length',
    'scale_embedding', 'normalize_embeddings', 'positional_offset'
]

HParamsTextTransformerBlock = HParamsTransformerBlock(
    ** HParamsTransformerTokenEmbedding,
    sos_token   = -1,
    eos_token   = -1,
    pad_token   = 0
)
HParamsTextTransformerEncoder = HParamsTextTransformerBlock(** HParamsTransformerEncoder)
HParamsTextTransformerDecoder = HParamsTextTransformerBlock(** HParamsTransformerDecoder)

class TransformerTokenEmbedding(tf.keras.layers.Layer):
    _attr_to_set = [
        'embedding_dim', 'vocab_size', 'norm_training', 'positional_offset', 'repeat_position'
    ]
    
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 max_input_length,
                 
                 token_embedding    = None,
                 positional_embedding   = None,
                 
                 name = 'embeddings',
                 ** kwargs
                ):
        super().__init__(name = name)
        
        self.hparams = HParamsTransformerTokenEmbedding.extract(kwargs)
        self.hparams = self.hparams(
            vocab_size      = vocab_size,
            embedding_dim   = embedding_dim,
            max_input_length    = max_input_length
        )
        
        self._max_input_length = self.hparams.max_input_length
        for attr_name in self._attr_to_set:
            setattr(self, attr_name, self.hparams[attr_name])
        
        self.embedding_factor = tf.math.sqrt(
            float(embedding_dim) if self.hparams.scale_embedding else 1.
        )
        
        # Set token embedding layer
        if token_embedding is None:
            token_embedding = FasterEmbedding(
                self.vocab_size, self.embedding_dim, name = 'token_embedding'
            )
        
        self.token_embedding_layer = token_embedding
        
        # Set token type embedding layer (if required)
        self.token_type_embedding_layer = None
        if self.hparams.max_token_types > 1:
            self.token_type_embedding_layer = FasterEmbedding(
                self.hparams.max_token_types, self.embedding_dim, name = "token_type_embedding"
            )
        
        # Set positional embedding layer
        if positional_embedding is None and self.max_input_length > 1:
            positional_embedding    = FasterEmbedding(
                self.max_input_length, self.embedding_dim, name = "pos_embeddings"
            )
        self.pos_embedding_layer    = positional_embedding
        
        # Set normalization layer
        self.norm       = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_embedding'
        ) if self.hparams.normalize_embeddings else None
        self.dropout    = tf.keras.layers.Dropout(
            self.hparams.drop_rate
        ) if self.hparams.drop_rate > 0. else None

    def change_vocabulary(self, vocab, ** kwargs):
        self.vocab_size = len(vocab)
        self.hparams.vocab_size = len(vocab)
        self.token_embedding_layer.change_vocabulary(vocab, ** kwargs)
    
    @property
    def max_input_length(self):
        return self._max_input_length + self.positional_offset

    @property
    def use_token_type(self):
        return self.token_type_embedding_layer is not None

    @timer
    def linear(self, output):
        batch_size, seq_len = tf.shape(output)[0], tf.shape(output)[1]
        
        logits = tf.reshape(output, [-1, self.embedding_dim])
        logits = tf.matmul(
            logits, tf.cast(self.token_embedding_layer.embeddings, logits.dtype), transpose_b = True
        )
        logits = tf.reshape(logits, [batch_size, seq_len, self.vocab_size])
        
        return logits

    @timer
    def embed_tokens(self, text):
        embeddings = self.token_embedding_layer(text)
        return embeddings * tf.cast(self.embedding_factor, embeddings.dtype)
    
    @timer
    def embed_token_types(self, token_types, batch_size, seq_len):
        token_type_embedded = 0.
        if self.token_type_embedding_layer is not None:
            if token_types is None:
                token_types = tf.fill((batch_size, seq_len), value = 0)
            elif len(tf.shape(token_types)) == 0:
                token_types = tf.fill((batch_size, seq_len), value = token_types)
            token_type_embedded = self.token_type_embedding_layer(token_types)
        return token_type_embedded
    
    @timer
    def embed_positions(self,
                        position_ids,
                        seq_len,
                        positional_offset,
                        repeat_position,
                        initial_state   = None,
                        debug = False
                       ):
        if self.pos_embedding_layer is None: return 0
        if initial_state:
            positional_offset += _get_state_length(initial_state)
        
        if position_ids is None:
            if repeat_position > 1:
                position_ids = tf.repeat(
                    tf.range(seq_len // repeat_position + 1), repeat_position
                )[:seq_len]
            else:
                position_ids = tf.range(seq_len)

            position_ids = tf.expand_dims(position_ids, axis = 0)
            if positional_offset > 0:
                position_ids = position_ids + positional_offset
        
        if debug: tf.print("Position ids :", position_ids)
        
        return self.pos_embedding_layer(position_ids)
    
    @timer(name = 'token embedding call')
    def call(self,
             inputs,
             input_length   = None,
             token_types    = None,
             position_ids   = None,
             initial_state  = None,
             
             prefix     = None,
             training   = False,
             
             positional_offset  = -1,
             repeat_position    = -1,
             
             debug      = False,
             ** kwargs
            ):
        if positional_offset == -1: positional_offset = self.positional_offset
        if repeat_position == -1:   repeat_position = self.repeat_position
        
        text = inputs
        if isinstance(inputs, (list, tuple)):
            text, input_length = inputs[:2]
            if len(inputs) > 2: token_types  = inputs[2]
            if len(inputs) > 3: position_ids = inputs[3]
        
        if debug:
            tf.print("Tokens shape :", tf.shape(text))
            tf.print("Positional offset :", positional_offset)
        
        if prefix is None or tf.shape(text)[1] > 0:
            # Embed tokens (text)
            token_embedded = self.embed_tokens(text)

            if prefix is not None and _get_state_length(initial_state) == 0:
                token_embedded = tf.concat([prefix, token_embedded], axis = 1)
        else:
            token_embedded = prefix
        
        # Embed token types (if necessary)
        token_type_embedded = self.embed_token_types(
            token_types, tf.shape(token_embedded)[0], tf.shape(token_embedded)[1]
        )
        
        # Embed positions 
        pos_embedded = self.embed_positions(
            position_ids, tf.shape(token_embedded)[1], positional_offset, repeat_position,
            initial_state = initial_state, debug = debug
        )
        
        if debug:
            tf.print('token embed shape :', tf.shape(token_embedded), 'pos shape :', tf.shape(pos_embedded), 'type shape :', tf.shape(token_type_embedded))
        
        # Combine all embeddings
        embeddings  = token_embedded + pos_embedded + token_type_embedded

        if self.norm is not None:
            embeddings = self.norm(embeddings, training = training and self.norm_training)
        if self.dropout is not None:
            embeddings = self.dropout(embeddings, training = training)

        return embeddings

    def get_output_shape(self, inputs):
        return tuple(inputs) + (self.embedding_dim, )
    
    def get_config(self):
        return (self.hparams + super().get_config()).get_config()

class TextTransformerBlock(TransformerBlock):
    """ Regular `TransformerBlock` with a `TransformerTokenEmbedding` layer applied on inputs """
    default_params  = HParamsTextTransformerBlock
    _attr_to_set    = TransformerBlock._attr_to_set + ['vocab_size', 'positional_offset']
    
    def __init__(self, vocab_size, embedding_dim, max_input_length, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim,
            max_input_length = max_input_length, ** kwargs
        )
        
        with tf.name_scope(self.name):
            self.sos_token   = tf.Variable(
                self.hparams.sos_token, trainable = False, dtype = tf.int32, name = 'sos_token'
            )
            self.eos_token   = tf.Variable(
                self.hparams.eos_token, trainable = False, dtype = tf.int32, name = 'eos_token'
            )
            self.pad_token   = tf.Variable(
                self.hparams.pad_token, trainable = False, dtype = tf.int32, name = 'pad_token'
            )
    
    def _init_input_layers(self,
                           token_embedding = None,
                           positional_embedding = None,
                           ** kwargs
                          ):
        self.embeddings = TransformerTokenEmbedding(
            token_embedding     = token_embedding,
            positional_embedding    = positional_embedding,
            name    = 'embeddings',
            ** self.hparams
        )
    
    def change_vocabulary(self, vocab, ** kwargs):
        self.vocab_size = len(vocab)
        self.hparams.vocab_size = len(vocab)
        self.embeddings.change_vocabulary(vocab, ** kwargs)

    @property
    def max_input_length(self):
        return self.embeddings.max_input_length
    
    @property
    def use_token_type(self):
        return self.embeddings.use_token_type
    
    @property
    def dummy_inputs(self):
        return tf.expand_dims(tf.range(9), axis = 0)
    
    def set_tokens(self, sos_token = None, eos_token = None, pad_token = None):
        if sos_token not in (-1, None): self.sos_token.assign(sos_token)
        if eos_token not in (-1, None): self.eos_token.assign(eos_token)
        if pad_token not in (-1, None): self.pad_token.assign(pad_token)

        self.hparams.update({
            'sos_token' : self.sos_token.numpy(),
            'eos_token' : self.eos_token.numpy(),
            'pad_token' : self.pad_token.numpy()
        })

    def embed_tokens(self, tokens):
        return self.embeddings.embed_tokens(tokens)
    
    def prepare_input(self,
                      inputs,
                      input_length  = None,
                      token_types   = None,
                      position_ids  = None,
                      additional_inputs = [],
                      initial_state = None,

                      mask  = None,
                      training  = False,
             
                      prefix     = None,
                      positional_offset  = -1,
                      repeat_position    = -1,
                      
                      ** kwargs
                     ):
        text = inputs
        if len(additional_inputs) > 0:
            if self.use_token_type:
                token_types     = additional_inputs[0]
                if len(additional_inputs) > 1: position_ids    = additional_inputs[1]
            else:
                position_ids    = additional_inputs[0]
        
        mask = build_padding_mask(
            text, mask = mask, lengths = input_length, pad_value = self.pad_token, dtype = tf.bool
        )
        
        if prefix is not None and _get_state_length(initial_state) == 0:
            prefix_length = tf.shape(prefix)[1]
            # maybe removes 'sos_token'
            if tf.reduce_all(text[:, 0] == self.sos_token):
                text = text[:, 1:]
                if input_length is not None: input_length = input_length - 1
                if mask is not None: mask = mask[..., 1:]
            
            if input_length is not None: input_length += prefix_length
            if mask is not None and tf.shape(mask)[-1] == tf.shape(text)[1]:
                mask = tf.concat([
                    tf.ones((tf.shape(text)[0], 1, 1, prefix_length), dtype = mask.dtype), mask
                ], axis = -1)

        embedded = self.embeddings(
            text,
            input_length    = input_length,
            token_types     = token_types,
            position_ids    = position_ids,
            initial_state   = initial_state,
            
            prefix  = prefix,
            
            repeat_position = repeat_position,
            positional_offset   = positional_offset,

            training    = training,
            mask    = mask,
            ** kwargs
        )
        embedded._keras_mask = mask
        
        return embedded

    def compute_output(self, output, training = False, mask = None, prefix = None, ** kwargs):
        output = super(TextTransformerBlock, self).compute_output(
            output, training = training, mask = mask, ** kwargs
        )
        if prefix is not None and tf.shape(output)[1] >= tf.shape(prefix)[1]:
            output = output[:, tf.shape(prefix)[1] - 1 :]
        
        return output

    def infer(self, * args, ** kwargs):
        kwargs.setdefault('max_length', self.max_input_length)
        return infer_method(
            self,
            * args,
            
            sos_token   = self.sos_token,
            eos_token   = self.eos_token,
            pad_token   = self.pad_token,
            vocab_size  = self.vocab_size,
            
            ** kwargs
        )
    
    def transfer_weights(self, * args, ** kwargs):
        kwargs.setdefault('skip_layers', ('sos_token', 'eos_token', 'pad_token'))
        return super(TextTransformerBlock, self).transfer_weights(* args, ** kwargs)
    
    def get_output_shape(self, inputs, * args, ** kwargs):
        return super().get_output_shape(
            self.embeddings.get_output_shape(inputs), * args, ** kwargs
        )
    

class TextTransformerEncoder(TextTransformerBlock):
    default_params = HParamsTextTransformerEncoder

class TextTransformerDecoder(TextTransformerBlock):
    default_params = HParamsTextTransformerDecoder

class TextTransformer(Transformer):
    encoder_class   = TextTransformerEncoder
    decoder_class   = TextTransformerDecoder
    _shared_keys    = Transformer._shared_keys + _shared_config
    
    def __init__(self, vocab_size, embedding_dim, max_input_length, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim,
            max_input_length = max_input_length, ** kwargs
        )
    
    def set_tokens(self, ** kwargs):
        """
            Call `set_tokens` on both the `encoder` and the `decoder`
            If you want to specify custom tokens for encoder / decoder, simply prefix its name by `encoder_` (or `decoder_`)
            For instance `set_tokens(encoder_pad_token = 0, pad_token = 1)` will set `pad_token` to 0 and 1 respectively for the encoder / decoder
        """
        if hasattr(self.encoder, 'set_tokens'):
            self.encoder.set_tokens(** {
                ** kwargs, ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')}
            })
        if hasattr(self.decoder, 'set_tokens'):
            self.decoder.set_tokens(** {
                ** kwargs, ** {k[8:] : v for k, v in kwargs.items() if k.startswith('decoder_')}
            })
