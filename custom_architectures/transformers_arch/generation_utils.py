
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

import logging
import collections
import tensorflow as tf

from loggers import timer
from utils import get_object
from custom_layers import log_softmax
from custom_architectures.transformers_arch.transformer_arch import format_output, build_padding_mask

TransformerInferenceOutput = collections.namedtuple(
    "TransformerInferenceOutput", [
        "tokens",
        "lengths",
        "output",
        "score",
        "attention_weights"
    ]
)

TransformerInferenceState   = collections.namedtuple(
    "TransformerInferenceState", [
        "t",
        "tokens",
        "lengths",
        "scores",
        "last_tokens",
        
        "padding_mask",
        "finished",
        
        "logits",
        "state",
        "attention_weights"
    ]
)

def get_shape_invariant(model,
                        encoder_output  = None,
                        
                        use_cache   = False,
                        return_attention    = False,
                        return_last_attention   = False,
                        return_only_cross_attention = False,
                        dtype   = tf.float32,
                        ** _
                       ):
    def _nested_map(shape):
        if isinstance(shape[0], tuple): return tuple(_nested_map(s) for s in shape)
        return tf.TensorSpec(shape = shape, dtype = dtype)
    
    out_shapes  = model.get_output_shape(
        (None, None),
        encoder_output      = (None, None, None) if encoder_output is not None else None,
        return_state        = use_cache,
        return_attention    = return_attention,
        return_last_attention   = return_last_attention,
        return_only_cross_attention = return_only_cross_attention,
        as_dict = True
    )
    if return_attention or return_last_attention:
        attn_shapes = {
            k : tf.TensorSpec(shape = v, dtype = dtype)
            for k, v in out_shapes.attention_weights.items()
        }
    
    if use_cache:
        state_shapes    = {
            k : _nested_map(v) for k, v in out_shapes.state.items()
        } if isinstance(out_shapes.state, dict) else [
            tf.TensorSpec(shape = s, dtype = dtype) for s in out_shapes.state
        ]
    
    return TransformerInferenceState(
        t               = tf.TensorSpec(shape = (),                 dtype = tf.int32),
        tokens          = tf.TensorSpec(shape = (None, None),       dtype = tf.int32),
        lengths         = tf.TensorSpec(shape = (None, 1),          dtype = tf.int32),
        scores          = tf.TensorSpec(shape = (None, ),           dtype = dtype),
        last_tokens     = tf.TensorSpec(shape = (None, 1),          dtype = tf.int32),
        
        padding_mask    = tf.TensorSpec(shape = (None, 1, 1, None), dtype = tf.bool),
        finished        = tf.TensorSpec(shape = (None, ),           dtype = tf.bool),
        
        logits          = tf.TensorSpec(shape = out_shapes.output,  dtype = dtype),
        state           = state_shapes if use_cache else {},
        attention_weights   = attn_shapes if return_attention or return_last_attention else {}
    )

@timer(name = 'decoder inference')
def infer(model, * args, method = 'greedy', ** kwargs):
    return get_object(
        _inference_methods, method, model, * args, ** kwargs
    )

@tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
def _infer(self,
           tokens       = None,
           input_length = None,
           encoder_output   = None,
           initial_state    = None,
           prefix       = None,
           
           enc_padding_mask = None,
           padding_mask = None,
           training     = False,
           use_cache    = False,
           
           temperature  : tf.Tensor = 0.,
           
           batch_size   : tf.Tensor = -1,
           max_length   : tf.Tensor = -1,
           early_stopping   = True,
           logits_filter    = None,
           
           return_state     = False,
           return_attention = False,
           return_last_attention    = False,
           return_hidden_states = False,
           return_mask      = False,

           ** kwargs
          ):
    def cond(t, tokens, lengths, scores, last_tokens, padding_mask, finished, logits, state, attn):
        return not (early_stopping and tf.reduce_all(finished))
    
    def body(t, tokens, lengths, scores, last_tokens, padding_mask, finished, logits, state, attn):
        #tf.print('tokens at t =', t, ':', tokens)
        outputs = self(
            tokens if not use_cache or t == 0 else last_tokens,
            input_length    = lengths,
            encoder_output  = encoder_output,
            initial_state   = state,
            prefix      = prefix,
            
            padding_mask    = padding_mask,
            enc_padding_mask    = enc_padding_mask,
            
            training    = training,
            apply_softmax   = False,
            
            return_state    = use_cache,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            
            ** kwargs
        )
        logits      = _compute_logits(
            outputs.output[:, -1, :], lengths, temperature = temperature,
            logits_filter = logits_filter, tokens = tokens, t = t
        )

        next_token  = _select_next_token(
            logits, n = 1, temperature = temperature, dtype = tokens.dtype
        )

        scores      = scores + tf.where(
            finished, tf.cast(0., logits.dtype), tf.gather(logits, next_token, batch_dims = 1)
        )

        tokens      = tf.concat([
            tokens, tf.expand_dims(next_token, axis = 1)
        ], axis = -1)

        finished    = tf.logical_or(finished, tf.math.equal(next_token, self.eos_token))

        lengths     = lengths + tf.expand_dims(
            tf.cast(tf.logical_not(finished), lengths.dtype), axis = 1
        )
        if use_cache:
            padding_mask = tf.reshape(tf.logical_not(finished), [-1, 1, 1, 1])
        else:
            padding_mask = tf.concat([
                padding_mask, tf.reshape(tf.logical_not(finished), [-1, 1, 1, 1])
            ], axis = -1)
        
        return TransformerInferenceState(
            t   = t + 1,
            tokens  = tokens,
            lengths = lengths,
            last_tokens     = tf.expand_dims(next_token, axis = 1),
            scores      = scores,
            
            padding_mask    = padding_mask,
            finished    = finished,
            
            logits      = outputs.output,
            state       = outputs.state if use_cache else state,
            attention_weights   = outputs.attention_weights if not skip_attention else attn
        )
    
    skip_attention = not (return_attention or return_last_attention)
    
    dtype = self.compute_dtype if encoder_output is None else encoder_output.dtype

    if batch_size == -1:
        batch_size = _get_batch_size(tokens, encoder_output, prefix = prefix)
    
    if tokens is None:
        tokens          = tf.fill((batch_size, 1), self.sos_token)
        input_length    = tf.fill((batch_size, 1), 1)
    elif isinstance(tokens, (list, tuple)):
        tokens, input_length    = tokens
    
    if input_length is None:
        input_length    = tf.fill((batch_size, 1), tf.shape(tokens)[1])
    
    if padding_mask is None:
        padding_mask    = build_padding_mask(
            tokens, lengths = input_length, pad_value = self.pad_token, dtype = tf.bool
        )

    if prefix is not None and tf.shape(padding_mask)[-1] == tf.shape(tokens)[1]:
        padding_mask = tf.concat([
            tf.ones((batch_size, 1, 1, tf.shape(prefix)[1]), dtype = padding_mask.dtype), padding_mask
        ], axis = -1)
    
    shapes_invariant    = get_shape_invariant(
        self,
        encoder_output  = encoder_output,
        return_attention    = return_attention,
        return_last_attention   = return_last_attention,
        use_cache   = use_cache,
        dtype   = dtype,
        ** kwargs
    )
    outputs = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = TransformerInferenceState(
            t   = tf.zeros((), dtype = tf.int32),
            tokens  = tokens,
            lengths = input_length,
            last_tokens     = tokens[:, -1:],
            scores      = tf.zeros((batch_size, ), dtype = dtype),
            
            padding_mask    = padding_mask,
            finished    = tf.zeros((batch_size,), dtype = tf.bool),
            
            logits      = tf.zeros(
                (batch_size, 1, shapes_invariant.logits.shape[-1]), dtype = dtype
            ),
            state       = self.initialize_cache(
                tf.zeros((batch_size, 0, self.embedding_dim), dtype = dtype), encoder_output
            ) if use_cache else {},
            attention_weights   = tf.nest.map_structure(
                lambda sign: tf.zeros(shape = _fix_shape(sign.shape, batch_size), dtype = sign.dtype),
                shapes_invariant.attention_weights
            ) if not skip_attention else {}
        ),
        shape_invariants    = shapes_invariant,
        maximum_iterations  = max_length
    ))
    
    return TransformerInferenceOutput(
        tokens  = outputs.tokens[:, tf.shape(tokens)[1] :],
        lengths = tf.squeeze(outputs.lengths - input_length, axis = 1),
        score   = outputs.scores,
        output  = outputs.logits,
        attention_weights   = outputs.attention_weights if not skip_attention else None
    )

@tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
def _infer_beam_search(self,
                       tokens    = None,
                       input_length  = None,
                       encoder_output    = None,
                       initial_state     = None,
                       prefix       = None,

                       padding_mask = None,
                       enc_padding_mask = None,
                       
                       num_beams    : tf.Tensor = 10,
                       num_sentences    : tf.Tensor = 5,

                       temperature  : tf.Tensor = 0.,
                       length_temperature   : tf.Tensor = 0.,
                       length_power : tf.Tensor = 0.,
                       logits_filter    = None,

                       batch_size   : tf.Tensor = -1,
                       max_length   : tf.Tensor = -1,
                       early_stopping    = True,
                       training     = False,
                       use_cache    = False,

                       return_state       = False,
                       return_attention   = False,
                       return_last_attention    = False,
                       return_hidden_states   = False,
                       return_mask        = False,

                       ** kwargs
                      ):
    def cond(t, tokens, lengths, scores, last_tokens, padding_mask, finished, logits, state, attn):
        if not early_stopping: return True
        return not tf.reduce_all(tf.reshape(finished, [batch_size, num_beams])[:, : num_sentences])
    
    def body(t, tokens, lengths, scores, last_tokens, padding_mask, finished, logits, state, attn):
        #tf.print('tokens at t =', t, ':', tokens)
        outputs = self(
            tokens if not use_cache or t == 0 else last_tokens,
            input_length    = lengths,
            encoder_output  = encoder_output,
            initial_state   = state,
            prefix  = prefix,
            
            training    = training,
            padding_mask    = padding_mask,
            enc_padding_mask    = enc_padding_mask,

            apply_softmax   = False,
            
            return_state    = use_cache,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            
            ** kwargs
        )

        logits      = _compute_logits(
            outputs.output[:, -1, :], lengths, temperature = temperature,
            length_temperature = length_temperature,
            logits_filter = logits_filter, tokens = tokens, t = t
        )
        
        logits_with_scores  = tf.where(
            tf.expand_dims(finished, axis = 1), eos_mask, logits
        ) + tf.expand_dims(scores, axis = 1)
        
        reshaped_logits  = logits_with_scores
        if length_power != 0.:
            reshaped_logits  = reshaped_logits / tf.cast(lengths, logits.dtype) ** length_power
        
        reshaped_logits = tf.reshape(reshaped_logits, [batch_size, -1])
        
        if t == 0: reshaped_logits = reshaped_logits[:, : tf.shape(logits)[1]]
        
        next_token  = tf.reshape(_select_next_token(
            reshaped_logits, n = num_beams, temperature = temperature, dtype = tokens.dtype
        ), [effective_batch_size])
        
        token_batch_idx = next_token // tf.shape(logits)[1] + batch_idx_add
        next_token      = next_token % tf.shape(logits)[1]

        tokens      = tf.gather(tokens,     token_batch_idx)
        lengths     = tf.gather(lengths,    token_batch_idx)
        finished    = tf.gather(finished,   token_batch_idx)
        logits_with_scores  = tf.gather(logits_with_scores, token_batch_idx)
        
        scores      = tf.gather(logits_with_scores, next_token, batch_dims = 1)

        tokens      = tf.concat([
            tokens, tf.expand_dims(next_token, axis = 1)
        ], axis = -1)

        finished    = tf.logical_or(finished, tf.math.equal(next_token, self.eos_token))

        lengths     = lengths + tf.expand_dims(
            tf.cast(tf.logical_not(finished), lengths.dtype), axis = 1
        )
        if use_cache:
            padding_mask = tf.reshape(tf.logical_not(finished), [-1, 1, 1, 1])
        else:
            padding_mask = tf.concat([
                tf.gather(padding_mask, token_batch_idx),
                tf.reshape(tf.logical_not(finished), [-1, 1, 1, 1])
            ], axis = -1)

        attn_weights = attn
        if not skip_attention:
            attn_weights = tf.nest.map_structure(
                lambda attn: tf.gather(attn, token_batch_idx), outputs.attention_weights
            )
        
        state = state
        if use_cache:
            state = tf.nest.map_structure(
                lambda s: tf.gather(s, token_batch_idx), outputs.state
            )
        
        return TransformerInferenceState(
            t   = t + 1,
            tokens  = tokens,
            lengths = lengths,
            last_tokens     = tf.expand_dims(next_token, axis = 1),
            scores      = scores,
            
            padding_mask    = padding_mask,
            finished    = finished,
            
            logits      = outputs.output,
            state       = state,
            attention_weights   = attn_weights
        )

    if max_length == -1: max_length = self.max_input_length
    
    skip_attention  = not return_attention and not return_last_attention
    
    if encoder_output is not None:
        dtype = encoder_output.dtype
    elif prefix is not None:
        dtype = prefix.dtype
    else:
        dtype = tf.float32
    
    if batch_size == -1:
        batch_size = _get_batch_size(tokens, encoder_output, prefix = prefix)
    
    if encoder_output is not None:
        encoder_output  = tf.repeat(encoder_output, num_beams, axis = 0)
        if enc_padding_mask is not None:
            enc_padding_mask  = tf.repeat(enc_padding_mask, num_beams, axis = 0)

    if prefix is not None:
        prefix = tf.repeat(prefix, num_beams, axis = 0)
    
    
    effective_batch_size    = batch_size * num_beams

    if tokens is None:
        tokens          = tf.fill((effective_batch_size, 1), self.sos_token)
        input_length    = tf.fill((effective_batch_size, 1), 1)
    elif isinstance(tokens, (list, tuple)):
        tokens, input_length    = tokens
    else:
        tokens          = tf.repeat(tokens, num_beams, axis = 0)
        if input_length is None:
            input_length    = tf.fill((effective_batch_size, 1), tf.shape(tokens)[1])
        else:
            input_length    = tf.repeat(input_length, num_beams, axis = 0)
    
    if padding_mask is None:
        padding_mask    = build_padding_mask(
            tokens, lengths = input_length, pad_value = self.pad_token, dtype = tf.bool
        )
    else:
        padding_mask    = tf.repeat(padding_mask, num_beams, axis = 0)
    
    if prefix is not None and tf.shape(padding_mask)[-1] == tf.shape(tokens)[1]:
        padding_mask = tf.concat([
            tf.ones((effective_batch_size, 1, 1, tf.shape(prefix)[1]), dtype = padding_mask.dtype),
            padding_mask
        ], axis = -1)
    
    n_init          = tf.shape(tokens)[1]
    batch_idx_add   = tf.repeat(tf.range(batch_size), num_beams, axis = 0) * num_beams
    eos_mask        = tf.tensor_scatter_nd_update(
        tf.fill((1, self.vocab_size), dtype.min), [[0, self.eos_token]], [0.]
    )

    shapes_invariant    = get_shape_invariant(
        self,
        encoder_output  = encoder_output,
        return_attention    = return_attention,
        return_last_attention   = return_last_attention,
        use_cache   = use_cache,
        dtype   = dtype
    )
    outputs = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = TransformerInferenceState(
            t   = tf.zeros((), dtype = tf.int32),
            tokens  = tokens,
            lengths = input_length,
            last_tokens     = tokens[:, -1:],
            scores      = tf.zeros((effective_batch_size, ), dtype = dtype),
            
            padding_mask    = padding_mask,
            finished    = tf.zeros((effective_batch_size,), dtype = tf.bool),
            
            logits      = tf.zeros(
                (effective_batch_size, 1, shapes_invariant.logits.shape[-1]), dtype = dtype
            ),
            state       = tf.nest.map_structure(
                lambda sign: tf.zeros(_fix_shape(sign.shape, effective_batch_size), dtype = sign.dtype),
                shapes_invariant.state
            ) if use_cache else {},
            attention_weights   = tf.nest.map_structure(
                lambda sign: tf.zeros(_fix_shape(sign.shape, effective_batch_size), dtype = sign.dtype),
                shapes_invariant.attention_weights
            ) if not skip_attention else {}
        ),
        shape_invariants    = shapes_invariant,
        maximum_iterations  = max_length
    ))
    
    scores  = outputs.scores
    if length_power != 0:
        lengths = tf.cast(tf.squeeze(outputs.input_length, axis = 1), scores.dtype)
        scores  = scores / (lengths ** length_power)
    
    attn_weights = outputs.attention_weights
    if not skip_attention:
        attn_weights = {
            k : tf.reshape(attn, [
                batch_size, num_beams, tf.shape(attn)[1], tf.shape(attn)[2], tf.shape(attn)[3]
            ])[:, : num_sentences]
            for k, attn in outputs.attention_weights.items()
        }
    
    return TransformerInferenceOutput(
        tokens  = tf.reshape(outputs.tokens,    [batch_size, num_beams, -1])[:, :num_sentences, n_init:],
        lengths = tf.reshape(outputs.lengths,   [batch_size, num_beams])[:, :num_sentences] - n_init,
        score   = tf.reshape(scores,            [batch_size, num_beams])[:, :num_sentences],
        output  = tf.reshape(outputs.logits,    [batch_size, num_beams, tf.shape(outputs.logits)[1], -1])[:, :num_sentences],
        attention_weights   = attn_weights
    )

@timer
def _compute_logits(scores,
                    lengths,
                    temperature  = 0.,
                    length_temperature   = 0.,
                    logits_filter    = None,
                    ** kwargs
                   ):
    """
        Computes logits (i.e. log-probabilities) based on models' output (scores)
        
        Arguments :
            - scores    : the models' last output with shape [batch_size, vocab_size]
            - lengths   : the tokens' lengths with shape [batch_size, 1]
            - temperature   : the softmax' temperature
                - a temperature < 1 will emphasize the scores' differences
                - a temperature > 1 will reduce the scores' difference
                - a temperature of 0 is equivalent to `argmax` (ignored here)
            - length_temperature    : a custom temperature based on the lengths
                - a temperature > 0 will encourage longer sentences
                - a temperature < 0 will encourage shorter sentences
                - a temperature of 0 has no effect (ignored)
            - logits_filter : a callable that takes `scores` (1st argument) and `kwargs` and returns the filtered `scores`
    """
    if temperature != 0.:
        scores = scores / temperature
    
    if length_temperature != 0.:
        lengths = tf.cast(lengths + 1, scores.dtype)
        
        scores = scores * (tf.cast(lengths + 1, scores.dtype) ** length_temperature)
    
    if logits_filter is not None:
        #tf.print('Scores (before filter) :', scores)
        scores = tf.cast(logits_filter(tf.cast(scores, tf.float32), ** kwargs), scores.dtype)
        #tf.print('Scores (after filter)  :', scores)

    return log_softmax(scores, axis = -1)

@timer
def _select_next_token(logits, n = 1, temperature = 0., dtype = tf.int32):
    """
        Returns top-`k` best scores either greedyly (if `temperature == 0.`) else randomly
        Arguments :
            - logits    : the unnormalized log-probabilities for each word ([batch_size, vocab_size])
            - n         : the number of samples to return
            - temperature   : the softmax' temperature (if `0` takes the argmax (or top-k))
            - dtype     : the result's dtype
        Returns :
            - if `n == 1`   : the selected token's index with shape [batch_size]
            - if `n > 1`    : the selected tokens' indexes with shape [batch_size, n]
    """
    if temperature == 0.:
        if n == 1: return tf.argmax(logits, axis = -1, output_type = dtype)
        return tf.cast(tf.nn.top_k(logits, k = n).indices, dtype)
    
    sample = tf.random.categorical(logits, n, dtype = dtype)
    return sample if n > 1 else sample[:, 0]

def _fix_shape(shape, batch_size, default = 0):
    return (batch_size, ) + tuple(s if s is not None else default for s in shape[1:])

def _get_batch_size(tokens, encoder_output, prefix, default = 1):
    if tokens is not None:
        return tf.shape(tokens)[0]
    elif encoder_output is not None:
        return tf.shape(encoder_output)[0]
    elif prefix is not None:
        return tf.shape(prefix)[0]
    else:
        return default

_inference_methods  = {
    'greedy'    : _infer,
    'sample'    : lambda * args, ** kwargs: _infer(* args, use_sampling = True, ** kwargs),
    'beam'      : _infer_beam_search
}