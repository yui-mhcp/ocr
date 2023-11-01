
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

import collections
import tensorflow as tf

from loggers import timer
from custom_layers import log_softmax

InferenceOutput = collections.namedtuple(
    "InferenceOutput", [
        "tokens", "lengths", "scores", "logits", "attention_weights"
    ]
)

InferenceState   = collections.namedtuple(
    "InferenceState", [
        "t", "finished", "state"
    ]
)

def infer(model, * args, method = 'greedy', ** kwargs):
    return _inference_methods[method](model, * args, ** kwargs)

@timer
@tf.function(reduce_retracing = True)
def _compute_logits(scores, lengths, temperature = 0., length_temperature = 0., logits_filter = None, ** kwargs):
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
        scores = scores / tf.cast(temperature, scores.dtype)
    
    if length_temperature != 0.:
        scores = scores * (tf.cast(lengths + 1, scores.dtype) ** tf.cast(length_temperature, scores.dtype))
    
    if logits_filter is not None:
        scores = tf.cast(logits_filter(scores, ** kwargs), scores.dtype)

    return log_softmax(scores, axis = -1)

@timer
@tf.function(reduce_retracing = True)
def _select_next_token(logits, n, temperature = 0., dtype = tf.int32, ** kwargs):
    """
        Returns top-`k` best scores either greedyly (if `temperature == 0.`) else randomly
        Arguments :
            - logits    : the unnormalized log-probabilities for each word ([batch_size, vocab_size])
            - n         : the number of samples to return
            - temperature   : the softmax' temperature (if `0` takes the argmax (or top-k))
            - dtype     : the result's dtype
        Returns :
            - token : `tf.Tensor` with shape [batch_size, n] and dtype `dtype`
    """
    if temperature == 0.:
        if n == 0: return tf.expand_dims(tf.argmax(logits, axis = -1, output_type = dtype), axis = 1)
        return tf.cast(tf.nn.top_k(logits, k = n).indices, dtype)
    
    sample = tf.random.categorical(logits, n, dtype = dtype)
    return sample

def _nested_map(fn, nest):
    if isinstance(nest, tuple): return fn(nest)
    if isinstance(nest, dict): return {k: _nested_map(fn, v) for k, v in nest.items()}
    return [_nested_map(fn, v) for v in nest]

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

@timer
#@tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
def infer_greedy(self,
                 tokens       = None,
                 input_length = None,
                 encoder_output   = None,
                 initial_state    = None,
                 prefix       = None,

                 sos_token  : tf.Tensor = -1,
                 eos_token  : tf.Tensor = -1,
                 pad_token  : tf.Tensor = -1,
                 vocab_size : tf.Tensor = -1,
           
                 enc_padding_mask = None,
                 padding_mask = None,
                 training     = False,
                 use_cache    = False,

                 top_p        : tf.Tensor = -1,
                 temperature  : tf.Tensor = 0.,
           
                 batch_size   : tf.Tensor = -1,
                 max_length   : tf.Tensor = -1,
                 early_stopping   = True,
                 logits_filter    = None,

                 step_fn          = None,
                 logit_processing = _compute_logits,
                 token_selector   = _select_next_token,
           
                 return_state     = False,
                 return_logits    = True,
                 return_attention = False,
                 return_last_attention    = False,
                 
                 ** kwargs
                ):
    def cond(inputs, outputs, loop_state):
        return not (early_stopping and tf.reduce_all(loop_state.finished))

    def body(inputs, outputs, loop_state):
        model_out = step_fn(
            inputs,
            input_length    = outputs.lengths,
            encoder_output  = encoder_output,
            initial_state   = loop_state.state,
            prefix      = prefix,
            
            padding_mask    = loop_state.finished[:, tf.newaxis],
            enc_padding_mask    = enc_padding_mask,
            
            training    = training,
            apply_softmax   = False,
            
            return_state    = True,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = False,
            return_mask = False,
            as_dict = True,
            
            ** kwargs
        )
        logits = logit_processing(
            model_out.output,
            lengths       = outputs.lengths,
            temperature   = temperature,
            logits_filter = logits_filter,
            tokens        = outputs.tokens,
            t             = loop_state.t
        )

        next_token  = token_selector(
            logits, n = tf.cast(1, tf.int32), top_p = top_p, temperature = temperature, dtype = tf.int32
        )[:, 0]
        next_token  = tf.where(loop_state.finished, pad_token, next_token)

        scores      = outputs.scores + tf.where(
            loop_state.finished, tf.cast(0., logits.dtype), tf.gather(logits, next_token, batch_dims = 1)
        )

        finished    = tf.logical_or(loop_state.finished, tf.math.equal(next_token, eos_token))
        lengths     = outputs.lengths + tf.cast(tf.logical_not(finished), tf.int32)

        t = loop_state.t
        next_outputs = InferenceOutput(
            tokens  = outputs.tokens.write(t, next_token),
            lengths = lengths,
            scores  = scores,
            logits  = outputs.logits.write(t, logits) if return_logits else outputs.logits,
            attention_weights = outputs.attention_weights if skip_attention else tf.nest.map_structure(
                lambda acc, attn: acc.write(t, attn), outputs.attention_weights, model_out.attention_weights
            )
        )
        next_state = InferenceState(
            t = t + 1, finished = finished, state = model_out.state
        )

        return next_token, next_outputs, next_state

    if step_fn is None: step_fn = self
    
    skip_attention = not (return_attention or return_last_attention)
    
    dtype = self.compute_dtype if encoder_output is None else encoder_output.dtype

    if batch_size == -1:
        batch_size = _get_batch_size(tokens, encoder_output, prefix = prefix)
    
    if tokens is None:
        tokens          = tf.fill((batch_size, 1), sos_token)
        input_length    = tf.fill((batch_size, 1), 1)
    elif isinstance(tokens, (list, tuple)):
        tokens, input_length    = tokens
    
    if input_length is None:
        input_length    = tf.fill((batch_size, 1), tf.reduce_sum(tf.cast(tokens != pad_token, tf.int32), axis = -1))

    inputs, outputs, state = body(
        tokens,
        InferenceOutput(
            tokens  = tf.TensorArray(dtype = tf.int32, element_shape = (None, ), size = max_length),
            lengths = input_length,
            scores  = tf.zeros((batch_size, ), dtype = tf.float32),
            logits  = tf.TensorArray(dtype = dtype, element_shape = (None, vocab_size), size = max_length) if return_logits else (),
            attention_weights = {} if skip_attention else _nested_map(
                lambda s: tf.TensorArray(dtype = dtype, size = max_length), self.get_output_shape(
                    tokens.shape, encoder_output = encoder_output.shape, prefix = prefix, return_attention = True
                ).attention_weights
            )
        ),
        InferenceState(
            t        = tf.zeros((), dtype = tf.int32),
            finished = tf.zeros((batch_size, ), dtype = tf.bool),
            state    = self.get_initial_state(batch_size = batch_size, dtype = dtype)
        )
    )
    
    last_output, outputs, state = tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = (inputs, outputs, state),
        maximum_iterations  = max_length - 1
    )

    length = tf.reduce_max(outputs.lengths) - 1
    return InferenceOutput(
        tokens  = tf.transpose(outputs.tokens.stack()[: length], [1, 0]),
        lengths = outputs.lengths,
        scores  = outputs.scores,
        logits  = tf.transpose(outputs.logits.stack()[: length], [1, 0, 2]) if return_logits else None,
        attention_weights = None if skip_attention else tf.nest.map_structure(
            lambda attn: tf.transpose(attn.stack()[: length], [1, 0, 2]), outputs.attention_weights
        )
    )

@timer
def infer_beam_search(self,
                      tokens    = None,
                      input_length  = None,
                      encoder_output    = None,
                      initial_state     = None,
                      prefix       = None,

                      sos_token  : tf.Tensor = -1,
                      eos_token  : tf.Tensor = -1,
                      pad_token  : tf.Tensor = -1,
                      vocab_size : tf.Tensor = -1,
                      
                      padding_mask = None,
                      enc_padding_mask = None,
                       
                      num_beams    : tf.Tensor = 10,
                      num_sentences    : tf.Tensor = 1,

                      temperature  : tf.Tensor = 0.,
                      length_temperature   : tf.Tensor = 0.,
                      length_power : tf.Tensor = 0.,
                      logits_filter    = None,

                      batch_size   : tf.Tensor = -1,
                      max_length   : tf.Tensor = -1,
                      early_stopping    = True,
                      training     = False,
                      use_cache    = False,

                      step_fn          = None,
                      logit_processing = _compute_logits,
                      token_selector   = _select_next_token,
                      
                      return_state       = False,
                      return_logits      = False,
                      return_attention   = False,
                      return_last_attention    = False,
                      return_hidden_states   = False,
                      return_mask        = False,

                      ** kwargs
                     ):
    def cond(inputs, outputs, loop_state):
        if not early_stopping: return True
        return not tf.reduce_all(tf.reshape(loop_state.finished, [batch_size, -1])[:, : num_sentences])

    def body(inputs, outputs, loop_state):
        model_out = step_fn(
            inputs,
            input_length    = outputs.lengths,
            encoder_output  = encoder_output,
            initial_state   = loop_state.state,
            prefix      = prefix,
            
            padding_mask    = loop_state.finished[:, tf.newaxis],
            enc_padding_mask    = enc_padding_mask,
            
            training    = training,
            apply_softmax   = False,
            
            return_state    = True,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = False,
            return_mask = False,
            as_dict = True,
            
            ** kwargs
        )
        # Shape [batch_size * num_beams, vocab_size]
        logits = logit_processing(
            model_out.output,
            lengths       = outputs.lengths,
            temperature   = temperature,
            logits_filter = logits_filter,
            tokens        = outputs.tokens,
            t             = loop_state.t
        )

        logits_with_scores  = tf.where(loop_state.finished[:, tf.newaxis], eos_mask, logits) + outputs.scores[:, tf.newaxis]
        
        reshaped_logits  = logits_with_scores
        if length_power != 0.:
            reshaped_logits  = reshaped_logits / tf.cast(outputs.lengths + 1, logits.dtype) ** length_power
        
        reshaped_logits = tf.reshape(reshaped_logits, [batch_size, -1])

        if loop_state.t == 0: reshaped_logits = reshaped_logits[:, :vocab_size]
        
        next_token = tf.reshape(token_selector(
            reshaped_logits, n = num_beams, temperature = temperature, dtype = tf.int32
        ), [-1])
        next_token = tf.where(loop_state.finished, pad_token, next_token)
        
        beam_index = next_token // vocab_size + batch_idx_add
        next_token = next_token % vocab_size

        tokens      = tf.gather(outputs.tokens,      beam_index)
        lengths     = tf.gather(outputs.lengths,     beam_index)
        finished    = tf.gather(loop_state.finished, beam_index)
        logits_with_scores  = tf.gather(logits_with_scores, beam_index)
        
        scores      = tf.gather(logits_with_scores, next_token, batch_dims = 1)

        tokens      = tf.concat([
            tokens, tf.expand_dims(next_token, axis = 1)
        ], axis = -1)
        finished    = tf.logical_or(finished, tf.math.equal(next_token, eos_token))
        lengths     = lengths + tf.cast(tf.logical_not(finished), lengths.dtype)[:, tf.newaxis]

        next_outputs = InferenceOutput(
            tokens  = tokens,
            lengths = lengths,
            scores  = scores,
            logits  = outputs.logits if not return_logits else tf.concat([
                tf.gather(outputs.logits, beam_index),
                tf.expand_dims(logits, axis = 1)
            ], axis = 1),
            attention_weights = outputs.attention_weights if skip_attention else tf.nest.map_structure(
                lambda acc, attn: tf.concat([
                    tf.gather(acc, beam_index), tf.reshape(attn, tf.concat([tf.shape(acc)[:-2], [1], [tf.shape(acc)[-1]]], axis = 0))
                ], axis = -2), outputs.attention_weights, model_out.attention_weights
            )
        )
        next_state = InferenceState(
            t = loop_state.t + 1, finished = finished, state = tf.nest.map_structure(
                lambda s: tf.gather(s, beam_index), model_out.state
            )
        )

        return tf.expand_dims(next_token, axis = 1), next_outputs, next_state

    if step_fn is None: step_fn = self

    skip_attention  = not return_attention and not return_last_attention
    
    if encoder_output is not None:
        dtype = encoder_output.dtype
    elif prefix is not None:
        dtype = prefix.dtype
    else:
        dtype = tf.float32
    
    if batch_size == -1:
        batch_size = _get_batch_size(tokens, encoder_output, prefix = prefix)

    effective_batch_size    = batch_size * num_beams

    batch_idx_add   = tf.repeat(tf.range(batch_size), num_beams, axis = 0) * num_beams
    eos_mask        = tf.tensor_scatter_nd_update(
        tf.fill((1, self.vocab_size), dtype.min), [[0, eos_token]], [0.]
    )
    
    if tokens is None:
        tokens          = tf.fill((effective_batch_size, 1), sos_token)
        input_length    = tf.fill((effective_batch_size, 1), 1)
    elif isinstance(tokens, (list, tuple)):
        tokens, input_length    = tokens
    
    if input_length is None:
        input_length    = tf.fill((effective_batch_size, 1), tf.reduce_sum(tf.cast(tokens != pad_token, tf.int32), axis = -1))

    if encoder_output is not None:
        encoder_output = tf.repeat(encoder_output, num_beams, axis = 0)

    if prefix is not None:
        prefix = tf.repeat(prefix, num_beams, axis = 0)

    inputs, outputs, state = body(
        tokens,
        InferenceOutput(
            tokens  = tf.zeros([effective_batch_size, 0], dtype = tf.int32),
            lengths = input_length,
            scores  = tf.zeros((effective_batch_size, ), dtype = tf.float32),
            logits  = tf.zeros([effective_batch_size, 0, vocab_size], dtype = dtype) if return_logits else (),
            attention_weights = {} if skip_attention else _nested_map(
                lambda s: tf.zeros(_fix_shape(s if len(s) > 2 else (None, None, s[-1]), effective_batch_size), dtype = dtype), self.get_output_shape(
                    tokens.shape, encoder_output = encoder_output.shape, prefix = prefix, return_attention = True
                ).attention_weights
            )
        ),
        InferenceState(
            t        = tf.zeros((), dtype = tf.int32),
            finished = tf.zeros((effective_batch_size, ), dtype = tf.bool),
            state    = self.get_initial_state(batch_size = effective_batch_size, dtype = dtype)
        )
    )
    
    last_output, outputs, state = tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = (inputs, outputs, state),
        shape_invariants = (
            tf.TensorShape((None, None)),
            InferenceOutput(
                tokens   = tf.TensorShape((None, None)),
                lengths  = tf.TensorShape((None, 1)),
                scores   = tf.TensorShape((None, )),
                logits   = () if not return_logits else tf.TensorShape((None, None, vocab_size)),
                attention_weights = tf.nest.map_structure(
                    lambda attn: tf.TensorShape(attn.shape[:-2] + [None, attn.shape[-1]]), outputs.attention_weights
                )
            ),
            InferenceState(
                t        = tf.TensorShape(()),
                finished = tf.TensorShape((None, )),
                state    = tf.nest.map_structure(lambda s: tf.TensorShape(s.shape), state.state)
            )
        ),
        maximum_iterations  = max_length - 1
    )

    return tf.nest.map_structure(
        lambda o: tf.reshape(o, tf.concat([[batch_size, num_beams], tf.shape(o)[1:]], axis = 0))[:, : num_sentences], outputs
    )


_inference_methods  = {
    'greedy'    : infer_greedy,
    'sample'    : lambda * args, ** kwargs: infer_greedy(* args, use_sampling = True, ** kwargs),
    'beam'      : infer_beam_search
}