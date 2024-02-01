
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

from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice, dynamic_slice

from utils import show_memory
from loggers import timer, time_logger
from custom_layers import log_softmax

InferenceConfig = collections.namedtuple(
    "InferenceConfig", [
        "use_xla",
        "use_cache",
        "max_length",
        "prefix_length",
        "encoder_seq_length",
        
        "is_transformer",
        "is_encoder_decoder",
        
        "skip_attention",
        "return_logits"
    ]
)

InferenceState  = collections.namedtuple(
    "InferenceState", [
        "t", "step", "finished", "state", "padding_mask"
    ]
)

InferenceOutput = collections.namedtuple(
    "InferenceOutput", [
        "tokens", "lengths", "scores", "logits", "attention_weights"
    ]
)

def infer(model, * args, method = 'greedy', ** kwargs):
    return _inference_methods[method](model, * args, ** kwargs)

@timer
def infer_greedy(self,
                 tokens     = None,
                 input_length   = None,
                 encoder_output = None,
                 initial_state  = None,
                 prefix     = None,

                 sos_token  : tf.Tensor = -1,
                 eos_token  : tf.Tensor = -1,
                 pad_token  : tf.Tensor = -1,
                 vocab_size : tf.Tensor = -1,
                 add_sos_token  = None,
           
                 step_fn    = None,
                 training   = False,
                 enc_padding_mask   = None,
           
                 batch_size : tf.Tensor = -1,
                 max_length : tf.Tensor = -1,
                 early_stopping = True,
                 is_transformer = False,
           
                 use_cache  = True,
                 return_mask    = False,
                 return_state   = False,
                 return_logits  = False,
                 return_attention   = False,
                 return_last_attention  = False,
                 return_only_cross_attention    = True,
                 return_hidden_states   = False,
                 
                 ** kwargs
                ):
    @timer
    def cond(inputs, outputs, loop_state):
        return not (early_stopping and tf.reduce_all(loop_state.finished))

    @timer
    def body(inputs, outputs, loop_state):
        mask = loop_state.padding_mask
        if not loop_state.state or not use_xla:
            inputs  = inputs[:, : loop_state.t - prefix_length]
            mask    = mask[:, : loop_state.t]

        model_out = step_fn(
            inputs if prefix is None or loop_state.state is not None or not empty_token else None,
            input_length    = outputs.lengths,
            encoder_output  = encoder_output,
            initial_state   = loop_state.state if use_cache else None,
            prefix      = prefix if not use_cache or loop_state.state is None else None,
            
            training    = training,
            apply_softmax   = False,
            padding_mask    = mask,
            enc_padding_mask    = enc_padding_mask,
            
            return_state    = use_cache,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_only_cross_attention = return_only_cross_attention,
            return_hidden_states    = False,
            return_mask = True,
            as_dict = True,
            
            ** kwargs
        )

        logits = model_out.output
        if len(tf.shape(logits)) == 3: logits = logits[:, -1, :]
        
        logits      = process_logits(
            logits, lengths = outputs.lengths, tokens = outputs.tokens, state = loop_state, ** kwargs
        )
        next_token, next_token_score = select_next_token(logits, n = 1, ** kwargs)
        
        next_token  = tf.where(loop_state.finished, pad_token, next_token[:, 0])
        scores      = outputs.scores + tf.where(loop_state.finished, 0., next_token_score[:, 0])

        
        finished    = tf.logical_or(loop_state.finished, tf.math.equal(next_token, eos_token))
        lengths     = outputs.lengths + tf.cast(tf.logical_not(finished)[:, tf.newaxis], tf.int32)
        
        tokens      = tf.tensor_scatter_nd_update(
            outputs.tokens,
            tf.stack([
                tf.range(batch_size), tf.broadcast_to(loop_state.t - prefix_length, [batch_size])
            ], axis = -1),
            next_token
        )
        
        next_inputs  = next_token[:, tf.newaxis] if use_cache else tokens
        next_outputs = InferenceOutput(
            tokens  = tokens,
            lengths = lengths,
            scores  = scores,
            logits  = update_logits(
                outputs.logits, logits, state = loop_state, config = config
            ),
            attention_weights = update_attention_weights(
                outputs.attention_weights, model_out.attention_weights, loop_state, config
            )
        )
        next_state = update_state(
            state = loop_state, output = model_out, finished = finished, config = config
        )
        return next_inputs, next_outputs, next_state

    if step_fn is None:         step_fn = self
    if add_sos_token is None:   add_sos_token = prefix is None
    empty_token = tokens is None and not add_sos_token
    
    use_xla         = not tf.executing_eagerly()
    skip_attention  = not (return_attention or return_last_attention)

    with time_logger.timer('initialization'):
        if encoder_output is not None:  dtype = encoder_output.dtype
        elif prefix is not None:        dtype = prefix.dtype
        else:                           dtype = self.compute_dtype

        if batch_size == -1:
            batch_size = _get_batch_size(tokens, encoder_output, prefix = prefix)

        if tokens is None:
            start_idx   = tf.constant(1 if add_sos_token else 0, dtype = tf.int32)
            tokens      = tf.concat([
                tf.fill((batch_size, start_idx), sos_token),
                tf.fill((batch_size, max_length - start_idx), pad_token)
            ], axis = 1)
            input_length    = tf.fill((batch_size, 1), start_idx)
        else:
            if isinstance(tokens, (list, tuple)):
                tokens, input_length = tokens
            elif input_length is None:
                input_length    = 1 + tf.reduce_sum(
                    tf.cast(tokens[:, 1:] != pad_token, tf.int32), axis = -1, keepdims = True
                )
            
            start_idx = tf.cast(tf.shape(tokens)[1], tf.int32)
            if start_idx <= max_length:
                tokens = tf.pad(
                    tokens, [(0, 0), (0, max_length - start_idx)], constant_values = pad_token
                )
        
        prefix_length   = tf.shape(prefix)[1] if prefix is not None else 0
        cur_len     = start_idx + prefix_length
        max_length  = max_length + prefix_length
        
        n = 1 if add_sos_token else 0
        padding_mask    = tf.concat([
            tf.ones((batch_size, prefix_length + n), dtype = tf.bool), tokens[:, n:] != pad_token
        ], axis = 1)
        if not use_xla: padding_mask = padding_mask[:, : cur_len]
    
    config  = InferenceConfig(
        use_xla = use_xla,
        use_cache   = use_cache,
        max_length  = max_length,
        prefix_length   = prefix_length,
        encoder_seq_length  = tf.shape(encoder_output)[1] if encoder_output is not None else 0,
        
        is_transformer  = is_transformer,
        is_encoder_decoder  = encoder_output is not None,
        
        skip_attention  = skip_attention,
        return_logits   = return_logits
    )

    with time_logger.timer('first step'):
        inputs, outputs, state = body(
            tokens,
            InferenceOutput(
                tokens  = tokens,
                lengths = input_length,
                scores  = tf.zeros((batch_size, ), dtype = dtype),
                logits  = None,
                attention_weights = None
            ),
            InferenceState(
                t       = cur_len,
                step    = tf.constant(0, tf.int32),
                state   = None,
                finished    = tf.zeros((batch_size, ), dtype = tf.bool),
                padding_mask    = padding_mask
            )
        )

    last_output, outputs, state = tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = (inputs, outputs, state),
        maximum_iterations  = max_length - state.t
    )

    return InferenceOutput(
        tokens  = outputs.tokens[:, start_idx :],
        lengths = outputs.lengths - start_idx,
        scores  = outputs.scores,
        logits  = outputs.logits if return_logits else None,
        attention_weights = None if skip_attention else outputs.attention_weights
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
                      add_sos_token = None,
                      
                      step_fn   = None,
                      training  = False,
                      padding_mask  = None,
                      enc_padding_mask = None,

                      num_beams    : tf.Tensor = 10,
                      num_sentences    : tf.Tensor = 1,

                      length_power : tf.Tensor = 0.,

                      batch_size   : tf.Tensor = -1,
                      max_length   : tf.Tensor = -1,
                      early_stopping    = True,
                      is_transformer    = False,
                      
                      use_cache    = False,
                      return_state   = False,
                      return_logits  = False,
                      return_attention   = False,
                      return_last_attention  = False,
                      return_only_cross_attention   = True,
                      return_hidden_states   = False,
                      return_mask        = False,

                      ** kwargs
                     ):
    @timer
    def cond(inputs, outputs, loop_state):
        return not (early_stopping and tf.reduce_all(tf.reshape(
            loop_state.finished, [batch_size, num_beams]
        )[:, : num_sentences]))

    @timer
    def body(inputs, outputs, loop_state):
        mask = loop_state.padding_mask
        if not loop_state.state or not use_xla:
            inputs  = inputs[:, : loop_state.t - prefix_length]
            mask    = mask[:, : loop_state.t]

        model_out = step_fn(
            inputs if prefix is None or loop_state.state is not None or not empty_token else None,
            input_length    = outputs.lengths,
            encoder_output  = encoder_output,
            initial_state   = loop_state.state if use_cache else None,
            prefix      = prefix if not use_cache or loop_state.state is None else None,
            
            training    = training,
            apply_softmax   = False,
            padding_mask    = mask,
            enc_padding_mask    = enc_padding_mask,

            return_state    = use_cache,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_only_cross_attention = return_only_cross_attention,
            return_hidden_states    = False,
            return_mask = False,
            as_dict = True,
            
            ** kwargs
        )
        logits = model_out.output
        if len(tf.shape(logits)) == 3: logits = logits[:, -1, :]
        
        logits      = process_logits(
            logits, lengths = outputs.lengths, tokens = outputs.tokens, state = loop_state, ** kwargs
        )
        # for finished sentences, only keep the EOS token with a score of 0
        # such that it does not decrease the current score of the sentence
        logits_with_scores  = outputs.scores[:, tf.newaxis] + tf.where(
            loop_state.finished[:, tf.newaxis], eos_mask, logits
        )
        # reshape logits to [batch_size, vocab_size * num_beams]
        reshaped_logits  = logits_with_scores
        if length_power != 0.:
            reshaped_logits  = reshaped_logits / tf.cast(
                outputs.lengths + 1, logits.dtype
            ) ** length_power

        reshaped_logits = tf.reshape(reshaped_logits, [batch_size, num_beams * vocab_size])
        if loop_state.state is None: reshaped_logits = reshaped_logits[:, : vocab_size]
        # the returned token scores are not used as they take into account the length normalization
        next_token  = select_next_token(reshaped_logits, n = num_beams, ** kwargs)[0]
        next_token  = tf.reshape(next_token, [effective_batch_size])
        
        beam_index  = next_token // vocab_size + batch_idx_add
        next_token  = next_token % vocab_size
        # for each data, the correct beams are gathered
        lengths     = tf.gather(outputs.lengths,     beam_index)
        finished    = tf.gather(loop_state.finished, beam_index)
        
        logits_with_scores  = tf.gather(logits_with_scores, beam_index)
        scores      = tf.gather(logits_with_scores, next_token, batch_dims = 1)
        
        next_token  = tf.where(finished, pad_token, next_token)

        finished    = tf.logical_or(finished, tf.math.equal(next_token, eos_token))
        lengths     = lengths + tf.cast(tf.logical_not(finished), lengths.dtype)[:, tf.newaxis]

        tokens      = tf.tensor_scatter_nd_update(
            tf.gather(outputs.tokens, beam_index),
            tf.stack([
                tf.range(effective_batch_size),
                tf.broadcast_to(loop_state.t - prefix_length, [effective_batch_size])
            ], -1),
            next_token
        )

        next_inputs  = next_token[:, tf.newaxis] if use_cache else tokens
        next_outputs = InferenceOutput(
            tokens  = tokens,
            lengths = lengths,
            scores  = scores,
            logits  = update_logits(
                outputs.logits, logits, state = loop_state, config = config, beam_index = beam_index
            ),
            attention_weights = update_attention_weights(
                outputs.attention_weights, model_out.attention_weights, loop_state, config, beam_index = beam_index
            )
        )
        next_state = update_state(
            state = loop_state, output = model_out, finished = finished, config = config, beam_index = beam_index
        )

        return next_inputs, next_outputs, next_state

    if step_fn is None:         step_fn = self
    if add_sos_token is None:   add_sos_token = prefix is None
    empty_token = tokens is None and not add_sos_token
    
    use_xla         = not tf.executing_eagerly()
    skip_attention  = not (return_attention or return_last_attention)
    
    with time_logger.timer('initialization'):
        if encoder_output is not None:  dtype = encoder_output.dtype
        elif prefix is not None:        dtype = prefix.dtype
        else:                           dtype = self.compute_dtype

        if batch_size == -1:
            batch_size = _get_batch_size(tokens, encoder_output, prefix = prefix)

        effective_batch_size    = batch_size * num_beams

        if tokens is None:
            start_idx   = tf.constant(1 if add_sos_token else 0, dtype = tf.int32)
            tokens      = tf.concat([
                tf.fill((effective_batch_size, start_idx), sos_token),
                tf.fill((effective_batch_size, max_length - start_idx), pad_token)
            ], axis = 1)
            input_length    = tf.fill((batch_size, 1), start_idx)
        else:
            if isinstance(tokens, (list, tuple)):
                tokens, input_length = tokens
            elif input_length is None:
                input_length    = 1 + tf.reduce_sum(
                    tf.cast(tokens[:, 1:] != pad_token, tf.int32), axis = -1, keepdims = True
                )
            
            #start_idx = tf.cast(tf.shape(tokens)[1], tf.int32)
            start_idx = tf.cast(tf.shape(tokens)[1], tf.int32)
            tokens = tf.pad(
                tokens, [(0, 0), (0, tf.maximum(0, max_length - start_idx))],
                constant_values = pad_token
            )
        
            tokens      = tf.repeat(tokens, num_beams, axis = 0)
            input_length    = tf.repeat(input_length, num_beams, axis = 0)
        
        if encoder_output is not None:
            encoder_output = tf.repeat(encoder_output, num_beams, axis = 0)
            if enc_padding_mask is not None:
                enc_padding_mask = tf.repeat(enc_padding_mask, num_beams, axis = 0)

        if prefix is not None:
            prefix = tf.repeat(prefix, num_beams, axis = 0)

        batch_idx_add   = tf.range(effective_batch_size) // num_beams * num_beams
        eos_mask        = tf.tensor_scatter_nd_update(
            tf.fill((1, vocab_size), dtype.min), [[0, pad_token]], [0.]
        )

        prefix_length   = tf.shape(prefix)[1] if prefix is not None else 0
        cur_len     = start_idx + prefix_length
        max_length  = max_length + prefix_length
        
        n = 1 if add_sos_token else 0
        padding_mask    = tf.concat([
            tf.ones((effective_batch_size, prefix_length + n), dtype = tf.bool),
            tokens[:, n:] != pad_token
        ], axis = 1)
        if not use_xla: padding_mask = padding_mask[:, : cur_len]

    config  = InferenceConfig(
        use_xla = use_xla,
        use_cache   = use_cache,
        max_length  = max_length,
        prefix_length   = prefix_length,
        encoder_seq_length  = tf.shape(encoder_output)[1] if encoder_output is not None else 0,
        
        is_transformer  = is_transformer,
        is_encoder_decoder  = encoder_output is not None,
        
        skip_attention  = skip_attention,
        return_logits   = return_logits
    )
    with time_logger.timer('first step'):
        inputs, outputs, state = body(
            tokens,
            InferenceOutput(
                tokens  = tokens,
                lengths = input_length,
                scores  = tf.zeros((effective_batch_size, ), dtype = dtype),
                logits  = None,
                attention_weights = None
            ),
            InferenceState(
                t       = cur_len,
                step    = tf.constant(0, tf.int32),
                state   = None,
                finished = tf.zeros((effective_batch_size, ), dtype = tf.bool),
                padding_mask    = padding_mask
            )
        )

    last_output, outputs, state = tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = (inputs, outputs, state),
        maximum_iterations  = max_length - state.t
    )

    outputs = tf.nest.map_structure(
        lambda o: tf.reshape(
            o, tf.concat([[batch_size, num_beams], tf.shape(o)[1:]], axis = 0)
        )[:, : num_sentences] if o is not None else o,
        outputs
    )
    return InferenceOutput(
        tokens  = outputs.tokens[:, :, start_idx :],
        lengths = outputs.lengths - start_idx,
        scores  = outputs.scores,
        logits  = outputs.logits if return_logits else None,
        attention_weights = None if skip_attention else outputs.attention_weights
    )


@timer
def process_logits(scores,
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
        scores = scores / tf.cast(temperature, scores.dtype)
    
    if length_temperature != 0.:
        scores = scores * (
            tf.cast(lengths + 1, scores.dtype) ** tf.cast(length_temperature, scores.dtype)
        )
    
    if logits_filter is not None:
        scores = tf.cast(logits_filter(scores, ** kwargs), scores.dtype)

    return log_softmax(scores, axis = -1)

@timer
def select_next_token(logits, n, temperature = 0., dtype = tf.int32, ** kwargs):
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
        if n <= 1:
            indices = tf.argmax(logits, axis = -1, output_type = tf.int32)
            return indices[:, tf.newaxis], tf.gather(logits, indices, batch_dims = 1)[:, tf.newaxis]
        values, indices = tf.nn.top_k(logits, k = n)
        return tf.cast(indices, tf.int32), values
    
    indices = tf.random.categorical(logits, n, dtype = dtype)
    scores  = tf.reshape(
        tf.gather(logits, tf.reshape(indices, [-1]), batch_dims = 1), tf.shape(indices)
    )
    return indices, scores

def update_logits(prev_logits, logits, config, state, beam_index = None):
    if not config.return_logits: return prev_logits if prev_logits is not None else ()
    
    if beam_index is not None:
        if prev_logits is not None: prev_logits = tf.gather(prev_logits, beam_index)
        logits = tf.gather(logits, beam_index)

    logits = logits[:, tf.newaxis, :]
    if not config.use_xla:
        return _update_logits_eager(prev_logits, logits, state = state, config = config)
    return _update_logits_xla(prev_logits, logits, state = state, config = config)

def _update_logits_eager(prev_logits, logits, state, config):
    return logits if prev_logits is None else tf.concat([prev_logits, logits], axis = 1)

def _update_logits_xla(prev_logits, logits, state, config):
    if prev_logits is None:
        padding = tf.constant([[0, 0], [0, 1], [0, 0]], tf.int32) * (config.max_length - 1)
        return tf.pad(logits, padding)
    
    return dynamic_update_slice(
        prev_logits, logits, tf.constant([0, 1, 0], tf.int32) * state.step
    )

def update_attention_weights(prev_attention, new_attention, state, config, beam_index = None):
    if config.skip_attention:
        return prev_attention if prev_attention is not None else ()
    
    if beam_index is not None:
        if prev_attention is not None:
            prev_attention = tf.nest.map_structure(
                lambda t: tf.gather(t, beam_index), prev_attention
            )
        new_attention   = tf.nest.map_structure(
            lambda t: tf.gather(t, beam_index), new_attention
        )
        
    if not config.use_xla:
        return _update_attention_weights_eager(prev_attention, new_attention, state, config)
    return _update_attention_weights_xla(prev_attention, new_attention, state, config)

def _update_attention_weights_eager(prev_attention, new_attention, state, config):
    if not config.use_cache: return new_attention
    # if the model is not a Transformer
    if not isinstance(new_attention, dict):
        if prev_attention is None: return new_attention[:, tf.newaxis, :]
        return tf.concat([prev_attention, new_attention[:, tf.newaxis, :]], axis = 1)
    
    if prev_attention is None:
        return {k : v[:, :, -1:, :] for k, v in new_attention.items()}

    def _concat_attentions(key, prev, attn):
        if 'enc' not in key: prev = tf.pad(prev, [[0, 0], [0, 0], [0, 0], [0, 1]])
        return tf.concat([prev, attn], axis = -2)
    
    return {
        k : _concat_attentions(k, prev_attention[k], new_attention[k])
        for k in prev_attention.keys()
    }

def _update_attention_weights_xla(prev_attention, new_attention, state, config):
    # if the model is not a Transformer
    if not isinstance(new_attention, dict):
        new_attention = new_attention[:, tf.newaxis, :]
        if prev_attention is None:
            padding = tf.constant([[0, 0], [0, 1], [0, 0]], tf.int32) * (config.max_length - 1)
            return tf.pad(new_attention, padding)
        
        start_indices = tf.constant([0, 1, 0], tf.int32) * state.step
        return dynamic_update_slice(prev_attention, new_attention, start_indices)
    
    if prev_attention is None:
        def init_attention(key, attn):
            padding = tf.constant(
                [[0, 0], [0, 0], [0, 1], [0, 0 if 'enc' in key else 1]], tf.int32
            ) * (config.max_length - 1)
            return tf.pad(attn[:, :, -1:, :], padding)
        
        return {k : init_attention(k, v) for k, v in new_attention.items()}
    
    def update_attention(key, attn, new_attn):
        new_attn = new_attn[:, :, -1:, :]
        if 'enc' not in key and config.use_cache:
            padding     = tf.constant([[0, 0], [0, 0], [0, 0], [0, 1]], tf.int32) * (
                config.max_length - state.t
            )
            new_value   = new_attn[:, :, :, -1:]
            
            last_idx    = tf.constant([0, 0, 0, 1], tf.int32) * (config.max_length - 1)
            start_idx   = tf.constant([0, 0, 0, 1], tf.int32) * (state.t - 1)
            new_attn    = dynamic_update_slice(new_attn, tf.zeros_like(new_value), last_idx)
            new_attn    = dynamic_update_slice(new_attn, new_value, start_idx)
        
        start_indices = tf.constant([0, 0, 1, 0], tf.int32) * state.step
        return dynamic_update_slice(attn, new_attn, start_indices)
    
    return {
        k : update_attention(k, prev_attention[k], new_attention[k])
        for k in prev_attention.keys()
    }

def update_state(state, output, finished, config, beam_index = None):
    _update_hidden_state_fn = _update_hidden_state_xla if config.use_xla else _update_hidden_state_eager
    _update_mask_fn = _update_padding_mask_xla if config.use_xla else _update_padding_mask_eager
    
    mask    = state.padding_mask
    next_hidden_state = output.state
    if beam_index is not None:
        mask = tf.gather(mask, beam_index)
        if config.use_cache:
            if not isinstance(next_hidden_state, dict) or not config.is_encoder_decoder:
                next_hidden_state = tf.nest.map_structure(
                    lambda t: tf.gather(t, beam_index), next_hidden_state
                )
            else:
                # the encoder output states are identical for all sentences within a beam
                # therefore, no need of gathering them, which safe time and memory !
                next_hidden_state = {
                    k : (
                        (tf.gather(v[0][0], beam_index), tf.gather(v[0][1], beam_index)), v[1]
                    ) for k, v in next_hidden_state.items()
                }

    return InferenceState(
        t   = state.t + 1,
        step    = state.step + 1,
        finished    = finished,
        state   = _update_hidden_state_fn(
            state, next_hidden_state, config = config
        ) if config.use_cache else (),
        padding_mask    = _update_mask_fn(state, mask, finished, config = config)
    )


def _update_hidden_state_eager(state, next_hidden_state, config):
    return next_hidden_state

def _update_hidden_state_xla(state, next_hidden_state, config):
    if not config.is_transformer:  return next_hidden_state
    
    if state.state is None:
        num_padding = config.max_length - state.t - 1
        padding     = tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]], tf.int32) * num_padding
        
        new_state   = {}
        for layer, layer_state in next_hidden_state.items():
            _state  = layer_state[0] if config.is_encoder_decoder else layer_state
            _shape  = tf.shape(_state[0])
            
            new_layer_state = (
                tf.pad(_state[0], padding), tf.pad(_state[1], padding)
            )
            new_state[layer] = new_layer_state if not config.is_encoder_decoder else (
                new_layer_state, layer_state[1]
            )
        return new_state

    start_slice = tf.constant([0, 0, 1, 0], tf.int32) * (state.t - 1)
    
    new_state   = {}
    for layer, layer_state in next_hidden_state.items():
        updated = layer_state[0] if config.is_encoder_decoder else layer_state

        updated = (
            dynamic_update_slice(updated[0][:, :, :-1, :], updated[0][:, :, -1:, :], start_slice),
            dynamic_update_slice(updated[1][:, :, :-1, :], updated[1][:, :, -1:, :], start_slice)
        )

        new_state[layer] = (updated, layer_state[1]) if config.is_encoder_decoder else updated
    
    return new_state

def _update_padding_mask_eager(state, mask, finished, config):
    return tf.concat([mask, ~finished[:, tf.newaxis]], axis = 1)

def _update_padding_mask_xla(state, mask, finished, config):
    if state.state is None and config.use_cache:
        mask = tf.concat([
            mask[:, :-1], tf.ones((tf.shape(mask)[0], 1), dtype = mask.dtype)
        ], axis = 1)
    
    update_idx = (state.t - 1) if config.use_cache else state.t
    return dynamic_update_slice(
        mask, ~finished[:, tf.newaxis], tf.constant([0, 1], tf.int32) * update_idx
    )

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
    'greedy'    : infer_greedy,
    'sample'    : lambda * args, ** kwargs: infer_greedy(* args, use_sampling = True, ** kwargs),
    'beam'      : infer_beam_search
}