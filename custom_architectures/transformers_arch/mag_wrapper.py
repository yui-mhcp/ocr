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

import enum
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import model_from_json

from loggers import timer
from hparams import HParams
from utils import get_enum_item
from utils.sequence_utils import pad_to_multiple
from custom_layers import FasterEmbedding
from custom_architectures.transformers_arch.transformer_arch import Transformer, TransformerBlock, build_mask, format_output
from custom_architectures.transformers_arch.bart_arch import Bart, BartEncoder
from custom_architectures.transformers_arch.text_transformer_arch import *

class SubsamplingMode(enum.IntEnum):
    SELECT  = 0
    DENSE   = 1
    CONV    = 2
    SEPARABLE   = 3
    MIN     = 4
    MAX     = 5
    MEAN    = 6

HParamsMAGWrapper = HParams(
    subsample_at    = -1,
    subsample_after = True,
    subsampling_step    = -1,
    subsampling_mode    = 'select',
    subsampling_offset  = 1,        # only used if `subsampling_mode == 'select'`
    subsampling_drop_rate   = 0.,

    repeat_pos_idx      = False,
    
    use_type_embedding      = False,
    random_training_type    = True,
    max_types   = 16
)

@timer
def pad_output_and_mask(output, mask, step):
    output = pad_to_multiple(output, step, axis = 1)
    if mask is not None:
        mask = pad_to_multiple(mask, step, axis = -1)
    
    return output, mask

@timer
@tf.function(reduce_retracing = True)
def concat_embeddings(embeddings,
                      mask      = None,
                      merge_embeddings  = False,
                      debug     = False,
                      ** kwargs
                     ):
    """
        Concat multiple embeddings into a single embedding of shape [batch_size, total_seq_len, embedding_dim]
        
        Arguments :
            - embeddings : list of `tf.Tensor`
                - The 1st one (`embeddings[0]`) is considered as the `query` and must be of shape [B, q_len, embedding_dim]
                - The others (`embeddings[1:]`) are considered as `contexts` and can be of shape :
                    1) [batch_size, ci_len, embedding_dim]
                    2) [batch_size, n_doc, ci_len, embedding_dim]
                Note : if multiple `contexts` are given, they **must** be of the 1st shape
    """
    query, contexts = embeddings[0], embeddings[1:]
    q_mask, c_masks = (mask[0], mask[1:]) if mask is not None else (None, None)

    check_padding, is_real_mask = False, c_masks is not None
    
    c_lengths   = [tf.shape(c)[-2] for c in contexts]
    contexts    = tf.concat(contexts, axis = 1) if len(contexts) > 1 else contexts[0]
    
    n_doc_per_batch = 1
    q_batch_size, c_batch_size = tf.shape(query)[0], tf.shape(contexts)[0]

    if c_masks is not None:
        if tf.shape(c_masks[0])[-2] > 1:
            c_masks = tuple([tf.reduce_any(m, axis = -2, keepdims = True) for m in c_masks])
        c_masks = tf.concat(c_masks, axis = -1) if len(c_masks) > 1 else c_masks[0]
    if q_mask is not None and tf.shape(q_mask)[-2] > 1:
        q_mask = tf.reduce_any(q_mask, axis = -2, keepdims = True)
    
    if c_masks is None:
        if len(tf.shape(contexts)) == 3:
            c_masks = tf.ones((tf.shape(contexts)[0], 1, 1, tf.shape(contexts)[1]), dtype = tf.bool)
        else:
            c_masks = tf.ones((tf.shape(contexts)[0], tf.shape(contexts)[1], 1, 1, tf.shape(contexts)[2]), dtype = tf.bool)
    if q_mask is None:
        q_mask = tf.ones((q_batch_size, 1, 1, tf.shape(query)[1]), dtype = c_masks.dtype)

    lengths     = [tf.shape(query)[1]] + c_lengths
    
    if debug:
        tf.print("Sequence lengths :", lengths)
        tf.print("1st input shape :", tf.shape(query))
        tf.print("2nd input shape :", tf.shape(contexts))
        tf.print("Masks shape :", tf.shape(c_masks))
    
    # flatten contexts from [B, n_doc, ctx_len, emb_dim] to [B, n_doc * ctx_len, emb_dim]
    if len(tf.shape(contexts)) == 4:
        if len(c_lengths) > 1:
            raise NotImplementedError("When passing multiple document / batch at once, you cannot pass multiple contexts, please flatten everything !")

        n_doc_per_batch = tf.shape(contexts)[1]
        
        ctx_types = tf.repeat(tf.range(1, n_doc_per_batch + 1), tf.shape(contexts)[2])
        
        contexts    = tf.reshape(contexts, [c_batch_size, -1, tf.shape(contexts)[-1]])
        c_masks     = tf.reshape(c_masks, [c_batch_size, 1, 1, -1])
        
        check_padding = True
        if debug:
            tf.print("Contexts (after flattening) shape :", tf.shape(contexts))
            tf.print("Masks (after flattening) shape :", tf.shape(c_masks))
                
    elif len(c_lengths) > 1:
        ctx_types   = tf.concat([
            tf.fill([length], i + 1) for i, length in enumerate(c_lengths)
        ], axis = -1)
    else:
        ctx_types   = tf.fill((tf.shape(contexts)[1], ), 1)
    
    # Merge contexts (if required)
    if merge_embeddings and q_batch_size > 1:
        if len(c_lengths) > 1:
            raise NotImplementedError("When merging contexts, you can only pass 1 context / batch !")
        
        ctx_add_type = tf.repeat(tf.range(q_batch_size), tf.shape(contexts)[1])

        contexts  = tf.reshape(
            tf.tile(contexts, [q_batch_size, 1, 1]), 
            [q_batch_size, -1, tf.shape(contexts)[-1]]
        )
        ctx_types = tf.tile(ctx_types, [q_batch_size]) + n_doc_per_batch * ctx_add_type
        c_masks   = tf.reshape(
            tf.tile(c_masks, [q_batch_size, 1, 1, 1]), 
            [q_batch_size, 1, 1, -1]
        )
        check_padding = True
        if debug:
            tf.print("Contexts (after merging) shape :", tf.shape(contexts))
            tf.print("Masks (after merging) shape :", tf.shape(c_masks))
        

    if check_padding and is_real_mask:
        not_padding = tf.reduce_any(tf.reshape(c_masks, [c_batch_size, -1]), axis = 0)

        contexts    = tf.boolean_mask(contexts,  not_padding, axis = 1)
        c_masks     = tf.boolean_mask(c_masks,   not_padding, axis = 3)
        ctx_types   = tf.boolean_mask(ctx_types, not_padding, axis = 0)
        
        if debug:
            tf.print("# padding :", tf.reduce_sum(tf.cast(tf.logical_not(not_padding), tf.int32)))
            tf.print("Contexts (after removing padding) shape :", tf.shape(contexts))
            tf.print("Masks (after removing padding) shape :", tf.shape(c_masks))
    
    if q_batch_size != c_batch_size and q_batch_size > 1:
        contexts = tf.tile(contexts, [q_batch_size, 1, 1])
        c_masks  = tf.tile(c_masks, [q_batch_size, 1, 1, 1])

    types   = tf.concat([tf.fill([tf.shape(query)[1]], 0), ctx_types], axis = -1)
    
    memory  = tf.concat([query, contexts], axis = 1)
    masks   = tf.concat([q_mask, c_masks], axis = -1)
    types   = tf.tile(tf.expand_dims(types, axis = 0), [q_batch_size, 1])

    return (memory, masks, types)

class MAGModelWrapper(tf.keras.Model):
    """
        This class is a `wrapper`, meaning that it takes as argument a regular `TransformerBlock` model
        Its objective is simply to modify the way data is given to the wrapped model by passing separately all inputs to the `M` first layers (named the `memory layers`), (possibly) subsample them, then concatenate them all, and finally pass the result to the `N` remaining layers (named the `embedding layers`)
        
        The benefit of this approach are multiple :
        - There are no restriction on the total input size (for the positional encoding)
        - It allows to reduce the memory impact of long inputs by subsampling (optional)
        - It reduces the impact of padding by removing it after the concatenation
        - It does not degrade performances (at least if trained with this strategy and with well-tuned parameters)
    """
    default_params  = HParamsMAGWrapper
    _attr_to_set    = [
        'subsample_at', 'subsample_after', 'subsampling_mode', 'subsampling_step',
        'subsampling_offset', 'max_types', 'random_training_type', 'repeat_pos_idx'
    ]

    def __init__(self, model, name = 'encoder', ** kwargs):
        super().__init__(name = name)
        self.hparams    = self.default_params.extract(kwargs)
        
        for attr_name in self._attr_to_set:
            setattr(self, attr_name, self.hparams[attr_name])
        
        self.model  = model
        
        for attr_name in self.model._attr_to_set:
            setattr(self, attr_name, getattr(self.model, attr_name))
        
        layer_idx = self.subsample_at
        if layer_idx < 0: layer_idx = len(self.model._layers) + layer_idx
        if self.subsample_after: layer_idx += 1
        self.M = max(0, min(len(self.model._layers), layer_idx))
        
        
        self.subsampling_layer  = None
        self.subsampling_drop_layer = tf.keras.layers.Dropout(
            self.hparams.subsampling_drop_rate
        ) if self.hparams.subsampling_drop_rate > 0 else None
        self.type_embedding_layer = None
        
        if self.subsampling_step > 1:
            self.subsampling_mode = get_enum_item(self.subsampling_mode, SubsamplingMode)
            
            if self.subsampling_mode in (SubsamplingMode.CONV, SubsamplingMode.SEPARABLE):
                if self.subsampling_mode == SubsamplingMode.CONV:
                    cls = tf.keras.layers.Conv1D
                else:
                    cls = tf.keras.layers.SeparableConv1D
                
                self.subsampling_layer = cls(
                    filters     = self.embedding_dim,
                    kernel_size = self.subsampling_step,
                    strides     = self.subsampling_step,
                    padding     = 'valid',
                    name    = 'subsampling_layer'
                )
            elif self.subsampling_mode == SubsamplingMode.DENSE:
                self.subsampling_layer = tf.keras.layers.Dense(
                    units   = self.embedding_dim,
                    name    = 'subsampling_layer',
                    kernel_initializer  = self._mean_initializer
                )
        
        if self.hparams.use_type_embedding:
            self.type_embedding_layer = FasterEmbedding(
                self.max_types, self.embedding_dim, name = "type_embedding"
            )

    
    def _mean_initializer(self, shape, dtype = None):
        w = np.zeros(shape)
        for i in range(self.embedding_dim):
            w[i::self.embedding_dim, i] = 1
        w /= self.subsampling_step
        return tf.cast(w, dtype)

    def _build(self, ** kwargs):
        return self(self.dummy_inputs, ** kwargs)

    @property
    def memory_layers(self):
        return self.model._layers[: self.M]
    
    @property
    def embedding_layers(self):
        return self.model._layers[self.M :]
    
    @property
    def dummy_inputs(self):
        dummy_inputs    = self.model.dummy_inputs
        multi_dummy_inputs = tf.nest.map_structure(
            lambda inp: tf.expand_dims(inp, axis = 1), dummy_inputs
        )
        
        return [dummy_inputs, multi_dummy_inputs]

    def set_tokens(self, * args, ** kwargs):
        return self.model.set_tokens(* args, ** kwargs)
    
    @timer
    def subsample(self, output, mask = None, training = False):
        if self.subsampling_step <= 1: return output, mask
        
        if self.subsampling_drop_layer is not None:
            output = self.subsampling_drop_layer(output, training = training)
        
        if self.subsampling_mode == SubsamplingMode.SELECT:
            indices = tf.range(self.subsampling_offset, tf.shape(output)[1], self.subsampling_step)
            indices = tf.tile(tf.expand_dims(indices, axis = 0), [tf.shape(output)[0], 1])

            output = tf.gather(output, indices, batch_dims = 1)

            if mask is not None:
                mask = tf.gather(tf.squeeze(mask, [1, 2]), indices, batch_dims = 1)
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1])
        elif self.subsampling_mode in (SubsamplingMode.CONV, SubsamplingMode.SEPARABLE):
            output = self.subsampling_layer(output, training = training)

            if mask is not None:
                indices = tf.range(0, tf.shape(output)[1]) * self.subsampling_step
                indices = tf.tile(tf.expand_dims(indices, axis = 0), [tf.shape(output)[0], 1])

                mask = tf.gather(tf.squeeze(mask, [1, 2]), indices, batch_dims = 1)
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1])
        else:
            output, mask = pad_output_and_mask(output, mask, self.subsampling_step)
            
            if mask is not None:
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1, self.subsampling_step])
                mask = tf.reduce_all(mask, axis = -1)

            if self.subsampling_mode == SubsamplingMode.DENSE:
                output = tf.reshape(
                    output, [tf.shape(output)[0], -1, self.subsampling_step * tf.shape(output)[-1]]
                )
                output = self.subsampling_layer(output)
            else:
                output = tf.reshape(
                    output, [tf.shape(output)[0], -1, self.subsampling_step, tf.shape(output)[-1]]
                )
                if self.subsampling_mode == SubsamplingMode.MIN:
                    output = tf.reduce_min(output, axis = 2)
                elif self.subsampling_mode == SubsamplingMode.MAX:
                    output = tf.reduce_max(output, axis = 2)
                else:
                    output = tf.reduce_mean(output, axis = 2)
        
        return output, mask

    @timer
    def embed_types(self, memory, types, training = False, debug = False, ** kwargs):
        if self.type_embedding_layer is None: return memory, types
        
        if self.max_types == 2:
            types = tf.cast(types > 0, tf.int32)
        elif self.random_training_type and training and tf.reduce_max(types) < self.max_types:
            random_offset = tf.random.uniform(
                (tf.shape(types)[0], 1),
                minval  = 0,
                maxval  = self.max_types - tf.reduce_max(types),
                dtype   = tf.int32
            )
            types = types + (random_offset * tf.cast(types > 0, tf.int32))
        
        if debug: tf.print("Types used :", types)
        
        memory = memory + self.type_embedding_layer(types)
        
        return memory, types
    
    @timer
    def embed_memory(self,
                     inputs,

                     mask  = None,
                     training  = False,

                     force_not_subsampling = False,
                     
                     return_mask    = True,
                     as_dict    = True,

                     debug = False,
                     ** kwargs
                    ):
        if isinstance(inputs, (list, tuple)):
            kwargs['force_not_subsampling'] = force_not_subsampling
            
            outputs = []
            for i in range(len(inputs)):
                outputs.append(self.embed_memory(
                    inputs[i],
                    
                    mask    = mask[i] if mask is not None else None,
                    training    = training,
                    
                    as_dict = True,
                    
                    debug   = debug,
                    ** {k : v[i] if isinstance(v, list) else v for k, v in kwargs.items()}
                ))
            return tf.nest.map_structure(
                lambda * args: args, * outputs
            )

        text = inputs

        if debug: tf.print("Input tokens shape :", tf.shape(text))
        
        batch_size = tf.shape(text)[0]
        n_doc_per_batch = -1
        if len(tf.shape(text)) == 3:
            n_doc_per_batch = tf.shape(text)[1]
            text            = tf.reshape(text, [-1, tf.shape(text)[-1]])
            if debug:
                tf.print("Input tokens reshaped shape :", tf.shape(text))

        outputs = self.model(
            text,
            mask    = mask,
            training    = training,

            first_layer_idx = -1,
            last_layer_idx  = self.M,
            
            return_mask = True,
            as_dict = True,
            ** kwargs
        )
        
        output, mask = outputs.output, outputs.mask

        if not force_not_subsampling:
            output, mask = self.subsample(output, mask = mask, training = training)
            if debug: tf.print("Output subsampled shape :", tf.shape(output))
        
        if n_doc_per_batch != -1:
            output  = tf.reshape(output, [
                batch_size, n_doc_per_batch, tf.shape(output)[1], tf.shape(output)[2]
            ])
            mask    = tf.reshape(mask,   [batch_size, n_doc_per_batch, 1, 1, tf.shape(mask)[-1]])

            if debug: tf.print("Output reshaped shape :", tf.shape(output))
        
        return format_output(
            output,
            mask    = mask,
            state   = outputs.state,
            logits  = outputs.logits,
            attention_weights   = outputs.attention_weights,
            
            return_mask = return_mask,
            as_dict = as_dict,
            ** kwargs
        )
    
    @timer
    def process_memory(self,
                       embeddings,
                       mask     = None,
                       training = False,
                       memory   = None,
                       ** kwargs
                      ):
        if memory is not None: embeddings = [embeddings, memory]
        memory, mask, types = concat_embeddings(embeddings, mask = mask, training = training, ** kwargs)
        
        memory, types = self.embed_types(memory, types, training = training, ** kwargs)
        
        return self.model(
            memory, first_layer_idx = self.M, training = training, padding_mask = mask, ** kwargs
        )
    
    @timer
    def call(self,
             inputs,
             memory     = None,
             mask       = None,
             training   = False,
             
             merge_embeddings   = False,
             
             return_state       = False,
             return_attention   = False,
             return_last_attention  = False,
             return_hidden_states   = False,
             return_mask        = False,
             as_dict    = False,
             
             ** kwargs
            ):
        """
            Computes the `MAG`-style prediction by :
            1) encoding separately all `inputs` (with the `memory layers`)
            2) (optional) subsampling all encoded `inputs`
            3) Concatenating all the encoded (possibly subsampled) `inputs`
            4) Passing all the concatenated result to the remaining `embedding layers`
            
            Arguments :
                - inputs    : the input tokens
                    - single `tf.Tensor`    : the `memory` argument must be given
                    - list of `tf.Tensor`   : the list of inputs, the 1st is the `main` input while the others are considered as `memory` (i.e. the contextuals information)
                - merge_embeddings  : whether to merge the *multi* input embeddings (i.e. the memory)
                other arguments are equivalent to `TransformerBlock.call`
        """
        memory_outputs = self.embed_memory(
            inputs,
            mask    = mask,
            training    = training,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = True,
            as_dict = True,
            ** kwargs
        )
        embeddings, masks = memory_outputs.output, memory_outputs.mask

        outputs = self.process_memory(
            embeddings,
            memory  = memory,
            mask    = masks,
            training    = training,
            merge_embeddings    = merge_embeddings,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            ** kwargs
        )
        
        return format_output(
            outputs.output,
            state   = (memory_outputs.state, outputs.state),
            attn_weights    = (memory_outputs.attention_weights, outputs.attention_weights),
            hidden_states   = (memory_outputs.hidden_states, outputs.hidden_states),
            mask    = outputs.mask,
            
            return_state    = return_state,
            return_attention    = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = as_dict
        )
    
    def get_config(self):
        config = self.hparams.get_config()
        config['model'] = json.loads(self.model.to_json())
        return config

    def transfer_weights(self, * args, ** kwargs):
        self.model.transfer_weights(* args, ** kwargs)
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        config['model'] = model_from_json(
            json.dumps(config['model']), custom_objects = custom_objects
        )
        return cls(** config)

class MAGWrapper(tf.keras.Model):
    def __init__(self, model = None, ** kwargs):
        super().__init__(
            name = 'mag_{}'.format(model.name if model is not None else kwargs.get('name', 'wrapper'))
        )
        
        if model is None:
            from custom_architectures.transformers_arch.bart_arch import Bart
            kwargs.update(MAGWrapper.get_wrapper_kwargs())
            model = Bart(** kwargs)
        
        if not isinstance(model, Transformer):
            if not isinstance(model, MAGModelWrapper):
                model = MAGModelWrapper(model, ** kwargs)
        elif not isinstance(model.encoder, MAGModelWrapper):
            model.encoder = MAGModelWrapper(model.encoder, ** kwargs)
        
        self.model = model
        
        for config in self.model._attr_to_set:
            setattr(self, config, getattr(self.model, config))
    
    @property
    def hparams(self):
        return self.model.hparams
    
    @property
    def dummy_inputs(self):
        return self.model.dummy_inputs
    
    @property
    def dummy_encoder_output(self):
        return self.model.dummy_encoder_output
    
    @property
    def encoder(self):
        return self.model.encoder if isinstance(self.model, Transformer) else self.model
    
    @property
    def decoder(self):
        return self.model.decoder if isinstance(self.model, Transformer) else None
    
    def _build(self, ** kwargs):
        return self(self.dummy_inputs, ** kwargs)

    def set_tokens(self, * args, ** kwargs):
        return self.model.set_tokens(* args, ** kwargs)
    
    def call(self, * args, ** kwargs):
        return self.model(* args, ** kwargs)
    
    def infer(self, * args, ** kwargs):
        return self.model.infer(* args, ** kwargs)
    
    def get_config(self):
        return {'model' : json.loads(self.model.to_json())}
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        if 'model' in config:
            from custom_architectures import get_architecture
            
            class_name  = config['model']['class_name']
            class_conf  = config['model']['config']
            class_conf.update(MAGWrapper.get_wrapper_kwargs())

            config['model'] = get_architecture(class_name, ** class_conf)
        
        return cls(** config)

    @classmethod
    def from_pretrained(cls, pretrained_name, * args, ** kwargs):
        from custom_architectures.transformers_arch import get_pretrained_transformer
        
        kwargs.update(MAGWrapper.get_wrapper_kwargs())
        return cls(get_pretrained_transformer(
            pretrained_name, * args, ** kwargs
        ))
    
    @staticmethod
    def get_wrapper_kwargs():
        return {'encoder_wrapper'   : MAGModelWrapper}

custom_objects  = {
    'MAG'   : MAGWrapper,
    'MAGWrapper'    : MAGWrapper,
    'MAGModelWrapper'   : MAGModelWrapper
}

custom_functions    = custom_objects
