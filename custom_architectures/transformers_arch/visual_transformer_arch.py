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

""" TF 2.0 CLIP (Visual Transformer), compatible with the official CLIP implementation """

import tensorflow as tf

from custom_layers import get_activation
from custom_architectures.current_blocks import _get_pooling_layer
from custom_architectures.transformers_arch.transformer_arch import (
    HParamsTransformerEncoder, TransformerEncoder
)

HParamsVisualTransformer  = HParamsTransformerEncoder(
    input_dim   = -1,
    input_channels  = 3,
    add_row_embedding   = False,
    add_col_embedding   = False,
    add_class_embedding = False,
    
    filters = -1,
    kernel_size = 3,
    strides     = -1,
    conv_bias   = False,
    padding     = 'valid',
    
    conv_normalize  = True,
    conv_activation = None,
    
    pooling     = None,
    pool_size   = 2,
    pool_strides    = 2,
    
    conv_drop_rate  = 0.1,
    
    normalize   = 'middle',
    normalize_output    = True,
    mha_normalize   = False,
    mha_normalize_input = True,
    
    ffn_activation  = 'quick_gelu',
    
    mha_epsilon     = 1e-5,
    epsilon     = 1e-5
)

class VisualTransformer(TransformerEncoder):
    default_params  = HParamsVisualTransformer
    _attr_to_set    = TransformerEncoder._attr_to_set + [
        'input_dim', 'add_class_embedding', 'add_row_embedding', 'add_col_embedding'
    ]
    
    def __init__(self, input_dim, embedding_dim, ** kwargs):
        super().__init__(
            input_dim = input_dim, embedding_dim = embedding_dim, ** kwargs
        )
    
    def _init_input_layers(self, ** kwargs):
        strides = self.hparams.strides if self.hparams.strides != -1 else self.hparams.filters
        self.conv   = tf.keras.layers.Conv2D(
            filters     = self.hparams.filters,
            kernel_size = self.hparams.kernel_size,
            use_bias    = self.hparams.conv_bias,
            strides     = strides,
            padding     = self.hparams.padding,
            name        = 'conv1'
        )
        self.conv_norm  = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_conv'
        ) if self.hparams.conv_normalize else None
        self.conv_act   = get_activation(self.hparams.conv_activation)
        self.pooling    = _get_pooling_layer(
            self.hparams.pooling, dim = '2d',
            pool_size   = self.hparams.pool_size,
            pool_strides    = self.hparams.pool_strides
        )
        self.conv_drop  = tf.keras.layers.Dropout(self.hparams.conv_drop_rate) if self.hparams.conv_drop_rate > 0 else None
        
        self.row_embedding  = None
        self.col_embedding  = None
        self.class_embedding    = None
        
        grid_size   = self.hparams.input_dim // strides
        ctx_length  = grid_size ** 2
        
        if self.add_class_embedding: ctx_length += 1
        with tf.name_scope(self.name):
            if self.add_row_embedding:
                self.row_embedding  = self.add_weight(
                    shape = (grid_size, self.embedding_dim), name = 'row_embedding'
                )
            if self.add_col_embedding:
                self.col_embedding  = self.add_weight(
                    shape = (grid_size, self.embedding_dim), name = 'col_embedding'
                )

            if self.add_class_embedding:
                self.class_embedding    = self.add_weight(
                    shape = (self.embedding_dim, ), name = 'class_embedding'
                )
            
            self.positional_embedding   = self.add_weight(
                shape = (ctx_length, self.embedding_dim), name = 'positional_embedding'
            )
    
    @property
    def input_signature(self):
        return tf.TensorSpec(
            shape = (None, self.input_dim, self.input_dim, self.hparams.input_channels),
            dtype = tf.float32
        )

    def prepare_input(self, inputs, training = False, mask = None, ** kwargs):
        embedded = self.conv(inputs)
        if self.conv_act is not None:   embedded = self.conv_act(embedded)
        if self.pooling is not None:    embedded = self.pooling(embedded)
        
        if self.add_row_embedding:
            embedded = embedded + tf.reshape(self.row_embedding[: tf.shape(embedded)[1]], [
                1, tf.shape(embedded)[1], 1, self.embedding_dim
            ])
        if self.add_col_embedding:
            embedded = embedded + tf.reshape(self.col_embedding[: tf.shape(embedded)[2]], [
                1, 1, tf.shape(embedded)[2], self.embedding_dim
            ])

        embedded = tf.reshape(embedded, [tf.shape(embedded)[0], -1, self.embedding_dim])
        
        if self.add_class_embedding:
            embedded = tf.concat([
                tf.broadcast_to(
                    self.class_embedding, [tf.shape(embedded)[0], 1, tf.shape(embedded)[-1]]
                ),
                embedded
            ], axis = 1)
        
        embedded = embedded + tf.expand_dims(
            self.positional_embedding[: tf.shape(embedded)[1]], axis = 0
        )
        if self.conv_norm is not None:  embedded = self.conv_norm(embedded, training = training)
        if self.conv_drop is not None:  embedded = self.conv_drop(embedded, training = training)
        
        embedded._keras_mask = tf.ones(
            (tf.shape(embedded)[0], tf.shape(embedded)[1]), dtype = tf.bool
        )
        return embedded
    
    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import (
            _transformer_patterns, _attn_split, name_based_partial_transfer_learning
        )

        kwargs.setdefault('transforms', _attn_split)

        if isinstance(pretrained, dict):
            pretrained = {k : v for k, v in pretrained.items() if 'visual' in k}
        
        return name_based_partial_transfer_learning(
            self, pretrained, skip_root = False, patterns = {
                ** _transformer_patterns, 'norm_pre' : 'transformer/norm_conv'
            }, ** kwargs
        )

    @classmethod
    def from_pretrained(cls, pretrained_name = 'RN50', pretrained = None,** kwargs):
        from custom_architectures.clip_arch import load_clip
        from models.weights_converter import get_pt_variables, get_pt_layers, transpose_weights

        state_dict  = load_clip(pretrained_name, pretrained = pretrained)
        
        embedding_dim   = state_dict["visual.conv1.weight"].shape[0]
        kernel_size = state_dict["visual.conv1.weight"].shape[-1]

        num_layers  = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        input_dim = kernel_size * grid_size
        output_dim  = state_dict["visual.proj"].shape[1]
        
        config = cls.default_params(
            input_dim   = input_dim,
            output_dim  = output_dim,
            output_bias = False,
            
            filters     = embedding_dim,
            kernel_size = kernel_size,
            strides     = kernel_size,
            padding     = 'valid',
            
            embedding_dim   = embedding_dim,
            num_layers      = num_layers,
            mha_num_heads   = embedding_dim // 64,
            ffn_dim         = embedding_dim * 4
        )
        with tf.device('cpu'):
            instance = cls(** config(** kwargs))
            instance._build()

        instance.transfer_weights(state_dict, ** kwargs)
        
        return instance

custom_functions    = {
    'ViT'   : VisualTransformer,
    'VisualTransformer' : VisualTransformer
}

custom_objects  = custom_functions

_encoders   = custom_functions
_transformers   = _encoders