
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

import enum
import tensorflow as tf

from utils import get_enum_item
from hparams.hparams import HParams
from custom_layers import get_activation
from custom_architectures.current_blocks import _get_layer, _get_pooling_layer

class TokenSelector(enum.IntEnum):
    NONE    = 0
    FIRST   = 1
    LAST    = 2
    MIN     = 3
    MAX     = 4
    
HParamsEmbeddingHead   = HParams(
    output_dim = -1,
    token_selector  = TokenSelector.NONE,
    
    hidden_dim  = 0,
    hidden_name = 'hidden_layer',
    hidden_activation   = None,
    hidden_drop_rate    = 0.1,
    hidden_layer_type   = 'bi_lstm',
    hidden_layer_kwargs = {},
    
    final_pooling   = None,
    use_final_dense = True,
    output_bias      = True,
    output_name      = 'output_layer',
    output_activation    = None,
    output_normalize    = False # for l2-normalization
)

class EmbeddingHead(tf.keras.layers.Layer):
    def __init__(self, output_dim, name = 'output', ** kwargs):
        super().__init__(name = name)
        kwargs.update({'output_dim' : output_dim})
        self.hparams = HParamsEmbeddingHead.extract(kwargs)

        self.token_selector = get_enum_item(self.hparams.token_selector, TokenSelector)
        
        # Layers are applied in this order (initialized only if required)
        self.hidden_layer   = None
        self.hidden_act_layer   = None
        self.hidden_drop_layer  = None
        
        self.final_pooling  = None
        self.concat_layer   = None
        
        self.final_dense    = None
        self.final_act_layer    = None
        self.final_norm_layer   = None
        
        if not self.hparams.use_final_dense or self.hparams.hidden_dim > 0:
            if self.hparams.use_final_dense:
                units   = self.hparams.hidden_dim
                name    = self.hparams.hidden_name
                act     = self.hparams.hidden_activation
                drop    = self.hparams.hidden_drop_rate
            else:
                units   = self.hparams.output_dim
                name    = self.hparams.output_name
                act     = self.hparams.output_activation
                drop    = 0.

            if units > 0:
                self.hidden_layer = _get_layer(
                    self.hparams.hidden_layer_type, units, name = name,
                    ** self.hparams.hidden_layer_kwargs
                )
                self.hidden_act_layer   = get_activation(act)
                if drop > 0: self.hidden_drop_layer = tf.keras.layers.Dropout(drop)
        
        if self.hparams.final_pooling:
            self.final_pooling = _get_pooling_layer(
                self.hparams.final_pooling, dim = '1d', global_pooling = True
            )
            if isinstance(self.hparams.final_pooling, (list, tuple)):
                self.concat_layer   = tf.keras.layers.Concatenate(axis = -1)
        
        if self.hparams.use_final_dense and output_dim > 0:
            self.final_dense    = tf.keras.layers.Dense(
                output_dim, use_bias = self.hparams.output_bias, name = self.hparams.output_name
            )
            self.final_act_layer    = get_activation(self.hparams.output_activation)
        
        if self.hparams.output_normalize:
            self.final_norm_layer = tf.keras.layers.Lambda(
                lambda x: tf.math.l2_normalize(x, axis = -1)
            )

    def select_token(self, inputs, text = None):
        if self.token_selector == TokenSelector.NONE:
            return inputs
        elif self.token_selector == TokenSelector.FIRST:
            return inputs[:, 0]
        elif self.token_selector == TokenSelector.LAST:
            return inputs[:, -1]
        elif self.token_selector == TokenSelector.MIN:
            assert len(tf.shape(text)) == 2, 'Invalid tokens shape : {}'.format(tf.shape(text))
            return tf.gather(inputs, tf.argmin(text, axis = -1))
        elif self.token_selector == TokenSelector.MAX:
            assert len(tf.shape(text)) == 2, 'Invalid tokens shape : {}'.format(tf.shape(text))
            return tf.gather(inputs, tf.argmax(text, axis = -1), batch_dims = 1)

    def call(self, inputs, mask = None, training = False, text = None, ** kwargs):
        output = self.select_token(inputs, text = text)
        
        if mask is not None and len(tf.shape(output)) == 3:
            mask = tf.cast(mask[:, 0, 0, :], tf.bool)
        
        if self.hidden_layer is not None:
            if self.hparams.hidden_layer_type != 'dense':
                output  = self.hidden_layer(output, mask = mask, training = training)
            else:
                output = self.hidden_layer(output, training = training)
            if self.hidden_act_layer is not None:
                output = self.hidden_act_layer(output)
            if self.hidden_drop_layer is not None:
                output = self.hidden_drop_layer(output, training = training)
        
        
        if self.final_pooling is not None:
            if isinstance(self.final_pooling, (list, tuple)):
                pooled = []
                for pool_layer, pool_type in zip(self.final_pooling, self.hparams.final_pooling):
                    masking = {} if pool_type == 'max' else {'mask' : mask}
                    pooled.append(pool_layer(output, ** masking))
                output = self.concat_layer(pooled)
            else:
                masking = {} if self.hparams.final_pooling == 'max' else {'mask' : mask}
                output = self.final_pooling(output, ** masking)
        
        
        if self.final_dense is not None: output = self.final_dense(output)
        if self.final_act_layer is not None: output = self.final_act_layer(output)
        
        if self.final_norm_layer is not None: output = self.final_norm_layer(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()
