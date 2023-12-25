
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

import tensorflow as tf

from custom_layers.multi_head_attention import MultiHeadAttention


class RotaryMultiHeadAttention(MultiHeadAttention):
    def __init__(self, max_position = 2048, base = 10000, ** kwargs):
        super().__init__(** kwargs)
        
        self.base   = base
        self.max_position   = max_position
        
        inv_freq    = 1. / self.base ** tf.cast(tf.range(0, self.depth, 2) / self.depth, tf.float32)
        self.inv_freq   = tf.Variable(
            inv_freq, dtype = tf.float32, trainable = False, name = 'inv_freq'
        )
        
        self.sin, self.cos = self._build_sin_cos(max_position)
    
    def _build_sin_cos(self, seq_len):
        t = tf.range(seq_len, dtype = tf.float32)
        
        freqs = tf.einsum('i,j->ij', t, self.inv_freq)
        emb = tf.concat([freqs, freqs], axis = 1)[tf.newaxis, tf.newaxis, :, :]
        return tf.math.sin(emb), tf.math.cos(emb)
    
    def apply_rotary_embedding(self, q, k, position_ids):
        seq_len = tf.shape(k)[-2]
        if position_ids is None: position_ids = tf.range(seq_len)

        sin = tf.gather(self.sin[0, 0, :seq_len], position_ids)[tf.newaxis, tf.newaxis, :, :]
        cos = tf.gather(self.cos[0, 0, :seq_len], position_ids)[tf.newaxis, tf.newaxis, :, :]
        
        sin = tf.cast(sin, q.dtype)
        cos = tf.cast(cos, q.dtype)
        
        q   = (q * cos) + (rotate_half(q) * sin)
        k   = (k * cos) + (rotate_half(k) * sin)
        return q, k

    def process_qkv(self, * args, ** kwargs):
        q, k, v = super().process_qkv(* args, ** kwargs)
        
        q, k = self.apply_rotary_embedding(q, k, None)
        
        return q, k, v

def rotate_half(x):
    return tf.concat([
        - x[..., tf.shape(x)[-1] // 2 :],
        x[..., : tf.shape(x)[-1] // 2]
    ], axis = -1)
