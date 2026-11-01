# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras
import keras.ops as K

from keras import tree
from keras.src.losses.loss import reduce_weighted_values
from keras.src.backend.common.masking import get_keras_mask

class LossWithMultipleOutputs(keras.losses.Loss):
    @property
    def output_names(self):
        raise NotImplementedError()
    
    def __call__(self, y_true, y_pred, sample_weight = None):
        """
            This function is adapted from the keras 3.1.1 official repository
            https://github.com/keras-team/keras/blob/master/keras/losses/loss.py
            
            This redefines the reduction strategy to enable subclasses to output multiple losses
        """
        # adapted from https://github.com/keras-team/keras/blob/master/keras/losses/loss.py v3.1.1
        in_mask = get_keras_mask(y_pred)

        with keras.name_scope(self.name):
            y_pred = tree.map_structure(
                lambda x: K.convert_to_tensor(x, dtype = self.dtype), y_pred
            )
            y_true = tree.map_structure(
                lambda x: K.convert_to_tensor(x, dtype = self.dtype), y_true
            )

            losses = self.call(y_true, y_pred)
            if isinstance(losses, (list, tuple)):
                losses = {name : loss for name, loss in zip(self.output_names, losses)}
                
            if isinstance(losses, dict):
                return {
                    k : self.reduce(loss, sample_weight, in_mask) for k, loss in losses.items()
                }
            else:
                return self.reduce(losses, sample_weight, in_mask)
    
    def reduce(self, loss, sample_weight = None, in_mask = None):
        with keras.name_scope('reduction'):
            out_mask = get_keras_mask(loss)

            if in_mask is not None and out_mask is not None:
                mask = in_mask & out_mask
            elif in_mask is not None:
                mask = in_mask
            elif out_mask is not None:
                mask = out_mask
            else:
                mask = None

            return reduce_weighted_values(
                loss,
                mask    = mask,
                sample_weight   = sample_weight,
                reduction   = self.reduction,
                dtype       = self.dtype
            )