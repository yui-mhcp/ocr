
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

""" TF 2.0 BART model, compatible with the `transformers`' model. """

import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from custom_layers import FasterEmbedding, get_activation
from custom_architectures.transformers_arch.embedding_head import EmbeddingHead, HParamsEmbeddingHead
from custom_architectures.transformers_arch.bart_arch import *

HParamsMBartEncoder     = HParamsBartEncoder(
    normalize   = 'middle',
    mha_normalize   = False,
    mha_normalize_input = True,
    normalize_output    = True,
    
    epsilon = 1e-5,
    mha_epsilon = 1e-5,
    ffn_activation  = 'gelu'
)
HParamsMBartEmbedding   = HParamsMBartEncoder(** HParamsEmbeddingHead)

HParamsMBartDecoder  = HParamsBartDecoder(
    normalize   = 'middle',
    normalize_output    = True,
    mha_normalize   = False,
    mha_normalize_input = True,
    enc_mha_normalize   = False,
    enc_mha_normalize_input = True,
    
    epsilon = 1e-5,
    mha_epsilon = 1e-5,
    enc_mha_epsilon = 1e-5,

    ffn_activation  = 'gelu'
)

class MBartEncoder(BartEncoder):
    default_params = HParamsMBartEncoder

class MBartEmbedding(MBartEncoder):
    default_params = HParamsBartEmbedding
    
    def __init__(self, output_dim, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.embedding_head = EmbeddingHead(** self.hparams)

    def compute_output(self, output, training = False, mask = None, ** kwargs):
        output = super().compute_output(output, training = training)
        return self.embedding_head(output, mask = mask, training = training)

class MBartDecoder(BartDecoder):
    default_params = HParamsMBartDecoder
    
class MBart(Bart):
    encoder_class   = MBartEncoder
    decoder_class   = MBartDecoder

def transformers_mbart(name = 'moussaKam/barthez', task = 'generation'):
    import transformers
    if task == 'generation':
        return transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))

_mbart_classes   = {
    'MBartEncoder'   : MBartEncoder,
    'MBartEmbedding' : MBartEmbedding,
    'MBartDecoder'   : MBartDecoder,
    'MBart'          : MBart,
    'BarthezEncoder'   : MBartEncoder,
    'BarthezEmbedding' : MBartEmbedding,
    'BarthezDecoder'   : MBartDecoder,
    'Barthez'          : MBart
}
        
custom_functions    = {
    ** _mbart_classes,
    'transformers_bart' : transformers_bart
}

custom_objects  = {
    ** _mbart_classes
}

_encoders   = {'Barthez' : MBartEmbedding}
_decoders   = {'Barthez' : MBartDecoder}
_transformers   = {'Barthez' : MBart}