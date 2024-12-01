# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import inspect
import logging
import collections

from .runtime import Runtime
from loggers import timer
from utils import time_to_string, args_to_str
from utils.keras_utils import ops

TRTLLMInferenceOutput = collections.namedtuple(
    "TRTLLMInferenceOutput", [
        "tokens", "lengths", "offset"
    ]
)

logger = logging.getLogger(__name__)

_default_enc_dec_config = {
    'max_input_len' : 32,
    'max_output_len'    : 512,
    'max_batch_size'    : 16,
    'max_beam_width'    : 3
}



class TensorRTLLMRuntime(Runtime):
    def __init__(self, * args, ** kwargs):
        super().__init__(* args, ** kwargs)

        self.is_enc_dec = 'encoder' in os.listdir(self.path)
        
        self.infer_signature    = inspect.signature(self.engine.generate).parameters.keys()
        
        self.sos_token  = -1
        self.eos_token  = -1
        self.pad_token  = -1
    
    def set_tokens(self, sos_token = None, eos_token = None, pad_token = None):
        if sos_token not in (-1, None): self.sos_token = sos_token
        if eos_token not in (-1, None): self.eos_token = eos_token
        if pad_token not in (-1, None): self.pad_token = pad_token

    @timer(name = 'TRT-LLM inference')
    def __call__(self,
                 inputs,
                 *,
                 
                 tokens = None,
                 max_input_len  = None,
                 stream_callback    = None,
                 encoder_output_lengths = None,

                 ** kwargs
                ):
        if max_input_len: self.engine.max_input_len = max_input_len
        
        inputs = self.prepare_tensor(
            inputs, self.is_enc_dec, pad_token = self.pad_token, dtype = self.engine.dtype
        )
        if tokens is not None:
            tokens = self.prepare_tensor(tokens, pad_token = self.pad_token)
        
        if 'kwargs' not in self.infer_signature:
            kwargs = {k : v for k, v in kwargs.items() if k in self.infer_signature}
        else:
            kwargs = {k : v for k, v in kwargs.items() if 'prompt' not in k and 'format' not in k}
        
        kwargs.update({
            'end_id'    : self.eos_token,
            'pad_id'    : self.pad_token,
            'streaming' : stream_callback is not None,
            'return_dict'   : True,
            'output_sequence_lengths'   : True
        })
        if self.is_enc_dec:
            kwargs['batch_input_ids'] = tokens
            kwargs['encoder_input_features' if ops.is_float(inputs[0]) else 'encoder_input_ids'] = inputs
            if encoder_output_lengths:
                if not isinstance(encoder_output_lengths, list):
                    encoder_output_lengths = [encoder_output_lengths] * len(inputs)
                kwargs['encoder_output_lengths'] = encoder_output_lengths
        else:
            kwargs['batch_input_ids'] = inputs
        
        inp_lengths = [tok.size(0) for tok in kwargs['batch_input_ids']]
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Calling `TRT-LLM` generate with {}'.format(args_to_str(kwargs)))
        t0 = time.time()
        output = self.engine.generate(** kwargs)
        
        if stream_callback is not None:
            stream = output
            for output in stream:
                stream_callback(TRTLLMInferenceOutput(
                    tokens  = output['output_ids'],
                    lengths = output['sequence_lengths'],
                    offset  = inp_lengths
                ))
        
        if logger.isEnabledFor(logging.INFO):
            n = output['sequence_lengths'].sum().cpu().numpy() - sum(inp_lengths)
            t1 = time.time()
            logger.info('{} tokens generated in {} ({:.3f} tokens/sec)'.format(
                n, time_to_string(t1 - t0), n / (t1 - t0)
            ))
        
        return TRTLLMInferenceOutput(
            tokens  = output['output_ids'],
            lengths = output['sequence_lengths'],
            offset  = inp_lengths
        )
    
    @staticmethod
    def prepare_tensor(tensor, is_encoder_input = False, pad_token = -1, dtype = None):
        if not is_encoder_input: dtype = 'int32'
        
        if not isinstance(tensor, list):
            tensor = ops.convert_to_torch_tensor(tensor)

            # batched rank is equal to 3 for encoder features (like whisper)
            batched_rank = 2 + int(is_encoder_input and ops.is_float(tensor))
            if len(tensor.shape) == batched_rank:
                if len(tensor) == 1:
                    tensor = [tensor[0]]
                elif not is_encoder_input:
                    tensor = [
                        inp[:length] for length in (tensor != pad_token).count_nonzero(1)
                    ]
            else:
                tensor = [tensor]
            
        elif isinstance(tensor[0], int):
            tensor = [ops.convert_to_torch_tensor(tensor, dtype = 'int32')]
        else:
            tensor = [ops.convert_to_torch_tensor(t, dtype = dtype) for t in tensor]
        
        if is_encoder_input and dtype and any(ops.is_float(t) for t in tensor):
            tensor = [t.to(dtype = dtype) for t in tensor]
        
        return tensor

    @staticmethod
    def load_engine(path, *, use_cpp = True, kv_cache_free_gpu_memory_fraction = 0.3, ** kwargs):
        from tensorrt_llm.runtime import ModelRunnerCpp, ModelRunner

        if 'encoder' in os.listdir(path):
            kwargs['is_enc_dec'] = True
            for k, v in _default_enc_dec_config.items():
                if k not in kwargs: kwargs[k] = v

        if use_cpp: kwargs['kv_cache_free_gpu_memory_fraction'] = kv_cache_free_gpu_memory_fraction

        runner_cls = ModelRunnerCpp if use_cpp else ModelRunner
        kwargs     = {
            k : v for k, v in kwargs.items()
            if k in inspect.signature(runner_cls.from_dir).parameters
        }
        return runner_cls.from_dir(engine_dir = path, ** kwargs)
