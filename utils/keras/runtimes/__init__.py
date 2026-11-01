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

import os
import json

from .runtime import Runtime
from .hf_runtime import HFRuntime
from .keras_runtime import KerasRuntime
from .onnx_runtime import ONNXRuntime
from .tensorrt_runtime import TensorRTRuntime
from .saved_model_runtime import SavedModelRuntime
from .tensorrt_llm_runtime import TensorRTLLMRuntime
from .tensorrt_llm_bert_runtime import TensorRTLLMBertRuntime

def build_runtime(runtime, path = None, * args, ** kwargs):
    if runtime not in _runtimes:
        raise ValueError('Unsupported runtime !\n  Accepted : {}\n  Got : {}'.format(
            tuple(_runtimes.keys()), runtime
        ))

    if runtime == 'trt_llm' and path and os.path.exists(os.path.join(path, 'config.json')):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)

        # `pretrained_config` only exists in a compiled TRT-LLM engine config : an HF checkpoint
        # `config.json` (what `trt_llm_api` consumes) has none, hence the defensive lookups
        architecture = config.get('pretrained_config', {}).get('architecture', '')
        if 'bert' in architecture.lower():
            return TensorRTLLMBertRuntime(path, * args, ** kwargs)

    return _runtimes[runtime](path, * args, ** kwargs)

def _build_trt_llm_api_runtime(path = None, * args, encode_only = None, ** kwargs):
    """
        Lazy factory for the LLM API runtimes : importing their module pins the Open MPI TCP
        interfaces (which has to happen before `tensorrt_llm` / `mpi4py` are imported), a side
        effect the other runtimes should not pay for.

        An encoder-only checkpoint (embedding model) is served by `TensorRTLLMEncoderRuntime`
        (`LLM(encode_only = True)`), anything else by `TensorRTLLMAPIRuntime` (generation).
        `encode_only` forces the choice ; by default it is inferred from the already instantiated
        `engine`, else from the checkpoint architecture.
    """
    from .tensorrt_llm_api_runtime import TensorRTLLMAPIRuntime
    from .tensorrt_llm_encoder_runtime import TensorRTLLMEncoderRuntime, is_encoder_checkpoint

    if encode_only is None:
        engine = kwargs.get('engine', None)
        if engine is not None:
            encode_only = bool(getattr(engine, '_encode_only', False))
        else:
            encode_only = path is not None and is_encoder_checkpoint(path)

    if encode_only:
        return TensorRTLLMEncoderRuntime(path, * args, ** kwargs)
    return TensorRTLLMAPIRuntime(path, * args, ** kwargs)


_runtimes    = {
    'hf'    : HFRuntime,
    'keras' : KerasRuntime,
    'onnx'  : ONNXRuntime,
    'trt'   : TensorRTRuntime,
    'trt_llm'   : TensorRTLLMRuntime,
    # TRT-LLM through the new `LLM API` (pytorch backend) ; `trt_llm` is the legacy
    # `ModelRunnerCpp` path, removed from `tensorrt_llm` in 1.3
    'trt_llm_api'   : _build_trt_llm_api_runtime,
    'saved_model'   : SavedModelRuntime
}