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

""" `TensorRTLLMAPIRuntime.max_input_length` : the prompt bound the LLM API actually enforces, per
backend (no `tensorrt_llm` import, the engine arguments are stubbed). """

from types import SimpleNamespace

import pytest

from utils.keras.runtimes.tensorrt_llm_api_runtime import TensorRTLLMAPIRuntime, _backend


def _max_input_length(** args):
    runtime = SimpleNamespace(_args = SimpleNamespace(** args))
    return TensorRTLLMAPIRuntime.max_input_length.fget(runtime)


class TorchLlmArgs(SimpleNamespace):
    """ Stands for the `TorchLlmArgs` of a version without the `backend` field """


def test_pytorch_backend_ignores_the_unchecked_max_input_len():
    # the TRT-LLM default `max_input_len = 1024` is never checked by the pytorch backend : it must
    # not cap the prompt budget (it used to, silently)
    assert _max_input_length(backend = 'pytorch', max_input_len = 1024, max_num_tokens = 4096) == 4096

def test_chunked_prefill_lifts_the_max_num_tokens_bound():
    # the prompt is prefilled in chunks : only `max_seq_len` bounds it
    assert _max_input_length(
        backend = 'pytorch', max_input_len = 1024, max_num_tokens = 4096, max_seq_len = 12288, enable_chunked_prefill = True
    ) == 12288
    assert _max_input_length(backend = 'pytorch', max_input_len = 1024, max_num_tokens = 4096, enable_chunked_prefill = True) is None

def test_max_seq_len():
    assert TensorRTLLMAPIRuntime.max_seq_len.fget(SimpleNamespace(_args = SimpleNamespace(max_seq_len = 12288))) == 12288
    assert TensorRTLLMAPIRuntime.max_seq_len.fget(SimpleNamespace(_args = SimpleNamespace())) is None
    assert TensorRTLLMAPIRuntime.max_seq_len.fget(SimpleNamespace(_args = None)) is None

@pytest.mark.parametrize('backend', [None, '_autodeploy'])
def test_other_backends_keep_max_input_len(backend):
    """ The legacy TRT engines (`TrtLlmArgs`, no `backend`) and `_autodeploy` check `max_input_len` """
    assert _max_input_length(backend = backend, max_input_len = 1024, max_num_tokens = 4096) == 1024
    assert _max_input_length(backend = backend, max_input_len = 8192, max_num_tokens = 4096) == 4096
    assert _max_input_length(
        backend = backend, max_input_len = 1024, max_num_tokens = 4096, max_seq_len = 12288, enable_chunked_prefill = True
    ) == 1024

def test_backend_resolution():
    assert _backend(SimpleNamespace(backend = 'pytorch')) == 'pytorch'
    assert _backend(TorchLlmArgs(backend = None)) == 'pytorch'
    assert _backend(SimpleNamespace()) is None

def test_no_engine():
    assert TensorRTLLMAPIRuntime.max_input_length.fget(SimpleNamespace(_args = None)) is None

def test_min_new_tokens_is_the_min_tokens_of_the_llm_api():
    from utils.keras.runtimes.tensorrt_llm_api_runtime import _rename_params
    assert _rename_params['min_new_tokens'] == 'min_tokens' and _rename_params['max_new_tokens'] == 'max_tokens'

def test_logits_processors_is_the_logits_processor_of_the_llm_api():
    from utils.keras.runtimes.tensorrt_llm_api_runtime import _rename_params
    assert _rename_params['logits_processors'] == 'logits_processor'

def test_audio_encoder_decoder_detection():
    from utils.keras.runtimes.tensorrt_llm_api_runtime import _is_audio_encoder_decoder
    assert _is_audio_encoder_decoder(SimpleNamespace(model_type = 'whisper', is_encoder_decoder = True))
    assert not _is_audio_encoder_decoder(SimpleNamespace(model_type = 't5', is_encoder_decoder = True))
    assert not _is_audio_encoder_decoder(SimpleNamespace())

def _audio_runtime():
    """ A whisper-like engine : the decoded prompt spells the ids, at 16 kHz """
    return SimpleNamespace(engine = SimpleNamespace(
        tokenizer   = SimpleNamespace(decode = lambda ids, skip_special_tokens = True: ''.join(
            '<|{}|>'.format(i) for i in ids
        )),
        input_processor = SimpleNamespace(sampling_rate = 16000)
    ))

def test_audio_prompts_carry_the_raw_audio_and_the_text_decoder_prompt():
    import numpy as np

    audios  = np.ones((2, 100, 1), dtype = 'float64')
    prompts = TensorRTLLMAPIRuntime._prepare_audio_prompts(_audio_runtime(), audios, [[1, 2, 3], [1, 4, 3]])

    assert [p['prompt'] for p in prompts] == ['<|1|><|2|><|3|>', '<|1|><|4|><|3|>']
    for prompt in prompts:
        (audio, rate), = prompt['multi_modal_data']['audio']
        assert rate == 16000 and audio.shape == (100, ) and audio.dtype == np.float32

def test_audio_prompts_default_to_the_checkpoint_prompt():
    import numpy as np

    prompts = TensorRTLLMAPIRuntime._prepare_audio_prompts(_audio_runtime(), np.zeros((50, ), 'float32'))
    assert len(prompts) == 1 and prompts[0]['prompt'] == ''

def test_audio_prompts_need_one_decoder_prompt_per_audio():
    import numpy as np

    with pytest.raises(AssertionError):
        TensorRTLLMAPIRuntime._prepare_audio_prompts(_audio_runtime(), np.zeros((2, 50)), [[1, 2]])

def test_kv_budget_fraction_discounts_the_weights(tmp_path, monkeypatch):
    from utils.keras import gpu
    from utils.keras.runtimes.tensorrt_llm_api_runtime import _kv_budget_fraction

    (tmp_path / 'model.safetensors').write_bytes(b'\0' * 1000)
    # a legacy copy of the weights next to the safetensors is not loaded : it must not be discounted
    (tmp_path / 'original_model.pt').write_bytes(b'\0' * 5000)
    monkeypatch.setattr(gpu, 'get_gpu_memory_infos', lambda gpu = 0: {'free' : 11000})

    assert _kv_budget_fraction(2500, str(tmp_path)) == 0.25
    assert _kv_budget_fraction(50000, str(tmp_path)) == 1.
