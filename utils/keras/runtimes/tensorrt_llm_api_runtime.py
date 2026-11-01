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
import re
import glob
import json
import time
import hashlib
import logging
import threading
import numpy as np

# Open MPI excludes the loopback from its default TCP BTL/OOB interface set and keeps only
# routable NIC(s). When TRT-LLM's LLM API spawns its worker (MpiPoolSession, used whenever the
# script is not launched under mpirun — always the case here, even at tp=1 with the pytorch
# backend), the parent<->worker TCP wire-up then fails to find a reachable interface pairing on
# a single-node host, crashing with "Unable to find reachable pairing between local and remote
# interfaces" -> MPI_ERR_INTERN. Pinning both the BTL and OOB TCP to `lo` is safe and sufficient
# here (single node, 1 GPU): the spawned worker lives on the same machine. `setdefault` keeps any
# user-provided override (e.g. a real NIC for a future multi-node setup). Must run before any
# `tensorrt_llm`/`mpi4py` import (mpi4py runs `MPI_Init` at import) — this module is loaded
# lazily when the runtime is selected, hence before `load_engine`/`LLM()`.
# Note : `load_engine(use_single_process = True)` removes the MPI question altogether (the engine
# then runs in-process), which is the friendlier mode for notebooks / debugging.
for _mpi_var in ('OMPI_MCA_btl_tcp_if_include', 'OMPI_MCA_oob_tcp_if_include'):
    os.environ.setdefault(_mpi_var, 'lo')

from ...generic_utils import time_to_string, resolve_hf_snapshot
from .. import Timer, timer
from .runtime import Runtime

logging.getLogger('tensorrt_llm').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

_default_kv_cache_free_gpu_memory_fraction  = 0.25

# Recurrent-state snapshot policy applied to a **hybrid** model (gated-delta-net / mamba layers,
# e.g. qwen3.5 / qwen3.8) loaded with block reuse and no policy of its own (see `load_engine`).
# Without any snapshot the recurrent state of a prefix cannot be restored : TRT-LLM then turns
# block reuse off by itself (`_validate_and_adjust_mamba_snapshot_config`, a warning the
# `tensorrt_llm` log level hides). The two rules cover the two reuse patterns of a chat model :
#   - `additional_snapshot_offsets_from_end = [0]` : one snapshot at the end of every prompt, so
#     that a prompt extending the previous one (the next pass of a tool-calling turn) resumes
#     right where the previous pass started generating. Free : the end of the prompt is a chunk
#     boundary anyway ;
#   - `periodic_snapshot_interval` : a snapshot every N tokens, so that a prefix shared up to an
#     arbitrary point (same system prompt / document, different question) is reused up to the
#     last multiple of N before the divergence.
# Every other snapshot point **splits the prefill** : the context phase stops at each one, and a
# context iteration has a high fixed cost (~39 ms + 0.03 ms / token on qwen3.8-27b, RTX 5090,
# against 12.6 ms for a generation step). A snapshot only pays when it later saves ~1k prefill
# tokens or more. `enable_branch_snapshot` (a snapshot where each request leaves the cached prefix)
# is left out for that reason : it splits every request that diverges, however short the saved
# prefix (+16 ms per classifier call for ~50 reused tokens, +12 ms per chat turn).
# Each snapshot holds the whole recurrent state (~75 MiB per snapshot in bfloat16 on
# qwen3.8-27b), taken from the recurrent pool of the KV cache : the periodic ones are evictable,
# the end-of-prompt one is kept alive with its request. That pool is small by default (see
# `load_engine`, note on the pool split) : evicted snapshots survive in the host tier
# (`kv_cache_host_memory`), from where they are reloaded instead of recomputed.
_default_hybrid_state_config    = {
    'periodic_snapshot_interval'    : 1024,
    'additional_snapshot_offsets_from_end'  : [0]
}
# the only rule the V1 (`CppMambaHybridCacheManager`) manager supports
_v1_hybrid_state_config     = {'periodic_snapshot_interval' : 1024}

# legacy runtime kwargs (`ModelRunnerCpp` flavor) -> LLM API `SamplingParams` fields
_rename_params  = {
    'max_new_tokens'    : 'max_tokens',
    # forbids `end_id` (only) before this length, see `_add_min_length_bans`
    'min_new_tokens'    : 'min_tokens',
    'random_seed'   : 'seed',
    'num_return_sequences'  : 'n',
    # one callback or a list of them. /!\ The `SamplingParams` of a call is shared by every request
    # of its batch : a *stateful* processor (e.g. `WhisperTimestampLogitsProcessor`) is only valid
    # for a single-request call (the worker process unpickles one copy per request, the in-process
    # mode does not)
    'logits_processors' : 'logits_processor'
}

# weight file patterns, used to tell a *complete* HF snapshot from a config-only one
_weight_patterns    = ('*.safetensors', '*.bin', '*.pth', '*.pt')

# any one of them is enough for `LLM` to build its tokenizer (see `_validate_checkpoint`)
_tokenizer_files    = ('tokenizer.json', 'tokenizer_config.json', 'tokenizer.model', 'vocab.json')

# any one of them is enough for `AutoProcessor` to build the image processor of a VLM
_processor_files    = ('preprocessor_config.json', 'processor_config.json')

# weight name suffixes betraying a quantized checkpoint (modelopt / compressed-tensors), used to
# detect a checkpoint whose quantization config did not make it to the local snapshot
_quantized_suffixes = ('weight_scale', 'input_scale', 'weight_scale_inv', 'weight_scale_2')

# Additive mask value for `allowed_tokens` (see `_build_embedding_bias`). A large *finite*
# negative is used rather than `-inf` : the bias is applied before the penalties / temperature
# scaling and before any logprob computation, where `-inf` can turn into a `NaN`. In float32,
# `exp(logit - 1e9)` underflows to 0, so the token is excluded just as firmly.
_mask_value = -1e9

class TensorRTLLMAPIRuntime(Runtime):
    """
        TRT-LLM runtime based on the **LLM API** (pytorch backend, the only one left as of
        `tensorrt_llm >= 1.3`), exposing the exact same interface as the legacy
        `TensorRTLLMRuntime` (`ModelRunnerCpp`) so that both can be swapped and benchmarked
        without touching `BaseLLM` / `RequestController`.

        Main differences with the legacy runtime :
            - `path` is a **model name** (HF repo id) or a checkpoint directory, not a compiled
              engine directory : there is no `trtllm-build` step any more. A repo id already
              fully cached locally is resolved to its snapshot (no re-download), anything else
              is left untouched for the LLM API to fetch
            - the generation controls are translated to their *native* LLM API counterparts
              rather than to a python `logits_post_processor` (see `prepare_sampling_params`) :
                * `allowed_tokens`  -> `SamplingParams.embedding_bias`, an additive vocab-sized
                  mask applied on-GPU by the sampler
                * `banned_tokens`   -> the same mask, the other way around (the given ids are
                  never generated, e.g. the opening of a tool call once the tool rounds are spent)
                * `stop_words`      -> `SamplingParams.stop`, enforced in-engine
                * `stop_condition`  -> evaluated **host-side** on the decoded text, aborting the
                  request on match (the runtime streams internally when the caller asked for a
                  non-streaming generation)
                * `logits_processors`   -> `SamplingParams.logits_processor`
                * `thinking_budget` -> the native `ThinkingBudgetLogitsProcessor`, forcing the
                  end of the reasoning block once the budget is spent
              This matters because the LLM API runs the engine in a *separate process* by
              default : a `logits_post_processor` closure would be pickled — along with the
              `Tokenizer` it captures — on every single request
            - the only encoder-decoder models supported are the **audio** ones (whisper) : the
              inputs are then the raw audios and `tokens` the decoder prompts (see
              `prepare_inputs`), the log-mel spectrogram being computed by the engine itself
    """
    # generation capabilities (see `Runtime`)
    supports_streaming  = True
    supports_guided_decoding    = True
    supports_stop_condition     = True
    supports_stop_words         = True
    supports_thinking_budget    = True
    supports_banned_tokens      = True

    def __init__(self, path, ** kwargs):
        super().__init__(path, ** kwargs)

        self._checkpoint    = None
        self._hf_config     = None
        self._generation_config = None

        # the `last_*` values are per thread : the memory workers call the runtime concurrently
        # with the conversation, and a value describing another thread's call is worse than none
        self._local = threading.local()

        # perf metrics of the **last** call, when it asked for them (see `_summarize_perf_metrics`)
        self.last_perf_metrics  = None
        # `(cached, prompt)` token counts of each request of the **last** completed call, i.e. how
        # much of each prompt was reused from the KV cache (see `_record_kv_reuse`)
        self.last_cached_tokens = None

        # `callable(record)` called once per completed request with its profiling record (see
        # `_perf_records`) ; `None` (default) = no profiling (the engine is then not asked for
        # the per-request perf metrics)
        self.perf_recorder  = None

        self.sos_token  = -1
        self.eos_token  = -1
        self.pad_token  = -1

    @property
    def _args(self):
        """ The engine's `TorchLlmArgs` (the resolved LLM API configuration) """
        return getattr(self.engine, 'args', None)

    @property
    def last_perf_metrics(self):
        return getattr(self._local, 'perf_metrics', None)

    @last_perf_metrics.setter
    def last_perf_metrics(self, value):
        self._local.perf_metrics = value

    @property
    def last_cached_tokens(self):
        return getattr(self._local, 'cached_tokens', None)

    @last_cached_tokens.setter
    def last_cached_tokens(self, value):
        self._local.cached_tokens = value

    @property
    def checkpoint(self):
        """
            Local checkpoint directory of the model, resolved on demand.

            The engine's own `_hf_model_dir` is authoritative (it is what the `LLM` actually
            loaded) and `path` is only the fallback : the engine may come from the shared
            `Runtime._engines` cache, in which case `load_engine` never ran here.

            Lazy so that a runtime built around an already instantiated engine
            (`build_runtime(..., engine = ...)`, hence `path = None`) still constructs.
        """
        if self._checkpoint is None:
            self._checkpoint = _resolve_hf_checkpoint(
                getattr(self.engine, '_hf_model_dir', None) or self.path
            )
        return self._checkpoint

    @property
    def hf_config(self):
        """ The checkpoint's `transformers` config (`base_dtype`, `vocab_size`, modalities, ...) """
        if self._hf_config is None:
            import transformers

            self._hf_config = transformers.AutoConfig.from_pretrained(self.checkpoint)
        return self._hf_config

    @property
    def generation_config(self):
        """
            The checkpoint's `generation_config.json`, i.e. the **authoritative** list of end ids.

            Kept separate from `hf_config` : `config.json:eos_token_id` is the config of the base
            model, and a chat fine-tune only updates the generation config (llama-3.1-instruct
            adds `<|eot_id|>` there and nowhere else).
        """
        if self._generation_config is None:
            import transformers

            try:
                self._generation_config = transformers.GenerationConfig.from_pretrained(
                    self.checkpoint
                )
            except Exception as e:
                # not every checkpoint ships one : the caller-provided tokenizer then remains
                # the only source of end ids (see `end_ids`)
                logger.info('No usable generation config for `{}` : {}'.format(self.path, e))
                self._generation_config = transformers.GenerationConfig()

        return self._generation_config

    @property
    def end_ids(self):
        """
            Every id that must end the generation, the caller's own one first.

            `SamplingParams.end_id` is a single `int` while a chat checkpoint declares several
            end ids (`<|eot_id|>` **and** `<|end_of_text|>` for llama-3.1-instruct) : the
            remaining ones are passed as `stop_token_ids`, which `_get_stop_words` cumulates
            with `end_id` — and which, unlike `stop`, needs no tokenizer.

            Stopping on a strict subset is not a cosmetic detail : the generation then runs to
            `max_tokens` on every request, which inflates the latency and silently ruins any
            tokens/sec comparison.

            /!\\ `SamplingParams._setup` (the LLM API fallback) reads `tokenizer.eos_token_id`
            alone — the generation config is only consulted for `kimi_k2` — so this is *more*
            complete than letting the LLM API resolve the end ids itself.
        """
        ids = [self.eos_token] if self.eos_token != -1 else []

        eos = getattr(self.generation_config, 'eos_token_id', None)
        if eos is not None:
            if not isinstance(eos, (list, tuple)): eos = [eos]
            ids.extend(int(token) for token in eos)

        return list(dict.fromkeys(ids))

    @property
    def is_multimodal(self):
        return hasattr(self.hf_config, 'vision_config')

    @property
    def is_audio_encoder_decoder(self):
        """ Whether the model is an audio encoder-decoder (whisper), see `prepare_inputs` """
        return _is_audio_encoder_decoder(self.hf_config)

    @property
    def supports_multimodal_data(self):
        """
            A property rather than the class flag of `Runtime` : the LLM API consumes
            `multi_modal_data` for any VLM it supports (its input processor expands the
            placeholders and runs the vision tower), but a text-only checkpoint has nothing to
            feed it to
        """
        return self.is_multimodal

    @property
    def base_dtype(self):
        """ Native float precision of the checkpoint (the pytorch backend runs the model in it) """
        # `torch_dtype` was renamed `dtype` in transformers 5 (the old name is kept as a
        # deprecated alias, and dropped from some configs) : both are read, newest first
        config  = self.hf_config
        dtype   = getattr(config, 'dtype', None) or getattr(config, 'torch_dtype', None)
        return str(dtype).rsplit('.', 1)[-1] if dtype is not None else 'float32'

    @property
    def max_input_length(self):
        """
            Effective input token budget, i.e. the bound a prompt is actually checked against.

            On the pytorch backend that bound is `max_num_tokens` (`LLM._check_arguments`
            rejects any prompt longer than it unless chunked prefill is enabled) ; `max_input_len`
            is inherited from the TRT builder, defaults to 1024 and is **not** checked there :
            folding it in would silently cap every prompt budget at 1024 tokens. It only bounds
            the prompt on the other backends (the legacy `tensorrt` engines of the LLM API before
            1.3, `_autodeploy`), where it is kept. The smallest of the applicable bounds is the
            only safe answer for `BaseLLM._resolve_max_input_length`.

            With chunked prefill, a prompt is prefilled in chunks of `max_num_tokens` : its only
            bound is then `max_seq_len`. Note : `max_seq_len` bounds *prompt + generation* (see
            `max_seq_len`) — the generation budget is the caller's business
            (`BaseLLM._resolve_max_input_length` subtracts it).
        """
        args    = self._args
        if args is None: return None

        lengths = []
        if _backend(args) != 'pytorch':
            lengths.append(getattr(args, 'max_input_len', None))
        if getattr(args, 'enable_chunked_prefill', False):
            lengths.append(getattr(args, 'max_seq_len', None))
        else:
            lengths.append(getattr(args, 'max_num_tokens', None))

        lengths = [length for length in lengths if length]
        return min(lengths) if lengths else None

    @property
    def max_seq_len(self):
        """ The bound of *prompt + generation* of a request (`None` : unknown) """
        args = self._args
        if args is None: return None
        return getattr(args, 'max_seq_len', None) or None

    def set_tokens(self, sos_token = None, eos_token = None, pad_token = None):
        if sos_token not in (-1, None): self.sos_token = sos_token
        if eos_token not in (-1, None): self.eos_token = eos_token
        if pad_token not in (-1, None): self.pad_token = pad_token

    def release(self):
        """
            Shuts the engine down, on top of the cache eviction performed by `Runtime.release`.

            Unlike the legacy runtime, an `LLM` owns a **worker process** (and its share of the
            VRAM) : dropping the last reference to it is not enough, the process has to be told
            to exit.
        """
        engine = super().release()
        if engine is None: return

        if hasattr(engine, 'shutdown'):
            engine.shutdown()
            logger.info('Engine `{}` has been released'.format(self.path))

    @timer(name = 'TRT-LLM API inference')
    def __call__(self,
                 inputs,
                 *,

                 tokens     = None,
                 streaming  = False,
                 num_beams  = None,

                 tokenizer  = None,
                 stop_words = None,
                 stop_condition = None,
                 allowed_tokens = None,
                 banned_tokens  = None,
                 thinking_budget    = None,
                 reasoning_token_ids    = None,

                 ** kwargs
                ):
        # the profiling context of the call (see `_perf_records`) : the caller kwargs names tell
        # the call sites apart (e.g. `classifier_question`, `writer_index`)
        perf = {
            'enter' : _steady_now(), 'keys' : tuple(sorted(kwargs)), 'streaming' : streaming
        } if self.perf_recorder is not None else None

        # the end ids come from the *caller's* tokenizer (the one that encoded the prompt),
        # exactly like the legacy runtime does : the LLM API would otherwise have to resolve
        # them from its own tokenizer, which does not exist when the checkpoint ships none
        if tokenizer is not None and self.eos_token in (None, -1):
            self.set_tokens(
                sos_token = tokenizer.sos_token_idx,
                eos_token = tokenizer.eos_token_idx,
                pad_token = tokenizer.blank_token_idx
            )

        # dropped up front : a stale value from the previous call, read as if it described this
        # one, is worse than no value at all
        self.last_perf_metrics  = None
        self.last_cached_tokens = None

        with Timer('prepare inputs'):
            prompts = self.prepare_inputs(inputs, tokens = tokens, ** kwargs)

        if tokens is not None and kwargs.get('max_new_tokens', None) and self.is_audio_encoder_decoder:
            # the decoder position table bounds prompt + answer (448 rows for whisper) : the input
            # processor caps `max_tokens` too, but with a warning on every single request
            budget  = self.hf_config.max_target_positions - max(
                len(tok) for tok in _normalize_batch_tokens(tokens)
            )
            kwargs['max_new_tokens'] = min(kwargs['max_new_tokens'], budget)

        with Timer('sampling params'):
            sampling_params = self.prepare_sampling_params(
                num_beams   = num_beams,
                stop_words  = stop_words,
                allowed_tokens  = allowed_tokens,
                banned_tokens   = banned_tokens,
                thinking_budget = thinking_budget,
                reasoning_token_ids = reasoning_token_ids,
                ** kwargs
            )
        # `stop_condition` is an arbitrary predicate on the decoded text : it has no native
        # equivalent and is therefore enforced here, on the *cumulated* output ids of each
        # request (see `_StopWatcher`)
        stop_watcher    = _StopWatcher(
            stop_condition, tokenizer, len(prompts)
        ) if stop_condition is not None else None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Calling `TRT-LLM API` inference with lengths {} and {}'.format(
                # an audio request carries a text decoder prompt (see `_prepare_audio_prompts`)
                [
                    len(prompt['prompt_token_ids']) if 'prompt_token_ids' in prompt else prompt['prompt']
                    for prompt in prompts
                ], sampling_params
            ))

        if streaming:
            return self._generate_stream(prompts, sampling_params, stop_watcher, perf = perf)

        t0 = time.perf_counter()
        if stop_watcher is not None:
            # the condition can only be evaluated on generated text : the generation is streamed
            # internally and the (cumulated) result is returned as if it had not been
            if perf is not None: perf['streaming'] = 'internal'
            with Timer('engine generate (streamed)'):
                stream = self._generate_stream(prompts, sampling_params, stop_watcher, perf = perf)
                result = stream.consume()
            self.last_perf_metrics = stream.perf_metrics
        else:
            with Timer('engine generate'):
                if perf is not None: perf['submit'] = _steady_now()
                outputs = self.engine.generate(
                    prompts, sampling_params = sampling_params, use_tqdm = False
                )
            self.last_perf_metrics = _summarize_perf_metrics(outputs)
            self._on_finish(outputs, perf)
            result = [[list(seq.token_ids) for seq in out.outputs] for out in outputs]

        if logger.isEnabledFor(logging.INFO):
            t1 = time.perf_counter()
            n  = sum(sum(len(beam) for beam in beams) for beams in result)
            logger.info('[TRT-LLM API] {} tokens generated in {} ({:.3f} tokens/sec)'.format(
                n, time_to_string(t1 - t0), n / (t1 - t0)
            ))

        return result

    def _generate_stream(self, prompts, sampling_params, stop_watcher = None, *, perf = None):
        if perf is not None: perf['submit'] = _steady_now()
        return APIInferenceStream([
            self.engine.generate_async(
                prompt, sampling_params = sampling_params, streaming = True
            ) for prompt in prompts
        ], stop_watcher = stop_watcher, on_finish = lambda results: self._on_finish(results, perf))

    def _on_finish(self, results, perf = None):
        """ The bookkeeping of a completed call : the KV cache reuse, and the profiling records
            when a `perf_recorder` is set (`perf` : the profiling context of the call) """
        self._record_kv_reuse(results)
        if perf is None or self.perf_recorder is None: return

        done = _steady_now()
        for record in _perf_records(results, perf, done):
            try:
                self.perf_recorder(record)
            except Exception as e:
                # profiling must never break a generation
                logger.warning('The `perf_recorder` failed : {}'.format(e))

    def get_iteration_stats(self, timeout = 0.1):
        """
            Drains the per-iteration statistics of the engine (`LLM.get_stats`), a `list` of
            `dict` (one per engine iteration since the previous call : timings `iterLatencyMS` /
            `hostStepTimeMS` / `gpuForwardTimeMS`, KV cache pool usage and movements, and for a
            hybrid model the recurrent-state snapshot lookups / hits / reused tokens in
            `kvCacheIterationStatsByLifecycle`).

            Requires the engine to be loaded with `enable_iter_perf_stats = True` (`[]`
            otherwise). The engine buffers a bounded number of iterations : drain regularly
            (e.g. after each scenario of a benchmark)
        """
        if self.engine is None or not getattr(self._args, 'enable_iter_perf_stats', False):
            return []
        return list(self.engine.get_stats(timeout = timeout))

    def _record_kv_reuse(self, results):
        """
            Records (`last_cached_tokens`) and logs how many prompt tokens of each request were
            reused from the KV cache.

            `RequestOutput.cached_tokens` is set by the engine from the KV cache position the
            first context chunk started at, so it is the reuse **actually obtained** — not the
            attention match, which a hybrid model prunes back to its latest recurrent-state
            snapshot (see `_default_hybrid_state_config`). It is the only direct evidence that
            block reuse works : a TTFT gain can as well come from a warmer GPU.

            The prompt length is the engine's own (`prompt_token_ids` of the output), which
            includes the expansion of the multimodal placeholders.
        """
        counts = []
        for result in results:
            prompt = getattr(result, 'prompt_token_ids', None)
            counts.append((
                int(getattr(result, 'cached_tokens', 0) or 0),
                len(prompt) if prompt is not None else None
            ))

        self.last_cached_tokens = counts
        if logger.isEnabledFor(logging.INFO):
            logger.info('[TRT-LLM API] KV cache : {} prompt tokens reused'.format(
                ', '.join('{} / {}'.format(cached, prompt) for cached, prompt in counts)
            ))

    def prepare_inputs(self, inputs, *, tokens = None, prompt_table = None, ** kwargs):
        """
            Returns a `list` of `TokensPrompt` dicts, one per request of the batch.

            For an audio encoder-decoder (whisper), `inputs` are the raw audios and `tokens` the
            decoder prompts instead (see `_prepare_audio_prompts`).

            The multimodal inputs are forwarded through the LLM API's own keys, so that
            TRT-LLM runs the vision tower itself (it then benefits from the encoder embedding
            cache, the encoder side-stream and the image-hash-based KV cache reuse, none of
            which a caller-computed `prompt_table` can use) :
                - `multi_modal_data`    : `{'image' : [...], ...}`, the raw items
                - `multi_modal_uuids`   : `{'image' : ['<stable key>', ...]}`, the caller's own
                  cache identifiers (e.g. an image filename), which replace the legacy
                  `input_token_extra_ids` trick
        """
        if self.is_audio_encoder_decoder:
            return self._prepare_audio_prompts(inputs, tokens)

        if prompt_table is not None:
            # `multi_modal_embeddings` (the LLM API's externally-computed-embeddings entry) is
            # only implemented by a handful of input processors (LlavaNext, Qwen2-VL, Llama 4) ;
            # `_attach_multimodal_embeddings_impl` raises for every other one, Mistral 3
            # included. Failing here, with the alternative spelled out, beats a `NotImplementedError`
            # coming out of the worker process on the first request
            raise NotImplementedError(
                'Pre-computed multimodal embeddings (`prompt_table`) are not supported by the '
                'LLM API for this model : pass the raw items in `multi_modal_data` and let '
                'TRT-LLM encode them (see `prepare_inputs`)'
            )

        if 'input_token_extra_ids' in kwargs:
            # p-tuning artefact of the legacy runtime : multimodal KV cache reuse is now keyed
            # on the content hash of each item, or on `multi_modal_uuids` when provided
            logger.warning(
                '`input_token_extra_ids` has no LLM API equivalent and is ignored : use '
                '`multi_modal_uuids` to provide your own KV cache identifiers'
            )

        if kwargs.get('multi_modal_data', None) is not None and getattr(
                self.engine, 'tokenizer', None) is None:
            # VLMs without a token-id fast path (Qwen3.5 / 3.8, Mistral 3, ...) detokenize the
            # prompt with the engine tokenizer, then let the HF processor expand the placeholders :
            # without it the request fails in the worker, far from its cause
            raise ValueError(
                '`multi_modal_data` requires the LLM API tokenizer : load the runtime without '
                '`skip_tokenizer_init`'
            )

        prompts = [
            {'prompt_token_ids' : tok} for tok in _normalize_batch_tokens(inputs)
        ]

        for key in ('multi_modal_data', 'multi_modal_uuids', 'mm_processor_kwargs'):
            value = kwargs.get(key, None)
            if value is None: continue

            if not isinstance(value, list): value = [value] * len(prompts)
            assert len(value) == len(prompts), (
                'There should be exactly 1 `{}` per request, got {} for a batch of {}'.format(
                    key, len(value), len(prompts)
            ))
            for prompt, item in zip(prompts, value): prompt[key] = item

        return prompts

    def _prepare_audio_prompts(self, audios, tokens = None):
        """
            Returns the `TextPrompt` dicts of an audio encoder-decoder (whisper), one per request.

            Arguments :
                - audios    : the batch of raw audios at the model's sampling rate (`(batch,
                  samples)`, `(batch, samples, 1)` or a `list` of waveforms), at most one window
                  (30 sec) each : the input processor zero-pads them, and rejects longer ones
                - tokens    : the decoder prompts (one per audio), `None` for the checkpoint default
                  (/!\\ english transcription without timestamps)

            The input processor ignores `prompt_token_ids` : the decoder prompt is passed as
            **text**, which it re-tokenizes. The ids are thus decoded with their special tokens,
            which is lossless for the task prefixes (`<|startoftranscript|><|fr|><|transcribe|>`),
            made of special tokens only. The processor requires that text to start with
            `<|startoftranscript|>` : a `<|startofprev|>` conditioning is rejected at submission.
        """
        if not isinstance(audios, (list, tuple)) and len(np.shape(audios)) == 1: audios = [audios]
        audios  = [np.asarray(audio, dtype = np.float32).reshape(-1) for audio in audios]

        prompts = [''] * len(audios)
        if tokens is not None:
            tokens  = _normalize_batch_tokens(tokens)
            assert len(tokens) == len(audios), (
                'There should be exactly 1 decoder prompt per audio, got {} for {} audios'.format(
                    len(tokens), len(audios)
            ))
            prompts = [
                self.engine.tokenizer.decode(tok, skip_special_tokens = False) for tok in tokens
            ]

        rate    = self.engine.input_processor.sampling_rate
        return [
            {'prompt' : prompt, 'multi_modal_data' : {'audio' : [(audio, rate)]}}
            for prompt, audio in zip(prompts, audios)
        ]

    def prepare_sampling_params(self,
                                *,

                                num_beams   = None,
                                stop_words  = None,
                                allowed_tokens  = None,
                                banned_tokens   = None,
                                thinking_budget = None,
                                reasoning_token_ids = None,

                                ** kwargs
                               ):
        """
            Builds the (single) `SamplingParams` for the batch.

            A single instance is shared by every request of the call — which the LLM API
            explicitly supports — because, unlike the legacy `logits_processors`, none of the
            translated controls is stateful any more. Concurrent `__call__` (in-flight batching
            through `BaseModel.predict`) each build their own, so nothing is shared across
            requests that could diverge.
        """
        from tensorrt_llm import SamplingParams

        # `max_beam_width` is the *upper bound* configured on the engine, not a sensible
        # default : a single sequence is returned unless the caller explicitly asks for more
        if num_beams is None: num_beams = 1

        config  = {_rename_params.get(k, k) : v for k, v in kwargs.items()}

        # the legacy runtime spells this one the other way around
        if 'include_input_in_output' in config:
            config['exclude_input_from_output'] = not config.pop('include_input_in_output')

        argnames    = _get_sampling_argnames()
        dropped     = [k for k in config if k not in argnames]
        if dropped and logger.isEnabledFor(logging.DEBUG):
            # silently swallowing unknown kwargs is how a typo, or a legacy-only argument,
            # turns into a mysterious behavior change
            logger.debug('Ignored (non-`SamplingParams`) inference kwargs : {}'.format(dropped))
        config  = {k : v for k, v in config.items() if k in argnames}

        config.update({
            # the prompts are already tokenized and only the ids are consumed downstream :
            # detokenizing here would be pure overhead the legacy runtime does not pay
            'add_special_tokens'    : False,
            'detokenize'    : False
        })
        if self.perf_recorder is not None and 'return_perf_metrics' in argnames:
            # the engine-side timings of the profiling records (see `_perf_records`)
            config.setdefault('return_perf_metrics', True)

        # The end ids (see `end_ids`) are split between `end_id` (a single int) and
        # `stop_token_ids` (the rest), which `_get_stop_words` cumulates with it.
        # /!\ `BaseLLM._prepare_sampling_params` only runs `SamplingParams._setup(tokenizer)` —
        # which is what turns `stop` into token ids — when `end_id` is **None** : the two are
        # therefore mutually exclusive, and the native stop words win. `stop_token_ids` is not
        # concerned (it is already made of ids), so the end ids survive either way
        end_ids = self.end_ids
        if stop_words:
            if getattr(self.engine, 'tokenizer', None) is None:
                # the stop strings are turned into token ids by `SamplingParams._setup`, which
                # needs the LLM API's own tokenizer : without it the request fails much later,
                # on an opaque `stop (...) is not processed by tokenizer` RuntimeError
                raise ValueError(
                    '`stop_words` requires the LLM API tokenizer : load the runtime without '
                    '`skip_tokenizer_init` (or enforce them through `stop_condition`)'
                )

            config['stop'] = [stop_words] if isinstance(stop_words, str) else list(stop_words)
            config.pop('end_id', None)
            if end_ids: config.setdefault('stop_token_ids', end_ids)
        elif end_ids:
            config.setdefault('end_id', end_ids[0])
            if len(end_ids) > 1: config.setdefault('stop_token_ids', end_ids[1 :])
            if self.pad_token != -1: config.setdefault('pad_id', self.pad_token)
        elif 'end_id' not in config and getattr(self.engine, 'tokenizer', None) is None:
            # nothing to stop on, and no engine-side tokenizer to fall back on : failing here,
            # naming the two ways out, beats the LLM API's `tokenizer is required to reset
            # end_id if it is None` raised from `_prepare_sampling_params`
            raise ValueError(
                'No end id available for `{}` : pass the `tokenizer` that encoded the prompt '
                '(or an explicit `end_id`). The checkpoint ships no tokenizer, so the LLM API '
                'cannot resolve one by itself'.format(self.path)
            )

        if allowed_tokens is not None or banned_tokens is not None:
            config['embedding_bias'] = self._build_embedding_bias(
                allowed_tokens, end_ids, banned_tokens = banned_tokens
            )

        if thinking_budget is not None:
            config['logits_processor'] = _add_logits_processor(
                config.get('logits_processor', None),
                _build_thinking_budget_processor(thinking_budget, reasoning_token_ids)
            )

        max_beam_width = getattr(self._args, 'max_beam_width', None) or 1
        if num_beams > max_beam_width:
            # `_check_arguments` returns early on the pytorch backend, so there is no
            # friendly validation upstream : the sampler simply sizes its buffers from
            # `max_beam_width` and fails (or truncates) much further down
            raise ValueError(
                '`num_beams = {}` exceeds the engine `max_beam_width = {}` : pass '
                '`max_beam_width = {}` when loading the runtime'.format(
                    num_beams, max_beam_width, num_beams
            ))
        if max_beam_width > 1:
            # the pytorch backend runs every request at exactly `max_beam_width` (`PyExecutor`
            # rejects any other width) : a narrower request — greedy included — is therefore a
            # beam search at the engine width, of which the `num_beams` best sequences are
            # returned (`n = num_beams` : every requested beam, like the legacy
            # `(batch, beams, tokens)` output)
            config.update({'n' : num_beams, 'best_of' : max_beam_width, 'use_beam_search' : True})

        return SamplingParams(** config)

    def _build_embedding_bias(self, allowed_tokens, end_ids = (), *, banned_tokens = None):
        """
            Turns `allowed_tokens` into an additive `[vocab_size]` mask (`embedding_bias`), and
            masks the `banned_tokens` (ids never generated : `allowed_tokens = None` allows every
            other one).

            `allowed_tokens` is a **padded batch** of encoded answers (see
            `BaseLLM._get_generation_config`), hence the flatten + unique. The padding is
            dropped, and the `end_ids` are always allowed : the mask applies at *every* step, so
            masking them leaves the model no way to stop once an answer is complete — it then
            repeats allowed tokens up to `max_tokens` (`'ouiouioui...'` for `['oui', 'non']`).
            The padding used to hide this whenever the answers had different lengths, the pad
            id being an end id on most chat tokenizers.

            This replaces the legacy per-token `logits_post_processor` : the mask is built once,
            host-side, and applied on-GPU by the sampler (`apply_embedding_bias`). Note that a
            biased request leaves the in-graph sampling fast path — exactly as a logits processor
            would, so nothing is lost.
        """
        import torch

        vocab_size  = self._get_vocab_size()
        if allowed_tokens is None:
            bias = torch.zeros((vocab_size, ), dtype = torch.float32)
            return _ban_tokens(bias, banned_tokens, vocab_size)

        allowed     = np.unique(np.asarray(allowed_tokens).reshape(-1))
        if self.pad_token != -1 and self.pad_token not in end_ids:
            allowed = allowed[allowed != self.pad_token]
        allowed     = np.union1d(allowed, np.asarray(list(end_ids), dtype = allowed.dtype))
        if len(allowed) and (allowed.min() < 0 or allowed.max() >= vocab_size):
            raise ValueError('`allowed_tokens` out of the [0, {}) vocab range : [{}, {}]'.format(
                vocab_size, allowed.min(), allowed.max()
            ))

        bias    = torch.full((vocab_size, ), _mask_value, dtype = torch.float32)
        bias[torch.as_tensor(allowed, dtype = torch.long)] = 0.
        return _ban_tokens(bias, banned_tokens, vocab_size)

    def _get_vocab_size(self):
        """
            Width of the model's logits, which `embedding_bias` must match exactly (the sampler
            adds it to the logits with no broadcasting slack).

            Unlike the removed TRT builder, the pytorch backend does not pad the vocabulary at
            `tp = 1`, so the checkpoint config is authoritative here. `text_config` covers the
            multimodal wrappers, whose top-level config carries no `vocab_size`.
        """
        config  = self.hf_config
        for candidate in (config, getattr(config, 'text_config', None)):
            vocab_size = getattr(candidate, 'vocab_size', None)
            if vocab_size: return vocab_size

        raise ValueError(
            'Unable to determine the vocab size of `{}`, which is required by `allowed_tokens`'.format(
                self.path
        ))

    @staticmethod
    def load_engine(path,
                    *,

                    kv_cache_dtype  = None,
                    kv_cache_kwargs = None,
                    kv_cache_free_gpu_memory    = None,
                    kv_cache_enable_block_reuse = True,
                    kv_cache_free_gpu_memory_fraction = None,
                    kv_cache_host_memory    = None,

                    use_single_process  = False,

                    ** kwargs
                   ):
        """
            Instantiates the `LLM`.

            Note on quantization : there is **no** `quant_algo` argument on purpose. On the
            pytorch backend the weight quantization comes from the checkpoint itself
            (`hf_quant_config.json`, or `config.json:quantization_config`) ; announcing a format
            the checkpoint does not have makes the loader misread the weights, which produces
            fluent-looking garbage rather than an error. The only supported runtime knob is
            `kv_cache_dtype` (`'auto'` / `'fp8'` / `'nvfp4'`), from which TRT-LLM derives
            `quant_config.kv_cache_quant_algo` itself.

            Note on `max_seq_len` : it is forwarded (like any other `LLM` argument) and is worth
            setting explicitly on a long-context checkpoint. Left to the checkpoint's own
            `max_position_embeddings`, TRT-LLM sizes the whole cache for it — 262144 tokens on
            qwen3.5 / qwen3.8, which does not fit on a single consumer GPU.

            Note on hybrid models : a mamba / gated-delta-net decoder keeps a **recurrent state**
            next to the attention KV cache, and the knobs that size it (`mamba_ssm_cache_dtype`,
            `use_kv_cache_manager_v2`, `avg_seq_len`, ...) are `KvCacheConfig` fields rather than
            `LLM` ones : passed through `** kwargs` they would be silently dropped by
            `_get_llm_argnames`. `kv_cache_kwargs` is the way in — a `dict` of any
            `KvCacheConfig` field, merged under the named `kv_cache_*` arguments above.
            That state is not a rounding error : on qwen3.8-27b it is 48 linear-attention layers
            x 48 value heads x 128 x 128, i.e. ~151 MiB *per sequence* in float32 and half that
            in bfloat16.

            Note on block reuse with a hybrid model : the recurrent state of a prefix can only be
            restored where it was **snapshotted**, and TRT-LLM takes no snapshot by default —
            it then silently turns block reuse off. The missing `KvCacheConfig` entries (snapshot
            policy, V2 manager) are filled by `_set_hybrid_reuse_defaults`, overridable through
            `kv_cache_kwargs`. The reuse actually obtained is logged and kept in
            `last_cached_tokens`.

            Note on the pool split of a hybrid model (V2 manager) : the GPU budget is divided
            between the attention pages and the recurrent-state slots according to a **typical
            request** of `avg_seq_len` tokens (a `KvCacheConfig` field, default `max_seq_len / 2`)
            holding its live state plus one per fixed snapshot rule. The evictable snapshots
            (periodic, branch) are **not** budgeted : with `max_seq_len = 12288` and
            `max_batch_size = 4`, qwen3.8-27b gets 9 recurrent slots (8 live + 1 dummy) for 778
            attention blocks of 32 tokens, and any snapshot evicts another one. A smaller
            `avg_seq_len` (the actual mean prompt + answer length of the workload, e.g. 2048 for a
            chat agent) moves budget from the attention pages, mostly unused at these lengths, to the
            recurrent slots. `pool_ratio` sets the split explicitly instead.

            Note on `kv_cache_host_memory` : the size of the host (CPU, pinned) tier where the
            evicted pages and snapshots are moved instead of being dropped, then reloaded on reuse
            (a 75 MiB snapshot reloads in a few ms, where its recompute is a whole prefill). Given
            in bytes, Mb or Gb (like `kv_cache_free_gpu_memory`), `0` disables the tier. `None`
            (default) leaves TRT-LLM's own policy : the V2 manager then provisions a host tier as
            large as the GPU budget (capped by the available memory), with the same pool split.

            Note on audio encoder-decoder models (whisper, the only encoder-decoder supported) :
            block reuse is always disabled, and the settings TRT-LLM documents for them are
            defaulted (see `_set_audio_enc_dec_defaults`).
        """
        import transformers

        if use_single_process:
            # runs the engine in-process instead of spawning an MPI worker : no request
            # pickling, no MPI wire-up, and the VRAM is released with the object. TRT-LLM warns
            # that streaming may be less smooth, the engine loop then sharing the GIL
            os.environ['TLLM_WORKER_USE_SINGLE_PROCESS'] = '1'

        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        path    = _resolve_hf_checkpoint(path)

        # a *directory* is the caller's own : there is nothing to re-download, so an incomplete
        # one is a hard error (a repo id left unresolved by `_resolve_hf_checkpoint` is not a
        # directory, and the LLM API will complete the download itself)
        missing = _missing_checkpoint_files(path) if os.path.isdir(path) else []
        if missing:
            raise ValueError('The checkpoint `{}` is incomplete : {} is/are missing.{}'.format(
                path, ', '.join(missing),
                '\nRun `snapshot_download(<repo id>, allow_patterns = ["*.json"])` to complete '
                'a partially downloaded snapshot.'
            ))

        config  = transformers.AutoConfig.from_pretrained(path)
        audio_enc_dec   = _is_audio_encoder_decoder(config)
        if getattr(config, 'is_encoder_decoder', False) and not audio_enc_dec:
            # the text ones (BART / T5) feed their encoder from the prompt tokens, a path no model
            # of the project exercises : refusing at load time beats shipping it untested
            raise NotImplementedError(
                'The only encoder-decoder models supported by `TensorRTLLMAPIRuntime` are the '
                'audio ones (whisper) : use `TensorRTLLMRuntime` (tensorrt_llm <= 1.2) for the '
                'text ones'
            )
        if audio_enc_dec:
            # the decoder prompt is the same for every audio, but not its KV states (every layer
            # past the first attends to the audio) : reusing them across requests would be wrong,
            # and the whisper input processor provides no audio hash to key the blocks on
            kv_cache_enable_block_reuse = False

        # the caller's `KvCacheConfig` passthrough (see the docstring), which the named
        # `kv_cache_*` arguments below are layered on top of
        cache_config    = dict(kv_cache_kwargs or {})

        named   = {'enable_block_reuse' : kv_cache_enable_block_reuse}
        if kv_cache_dtype: named['dtype'] = kv_cache_dtype
        if kv_cache_host_memory is not None:
            named['host_cache_size'] = _to_bytes(kv_cache_host_memory) if kv_cache_host_memory else 0

        if kv_cache_free_gpu_memory:
            # an absolute budget : `max_gpu_total_bytes` expresses it natively, no need to
            # convert it to a fraction of the free memory (which the legacy runtime had to do
            # by estimating the engine size from the `.engine` file sizes)
            named['max_gpu_total_bytes'] = _to_bytes(kv_cache_free_gpu_memory)
        elif kv_cache_free_gpu_memory_fraction:
            named['free_gpu_memory_fraction'] = kv_cache_free_gpu_memory_fraction
        elif not any(k in cache_config for k in ('free_gpu_memory_fraction', 'max_gpu_total_bytes')):
            # only when the caller budgeted nothing at all : `KvCacheConfig` allocates the
            # **minimum** of the two when both are set, so defaulting the fraction on top of a
            # `max_gpu_total_bytes` given through `kv_cache_kwargs` would quietly shrink it
            named['free_gpu_memory_fraction'] = _default_kv_cache_free_gpu_memory_fraction

        # a key given both ways is a genuine ambiguity, not a preference : the named argument is
        # the documented one and wins, but silently is exactly how a kwargs typo becomes a
        # mysterious behavior change
        conflicts   = [k for k, v in named.items() if cache_config.get(k, v) != v]
        if conflicts:
            logger.warning('`kv_cache_kwargs` {} overridden by the named argument(s)'.format(
                {k : cache_config[k] for k in conflicts}
            ))
        cache_config.update(named)

        if cache_config['enable_block_reuse'] and _is_hybrid_linear(config):
            _set_hybrid_reuse_defaults(cache_config)
        if audio_enc_dec:
            _set_audio_enc_dec_defaults(config, cache_config, kwargs)
            if kv_cache_free_gpu_memory and cache_config.get('use_kv_cache_manager_v2', False) is not True:
                # the V1 manager (the default of an encoder-decoder, and the only one supporting beam
                # search) sizes its pools from `max_tokens`, which TRT-LLM derives from
                # `free_gpu_memory_fraction` *before* applying `max_gpu_total_bytes` : the absolute
                # budget alone is ignored (0.9 of the free memory gets allocated)
                cache_config['free_gpu_memory_fraction'] = _kv_budget_fraction(
                    cache_config['max_gpu_total_bytes'], path
                )

        kwargs['kv_cache_config'] = KvCacheConfig(** _filter_kwargs(
            cache_config, _get_field_names(KvCacheConfig), 'KvCacheConfig'
        ))

        kwargs  = _filter_kwargs(kwargs, _get_llm_argnames(), 'LLM')

        return LLM(model = path, ** kwargs)

class APIInferenceStream:
    """
        Stream-like wrapper around a batch of streaming `generate_async` results, mirroring the
        legacy `InferenceStream` interface : iterating yields the (cumulated)
        `(batch, num_sequences, tokens)` ids at each generation step, `abort()` cancels the
        underlying requests, and `tokens` holds the final (cumulated) ids once consumed.

        The `token_ids` exposed by the LLM API are already cumulated (`token_ids_diff` carries
        the step delta), which is exactly what the legacy stream contract expects.

        `on_finish(results)` is called once, when every request of the batch has ended (aborted
        by the stop watcher included) — not when the *consumer* stops iterating early, the
        stream being then abandoned rather than finished.
    """
    def __init__(self, results, *, stop_watcher = None, on_finish = None):
        self.results    = results
        self.stop_watcher   = stop_watcher
        self.on_finish  = on_finish

        self._aborted   = False
        self._outputs   = None

    @property
    def tokens(self):
        return self._outputs

    @property
    def perf_metrics(self):
        """
            Perf metrics of the batch, meaningful once the stream has been consumed.

            The counters are cumulated by the engine and only complete on the final response, so
            reading this mid-stream returns what has accumulated so far — which is exactly what
            an aborted request (`stop_condition`) leaves behind.
        """
        return _summarize_perf_metrics(self.results)

    @property
    def request_id(self):
        """ The ids of the enqueued requests (available before the first step is consumed) """
        return [result.request_id for result in self.results]

    def __iter__(self):
        outputs = [[[]] for _ in self.results]
        # the results are polled round-robin : each `next` blocks on its own response queue, so
        # the yield rate is gated by the slowest request of the batch. Harmless in practice —
        # streaming is a single-request path (`BaseModel.infer` is per-sample)
        pending = list(range(len(self.results)))
        while pending:
            progressed = False
            for i in tuple(pending):
                try:
                    item = next(self.results[i])
                except StopIteration:
                    pending.remove(i)
                    continue

                progressed  = True
                outputs[i]  = [list(seq.token_ids) for seq in item.outputs]

                if self.stop_watcher is not None and self.stop_watcher.should_stop(i, outputs[i][0]):
                    # only *this* request is done : aborting it (rather than forcing its EOS as
                    # the legacy processor did) leaves the rest of the batch running
                    self.results[i].abort()
                    pending.remove(i)

            # nothing new : every pending request just ended, so the last state has already
            # been yielded (a duplicate step would be an empty `delta` for the consumers)
            if not progressed: break

            # the per-request entries are re-assigned (never mutated) at each step, so a shallow
            # copy of the outer list is enough to keep a yielded item valid afterwards
            self._outputs = list(outputs)
            yield self._outputs

            if self.is_aborted(): break

        if self.on_finish is not None and not self.is_aborted():
            on_finish, self.on_finish = self.on_finish, None
            on_finish(self.results)

    def consume(self):
        """ Drains the stream and returns the final (cumulated) ids """
        for _ in self: pass
        return self._outputs

    def abort(self):
        if self._aborted: return
        self._aborted = True

        for result in self.results:
            if not result.finished: result.abort()

    def is_aborted(self):
        return self._aborted

class _StopWatcher:
    """
        Host-side `stop_condition` enforcement.

        `stop_condition` is either a regex (`str`) or a callable on the decoded text ; either
        way it has no native LLM API counterpart. Evaluating it here rather than in a
        `logits_post_processor` :
            - keeps the (stateful) incremental decoding in this process, with the project
              `Tokenizer` — the legacy version had to ship it to the engine worker, where a
              `UnicodeDecodeError` fallback made accented stop words never match ;
            - needs no prompt-length bookkeeping : the LLM API excludes the prompt from the
              output by default, so the ids seen here are the answer alone ;
            - costs one generation step of latency, the request being aborted rather than
              forced to emit EOS.

        The decoding state is per request (`decode_stream` is stateful and must never be shared).
    """
    def __init__(self, condition, tokenizer, batch_size):
        if tokenizer is None:
            raise ValueError('`stop_condition` requires a `tokenizer` to decode the generation')

        if isinstance(condition, str):
            _regex    = condition
            condition = lambda text: re.search(_regex, text) is not None

        self.condition  = condition
        self.tokenizer  = tokenizer

        self._states    = [None] * batch_size

    def should_stop(self, index, tokens):
        text, _, self._states[index] = self.tokenizer.decode_stream(tokens, self._states[index])
        return bool(self.condition(text))


def _ban_tokens(bias, banned_tokens, vocab_size):
    """ Masks the `banned_tokens` ids in the `embedding_bias` `bias` (in place), returns it """
    import torch

    if banned_tokens is None: return bias
    banned = np.unique(np.asarray(banned_tokens, dtype = np.int64).reshape(-1))
    if len(banned) and (banned.min() < 0 or banned.max() >= vocab_size):
        raise ValueError('`banned_tokens` out of the [0, {}) vocab range : [{}, {}]'.format(
            vocab_size, banned.min(), banned.max()
        ))
    bias[torch.as_tensor(banned, dtype = torch.long)] = _mask_value
    return bias

def _build_thinking_budget_processor(thinking_budget, reasoning_token_ids):
    """
        The native TRT-LLM `ThinkingBudgetLogitsProcessor`, built **here** from the caller's
        delimiter ids rather than through `SamplingParams.thinking_token_budget` : the latter
        requires the engine to be loaded with a `reasoning_parser` (to find the delimiters),
        whereas the model already knows them (`BaseLLM.reasoning_tokens`).

        The processor sees the whole sequence, **prompt included** (the pytorch backend passes
        `request.get_tokens(beam)`) : a block opened by the chat template (Qwen3.x
        `<think>\\n` generation prompt) is therefore budgeted as well. Once the budget is spent,
        the end delimiter is forced (abruptly : no transition sentence).

        Performance note : it is a python callback called at every step, on a copy of the
        whole sequence — only paid by the requests that set a budget.
    """
    from tensorrt_llm.llmapi import ThinkingBudgetLogitsProcessor

    if not reasoning_token_ids or len(reasoning_token_ids) != 2:
        raise ValueError('`thinking_budget` requires the `(start, end)` `reasoning_token_ids`')

    start, end = reasoning_token_ids
    return ThinkingBudgetLogitsProcessor(
        thinking_token_budget   = int(thinking_budget),
        reasoning_start_token_ids   = [int(start)],
        reasoning_end_token_ids     = [int(end)]
    )

def _add_logits_processor(existing, processor):
    """ Chains `processor` after the `existing` `SamplingParams.logits_processor` (if any) """
    if existing is None: return processor
    if isinstance(existing, list): return existing + [processor]
    return [existing, processor]

_llm_argnames, _sampling_argnames = None, None

def _backend(args):
    """
        The LLM API backend of the engine `args` : `pytorch` (`TorchLlmArgs`, the only one as of
        1.3), `_autodeploy`, or `None` for the legacy TRT engines (`TrtLlmArgs`, before 1.3)
    """
    backend = getattr(args, 'backend', None)
    if backend is None and type(args).__name__ == 'TorchLlmArgs': backend = 'pytorch'
    return backend

def _is_hybrid_linear(config):
    """
        Whether the (HF) `config` describes a hybrid decoder, i.e. one keeping a recurrent state
        next to its attention KV cache : qwen3-next / qwen3.5 / qwen3.8 (`layer_types` with
        `linear_attention`, or a `full_attention_interval`), nemotron-h
        (`hybrid_override_pattern`), kimi-linear (`linear_attn_config`). The multimodal wrappers
        nest it in `text_config`.

        TRT-LLM's own `is_hybrid_linear` is not reused : it runs on the config *after* the
        model loader normalized it (`Qwen3_5ForCausalLM`), and returns `False` on the checkpoint's
        `Qwen3_5ForConditionalGeneration` read here.
    """
    for candidate in (config, getattr(config, 'text_config', None)):
        if candidate is None: continue
        layer_types = getattr(candidate, 'layer_types', None) or ()
        if any(t in ('linear_attention', 'mamba') for t in layer_types): return True
        if getattr(candidate, 'full_attention_interval', None): return True
        if getattr(candidate, 'hybrid_override_pattern', None): return True
        if getattr(candidate, 'linear_attn_config', None) is not None: return True
    return False

def _set_hybrid_reuse_defaults(cache_config):
    """
        Completes the `KvCacheConfig` kwargs of a hybrid model loaded with block reuse, leaving
        every entry the caller set untouched :
            - `use_kv_cache_manager_v2 = True` : the end-of-prompt snapshots require the V2
              manager, which is also what TRT-LLM prefers for the qwen3 hybrids
              (`get_preferred_kv_cache_manager_version`). It has to be explicit : the
              `nvidia/Qwen3.8-27B-NVFP4` config (`language_model_only = false` + the vision token
              ids) matches `is_qwen_image_bench_config`, so TRT-LLM routes it to
              `QwenImageBenchModel`, which declares no preference — and `'auto'` then resolves
              to V1 ;
            - `mamba_state_config` : `_default_hybrid_state_config` (end-of-prompt and periodic
              snapshots), or its periodic-only subset when the caller forced the V1 manager.
    """
    cache_config.setdefault('use_kv_cache_manager_v2', True)
    if any(k in cache_config for k in ('mamba_state_config', 'mamba_state_cache_interval')):
        return

    use_v2 = cache_config['use_kv_cache_manager_v2'] is True
    # the end-of-prompt snapshots require the V2 manager
    cache_config['mamba_state_config'] = dict(
        _default_hybrid_state_config if use_v2 else _v1_hybrid_state_config
    )
    logger.info('Hybrid model with block reuse : recurrent-state snapshot policy {} ({} manager)'.format(
        cache_config['mamba_state_config'], 'V2' if use_v2 else 'V1'
    ))

def _is_audio_encoder_decoder(config):
    """
        Whether the (HF) `config` describes an audio encoder-decoder (whisper) : the LLM API feeds
        its encoder from `multi_modal_data['audio']` (the raw audio, the engine computing the
        log-mel itself) and its decoder from the text prompt
    """
    return getattr(config, 'model_type', None) == 'whisper'

def _set_audio_enc_dec_defaults(config, cache_config, kwargs):
    """
        Completes the `KvCacheConfig` and `LLM` kwargs of an audio encoder-decoder (whisper) with
        the settings TRT-LLM documents for it (`docs/source/models/encoder-decoder.md`), leaving
        every entry the caller set untouched :
            - `cross_kv_cache_fraction` (required for an encoder-decoder) : the share of the KV
              budget held by the cross-attention pool. Every request holds `max_source_positions`
              (1500) cross positions and at most `max_target_positions` (448) decoder ones, hence
              their ratio (~0.77) : the documented 0.5 starves the cross pool first ;
            - `max_input_len = max_source_positions` : the cross pool is sized from it, and its
              default (1024) is smaller than the 1500 encoder positions of every request ;
            - `scheduler_config = SchedulerConfig(use_python_scheduler = True)`, the scheduler
              every encoder-decoder example of TRT-LLM runs with ;
            - `disable_overlap_scheduler = True` : with the overlap scheduler, a logits processor
              sees the generated ids one step late (twice the prompt alone, then always missing
              the last token). The whisper timestamp rules (`WhisperTimestampLogitsProcessor`)
              then apply to a stale history : a segment gets closed right after its first word.
    """
    from tensorrt_llm.llmapi import SchedulerConfig

    source, target = config.max_source_positions, config.max_target_positions
    cache_config.setdefault('cross_kv_cache_fraction', round(source / (source + target), 2))
    kwargs.setdefault('max_input_len', source)
    kwargs.setdefault('scheduler_config', SchedulerConfig(use_python_scheduler = True))
    kwargs.setdefault('disable_overlap_scheduler', True)

def _kv_budget_fraction(budget, path):
    """
        The `free_gpu_memory_fraction` equivalent to an absolute KV cache `budget` (in bytes) : TRT-LLM
        applies the fraction to the memory left free once the weights are loaded, estimated here as
        the currently free memory minus the size of the checkpoint weights
    """
    from ..gpu import get_gpu_memory_infos

    files   = glob.glob(os.path.join(path, '*.safetensors')) if os.path.isdir(path) else []
    if not files and os.path.isdir(path): files = _get_weight_files(path)
    weights = sum(os.path.getsize(f) for f in files)

    available = get_gpu_memory_infos()['free'] - weights
    return min(1., budget / available) if available > 0 else 1.

def _get_llm_argnames():
    global _llm_argnames
    if _llm_argnames is None:
        import inspect

        from tensorrt_llm import LLM

        _llm_argnames = set(inspect.signature(LLM.__init__).parameters)
        try:
            # `LLM.__init__(model, tokenizer, ..., ** kwargs)` forwards its kwargs to the
            # (pydantic) `TorchLlmArgs` : accept its fields as well
            from tensorrt_llm.llmapi import TorchLlmArgs
            _llm_argnames.update(_get_field_names(TorchLlmArgs))
        except ImportError:
            pass
        _llm_argnames.difference_update(('self', 'model', 'kwargs'))
    return _llm_argnames

def _get_sampling_argnames():
    global _sampling_argnames
    if _sampling_argnames is None:
        from tensorrt_llm import SamplingParams

        _sampling_argnames = _get_field_names(SamplingParams)
    return _sampling_argnames

def _get_field_names(cls):
    """
        Constructor-settable field names of `cls`, whichever flavor it is.

        `SamplingParams` is a `dataclass` whose private fields are declared `init = False`
        (`_stop_word_ids`, `_stream_interval`, ...) : `dataclasses.fields` lists them, but
        passing one to the constructor raises. Hence the `f.init` filter.
    """
    import dataclasses

    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls) if f.init}
    return set(getattr(cls, 'model_fields', {}))   # pydantic flavor

def _filter_kwargs(kwargs, argnames, name):
    dropped = [k for k in kwargs if k not in argnames]
    if dropped:
        logger.warning('Ignored (non-`{}`) arguments : {}'.format(name, dropped))
    return {k : v for k, v in kwargs.items() if k in argnames}

def _summarize_perf_metrics(results):
    """
        Aggregates the `RequestPerfMetrics` of a batch into a plain (json-able) `dict`, or
        returns `None` when the call did not ask for them.

        The `(batch, sequences, ids)` return contract of `__call__` has no room for them, and
        they are what makes a **speculative decoding** measurement interpretable : `tokens/sec`
        alone cannot distinguish a drafter that proposes badly from one whose proposals are
        rejected, and `acceptance_rate` is exactly that split. `tokens_per_iteration` is the
        other half — it is the number of tokens the engine produced per forward pass, i.e. the
        factor by which speculation broke the memory-bandwidth bound.

        Metrics only exist when the caller passed `return_perf_metrics = True` (a
        `SamplingParams` field, so it reaches the engine untouched) : everything is guarded
        rather than assumed, since a runtime returning `None` here is normal.
    """
    accepted = drafted = tokens = iterations = 0
    found    = False

    for result in results:
        for sequence in getattr(result, 'outputs', ()):
            metrics = getattr(sequence, 'request_perf_metrics', None)
            if metrics is None: continue

            found   = True
            tokens += len(sequence.token_ids)

            # `first_iter` / `last_iter` bound the forward passes the request took, so
            # `tokens / iterations` is > 1 exactly when speculation accepted something
            first, last = getattr(metrics, 'first_iter', None), getattr(metrics, 'last_iter', None)
            if first is not None and last is not None and last >= first:
                iterations += last - first + 1

            spec = getattr(metrics, 'speculative_decoding', None)
            if spec is not None:
                accepted += getattr(spec, 'total_accepted_draft_tokens', 0) or 0
                drafted  += getattr(spec, 'total_draft_tokens', 0) or 0

    if not found: return None

    summary = {'tokens' : tokens, 'iterations' : iterations}
    if iterations:
        summary['tokens_per_iteration'] = tokens / iterations
    if drafted:
        summary.update({
            'accepted_draft_tokens' : accepted,
            'draft_tokens'  : drafted,
            'acceptance_rate'   : accepted / drafted
        })
    return summary

_steady_clock = None

def _steady_now():
    """
        The clock of the engine's timing metrics (`RequestPerfMetrics.timing_metrics`) : the
        rank-adjusted C++ steady clock, in seconds. Host-side time points taken with it are
        directly comparable to the engine-side ones, the worker process included (the steady
        clock is system-wide)
    """
    global _steady_clock
    if _steady_clock is None:
        try:
            from tensorrt_llm._utils import get_global_steady_clock_now_in_seconds as _steady_clock
        except ImportError:
            # the same clock on Linux (`std::chrono::steady_clock` = `CLOCK_MONOTONIC`), without
            # the rank adjustment (zero on a single rank)
            _steady_clock = time.monotonic
    return _steady_clock()

def _seconds(value):
    """ A timing metric in seconds, `None` when unset (a zero time point = never reached) """
    if value is None: return None
    value = value.total_seconds() if hasattr(value, 'total_seconds') else float(value)
    return value if value > 0 else None

def _delta(start, end):
    return end - start if start is not None and end is not None else None

def _ids_hash(ids):
    """ Short digest of a token id sequence (`None` for none). Of the **ids**, not of the text :
        `Turn.prompt_hash` is the digest of the rendered prompt """
    if ids is None: return None
    return hashlib.sha1(np.asarray(list(ids), dtype = np.int64).tobytes()).hexdigest()[:16]

def _perf_records(results, perf, done):
    """
        The profiling records of a completed call, one per request (`dict`, json-able) :
            - the call : `thread`, `keys` (the caller kwargs names), `streaming` (`True` = the
              caller streams, `'internal'` = streamed for a `stop_condition`, `False`),
              `batch_index` / `batch_size`
            - the tokens : `n_prompt` (the engine's own count), `n_cached` (reused from the KV
              cache), `n_output`, `finish_reason`, and the digests of the prompt and output ids
              (`prompt_ids_hash` / `output_ids_hash`, see `_ids_hash`) : two runs that got the
              same prompt but generated differently diverged numerically (a different chunking of
              the prefill, a restored snapshot, another batch composition), not by their inputs
            - the host side : `host_prepare_s` (from the call to the submission : inputs, sampling
              params, embedding bias), `total_s` (from the call to the completion, as seen by
              this process)
            - the engine side (when the engine returned its timing metrics) : `submit_to_arrival_s`
              (submission to the engine's arrival : LLM API preprocessing + IPC to the worker),
              `queue_s` (arrival to the first scheduling), `prefill_s` (first scheduling to the
              first token : the context phase, every chunk included), `decode_s` (first to last
              token), `last_token_to_done_s` (last token to the completion here : postprocessing +
              IPC back + this process' own consumption), and the KV cache block counts of the
              request (`kv_reused_blocks`, `kv_missed_blocks`, `kv_hit_rate`)
        Every duration is in seconds. A request aborted by the stop watcher has no `last_token`
        time point : its `decode_s` / `last_token_to_done_s` are `None`, `decode_until_done_s`
        (first token to the completion here) being the upper bound of its decoding
    """
    records = []
    thread  = threading.current_thread().name
    for index, result in enumerate(results):
        prompt  = getattr(result, 'prompt_token_ids', None)
        outputs = getattr(result, 'outputs', None) or []
        seq     = outputs[0] if outputs else None

        record  = {
            'thread'    : thread,
            'keys'      : list(perf.get('keys', ())),
            'streaming' : perf.get('streaming', False),
            'batch_index'   : index,
            'batch_size'    : len(results),
            'n_prompt'  : len(prompt) if prompt is not None else None,
            'n_cached'  : int(getattr(result, 'cached_tokens', 0) or 0),
            'n_output'  : len(seq.token_ids) if seq is not None else None,
            'finish_reason' : str(getattr(seq, 'finish_reason', None)),
            'prompt_ids_hash'   : _ids_hash(prompt),
            'output_ids_hash'   : _ids_hash(seq.token_ids if seq is not None else None),
            'host_prepare_s'    : _delta(perf.get('enter'), perf.get('submit')),
            'total_s'   : _delta(perf.get('enter'), done)
        }

        metrics = getattr(seq, 'request_perf_metrics', None)
        timing  = getattr(metrics, 'timing_metrics', None)
        if timing is not None:
            arrival = _seconds(getattr(timing, 'arrival_time', None))
            first_scheduled = _seconds(getattr(timing, 'first_scheduled_time', None))
            first_token     = _seconds(getattr(timing, 'first_token_time', None))
            last_token      = _seconds(getattr(timing, 'last_token_time', None))
            record.update({
                'submit_to_arrival_s'   : _delta(perf.get('submit'), arrival),
                'queue_s'   : _delta(arrival, first_scheduled),
                'prefill_s' : _delta(first_scheduled, first_token),
                'decode_s'  : _delta(first_token, last_token),
                'last_token_to_done_s'  : _delta(last_token, done),
                # the bound of the decoding seen from here : the only one of a request aborted
                # by the stop watcher (no `last_token` time point)
                'decode_until_done_s'   : _delta(first_token, done)
            })

        kv = getattr(metrics, 'kv_cache_metrics', None)
        if kv is not None:
            record.update({
                'kv_reused_blocks'  : getattr(kv, 'num_reused_blocks', None),
                'kv_missed_blocks'  : getattr(kv, 'num_missed_blocks', None),
                'kv_hit_rate'   : getattr(kv, 'kv_cache_hit_rate', None)
            })

        records.append(record)
    return records

def _normalize_batch_tokens(tokens):
    """ Uniformizes `tokens` to a batch of plain `List[int]` (like the legacy `batch_input_ids`) """
    if not isinstance(tokens, list): tokens = tokens.tolist()
    if not isinstance(tokens[0], list) and not hasattr(tokens[0], 'shape'):
        tokens = [tokens]
    return [tok if isinstance(tok, list) else np.asarray(tok).tolist() for tok in tokens]

def _to_bytes(memory):
    """ Interprets a memory budget given in bytes, Mb or Gb (like the legacy runtime did) """
    if memory < 128:            return int(memory * 1024 ** 3)
    elif memory < 128 * 1024:   return int(memory * 1024 ** 2)
    return int(memory)

def _get_weight_files(path):
    return [f for pattern in _weight_patterns for f in glob.glob(os.path.join(path, pattern))]

def _get_weight_names(path):
    """ Names of the checkpoint's tensors, read from its metadata alone (no weight is loaded) """
    index   = os.path.join(path, 'model.safetensors.index.json')
    if os.path.exists(index):
        with open(index, 'r', encoding = 'utf-8') as f:
            return list(json.load(f).get('weight_map', {}))

    files   = sorted(glob.glob(os.path.join(path, '*.safetensors')))
    if not files: return []

    # the safetensors header is a json blob prefixed by its length (8 bytes, little endian) :
    # reading it directly costs one small read, where `safetensors.safe_open` would pull torch in
    try:
        with open(files[0], 'rb') as f:
            length = int.from_bytes(f.read(8), 'little')
            header = json.loads(f.read(length).decode('utf-8'))
    except Exception as e:
        logger.warning('Unable to read the safetensors header of `{}` : {}'.format(files[0], e))
        return []

    return [name for name in header if name != '__metadata__']

def _missing_checkpoint_files(path):
    """
        Lists what a local checkpoint directory lacks to be *loadable*, `[]` when it is complete.

        Checking more than the presence of weights is what tells a usable checkpoint from the
        snapshot an interrupted staged download leaves behind (`architectures.transformers`
        `download_hf_model` fetches the `*.json` then the weights, and only re-checks
        `config.json` afterwards — so a half-downloaded snapshot is never repaired on its own).
        Both failure modes are silent where it hurts :
            - **no tokenizer file** : `LLM.tokenizer` is then `None`, and every request dies much
              later on an opaque `tokenizer is required to reset end_id if it is None` ;
            - **a quantized checkpoint whose quantization config is missing** : the pytorch
              backend has no way to know the weights are fp8 / int4, loads them as if they were
              the config's `dtype` and produces fluent-looking garbage — on top of a model far
              bigger than expected, which starves the KV cache.
    """
    missing = []

    if not _get_weight_files(path): missing.append('the model weights')

    if not os.path.exists(os.path.join(path, 'config.json')):
        # everything below reads it : no point listing what cannot be checked
        return missing + ['config.json']

    if not any(os.path.exists(os.path.join(path, f)) for f in _tokenizer_files):
        missing.append('a tokenizer file ({})'.format(' / '.join(_tokenizer_files)))

    with open(os.path.join(path, 'config.json'), 'r', encoding = 'utf-8') as f:
        config = json.load(f)

    # a VLM checkpoint is only usable with its HF processor : the LLM API builds its input
    # processor from it, and without it the failure only surfaces on the first image request
    if 'vision_config' in config and not any(
            os.path.exists(os.path.join(path, f)) for f in _processor_files):
        missing.append('a processor file ({}), required by the vision tower'.format(
            ' / '.join(_processor_files)
        ))

    is_quantized = any(
        name.endswith(_quantized_suffixes) for name in _get_weight_names(path)
    )
    if is_quantized and not os.path.exists(os.path.join(path, 'hf_quant_config.json')):
        # the two places TRT-LLM looks for it : the modelopt side-car, or the transformers-native
        # entry of `config.json`
        if 'quantization_config' not in config:
            missing.append(
                'hf_quant_config.json (the weights carry quantization scales, so the checkpoint '
                'is quantized : without it the weights would be silently misread)'
            )

    return missing

def _resolve_hf_checkpoint(path):
    """
        Resolves a model name (HF repo id) to its **local** cached snapshot (see
        `utils.resolve_hf_snapshot`), so an already downloaded checkpoint is reused instead of
        being re-downloaded :
            - a local directory is returned unchanged (already a checkpoint dir) ;
            - a fully cached repo id resolves to the snapshot of its current revision ;
            - anything else is returned unchanged, for the LLM API to download.

        An **incomplete** cached snapshot (see `_missing_checkpoint_files`) is treated as a miss
        rather than as a hit : handing such a directory to the LLM API yields anything from a
        puzzling `No weight files found in ...` to a silently misread model, whereas returning
        the repo id lets it download what is missing. This self-heals, and cheaply so : the HF
        cache is keyed by blob hash, so only the absent files are actually fetched.
    """
    return resolve_hf_snapshot(path, validate = _missing_checkpoint_files)
