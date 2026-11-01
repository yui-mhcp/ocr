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

import logging
import threading
import numpy as np

from .. import timer
from .runtime import Runtime
from .tensorrt_llm_api_runtime import _get_llm_argnames, _resolve_hf_checkpoint

logger = logging.getLogger(__name__)

# Default packed-batch budget. `LLM.encode` *rejects* a batch exceeding it rather than splitting
# it : `__call__` therefore splits the batch itself (see `_split_batch`), this only bounds the size
# of a single forward pass (and of the runner's activation buffers)
_default_max_num_tokens = 16384
_default_max_batch_size = 64

class TensorRTLLMEncoderRuntime(Runtime):
    """
        TRT-LLM runtime for **encoder-only** (embedding) models, based on the LLM API
        `encode_only` mode (`LLM(..., encode_only = True)` + `LLM.encode`).

        Unlike `TensorRTLLMAPIRuntime` (generation), the engine runs **in-process** and
        synchronously : no scheduler, no sampler, no KV cache, no MPI worker. A call is a single
        forward pass over a *packed* batch (the sequences are concatenated, without padding),
        which is where the gain over `HFRuntime` comes from on variable-length batches.

        The model itself returns the pooled embedding (`(batch, hidden)`) : pooling and
        normalization are part of the TRT-LLM model definition (e.g. `XLMRobertaModel` = CLS +
        L2 normalization for BGE-M3), mirroring what `HFRuntime` does host-side.

        Selected automatically by `build_runtime('trt_llm_api', path)` when the checkpoint is an
        encoder (see `utils.keras.runtimes._build_trt_llm_api_runtime`).
    """
    def __init__(self, path, ** kwargs):
        super().__init__(path, ** kwargs)

        self._hf_config = None
        # `LLM.encode` reuses the runner's static input / output buffers : two concurrent calls
        # would overwrite each other's inputs (and outputs, see `__call__`)
        self._lock      = threading.Lock()

        self.sos_token  = -1
        self.eos_token  = -1
        self.pad_token  = -1

    @property
    def _model_engine(self):
        """ The `PyTorchModelEngine` of the in-process `EncoderExecutor` """
        return self.engine._encoder_executor.model_engine

    @property
    def checkpoint(self):
        return _resolve_hf_checkpoint(getattr(self.engine, '_hf_model_dir', None) or self.path)

    @property
    def hf_config(self):
        if self._hf_config is None:
            import transformers

            self._hf_config = transformers.AutoConfig.from_pretrained(self.checkpoint)
        return self._hf_config

    @property
    def base_dtype(self):
        """ Precision the model actually runs in (which may differ from the checkpoint's one) """
        dtype = next(self._model_engine.model.parameters()).dtype
        return str(dtype).rsplit('.', 1)[-1]

    @property
    def embedding_dim(self):
        return self.hf_config.hidden_size

    @property
    def max_input_length(self):
        return self._model_engine.max_seq_len

    @property
    def max_num_tokens(self):
        return self._model_engine.max_num_tokens

    @property
    def max_batch_size(self):
        return self._model_engine.batch_size

    def set_tokens(self, sos_token = None, eos_token = None, pad_token = None):
        if sos_token not in (-1, None): self.sos_token = sos_token
        if eos_token not in (-1, None): self.eos_token = eos_token
        if pad_token not in (-1, None): self.pad_token = pad_token

    def release(self):
        engine = super().release()
        if engine is None: return

        engine.shutdown()
        logger.info('Engine `{}` has been released'.format(self.path))

    @timer(name = 'TRT-LLM encoder inference')
    def __call__(self, inputs, ** _):
        """
            Embeds a batch of token sequences.

            Arguments :
                - inputs    : a padded `(batch, length)` array / tensor (the `BaseEncoder.embed`
                              format, padded with `pad_token`), or a list of token lists
            Return :
                - embeddings    : a `torch.Tensor` `(batch, embedding_dim)` on the GPU, like
                                  `HFRuntime`
        """
        import torch

        sequences = self._unpad(inputs)

        outputs = []
        with self._lock:
            for batch in self._split_batch(sequences):
                out = self.engine.encode(
                    [{'prompt_token_ids' : tokens} for tokens in batch],
                    return_raw_logits   = True,
                    copy_logits_to_host = False
                )
                # `copy_logits_to_host = False` returns a *view* on the runner's output buffer,
                # which the next forward pass overwrites : it has to be copied before then
                outputs.append(out.clone())

        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim = 0)

    def _unpad(self, inputs):
        """ Turns `inputs` into a list of (unpadded) `List[int]` """
        if isinstance(inputs, list) and (not inputs or isinstance(inputs[0], (list, tuple))):
            return [list(tokens) for tokens in inputs]

        if hasattr(inputs, 'detach'): inputs = inputs.detach().cpu().numpy()
        inputs = np.asarray(inputs)
        if inputs.ndim == 1: inputs = inputs[None]

        pad = self.pad_token
        if pad == -1: pad = getattr(self.hf_config, 'pad_token_id', None)
        if pad is None: return inputs.tolist()

        # only the *trailing* padding is removed : the length of a row is the position of its
        # last non-padding token
        not_pad = inputs != pad
        lengths = np.where(
            not_pad.any(axis = 1), inputs.shape[1] - np.argmax(not_pad[:, ::-1], axis = 1), 0
        )
        return [row[: length].tolist() for row, length in zip(inputs, lengths)]

    def _split_batch(self, sequences):
        """
            Splits `sequences` in consecutive sub-batches fitting the engine budget
            (`max_batch_size` sequences, `max_num_tokens` tokens), the order being preserved
        """
        max_batch_size, max_num_tokens = self.max_batch_size, self.max_num_tokens

        batch, num_tokens = [], 0
        for tokens in sequences:
            if batch and (len(batch) == max_batch_size or num_tokens + len(tokens) > max_num_tokens):
                yield batch
                batch, num_tokens = [], 0
            batch.append(tokens)
            num_tokens += len(tokens)

        if batch: yield batch

    @staticmethod
    def load_engine(path,
                    *,

                    dtype   = 'float16',
                    model_kwargs    = None,
                    max_num_tokens  = _default_max_num_tokens,
                    max_batch_size  = _default_max_batch_size,
                    cuda_graph_config   = None,

                    ** kwargs
                   ):
        """
            Instantiates the `LLM` in `encode_only` mode.

            Arguments :
                - path  : HF repo id or checkpoint directory
                - dtype : the compute precision. /!\\ `LLM(dtype = ...)` is **not** applied to the
                          model on the pytorch backend (the checkpoint `config.json` wins, float32
                          for BGE-M3, which makes the attention fall back to an unfused kernel
                          with a workspace in O(seq_len²)) : it is forwarded as a config override
                          through `model_kwargs` instead
                - max_num_tokens / max_batch_size   : the budget of a single forward pass
                - cuda_graph_config : `EncodeCudaGraphConfig` (or its kwargs as a `dict`) to enable
                                      the encoder CUDA graphs, disabled by default. They speed up the
                                      small, launch-bound batches but pad each batch to its captured
                                      bucket, which cancels the benefit of the packing on
                                      variable-length batches
                - kwargs    : any other `LLM` argument (`max_seq_len`, `attn_backend`, ...), the
                              others are dropped
        """
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import EncodeCudaGraphConfig

        if isinstance(cuda_graph_config, dict):
            cuda_graph_config = EncodeCudaGraphConfig(** cuda_graph_config)

        model_kwargs = dict(model_kwargs or {})
        if dtype: model_kwargs.setdefault('dtype', dtype)

        # the model-level kwargs `BaseModel` forwards to every runtime (`pretrained`, ...) are
        # expected here : they are dropped silently
        argnames = _get_llm_argnames()
        dropped  = [k for k in kwargs if k not in argnames]
        if dropped: logger.debug('Ignored (non-`LLM`) arguments : {}'.format(dropped))
        kwargs   = {k : v for k, v in kwargs.items() if k in argnames}

        return LLM(
            model   = _resolve_hf_checkpoint(path),
            encode_only = True,
            model_kwargs    = model_kwargs,
            max_num_tokens  = max_num_tokens,
            max_batch_size  = max_batch_size,
            cuda_graph_config   = cuda_graph_config,
            ** kwargs
        )

def is_encoder_checkpoint(path):
    """
        Whether the checkpoint at `path` (HF repo id or directory) is an encoder-only model for
        the TRT-LLM pytorch backend, i.e. one served by `LLM.encode` rather than `LLM.generate`
    """
    import os
    import json

    path = str(_resolve_hf_checkpoint(path))
    config_file = os.path.join(path, 'config.json')
    if not os.path.exists(config_file):
        if os.path.isdir(path): return False
        # a repo id not (fully) cached yet : its config alone is enough to decide
        try:
            from huggingface_hub import hf_hub_download

            config_file = hf_hub_download(path, 'config.json')
        except Exception as e:
            logger.info('Unable to fetch the config of `{}` : {}'.format(path, e))
            return False

    with open(config_file, 'r', encoding = 'utf-8') as f:
        architectures = json.load(f).get('architectures', None)
    if not architectures: return False

    from tensorrt_llm._torch.model_config import ModelConfig

    return not ModelConfig.is_generation_model(architectures)
