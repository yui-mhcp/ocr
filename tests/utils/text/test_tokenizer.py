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

""" Pure-``pytest`` tests for :mod:`utils.text.tokenizer`.

Ported from the original ``unittest`` ``TestTokenizer`` :
  - ``setUp`` -> a ``tokenizer`` fixture (a 150-symbol char-level english tokenizer) ;
  - ``absl.parameterized`` -> ``@pytest.mark.parametrize`` ;
  - the ``tensorflow`` graph test is gated behind ``@pytest.mark.tensorflow`` ;
  - the ``transformers`` parity tests are gated behind ``pytest.importorskip('transformers')``
    + ``@pytest.mark.slow``, and skipped if the model is not in the HF cache (run them with
    ``HF_HUB_OFFLINE=1``, in both ``transformers`` 4 and 5 environments) ;
  - a cached repo id is loaded from its local snapshot directory (no hub request), checked by
    ``test_transformers_loads_the_local_snapshot`` without any model.

``transformers`` parity : ``Tokenizer.from_transformers_pretrained`` reads the serialized
``tokenizers`` pipeline (identical in ``transformers`` 4 and 5, which removed the "slow"
tokenizers). On a varied corpus (code, digits, NFD text, emojis / CJK, chat special tokens) the
ids must be identical with and without the default special tokens (inferred from the
post-processor), and the decoded text identical to ``decode(clean_up_tokenization_spaces =
False)``. Accepted differences are documented in the tests (BERT decoder clean-up, trailing
``▁`` of the ``transformers`` 4 sentencepiece fast tokenizers, Llama-2 / vigostral compared to
the sentencepiece model as ``transformers`` 5 drops its leading ``▁``).

NOTE : the original ``_test_format`` (disabled via its leading underscore) is **not**
ported — it exercised a ``Tokenizer.format`` method that no longer exists, so it would
only raise ``AttributeError``. It is dropped rather than resurrected against a missing
API ; chat/templating is the job of ``encode_chat`` and should get its own tests.

Coverage was extended beyond the original port to harden the ``encode`` / ``decode``
hot path : ``decode`` / ``decode_ids`` (round-trip, padding skip & the consecutive-pad
early-break, ``remove_tokens``, batched / list-of-lists, float-logits ``argmax``),
``encode`` edge cases (``__call__`` alias, ``return_text``, independent ``add_sos`` /
``add_eos`` + idempotency, ``dict`` input, the cleaning path, out-of-vocab dropping,
empty input, invalid ``return_type``, ``tf`` / ``tensor`` return types), ``__getitem__``
/ ``__contains__`` and a ``get_config`` / ``save`` -> ``load_from_file`` round-trip.

Source bug found & fixed while writing these (``utils/text/tokenizer.py`` ``decode``) :
the ``remove_tokens`` branch set ``_skip = self.token_indexes``, a dict keyed by token
*string*, then tested the iterated integer ids against it (``token not in _skip``) — so
special tokens were never actually removed. Now ``_skip`` is a ``set`` of the id
*values*, unioned with the padding id (``remove_tokens`` also implies ``skip_padding``).
"""

import os
import functools

import numpy as np
import pytest

from utils.text import default_english_tokenizer, en_symbols, Tokenizer
from utils.text.tokenizer import _gpt_pattern
from utils.text.text_processing import bytes_to_unicode

_default_texts = [
    "Hello World !",
    "Bonjour à tous !",
    "1, 2, 3, 4, 5, 6, 7, 8, 9 et 10 !",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
    "mollit anim id est laborum.",
]


@pytest.fixture
def tokenizer():
    """ 150-symbol char-level english tokenizer with ``<s>`` / ``</s>`` at the vocab end. """
    return default_english_tokenizer(
        vocab_size      = 150,
        pad_token       = '_',
        sos_token       = '<s>',
        eos_token       = '</s>',
        use_sos_and_eos = True,
    )


# --- attributes / special tokens ---------------------------------------------------

def test_attributes(asserts, tokenizer):
    asserts.assert_equal(150, len(tokenizer))
    asserts.assert_equal(150, tokenizer.vocab_size)

    asserts.assert_equal(en_symbols, tokenizer.vocab[: len(en_symbols)])
    asserts.assert_equal(en_symbols, [tokenizer[i] for i in range(len(en_symbols))])

    asserts.assert_equal(en_symbols[0], tokenizer.blank_token)
    asserts.assert_equal(0,   tokenizer.blank_token_idx)
    asserts.assert_equal(148, tokenizer.sos_token_idx)
    asserts.assert_equal(149, tokenizer.eos_token_idx)


# --- encoding ----------------------------------------------------------------------

@pytest.mark.parametrize('text', _default_texts, ids = ['hello', 'bonjour', 'numbers', 'lorem'])
def test_tokenize_keeps_special_tokens(asserts, tokenizer, text):
    asserts.assert_equal(
        ['<s>'] + list(text) + ['</s>'],
        tokenizer.tokenize(tokenizer.sos_token + text + tokenizer.eos_token, cleaned = True)
    )

@pytest.mark.parametrize('text', _default_texts, ids = ['hello', 'bonjour', 'numbers', 'lorem'])
def test_encode(asserts, tokenizer, text):
    in_vocab = [tokenizer[c] for c in text if c in tokenizer]

    asserts.assert_equal(in_vocab, tokenizer.encode(text, cleaned = True, add_sos_and_eos = False))
    asserts.assert_equal(
        [tokenizer.sos_token_idx] + in_vocab + [tokenizer.eos_token_idx],
        tokenizer.encode(text, cleaned = True)
    )

def test_batched_encode(asserts, tokenizer):
    all_encoded = [tokenizer.encode(txt, cleaned = True) for txt in _default_texts]

    max_len = max(len(e) for e in all_encoded)
    all_encoded_padded = np.full(
        (len(all_encoded), max_len), tokenizer.blank_token_idx, dtype = np.int32
    )
    for i, enc in enumerate(all_encoded):
        all_encoded_padded[i, : len(enc)] = enc

    asserts.assert_equal(
        all_encoded, tokenizer.encode(_default_texts, cleaned = True, return_type = 'list')
    )
    asserts.assert_equal(
        all_encoded_padded, tokenizer.encode(_default_texts, cleaned = True, return_type = 'np')
    )


# --- `__getitem__` / `__contains__` ------------------------------------------------

def test_getitem(asserts, tokenizer):
    # int -> symbol and symbol -> int are inverse
    asserts.assert_equal('_', tokenizer[0])
    asserts.assert_equal(0,   tokenizer['_'])
    asserts.assert_equal('H', tokenizer[tokenizer['H']])

    # a multi-character string falls back to `encode` (no sos/eos) -> list of ids
    asserts.assert_equal([tokenizer['H'], tokenizer['i']], tokenizer['Hi'])

    # membership
    assert 'H' in tokenizer
    assert '中' not in tokenizer   # a CJK char, not in the english vocab

    # an unknown, non-string index raises
    with pytest.raises(KeyError):
        tokenizer[None]


# --- encoding (edge cases) ---------------------------------------------------------

def test_call_is_encode_alias(asserts, tokenizer):
    for text in _default_texts:
        asserts.assert_equal(tokenizer.encode(text), tokenizer(text))

def test_encode_return_text(asserts, tokenizer):
    text, encoded = tokenizer.encode('Hello', cleaned = True, return_text = True)
    asserts.assert_equal('Hello', text)
    asserts.assert_equal(tokenizer.encode('Hello', cleaned = True), encoded)

def test_encode_dict_input(asserts, tokenizer):
    asserts.assert_equal(
        tokenizer.encode('Hello', cleaned = True),
        tokenizer.encode({'text' : 'Hello'}, cleaned = True),
    )

def test_encode_add_sos_eos_independently(asserts, tokenizer):
    base = tokenizer.encode('Hello', cleaned = True, add_sos_and_eos = False, return_type = 'list')
    sos, eos = tokenizer.sos_token_idx, tokenizer.eos_token_idx

    asserts.assert_equal(
        [sos] + base,
        tokenizer.encode('Hello', cleaned = True, add_sos = True, add_eos = False, return_type = 'list')
    )
    asserts.assert_equal(
        base + [eos],
        tokenizer.encode('Hello', cleaned = True, add_sos = False, add_eos = True, return_type = 'list')
    )
    asserts.assert_equal(
        [sos] + base + [eos],
        tokenizer.encode('Hello', cleaned = True, return_type = 'list')
    )

def test_encode_sos_eos_idempotent(asserts, tokenizer):
    """ Re-encoding a text that already carries ``<s>`` / ``</s>`` must not double them. """
    asserts.assert_equal(
        tokenizer.encode('Hello', cleaned = True, return_type = 'list'),
        tokenizer.encode(
            tokenizer.sos_token + 'Hello' + tokenizer.eos_token, cleaned = True, return_type = 'list'
        )
    )

def test_encode_out_of_vocab_dropped(asserts, tokenizer):
    """ Symbols outside the 150-token vocab (here digits) are silently dropped (no ``ukn``). """
    asserts.assert_equal(
        tokenizer.encode('Hi', cleaned = True, return_type = 'list'),
        tokenizer.encode('H123i', cleaned = True, return_type = 'list'),
    )

def test_encode_empty(asserts, tokenizer):
    asserts.assert_equal(
        [tokenizer.sos_token_idx, tokenizer.eos_token_idx],
        tokenizer.encode('', cleaned = True, return_type = 'list')
    )
    asserts.assert_equal(
        [], tokenizer.encode('', cleaned = True, add_sos_and_eos = False, return_type = 'list')
    )

def test_encode_runs_cleaners(asserts, tokenizer):
    """ ``cleaned = False`` applies ``english_cleaners`` (lower-casing) before encoding. """
    asserts.assert_equal(
        tokenizer.encode('hello world', cleaned = True),
        tokenizer.encode('Hello World', cleaned = False),
    )
    asserts.assert_not_equal(
        tokenizer.encode('Hello World', cleaned = True),
        tokenizer.encode('Hello World', cleaned = False),
    )

def test_encode_invalid_return_type(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.encode('Hello', cleaned = True, return_type = 'banana')

@pytest.mark.tensorflow
def test_encode_return_type_tf(asserts, tokenizer):
    encoded = tokenizer.encode('Hello', cleaned = True, return_type = 'tf')
    asserts.assert_tf_tensor(encoded)
    asserts.assert_equal(tokenizer.encode('Hello', cleaned = True), encoded)

@pytest.mark.keras
def test_encode_return_type_tensor(asserts, tokenizer):
    encoded = tokenizer.encode('Hello', cleaned = True, return_type = 'tensor')
    asserts.assert_tensor(encoded)
    asserts.assert_equal(tokenizer.encode('Hello', cleaned = True), encoded)


# --- decoding ----------------------------------------------------------------------

@pytest.mark.parametrize('text', _default_texts, ids = ['hello', 'bonjour', 'numbers', 'lorem'])
def test_decode_roundtrip(asserts, tokenizer, text):
    # only in-vocab chars survive `encode` -> that's what `decode` can give back
    in_vocab = ''.join(c for c in text if c in tokenizer)
    encoded  = tokenizer.encode(text, cleaned = True)

    asserts.assert_equal(tokenizer.sos_token + in_vocab + tokenizer.eos_token, tokenizer.decode(encoded))
    asserts.assert_equal(in_vocab, tokenizer.decode(encoded, remove_tokens = True))

def test_decode_ids(asserts, tokenizer):
    # a single id -> its symbol ; a list -> the char-level (sep = '') concatenation
    asserts.assert_equal('H', tokenizer.decode_ids(tokenizer['H']))
    asserts.assert_equal('Hello', tokenizer.decode_ids([tokenizer[c] for c in 'Hello']))

def test_decode_skip_padding(asserts, tokenizer):
    h, i = tokenizer['H'], tokenizer['i']
    pad  = tokenizer.blank_token_idx

    # an isolated padding token is skipped, decoding continues
    asserts.assert_equal('Hi', tokenizer.decode([h, pad, i]))
    # two consecutive padding tokens mark the end -> the rest is dropped
    asserts.assert_equal('H', tokenizer.decode([h, pad, pad, i]))

def test_decode_keep_padding(asserts, tokenizer):
    h, i = tokenizer['H'], tokenizer['i']
    pad  = tokenizer.blank_token_idx
    asserts.assert_equal(
        'H' + tokenizer.blank_token + 'i', tokenizer.decode([h, pad, i], skip_padding = False)
    )

def test_decode_remove_tokens(asserts, tokenizer):
    """ Regression : ``remove_tokens`` must strip special tokens *and* padding. """
    encoded = tokenizer.encode('Hi', cleaned = True, return_type = 'list')   # [sos, H, i, eos]
    asserts.assert_equal('Hi', tokenizer.decode(encoded, remove_tokens = True))
    asserts.assert_equal(
        'Hi', tokenizer.decode(encoded + [tokenizer.blank_token_idx], remove_tokens = True)
    )

def test_decode_batched(asserts, tokenizer):
    expected = ['<s>Hi</s>', '<s>Hello</s>']
    # padded 2-D `np.ndarray` -> list of strings (the trailing padding is trimmed)
    asserts.assert_equal(
        expected, tokenizer.decode(tokenizer.encode(['Hi', 'Hello'], cleaned = True, return_type = 'np'))
    )
    # list of (ragged) id lists -> list of strings
    asserts.assert_equal(
        expected, tokenizer.decode(tokenizer.encode(['Hi', 'Hello'], cleaned = True, return_type = 'list'))
    )

def test_decode_argmax_logits(asserts, tokenizer):
    """ A float ``(T, vocab)`` array is ``argmax``-ed before decoding ; batched -> list. """
    ids    = tokenizer.encode('Hello', cleaned = True, add_sos_and_eos = False, return_type = 'list')
    logits = np.full((len(ids), len(tokenizer)), -1., dtype = np.float32)
    for t, idx in enumerate(ids):
        logits[t, idx] = 1.

    asserts.assert_equal('Hello', tokenizer.decode(logits))
    asserts.assert_equal(['Hello', 'Hello'], tokenizer.decode(np.stack([logits, logits])))


# --- serialization round-trip ------------------------------------------------------

def test_config_roundtrip(asserts, tokenizer):
    clone = Tokenizer(** tokenizer.get_config())
    for text in _default_texts:
        asserts.assert_equal(tokenizer.encode(text), clone.encode(text))

def test_save_load(asserts, tokenizer, temp_dir):
    path = os.path.join(temp_dir, 'test_tokenizer.json')
    tokenizer.save(path)
    loaded = Tokenizer.load_from_file(path)

    for text in _default_texts:
        asserts.assert_equal(tokenizer.encode(text), loaded.encode(text))

    encoded = tokenizer.encode('Hello', cleaned = True)
    asserts.assert_equal(tokenizer.decode(encoded), loaded.decode(encoded))


# --- tensorflow graph (`tf.data` pipeline) -----------------------------------------

@pytest.mark.tensorflow
def test_tf_function(asserts, tokenizer):
    import tensorflow as tf

    pipe = tf.data.Dataset.from_tensor_slices(_default_texts).map(tokenizer.encode)
    for text, encoded in zip(_default_texts, pipe):
        asserts.assert_tf_tensor(encoded)
        asserts.assert_equal(tokenizer.encode(text), encoded)

    asserts.assert_equal(
        tokenizer.encode(_default_texts, return_type = 'np'),
        tf.function(tokenizer.encode)(tf.reshape(_default_texts, [-1]), shape = (None, None)),
    )


# --- byte-level BPE (synthetic, no `transformers`) ---------------------------------

def _byte_bpe_tokenizer(merges, ** kwargs):
    """ Byte-level BPE tokenizer whose vocab = the 256 bytes + the merged symbols """
    byte_encoder = bytes_to_unicode()
    vocab = list(byte_encoder.values()) + [''.join(pair) for pair in merges]
    return Tokenizer(
        vocab,
        level           = 'token',
        bpe_pairs       = merges,
        byte_encoder    = byte_encoder,
        split_pattern   = _gpt_pattern,
        ** kwargs
    )

def test_bpe_without_special_tokens(asserts):
    """ Regression : without special tokens, `split_text` returned the words as if they were
    special tokens, so BPE was never applied. """
    tokenizer = _byte_bpe_tokenizer([('l', 'l'), ('e', 'll'), ('Ġ', 'w')])
    asserts.assert_equal(['h', 'ell', 'o', 'Ġw', 'o', 'r', 'l', 'd'], tokenizer.tokenize('hello world'))
    asserts.assert_equal('hello world', tokenizer.decode(tokenizer.encode('hello world')))

def test_bpe_ignore_merges(asserts):
    """ `ignore_merges` (Llama-3) : a word in the vocab is used as is, even if the merges would split it. """
    # `abc` is in the vocab, but the merges only produce `ab` + `c`
    merges = [('a', 'b'), ('b', 'c'), ('a', 'bc')]
    asserts.assert_equal(['ab', 'c'], _byte_bpe_tokenizer(merges).tokenize('abc'))
    asserts.assert_equal(['abc'], _byte_bpe_tokenizer(merges, ignore_merges = True).tokenize('abc'))

def test_bpe_cache_is_bounded(asserts, monkeypatch):
    tokenizer = _byte_bpe_tokenizer([('l', 'l')])
    monkeypatch.setattr(tokenizer, 'bpe_cache_size', 4)
    for i in range(20): tokenizer.tokenize('word{}'.format(i))
    assert len(tokenizer._bpe_cache) <= 4
    asserts.assert_equal(['h', 'e', 'll', 'o'], tokenizer.tokenize('hello'))

def test_byte_level_special_tokens_decode(asserts):
    """ Special tokens are decoded from their raw content (not byte-decoded), even with non-ASCII characters. """
    tokenizer = _byte_bpe_tokenizer([], additional_tokens = ['<|im_start|>', '<|é_tok|>'])
    ids = tokenizer.encode('<|im_start|>héllo<|é_tok|>', return_type = 'list')
    asserts.assert_equal(tokenizer['<|im_start|>'], ids[0])
    asserts.assert_equal(tokenizer['<|é_tok|>'], ids[-1])
    asserts.assert_equal('<|im_start|>héllo<|é_tok|>', tokenizer.decode(ids))
    asserts.assert_equal('héllo', tokenizer.decode(ids, remove_tokens = True))

def test_byte_level_truncated_utf8(asserts):
    """ A sequence cut in a multi-byte character drops it, the rest is decoded normally. """
    tokenizer = _byte_bpe_tokenizer([])
    ids = tokenizer.encode('café', return_type = 'list')
    asserts.assert_equal('caf', tokenizer.decode(ids[: -1]))


# --- default SOS / EOS and config compatibility ------------------------------------

def test_add_sos_add_eos_defaults(asserts, tokenizer):
    config = {** tokenizer.get_config(), 'add_sos' : True, 'add_eos' : False}
    sos_only = Tokenizer(** config)
    base = tokenizer.encode('Hi', cleaned = True, add_sos_and_eos = False, return_type = 'list')

    asserts.assert_equal([sos_only.sos_token_idx] + base, sos_only.encode('Hi', cleaned = True, return_type = 'list'))
    # explicit arguments still override the defaults
    asserts.assert_equal(base, sos_only.encode('Hi', cleaned = True, add_sos_and_eos = False, return_type = 'list'))
    asserts.assert_equal(
        base + [sos_only.eos_token_idx], sos_only.encode('Hi', cleaned = True, add_sos = False, add_eos = True, return_type = 'list')
    )

def test_legacy_config(asserts, tokenizer, temp_dir):
    """ A config saved before `ignore_merges` / `add_sos` / `add_eos` behaves as before (`use_sos_and_eos`). """
    config = tokenizer.get_config()
    for key in ('ignore_merges', 'add_sos', 'add_eos'): config.pop(key)
    legacy = Tokenizer(** config)

    asserts.assert_equal(None, legacy.add_sos)
    for text in _default_texts:
        asserts.assert_equal(tokenizer.encode(text), legacy.encode(text))


# --- decoding (input types) --------------------------------------------------------

def test_decode_input_types(asserts, tokenizer):
    ids = tokenizer.encode('Hi', cleaned = True, add_sos_and_eos = False, return_type = 'list')
    # list of `np.integer` (used to be handled as a nested batch)
    asserts.assert_equal('Hi', tokenizer.decode([np.int64(i) for i in ids]))
    asserts.assert_equal('Hi', tokenizer.decode(tuple(ids)))
    # single id and empty sequences
    asserts.assert_equal('H', tokenizer.decode(ids[0]))
    asserts.assert_equal('', tokenizer.decode([]))
    asserts.assert_equal('', tokenizer.decode(np.zeros((0, ), dtype = np.int32)))


# --- `transformers` parity (heavy, needs the models in the HF cache) ---------------

_parity_corpus = _default_texts + [
    "Bonjour à tous, comment ça va aujourd'hui ? Très bien, merci.",
    "Le prix est de 12345,67 € (soit 3.14159 * 2024).",
    "I'LL DO IT, DON'T WORRY. She's here, they're gone.",
    "def foo(x):\n    return x ** 2\n\n\nclass Bar:\n\tpass\r\n",
    "Emojis 😀🎉 et CJK 中文字符 테스트 русский",
    "   multiple   spaces   and trailing  ",
    "naïve café — « guillemets » … ½ ﬁ Å",
    "Décomposé (NFD) : été, café",
    "line1\n\nline2\n \n  line3",
    "CamelCaseIdentifier snake_case_name HTTPServerError 1234567890",
    "## Titre\n\n- **gras** et `code` : [lien](https://example.com/a?b=1&c=2)\n",
    "<|im_start|>user\nSalut !<|im_end|>\n<|im_start|>assistant\n",
]

_parity_models = [
    'nvidia/Qwen3.8-27B-NVFP4',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    'BAAI/bge-m3',
    'google/flan-t5-large',
    'bofenghuang/vigostral-7b-chat',
    'bert-base-cased',
    'bert-base-uncased',
    'gpt2',
    'facebook/bart-large',
    'openai/whisper-base',
    'moussaKam/barthez',
]

@functools.cache
def _load_parity_pair(model_name):
    """ Returns `(reference, tokenizer)`, `rstrip = False` being required for an exact equivalence """
    transformers = pytest.importorskip('transformers')
    try:
        reference = transformers.AutoTokenizer.from_pretrained(model_name)
    except OSError as e:
        pytest.skip('{} is not available : {}'.format(model_name, e))
    return reference, Tokenizer.from_transformers_pretrained(model_name, rstrip = False)

def _reference_fns(model_name, reference, tokenizer):
    """ Returns the reference `(encode(text, add_special_tokens), decode(ids))` functions """
    if 'vigostral' in model_name:
        # `transformers >= 5` drops the leading `▁` of Llama-2 sentencepiece models without
        # `tokenizer.json` : the sentencepiece model itself is the reference
        sp_model = tokenizer.tokenizer
        return (
            lambda text, special: ([reference.bos_token_id] if special else []) + sp_model.encode(text),
            sp_model.decode
        )
    return (
        lambda text, special: reference.encode(text, add_special_tokens = special),
        lambda ids: reference.decode(ids, clean_up_tokenization_spaces = False)
    )

def _skip_text(model_name, tokenizer, text):
    """ Known (and accepted) differences, see the comments """
    # transformers 4 fast sentencepiece tokenizers keep a final `▁` for trailing whitespaces,
    # while sentencepiece itself (and transformers 5) drops it
    return type(tokenizer).__name__ == 'SentencePieceTokenizer' and text != text.rstrip()

@pytest.mark.slow
@pytest.mark.parametrize('model_name', _parity_models)
def test_transformers_parity(asserts, model_name):
    transformers = pytest.importorskip('transformers')
    if model_name == 'moussaKam/barthez' and int(transformers.__version__.split('.')[0]) >= 5:
        pytest.xfail('`transformers >= 5` fails to convert the BARThez tokenizer (upstream bug)')

    reference, tokenizer = _load_parity_pair(model_name)
    ref_encode, ref_decode = _reference_fns(model_name, reference, tokenizer)

    for text in _parity_corpus:
        if _skip_text(model_name, tokenizer, text): continue

        ids = ref_encode(text, False)
        asserts.assert_equal(
            ids, tokenizer.encode(text, add_sos_and_eos = False, return_type = 'list'), msg = repr(text)
        )
        # default special tokens (inferred from the post-processor). `add_sos is None` means that
        # it cannot be represented (e.g., the 2-tokens Whisper prefix) : the legacy behavior is kept
        if tokenizer.add_sos is not None:
            asserts.assert_equal(ref_encode(text, True), tokenizer.encode(text, return_type = 'list'), msg = repr(text))
        # the WordPiece decoder of `transformers` always cleans up the spaces around punctuation
        if 'bert' not in model_name:
            asserts.assert_equal(ref_decode(ids), tokenizer.decode(ids, skip_padding = False), msg = repr(text))

@pytest.mark.slow
@pytest.mark.parametrize('model_name', [
    'nvidia/Qwen3.8-27B-NVFP4', 'BAAI/bge-m3', 'bofenghuang/vigostral-7b-chat'
])
def test_transformers_decode_stream(asserts, model_name):
    """ `decode_stream` (token per token) rebuilds exactly `decode_ids`, sentencepiece models included. """
    _, tokenizer = _load_parity_pair(model_name)
    assert tokenizer.supports_incremental_decoding

    for text in _parity_corpus:
        ids = tokenizer.encode(text, return_type = 'list')
        state, deltas = None, []
        for i in range(1, len(ids) + 1):
            full, delta, state = tokenizer.decode_stream(ids[: i], state)
            deltas.append(delta)
        asserts.assert_equal(tokenizer.decode_ids(ids), full, msg = repr(text))
        asserts.assert_equal(full, ''.join(deltas), msg = repr(text))

@pytest.mark.slow
@pytest.mark.parametrize('model_name,sos,eos', [
    ('nvidia/Qwen3.8-27B-NVFP4',              False, False),
    ('meta-llama/Meta-Llama-3.1-8B-Instruct', True,  False),
    ('bert-base-uncased',                     True,  True),
    ('google/flan-t5-large',                  False, True),
])
def test_transformers_special_tokens(asserts, model_name, sos, eos):
    """ The default SOS / EOS come from the post-processor, and `rstrip` is disabled for chat models. """
    tokenizer = Tokenizer.from_transformers_pretrained(model_name)
    asserts.assert_equal(sos, tokenizer.add_sos)
    asserts.assert_equal(eos, tokenizer.add_eos)
    if tokenizer.template is not None:
        asserts.assert_equal(False, tokenizer.rstrip)

@pytest.mark.slow
def test_transformers_save_load(asserts, temp_dir):
    """ The converted config (cleaners, split pattern, `ignore_merges`, ...) survives a save / load. """
    _, tokenizer = _load_parity_pair('nvidia/Qwen3.8-27B-NVFP4')
    path = os.path.join(temp_dir, 'qwen_tokenizer.json')
    tokenizer.save(path)
    loaded = Tokenizer.load_from_file(path)

    for text in _parity_corpus:
        ids = tokenizer.encode(text, return_type = 'list')
        asserts.assert_equal(ids, loaded.encode(text, return_type = 'list'))
        asserts.assert_equal(tokenizer.decode(ids), loaded.decode(ids))

@pytest.mark.parametrize('cached', [True, False])
def test_transformers_loads_the_local_snapshot(monkeypatch, cached):
    """ `transformers` receives the snapshot directory of a cached repo id, the repo id otherwise """
    transformers = pytest.importorskip('transformers')
    from utils.text import tokenizer as tokenizer_module

    snapshot = '/hf/cache/models--org--model/snapshots/abc'
    resolved = []
    def resolve(name, *, validate = None, ** kwargs):
        resolved.append((name, kwargs))
        return snapshot if cached else name

    loaded = []
    def from_pretrained(name, ** kwargs):
        loaded.append(name)
        return 'processor'

    monkeypatch.setattr(tokenizer_module, 'resolve_hf_snapshot', resolve)
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', from_pretrained)

    result = tokenizer_module._load_transformers_tokenizer('org/model', revision = 'v1', fix_mistral_regex = True)
    assert result == 'processor'
    assert resolved == [('org/model', {'revision' : 'v1'})]
    assert loaded == [snapshot if cached else 'org/model']
