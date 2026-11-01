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

""" Tests for the incremental decoding of :meth:`utils.text.Tokenizer.decode_stream`.

``decode_stream`` is the streaming counterpart of ``decode`` : it is fed the **cumulated**
token ids at each generation step and decodes only what is new, carrying the state (already
consumed ids, incomplete UTF-8 tail, decoded text) in an opaque ``dict`` held by the caller.

The contract exercised here is *equivalence* : whatever the chunking, the final text must be
byte-identical to ``decode_ids`` over the whole sequence, and the concatenation of the deltas
must rebuild exactly that text. Three properties are hardened in particular :

  - **byte-level tokens** : a step ending in the middle of a multi-byte character must emit
    nothing for that character rather than a broken/replacement one (the fixture decodes one
    **byte** per token, so every multi-byte character is necessarily split across steps) ;
  - **several tokens per step** : the runtime does not call back once per token, so the
    decoder must consume an arbitrary-length suffix, not "the last token" ;
  - **prefix invariant** : beam search may reorder or prune beams, so a sequence that does not
    extend the previous one must trigger a full re-decode instead of being appended.

Also covers the ``decode_ids`` fix : on a truncated multi-byte tail it used to swallow the
``UnicodeDecodeError`` and return the raw byte-encoded string, turning the **whole** text into
mojibake instead of just dropping the incomplete character.
"""

import pytest

from utils.text import Tokenizer
from utils.text.tokenizer import _gpt_pattern
from utils.text.text_processing import bytes_to_unicode

# Texts whose UTF-8 encoding spans 1-, 2-, 3- and 4-byte characters
_TEXTS = [
    'Hello World !',
    "Bonjour à tous, ça va ?",
    'Élève français : déjà vu, où ça ?',
    'emoji 🙂 mixed with accents éèê and CJK 日本語',
    'a' * 200 + ' é' * 50,
]


@pytest.fixture
def byte_tokenizer():
    """ Byte-level tokenizer (one token = one byte), as GPT-2 / Llama / Qwen / Mistral are.

        `split_pattern` is what makes `word_split` False (hence no space inserted between
        symbols when decoding), matching `Tokenizer.from_transformers_pretrained`.
    """
    byte_encoder = bytes_to_unicode()
    return Tokenizer(
        list(byte_encoder.values()),
        level           = 'token',
        byte_encoder    = byte_encoder,
        split_pattern   = _gpt_pattern,
    )


def _ids_for(tokenizer, text):
    """ One token id per UTF-8 byte of `text` """
    return [tokenizer._symbol_to_id[tokenizer.byte_encoder[b]] for b in text.encode('utf-8')]


def _stream(tokenizer, ids, step = 1):
    """ Feeds the **cumulated** ids by chunks of `step`, returning `(texts, deltas)` """
    texts, deltas, state = [], [], None
    for end in range(step, len(ids) + step, step):
        text, delta, state = tokenizer.decode_stream(ids[: min(end, len(ids))], state)
        texts.append(text)
        deltas.append(delta)
    return texts, deltas


# --- the fixture itself must round-trip, otherwise everything below is vacuous ------

@pytest.mark.parametrize('text', _TEXTS)
def test_fixture_round_trip(asserts, byte_tokenizer, text):
    asserts.assert_equal(text, byte_tokenizer.decode_ids(_ids_for(byte_tokenizer, text)))


def test_supports_incremental_decoding(asserts, byte_tokenizer):
    asserts.assert_equal(True, byte_tokenizer.supports_incremental_decoding)


# --- equivalence with the full decode ----------------------------------------------

@pytest.mark.parametrize('text', _TEXTS)
@pytest.mark.parametrize('step', [1, 2, 3, 5, 7, 64])
def test_matches_full_decode(asserts, byte_tokenizer, text, step):
    """ Whatever the chunking, the final text and the concatenated deltas must match `decode_ids` """
    ids = _ids_for(byte_tokenizer, text)
    texts, deltas = _stream(byte_tokenizer, ids, step)

    asserts.assert_equal(text, texts[-1])
    asserts.assert_equal(text, ''.join(deltas))


@pytest.mark.parametrize('text', _TEXTS)
def test_skip_padding_matches_decode(asserts, text):
    """ `skip_padding = True` : the padding tokens (e.g. the end token of llama-3.1) are dropped, like
        `decode` does, whatever the chunking """
    byte_encoder = bytes_to_unicode()
    tokenizer    = Tokenizer(
        list(byte_encoder.values()) + ['<pad>'], level = 'token', byte_encoder = byte_encoder,
        split_pattern = _gpt_pattern, pad_token = '<pad>'
    )
    pad = tokenizer.blank_token_idx
    ids = [pad] + _ids_for(tokenizer, text) + [pad]

    texts, deltas, state = [], [], None
    for end in range(1, len(ids) + 1):
        step_text, delta, state = tokenizer.decode_stream(ids[: end], state, skip_padding = True)
        texts.append(step_text)
        deltas.append(delta)

    asserts.assert_equal(text, tokenizer.decode(ids))
    asserts.assert_equal(text, texts[-1])
    asserts.assert_equal(text, ''.join(deltas))
    # without `skip_padding`, the padding token is decoded as is
    assert tokenizer.decode_stream(ids)[0] == '<pad>' + text + '<pad>'


@pytest.mark.parametrize('text', _TEXTS)
def test_every_step_is_a_prefix(asserts, byte_tokenizer, text):
    """ The emitted text only ever grows, and never exposes a broken character """
    ids = _ids_for(byte_tokenizer, text)
    texts, _ = _stream(byte_tokenizer, ids, 1)

    for step_text in texts:
        assert text.startswith(step_text), 'Not a prefix of the final text : {!r}'.format(step_text)
        assert '�' not in step_text, 'Broken character emitted : {!r}'.format(step_text)


@pytest.mark.parametrize('text', _TEXTS)
def test_delta_rebuilds_text_incrementally(asserts, byte_tokenizer, text):
    """ `text` is exactly the accumulation of the deltas emitted so far """
    ids = _ids_for(byte_tokenizer, text)
    texts, deltas = _stream(byte_tokenizer, ids, 1)

    accumulated = ''
    for step_text, delta in zip(texts, deltas):
        accumulated += delta
        asserts.assert_equal(step_text, accumulated)


def test_no_delta_when_no_new_id(asserts, byte_tokenizer):
    """ Re-feeding the same cumulated ids yields an empty delta (nothing to emit) """
    ids = _ids_for(byte_tokenizer, 'Bonjour à tous')

    text, delta, state = byte_tokenizer.decode_stream(ids, None)
    asserts.assert_equal('Bonjour à tous', text)

    text2, delta2, state = byte_tokenizer.decode_stream(ids, state)
    asserts.assert_equal(text, text2)
    asserts.assert_equal('', delta2)


def test_multi_byte_character_is_held_back(asserts, byte_tokenizer):
    """ A step cutting a multi-byte character emits nothing for it, and no replacement char """
    ids = _ids_for(byte_tokenizer, 'cafés')   # 'é' spans 2 bytes -> split across 2 steps

    texts, deltas = _stream(byte_tokenizer, ids, 1)

    # 'caf' + (nothing) + 'é' + 's'  -> the step holding the first byte of 'é' emits ''
    asserts.assert_equal('cafés', texts[-1])
    asserts.assert_equal(['c', 'a', 'f', '', 'é', 's'], deltas)


# --- prefix invariant (beam search) -------------------------------------------------

def test_full_redecode_when_prefix_breaks(asserts, byte_tokenizer):
    """ A sequence that does not extend the previous one must be re-decoded from scratch """
    first  = _ids_for(byte_tokenizer, 'the cat sat')
    second = _ids_for(byte_tokenizer, 'a dog ran')   # different beam : not a continuation

    _, _, state = byte_tokenizer.decode_stream(first, None)
    text, delta, state = byte_tokenizer.decode_stream(second, state)

    asserts.assert_equal('a dog ran', text)
    asserts.assert_equal('a dog ran', delta)


def test_full_redecode_when_sequence_shrinks(asserts, byte_tokenizer):
    """ A shorter sequence (pruned beam) is a prefix break too """
    ids = _ids_for(byte_tokenizer, 'Bonjour à tous')

    _, _, state = byte_tokenizer.decode_stream(ids, None)
    text, _, state = byte_tokenizer.decode_stream(ids[: 7], state)

    asserts.assert_equal(byte_tokenizer.decode_ids(ids[: 7]), text)


def test_state_is_not_shared(asserts, byte_tokenizer):
    """ Two concurrent requests must not interfere (the state lives in the caller) """
    ids_a, ids_b = _ids_for(byte_tokenizer, 'héllo'), _ids_for(byte_tokenizer, 'wörld')

    state_a = state_b = None
    for i in range(1, max(len(ids_a), len(ids_b)) + 1):
        if i <= len(ids_a): text_a, _, state_a = byte_tokenizer.decode_stream(ids_a[: i], state_a)
        if i <= len(ids_b): text_b, _, state_b = byte_tokenizer.decode_stream(ids_b[: i], state_b)

    asserts.assert_equal('héllo', text_a)
    asserts.assert_equal('wörld', text_b)


# --- nested (batch / beam) inputs ---------------------------------------------------

def test_accepts_nested_sequences(asserts, byte_tokenizer):
    """ Runtime streams yield `[batch][beam][ids]` : the first beam is followed """
    ids = _ids_for(byte_tokenizer, 'Bonjour')

    text, _, _ = byte_tokenizer.decode_stream([[ids]], None)
    asserts.assert_equal('Bonjour', text)


# --- the `decode_ids` truncation fix ------------------------------------------------

def test_decode_ids_drops_incomplete_tail(asserts, byte_tokenizer):
    """ A truncated multi-byte tail is dropped, instead of turning the whole text to mojibake """
    ids = _ids_for(byte_tokenizer, 'café')

    # drop the last byte of 'é' -> the character can no longer be decoded
    truncated = byte_tokenizer.decode_ids(ids[: -1])

    asserts.assert_equal('caf', truncated)
    assert 'Ã' not in truncated, 'Raw byte-encoded text leaked : {!r}'.format(truncated)


# --- tokenizers that cannot take the fast path --------------------------------------

def test_word_level_separator(asserts):
    """ `word_split` tokenizers join symbols with a space, including across chunks """
    tokenizer = Tokenizer(['hello', 'world', 'again'], level = 'word')

    texts, deltas = _stream(tokenizer, [0, 1, 2], 1)

    asserts.assert_equal(tokenizer.decode_ids([0, 1, 2]), texts[-1])
    asserts.assert_equal(texts[-1], ''.join(deltas))


def test_end_of_word_falls_back_to_full_decode(asserts):
    """ `bpe_end_of_word` is a whole-string replace : the fast path must be disabled """
    tokenizer = Tokenizer(
        ['a</w>', 'b</w>', 'ab'],
        level           = 'token',
        bpe_end_of_word = '</w>',
        split_pattern   = _gpt_pattern,
    )

    asserts.assert_equal(False, tokenizer.supports_incremental_decoding)

    texts, _ = _stream(tokenizer, [0, 1, 2], 1)
    asserts.assert_equal(tokenizer.decode_ids([0, 1, 2]), texts[-1])
