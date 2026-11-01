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

import regex as re

from functools import cached_property

from ..file_utils import dump_json
from .tokenizer import Tokenizer

_byte_piece_re  = re.compile(r'<0x[0-9A-Fa-f]{2}>')

class SentencePieceTokenizer(Tokenizer):
    def __init__(self, vocab, tokenizer, *, offset = 0, ** kwargs):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, str):
            import sentencepiece

            self.tokenizer = sentencepiece.SentencePieceProcessor()
            with open(tokenizer, 'rb') as f:
                model_proto = f.read()
            self.tokenizer.Load(model_proto = model_proto)

        self.offset = offset
        kwargs['level'] = 'token'

        super().__init__(vocab = vocab, ** kwargs)

    @property
    def index_to_token(self):
        return {v : k for k, v in self.token_indexes.items()}

    @cached_property
    def space_replacement(self):
        return self.tokenizer.encode_as_pieces(' !')[0][0]

    def _split_text(self, text):
        return [text]

    def _tokenize(self, token):
        return self.tokenizer.encode_as_pieces(token)

    @cached_property
    def _id_to_bytes(self):
        """
            `dict` `{id : bytes}` : the space replacement (`▁`) is converted to a space, and the
            byte-fallback pieces (`<0xAB>`, e.g., Llama-2) to their byte. Special tokens are kept as is
            (as `transformers`), and the leading space (sentencepiece dummy prefix) is removed by
            `decode_ids`. Decoding is then a simple concatenation, which also enables `decode_stream`
            to only decode the new tokens.
        """
        space   = self.space_replacement
        result  = {}
        for idx, piece in self._id_to_symbol.items():
            if piece in self._special_tokens_set:
                result[idx] = piece.encode('utf-8')
            elif _byte_piece_re.fullmatch(piece):
                result[idx] = bytes([int(piece[3 : 5], 16)])
            else:
                result[idx] = piece.replace(space, ' ').encode('utf-8')
        return result

    def decode_ids(self, tokens):
        text = super().decode_ids(tokens)
        return text[1 :] if text.startswith(' ') else text

    def _decode_stream_chunk(self, ids, state):
        delta = super()._decode_stream_chunk(ids, state)
        # the leading space is only removed from the first (non-empty) decoded chunk
        if delta and not state['text'] and delta.startswith(' '): delta = delta[1 :]
        return delta

    def get_config(self):
        config = super().get_config()
        config['offset'] = self.offset
        return config

    def save(self, filename):
        model_path  = filename.replace('.json', '.model')

        with open(model_path, 'wb') as file:
            file.write(self.tokenizer.serialized_model_proto())

        config = self.get_config()
        config['tokenizer'] = model_path

        dump_json(filename, config, indent = 4)

        return filename
