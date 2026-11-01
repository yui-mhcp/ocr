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
import enum
import json
import time
import codecs
import logging
import numpy as np
import regex as re

from itertools import repeat
from functools import cached_property, cache

from loggers import Timer, timer
from .. import load_json, dump_json, pad_batch, get_enum_item, is_dataframe, convert_to_str, timestamp_to_str, download_file, resolve_hf_snapshot
from ..keras import TensorSpec, ops, execute_eagerly
from .ctc_decoder import ctc_decode
from .cleaners import get_cleaners_fn, clean_text, strip
from .text_processing import bytes_to_unicode, bpe, format_text, split_and_join
from .tokens_processing import process_model_output

logger  = logging.getLogger(__name__)

text_signature          = TensorSpec(shape = (None, ), dtype = 'int32', name = 'text')
multi_text_signature    = TensorSpec(shape = (None, None), dtype = 'int32', name = 'text')
text_length_signature   = TensorSpec(shape = (None, ), dtype = 'int32', name = 'length')
multi_text_length_signature = TensorSpec(shape = (None, None), dtype = 'int32', name = 'length')

_clip_bpe_url   = 'https://raw.githubusercontent.com/openai/CLIP/master/clip/bpe_simple_vocab_16e6.txt.gz'
_whisper_url   = 'https://raw.githubusercontent.com/openai/whisper/master/whisper/assets/{}/{}'

# any one of them is enough for `transformers` to build the tokenizer of a cached snapshot
_tokenizer_files    = ('tokenizer.json', 'tokenizer_config.json', 'tokenizer.model', 'vocab.json')

_gpt_pattern    = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_clip_pattern   = r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
_llama_pattern  = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

        
class TokenizerLevel(enum.IntEnum):
    CHAR    = 0
    BPE     = 1
    TOKEN   = 1
    SUBWORD = 1
    WORD    = 2

class Tokenizer:
    # maximal number of words kept in the BPE cache (cleared once reached)
    bpe_cache_size  = 1 << 17

    def __init__(self,
                 vocab,
                 level,
                 
                 *,
                 
                 template   = None,
                 vocab_size     = None,
                 
                 lstrip     = False,
                 rstrip     = False,
                 cleaners   = [],
                 
                 split_pattern  = None,
                 
                 bpe_pairs      = None,
                 byte_encoder   = None,
                 bpe_end_of_word    = None,
                 ignore_merges  = False,

                 pad_token      = '',       # blank token
                 sos_token      = None,     # Start Of Sequence
                 eos_token      = None,     # End Of Sequence
                 ukn_token      = None,     # for unknown toekn (if not provided, skip them)
                 sep_token      = None,
                 mask_token     = None,
                 additional_tokens  = None,
                 
                 sub_word_prefix    = '',   # Add for inner sub-word part
                 use_sos_and_eos    = False,
                 add_sos    = None,
                 add_eos    = None,
                 add_special_tokens_at_end  = True,
                 
                 name           = 'Tokenizer'
                ):
        """
            Constructor for the `Tokenizer` class
            
            Arguments :
                - vocab     : list of tokens (words, sub-words or characters)
                - level     : tokenization level (char, token, word)
                - vocab_size    : special vocab_size (by default len(vocab))
                
                - lstrip / rstrip   : whether to strip left / right the text
                - cleaners      : cleaners to use for text-cleaning (see `cleaners.get_cleaners_fn`)
                - split_pattern : regex pattern used to split the text
                - bpe_pairs     : list containing the byte-pair encoding (BPE) pairs
                - byte_encoder  : mapping (`dict`) for byte-encoding
                - bpe_end_of_word   : special character to add at the end of words
                - ignore_merges     : whether to directly use a word if it is in the vocabulary, without applying BPE (Llama-3)

                - pad_token     : token to add for padding (also called `blank_token`)
                - ukn_token     : token to use for unknown (out-of vocabulary) tokens
                - sep_token     : special token to separate different parts
                - mask_token    : special token to mask some tokens (e.g., data augmentation or MLM)
                - {sos / eos}_token : start / end of sequence tokens
                
                - sub_word_prefix   : prefix to add at the start of sub-words (used in BERT)
                - use_sos_and_eos   : whether to add SOS and EOS at the encoded text extrema
                - add_sos / add_eos : default value for `encode(add_sos / add_eos)`, `None` means `use_sos_and_eos`
                                      (set by `from_transformers_pretrained` according to the HF post-processor)
                - add_special_tokens_at_end : whether to add special tokens at the end of the vocabulary (or at the beginning)
                
                - name      : special name for this encoder
        """
        self.name       = name
        self._vocab     = list(vocab)
        self.level      = get_enum_item(level, TokenizerLevel)
        self.template   = template
        
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.cleaners   = cleaners
        
        self.split_pattern  = split_pattern
        
        if bpe_pairs is not None and not byte_encoder: byte_encoder = bytes_to_unicode()
        self.bpe_pairs      =  [tuple(pair) for pair in bpe_pairs] if bpe_pairs is not None else None
        self.byte_encoder   =  {int(k) : v for k, v in byte_encoder.items()} if byte_encoder else None
        self.bpe_end_of_word    = bpe_end_of_word
        self.byte_encoder_inv   = {v : k for k, v in self.byte_encoder.items()} if byte_encoder else None
        self.ignore_merges  = ignore_merges
        # `str.translate` table equivalent to `byte_encoder` (applied on the `latin-1` view of the utf-8 bytes)
        self._byte_table    = {
            b : self.byte_encoder.get(b, None) for b in range(256)
        } if byte_encoder else None

        self.pad_token  = pad_token
        self.sos_token  = sos_token
        self.eos_token  = eos_token
        self.ukn_token  = ukn_token
        self.sep_token  = sep_token
        self.mask_token = mask_token
        self.sub_word_prefix    = sub_word_prefix
        self.use_sos_and_eos    = use_sos_and_eos
        self.add_sos    = add_sos
        self.add_eos    = add_eos
        self.add_special_tokens_at_end  = add_special_tokens_at_end

        if additional_tokens is None:            additional_tokens = []
        elif isinstance(additional_tokens, str): additional_tokens = [additional_tokens]
        self._additional_tokens  = additional_tokens
        
        self.splitter   = re.compile(split_pattern) if split_pattern else None
        self.bpe_ranks  = {pair : i for i, pair in enumerate(self.bpe_pairs)} if bpe_pairs is not None else None
        self.cleaners_fn    = get_cleaners_fn(cleaners)
        
        self._special_tokens    = list(self.tokens.values())
        self._special_tokens_set    = set(self._special_tokens)
        # longest first : a special token may be a prefix of another one
        self._tokens_split_re   = re.compile('({})'.format('|'.join([
            re.escape(tok) for tok in sorted(self._special_tokens_set, key = len, reverse = True)
        ]))) if self._special_tokens else None
        self._bpe_cache     = {}
        self._symbol_to_id  = {}
        self._id_to_symbol  = {}
        self.__build_indexes(vocab_size, add_special_tokens_at_end)
        # only the tokens modified by the cleaners have to be restored (cf `clean_text`)
        self._cleaned_tokens    = {}
        for token in self._special_tokens:
            cleaned = self.clean_text(token)
            if cleaned != token: self._cleaned_tokens[cleaned] = token

        for name, token in self.additional_tokens.items():
            if hasattr(self, name): continue

            setattr(self, name, token)
            setattr(self, name + '_idx', self[token])

        # `_tokenize` is called for each word : it is intentionally not wrapped by a `timer`,
        # whose overhead (even when disabled) was ~20% of the encoding time
        if not hasattr(self, '_tokenize'):
            if self.level != TokenizerLevel.TOKEN:
                self._tokenize = lambda token: [token]
            elif self.bpe_pairs is not None:
                self._tokenize = self._bpe_tokenize
            else:
                self._tokenize = self._sub_word_tokenize

        if not hasattr(self, '_split_text'):
            if self.splitter is not None:
                self._split_text = lambda text: re.findall(self.splitter, text)
            elif self.word_split:
                self._split_text = lambda text: text.split()
            else:
                self._split_text = lambda text: list(text)



    def __build_indexes(self, vocab_size, add_special_tokens_at_end):
        def _add_symbols(symbols):
            for s in symbols: self._symbol_to_id.setdefault(s, len(self._symbol_to_id))
        
        special_tokens = list(self.tokens.values())
        
        if vocab_size:
            for spe in special_tokens:
                if spe not in self._vocab: vocab_size -= 1
                
            if len(self._vocab) > vocab_size:
                logger.warning(
                    'Truncating vocab to size {} (from {})'.format(len(self._vocab), vocab_size)
                )
                self._vocab = self._vocab[: vocab_size]
            elif len(self._vocab) < vocab_size:
                self._vocab += ['ukn_{}'.format(i) for i in range(vocab_size - len(self._vocab))]
        
        self._symbol_to_id  = {}
        if not add_special_tokens_at_end:
            # Add special tokens (if required)
            _add_symbols(special_tokens)
        # Build `symbol to id` pairs
        _add_symbols(self._vocab)

        if add_special_tokens_at_end:
            # Add special tokens (if required)
            _add_symbols(special_tokens)

        # Build ID --> symbol pairs
        self._id_to_symbol  = {idx : s for s, idx in self._symbol_to_id.items()}
        
    @property
    def vocab_size(self):
        return len(self._id_to_symbol)
    
    @property
    def vocab(self):
        return sorted(self._symbol_to_id.keys(), key = self._symbol_to_id.get)
    
    @property
    def word_split(self):
        return self.level != TokenizerLevel.CHAR and self.splitter is None
    
    @cached_property
    def special_tokens(self):
        return set(v for k, v in self.get_config().items() if k.endswith('_token') and v is not None)
    
    @cached_property
    def additional_tokens(self):
        if isinstance(self._additional_tokens, (list, tuple)):
            return _create_tokens_name(self._additional_tokens)
        else:
            return self._additional_tokens
        
    @cached_property
    def tokens(self):
        _tokens = ('sep', 'ukn', 'sos', 'eos')
        
        tokens  = {
            k + '_token' : getattr(self, k + '_token') for k in _tokens
            if getattr(self, k + '_token', None) is not None
        }
        if not self.use_sos_and_eos:
            tokens.pop('sos_token', None)
            tokens.pop('eos_token', None)
        tokens.update(self.additional_tokens)
        return tokens
    
    @cached_property
    def token_indexes(self):
        return {v : self._symbol_to_id[v] for k, v in self.tokens.items()}
    
    @property
    def sos_token_idx(self):
        return self._symbol_to_id.get(self.sos_token, -1)
    
    @property
    def eos_token_idx(self):
        return self._symbol_to_id.get(self.eos_token, -1)
    
    @property
    def sep_token_idx(self):
        return self._symbol_to_id.get(self.sep_token, -1)
    
    @property
    def ukn_token_idx(self):
        return self._symbol_to_id.get(self.ukn_token, -1)
    
    @property
    def mask_token_idx(self):
        return self._symbol_to_id.get(self.mask_token, -1)
        
    @property
    def blank_token_idx(self):
        default = 0 if not self.use_sos_and_eos else self.eos_token_idx
        return self._symbol_to_id.get(self.pad_token, default)

    @property
    def blank_token(self):
        return self._id_to_symbol[self.blank_token_idx]
    
    def __str__(self):
        des = "========== {} ==========\n".format(self.name)
        des += "Vocab (size = {}) : {}\n".format(
            len(self), self.vocab if self.vocab_size <= 50 else '[{}, ...]'.format(
                str(self.vocab[:50])[1:-1]
            )
        )
        config = self.get_config()
        if self.template:
            des += 'Template :\n{}\n'.format(pretty_print_template(config.pop('template')))
        
        for k in ['name', 'vocab', 'bpe_pairs', 'byte_encoder']: config.pop(k)
        des += 'Config : {}'.format(json.dumps(config, indent = 2))
        return des
        
    def __len__(self):
        return self.vocab_size
    
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._id_to_symbol[idx]
        elif idx in self._symbol_to_id:
            return self._symbol_to_id[idx]
        elif isinstance(idx, str):
            res = self.encode(idx, add_sos_and_eos = False, cleaned = True, return_type = 'list')
            return res[0] if len(res) == 1 else res
        else:
            raise KeyError('Unknown index : {}'.format(idx))
    
    def __contains__(self, token):
        return token in self._symbol_to_id
    
    def _sub_word_tokenize(self, token):
        start, valid = 0, True
        sub_tokens = []
        while start < len(token):
            end = len(token)
            while start < end:
                sub_token = token[start : end]
                    
                sub = sub_token if start == 0 else self.sub_word_prefix + sub_token

                if sub in self._symbol_to_id:
                    sub_tokens.append(sub)
                    break
                end -= 1
                
            if end <= start:
                valid = False
                break
                
            start += len(sub_token)
        
        return sub_tokens if valid else [self.ukn_token]
    
    def _bpe_tokenize(self, token):
        tokens = self._bpe_cache.get(token)
        if tokens is None:
            byte_token = token.encode('utf-8').decode('latin-1').translate(self._byte_table)
            if self.ignore_merges and byte_token in self._symbol_to_id:
                tokens = (byte_token, )
            else:
                tokens = bpe(byte_token, self.bpe_ranks, end_of_word = self.bpe_end_of_word)

            # bounded cache, as the `Tokenizer` lives as long as the (server) process
            if len(self._bpe_cache) >= self.bpe_cache_size: self._bpe_cache.clear()
            self._bpe_cache[token] = tokens

        return tokens

    @timer(log_if_root = False)
    def clean_text(self, text, tokens = {}, ** kwargs):
        """ Apply all cleaners to 'text' """
        if self.cleaners_fn:
            if not isinstance(tokens, dict):
                tokens = {self.clean_text(token, ** kwargs) : token for token in tokens}

            text = clean_text(text, self.cleaners_fn, tokens = tokens, ** kwargs)
        return strip(text, lstrip = self.lstrip, rstrip = self.rstrip)
    
    @timer(log_if_root = False)
    def split_text(self, text):
        """
            Splits `text` into a list where each item is either a special token (`str`),
            either the `list` of words of the text between 2 special tokens
        """
        if not self._tokens_split_re:
            return [self._split_text(text)]

        splitted = []
        for part in re.split(self._tokens_split_re, text):
            if not part: continue
            elif part in self._special_tokens_set:
                splitted.append(part)
            else:
                splitted.append(self._split_text(part))

        return splitted

    @timer(log_if_root = False)
    def tokenize(self, text, cleaned = False, ** kwargs):
        if not cleaned:
            text = self.clean_text(text, self._cleaned_tokens, ** kwargs)

        splitted = self.split_text(text)

        tokens = []
        if self.level == TokenizerLevel.TOKEN:
            tokenize_fn = self._tokenize
            for part in splitted:
                if isinstance(part, str):
                    tokens.append(part)
                else:
                    for word in part: tokens.extend(tokenize_fn(word))
        else:
            for part in splitted:
                if isinstance(part, str): tokens.append(part)
                else:                     tokens.extend(part)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Cleaned  : {}\nSplitted : {}\nTokens  : {}'.format(text, splitted, tokens))
        
        return tokens
    
    @timer
    @execute_eagerly(signature = TensorSpec(shape = (None, ), dtype = 'int32'), numpy = True)
    def encode(self,
               text,
               
               *,
               
               cleaned  = False,
               
               add_sos  = None,
               add_eos  = None,
               add_sos_and_eos  = None,
               
               return_type  = 'np',
               return_text  = False
              ):
        """
            Encode text (str) into tokens (int)
            Arguments :
                - text : text to encode
                    Can be a string, dict, list / tuple / np.ndarray / Tensor of dict or str
            Return :
                - np.ndarray    : the encoded text (or list of np.ndarrays if multiple textwere given)
            
            1) Clean the text
            2) Split it into tokens (characters, words or sub-words) (cf `self.tokenize`)
            3) Convert all tokens to its corresponding id (with `self._symbol_to_id`)
            4) If necessary, add `sos` and `eos` tokens
        """
        return_type = convert_to_str(return_type)
        if add_sos is None:
            if add_sos_and_eos is not None: add_sos = add_sos_and_eos
            elif self.add_sos is not None:  add_sos = self.add_sos
            else:                           add_sos = self.use_sos_and_eos
        if add_eos is None:
            if add_sos_and_eos is not None: add_eos = add_sos_and_eos
            elif self.add_eos is not None:  add_eos = self.add_eos
            else:                           add_eos = self.use_sos_and_eos

        if is_dataframe(text):          text = text['text'].values.tolist()
        elif isinstance(text, dict):    text = text['text']
        text = convert_to_str(text)

        if isinstance(text, (list, tuple)):
            encoded = [
                self._encode(t, cleaned, add_sos, add_eos) if isinstance(t, str) else self.encode(
                    t, cleaned = cleaned, add_sos = add_sos, add_eos = add_eos, return_type = 'list'
                )
                for t in text
            ]

            if return_type == 'list': return encoded
            encoded = pad_batch(encoded, pad_value = self.blank_token_idx, dtype = 'int32')
            if return_type == 'np':     return encoded
            elif return_type == 'tf':   return ops.convert_to_tf_tensor(encoded, 'int32')
            elif return_type == 'tensor':   return ops.convert_to_tensor(encoded, 'int32')
            else:   raise ValueError("Unknown `return_type` : {}".format(return_type))

        tokens = self._encode(text, cleaned, add_sos, add_eos)

        if return_type == 'list': pass
        elif return_type == 'np': tokens = np.array(tokens, dtype = np.int32)
        elif return_type == 'tf': tokens = ops.convert_to_tf_tensor(tokens, dtype = 'int32')
        elif return_type == 'tensor':   tokens = ops.convert_to_tensor(tokens, dtype = 'int32')
        else:   raise ValueError("Unknown `return_type` : {}".format(return_type))
        
        return tokens if not return_text else (text, tokens)

    __call__    = encode

    def _encode(self, text, cleaned, add_sos, add_eos):
        """ Encodes a single `str` into a `list` of ids (see `encode`) """
        ukn_idx = self.ukn_token_idx
        get     = self._symbol_to_id.get
        tokens  = [get(token, ukn_idx) for token in self.tokenize(text, cleaned = cleaned)]
        if ukn_idx == -1: tokens = [token for token in tokens if token != -1]

        if (add_sos and self.sos_token) and (len(tokens) == 0 or tokens[0] != self.sos_token_idx):
            tokens.insert(0, self.sos_token_idx)
        if (add_eos and self.eos_token) and (len(tokens) == 0 or tokens[-1] != self.eos_token_idx):
            tokens.append(self.eos_token_idx)

        return tokens
    
    @timer
    @execute_eagerly(signature = TensorSpec(shape = (None, ), dtype = 'int32'), numpy = True)
    def encode_chat(self,
                    text = None,
                    *,

                    messages    = None,
                    paragraphs  = None,
                    answer_start    = None,
                    add_generation_prompt   = True,

                    system_prompt   = None,
                    prompt_format   = 'markdown',
                    
                    message_format  = None,
                    paragraphs_format   = None,
                    last_message_format = None,

                    encode  = True,
                    add_eos = None,
                    max_length  = None,
                    
                    return_text = False,
                    return_type = 'np',
                    
                    ** kwargs
                   ):
        """
            Encode `messages` according to `self.chat_template`. This is used for chat-based LLM.
            
            Arguments :
                - text  : the new message (handled as the new user message)
                
                - messages  : `list` of messages (`dict` or `models.nlu.Message`)
                              should have at least `content` and `role` entries
                - paragraphs    : `list` of paragraphs (`dict`) with a `type` entry (text, table, ...)
                - answer_start  : string added after the chat template, used to guide the start of generation
                - add_generation_prompt : bool, whether to add generation special tokens or not
                                          This special argument is used in most of modern templates
                
                
                - system_prompt : custom system message (used as the first message)
                                  if `messages[0]['role'] == 'system'`, this argument is ignored
                - prompt_format : special string used in custom system prompts (`models.nlu.prompts`)
                - message_format    : special format applied to each message
                - paragraphs_format : custom string to format paragraphs
                                      arguments are `paragraphs` and `kwargs`
                
                - last_message_format   : special format applied only on the last message
            Return :
                - tokens    : the encoded chat message
            
            **Important Note**: the `paragraphs` and `prompt_format` are custom arguments used in the custom system prompts defined in `models.nlu.prompts`. If these arguments are not exploited in any of the format (e.g., system_prompt, message_format, last_message_format), they will be ignored.
            If `paragraphs_format` is provided, the `paragraphs` argument forwarded to the chat template will be a `string`, while if `paragraphs_format` is not provided, `paragraphs` will be forwarded directly no matter what it is.
        """
        assert text or messages
        
        if add_eos is None: add_eos = not add_generation_prompt

        kwargs.update({
            'prompt_format' : prompt_format,
            'timestamp_to_str'  : timestamp_to_str
        })
        if 'date_string' in self.template and 'date_string' not in kwargs:
            kwargs['date_string'] = timestamp_to_str(time.time(), include_time = False)
                    
        kwargs.update(self.tokens)
        
        if paragraphs and paragraphs_format:
            with Timer('paragraphs preparation'):
                paragraphs = format_text(paragraphs_format, paragraphs = paragraphs, ** kwargs)
        
        kwargs['paragraphs'] = paragraphs

        with Timer('messages preparation'):
            if messages is None:                messages = []
            elif isinstance(messages, dict):    messages = [messages]
            elif isinstance(messages, str):     messages = [{'role' : 'user', 'content' : messages}]
            elif not isinstance(messages, list):
                raise ValueError('Unsupported `messages` type ({}) : {}'.format(
                    type(messages), messages
                ))

            if text:
                messages += [{'role' : 'user', 'content' : text}]

            if message_format:
                messages    = [{
                    ** (msg if isinstance(msg, dict) else msg.to_json()),
                    'content' : format_text(
                        message_format, text = msg['content'], message = msg, ** kwargs
                    )
                } if msg['role'] != 'system' else msg
                    for msg in messages]

            if last_message_format:
                messages[-1] = {** messages[-1], 'content' : format_text(
                    last_message_format, text = messages[-1]['content'], messages = messages, ** kwargs
                )}
            
            if system_prompt and messages[0]['role'] != 'system':
                messages = [{
                    'role'      : 'system',
                    'content'   : format_text(system_prompt, messages = messages, ** kwargs)
                }] + messages

        for _ in range(max(1, len(messages) - 1)):
            with Timer('apply template'):
                text = format_text(
                    self.template,
                    messages    = messages,
                    add_generation_prompt = add_generation_prompt,
                    ** kwargs
                )
                if add_generation_prompt and answer_start: text += answer_start
            
            if not encode: return text

            encoded = self.encode(text, add_sos = False, add_eos = add_eos, return_type = return_type)
            if not max_length or len(encoded) <= max_length:
                return encoded if not return_text else (text, encoded)
            
            messages.pop(1)
        
        raise ValueError('The message length ({}) exceeded the maximum length ({})'.format(
            len(encoded), max_length
        ))

    def decode(self, tokens, *, skip_padding = True, remove_tokens = False, ** _):
        """ Decode the given list of token ids into their corresponding token (str) """
        if hasattr(tokens, 'tokens'): tokens = process_model_output(tokens)
        
        if ops.is_tensor(tokens): tokens = ops.convert_to_numpy(tokens)
        if isinstance(tokens, np.ndarray):
            if np.issubdtype(tokens.dtype, np.floating) and all(s > 0 for s in tokens.shape):
                tokens = np.argmax(tokens, axis = -1)
            
            if tokens.ndim > 1:
                return [self.decode(
                    t, skip_padding = skip_padding, remove_tokens = remove_tokens
                ) for t in tokens]

            tokens = tokens.tolist()
            if not isinstance(tokens, list): tokens = [tokens]

        elif isinstance(tokens, (list, tuple)):
            if tokens and not isinstance(tokens[0], (int, np.integer)):
                return [self.decode(
                    t, skip_padding = skip_padding, remove_tokens = remove_tokens
                ) for t in tokens]
        else:
            tokens = [tokens]

        if remove_tokens: skip_padding = True

        if skip_padding:
            # 2 consecutive padding tokens mark the end of the sequence. `token_indexes` is
            # keyed by token *string* -> its id *values* are used to skip the special tokens
            blank   = self.blank_token_idx
            skip    = {blank}
            if remove_tokens: skip.update(self.token_indexes.values())

            tokens  = [token for token in tokens[: _find_padding_end(tokens, blank)] if token not in skip]

        return self.decode_ids(tokens)

    def decode_ids(self, tokens):
        """ Converts `tokens` (an id or a list of ids) into text, without any filtering (cf `decode`) """
        if not isinstance(tokens, (list, tuple)): tokens = [tokens]

        if self._id_to_bytes is not None:
            text = _decode_utf8(b''.join(map(self._id_to_bytes.get, tokens, repeat(b''))))
        else:
            sep  = ' ' if self.word_split else ''
            text = sep.join([self._id_to_symbol.get(t, '') for t in tokens])

        if self.level == TokenizerLevel.TOKEN:
            if self.sub_word_prefix:
                text = text.replace(' ' + self.sub_word_prefix, '')
            elif self.bpe_end_of_word:
                text = text.replace(self.bpe_end_of_word, ' ')

        return text

    @cached_property
    def _id_to_bytes(self):
        """
            `dict` `{id : bytes}` for byte-level tokenizers (`None` otherwise) : decoding is then a
            single `bytes.join`, instead of converting each character of each token

            Special / added tokens are not byte-encoded : their raw content is used (as `transformers`)
        """
        if self.byte_encoder is None: return None

        table   = {ord(c) : b for b, c in self.byte_encoder.items()}
        result  = {}
        for idx, symbol in self._id_to_symbol.items():
            if symbol not in self._special_tokens_set:
                try:
                    result[idx] = symbol.translate(table).encode('latin-1')
                    continue
                except UnicodeEncodeError:
                    pass
            result[idx] = symbol.encode('utf-8')
        return result

    @cached_property
    def supports_incremental_decoding(self):
        """
            Whether `decode_stream` can decode only the new tokens instead of re-decoding the
            whole sequence.

            The fast path requires the post-processing of `decode_ids` to be a no-op : both
            `sub_word_prefix` (BERT-like `##`) and `bpe_end_of_word` (CLIP-like `</w>`) are
            whole-string `str.replace` whose pattern may straddle a chunk boundary. Byte-level
            BPE tokenizers (GPT-2 / Llama / Qwen / Mistral), i.e. every LLM this is used with,
            set neither and therefore take the fast path.
        """
        if self.level != TokenizerLevel.TOKEN: return True
        return not self.sub_word_prefix and not self.bpe_end_of_word

    def decode_stream(self, tokens, state = None, *, skip_padding = False):
        """
            Incremental counterpart of `decode` for streaming generation.

            Arguments :
                - tokens    : the **cumulated** token ids, as yielded at each step by the
                              runtime streams (`InferenceStream`). Nested sequences are
                              descended into (batch / beam), like `decode(out)[0][0]`
                - state : the `dict` returned by the previous call (`None` on the first one).
                          It is **opaque** : callers only store it and pass it back
                - skip_padding  : drops the padding tokens, like `decode(skip_padding = True)`
                                  does : the streamed text then matches the final decoding
                                  (e.g. the `<|eot_id|>` end token of llama-3.1, which is its
                                  padding token, is never streamed)
            Return :
                - `(text, delta, state)` : the full decoded text, the part decoded by *this*
                  call, and the updated state. `delta` is `''` when the step produced no new
                  complete character (e.g. a multi-byte sequence still being buffered), so
                  consumers can skip emitting it

            The state is **per request** : it must never be shared across concurrent
            generations (`BaseModel.predict` may run several `infer` in parallel). This is why
            it lives in the caller rather than on this (shared, cached) `Tokenizer`.

            Falls back to a full re-decode whenever `tokens` does not extend the previously
            decoded ids : beam search may reorder or prune beams, which breaks the prefix
            invariant the incremental decoding relies on.
        """
        if state is None: state = {'ids' : [], 'text' : '', 'decoder' : None}

        ids = _to_id_list(tokens)
        if skip_padding:
            # the filtering of `decode` : it preserves the prefixes, hence the incremental decoding
            blank = self.blank_token_idx
            ids   = [token for token in ids[: _find_padding_end(ids, blank)] if token != blank]

        n = len(state['ids'])
        if len(ids) < n or ids[: n] != state['ids']:
            state.clear()
            state.update({'ids' : [], 'text' : '', 'decoder' : None})
            n = 0

        new_ids = ids[n :]
        if not new_ids:
            return state['text'], '', state

        if not self.supports_incremental_decoding:
            text  = self.decode_ids(ids)
            delta = text[len(state['text']) :] if text.startswith(state['text']) else text
        else:
            delta = self._decode_stream_chunk(new_ids, state)
            text  = state['text'] + delta

        state.update({'ids' : ids, 'text' : text})

        return text, delta, state

    def _decode_stream_chunk(self, ids, state):
        """ Decodes the new `ids` only, carrying the incomplete UTF-8 tail through `state` """
        if self._id_to_bytes is None:
            sep  = ' ' if self.word_split else ''
            text = sep.join([self._id_to_symbol.get(t, '') for t in ids])
            # the separator also goes *between* the already-decoded text and this chunk
            if sep and text and state['text']: text = sep + text
            return text

        if state['decoder'] is None:
            state['decoder'] = codecs.getincrementaldecoder('utf-8')('replace')

        # `final = False` : a trailing incomplete multi-byte sequence is buffered by the
        # decoder and emitted on the step that completes it
        return state['decoder'].decode(
            b''.join(map(self._id_to_bytes.get, ids, repeat(b''))), False
        )


    def ctc_decode(self, logits, lengths = None, method = 'beam', return_scores = False, ** kwargs):
        """ Decode a given np.ndarray by replacing each known id by its corresponding token """
        tokens, scores = ctc_decode(
            logits,
            method  = method,
            lengths = lengths,
            blank_index = self.blank_token_idx,
            ** kwargs
        )

        decoded = self.decode(tokens, ** kwargs)
        return decoded if not return_scores else (decoded, scores)

    def get_config(self):
        return {
            'name'      : self.name,
            'vocab'     : self._vocab,
            'level'     : self.level,
            'template'  : self.template,
            
            'lstrip'    : self.lstrip,
            'rstrip'    : self.rstrip,
            'cleaners'  : self.cleaners,
            
            'split_pattern' : self.split_pattern,
            
            'bpe_pairs' : self.bpe_pairs,
            'byte_encoder'  : self.byte_encoder,
            'bpe_end_of_word'   : self.bpe_end_of_word,
            'ignore_merges'     : self.ignore_merges,

            'pad_token' : self.pad_token,
            'sos_token' : self.sos_token,
            'eos_token' : self.eos_token,
            'sep_token' : self.sep_token,
            'ukn_token' : self.ukn_token,
            'mask_token'    : self.mask_token,
            'additional_tokens' : self._additional_tokens,
            
            'sub_word_prefix'   : self.sub_word_prefix,
            'use_sos_and_eos'   : self.use_sos_and_eos,
            'add_sos'   : self.add_sos,
            'add_eos'   : self.add_eos,
            'add_special_tokens_at_end' : self.add_special_tokens_at_end
        }
    
    def save(self, filename):
        """ Saves `self.get_config()` to `filename` (`json` format) """
        return dump_json(filename, self.get_config(), indent = 4)

    save_to_file = save
    
    @classmethod
    def load_from_file(cls, filename):
        config = load_json(filename)
        
        if 'tokenizer' in config:
            from .sentencepiece_tokenizer import SentencePieceTokenizer
            cls = SentencePieceTokenizer
        
        return cls(** config)

    @classmethod
    def from_transformers_pretrained(cls, name, *, load_kwargs = None, ** kwargs):
        """
            Converts a `transformers` tokenizer into a `Tokenizer`

            Arguments :
                - name  : the model name (or directory), or an already loaded tokenizer / processor
                          A cached model name is loaded from its local snapshot, without any
                          network request (see `_load_transformers_tokenizer`)
                - load_kwargs   : forwarded to `from_pretrained` (e.g., `{'fix_mistral_regex' : True}`)
                - kwargs    : overrides the converted config (e.g., `rstrip = False`)
            Return :
                - tokenizer : a `Tokenizer` (or `SentencePieceTokenizer`)

            The conversion relies on the serialized `tokenizers` pipeline (`backend_tokenizer.to_str()`), which is identical in `transformers` 4 (fast tokenizers) and 5 (where the "slow" tokenizers do not exist anymore) :
                - BPE + byte-level (GPT-2, BART, Whisper, Llama-3, Mistral, Qwen, ...) : vocab, merges, split regex and normalizer are read from the pipeline
                - WordPiece (BERT)  : `continuing_subword_prefix` and `BertNormalizer` options
                - Unigram / sentencepiece BPE (T5, XLM-R / BGE-M3, Llama-2, ...) : the original sentencepiece model (`vocab_file`) is used
            The SOS / EOS tokens added by default are inferred from the post-processor (e.g., only BOS for Llama-3, nothing for Qwen).

            Note : `rstrip = True` for models without chat template (GPT-2, BART, Whisper), as it is preferable in practice. It is disabled for chat models, as the generation prompt of most templates ends with `\n` (e.g., `<|im_start|>assistant\n`).
        """
        pretrained = name
        if isinstance(name, str):
            pretrained = _load_transformers_tokenizer(name, ** (load_kwargs or {}))

        chat_template = getattr(pretrained, 'chat_template', None)
        if hasattr(pretrained, 'tokenizer'):
            pretrained = pretrained.tokenizer
            if chat_template is None: chat_template = getattr(pretrained, 'chat_template', None)

        data = _get_backend_config(pretrained)
        sp_model, sp_offset = _get_sentencepiece_model(pretrained, data)

        config = {
            'vocab' : _get_vocab(pretrained),
            'level' : TokenizerLevel.TOKEN,
            'template'  : chat_template,
            'sos_token' : pretrained.bos_token,
            'eos_token' : pretrained.eos_token,
            'sep_token' : pretrained.sep_token,
            'ukn_token' : pretrained.unk_token,
            'mask_token'    : pretrained.mask_token
        }
        if data is not None:
            config['additional_tokens'] = [tok['content'] for tok in data.get('added_tokens', [])]
            config.update(_get_sos_and_eos_config(data.get('post_processor')))
        elif getattr(pretrained, 'additional_special_tokens', None):
            config['additional_tokens'] = list(pretrained.additional_special_tokens)

        model_type = data['model']['type'] if data is not None else None
        if sp_model is not None:
            from .sentencepiece_tokenizer import SentencePieceTokenizer
            cls = SentencePieceTokenizer

            config.update({'tokenizer' : sp_model, 'offset' : sp_offset})
        elif model_type == 'BPE':
            config.update(_get_bpe_config(data, chat_template))
        elif model_type == 'WordPiece':
            config.update(_get_wordpiece_config(data))
        else:
            raise ValueError('Unsupported tokenizer {} (model : {}) : {}'.format(
                type(pretrained).__name__, model_type,
                'no `tokenizers` backend' if data is None else 'the sentencepiece model is not available'
            ))

        config['use_sos_and_eos'] = config['sos_token'] is not None or config['eos_token'] is not None

        if pretrained.pad_token is not None:    config['pad_token'] = pretrained.pad_token
        elif config['eos_token']:               config['pad_token'] = config['eos_token']
        else:                                   config['pad_token'] = config['vocab'][0]

        config.update(kwargs)
        return cls(** config)

    @classmethod
    def from_clip_pretrained(cls, ** kwargs):
        import gzip
        
        from models import _pretrained_models_folder
        
        filename    = download_file(url = _clip_bpe_url, directory = os.path.join(
            _pretrained_models_folder, 'pretrained_weights'
        ))
        
        with gzip.open(filename) as file:
            pairs = file.read().decode('utf-8').split('\n')
        
        pairs = [tuple(pair.split()) for pair in pairs[1:49152-256-2+1]]
        
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for pair in pairs:
            vocab.append(''.join(pair))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        kwargs.update({
            'vocab' : vocab,
            'level' : 'bpe',
            'cleaners'  : {'name' : 'english_cleaners', 'to_lowercase' : False},
            'bpe_pairs' : {pair : i for i, pair in enumerate(pairs)},
            'byte_encoder'  : bytes_to_unicode(),
            'split_pattern' : _clip_pattern,
            'use_sos_and_eos'   : True,
            'pad_token'     : vocab[0],
            'sos_token'     : '<|startoftext|>',
            'eos_token'     : '<|endoftext|>',
            'bpe_end_of_word'   : '</w>'
        })
        return cls(** kwargs)

    @classmethod
    def from_whisper_pretrained(cls, name = 'openai/whisper-base', ** kwargs):
        """ `Whisper` tokenizer of the `name` checkpoint (`large-v3` has 1 more language token) """
        return cls.from_transformers_pretrained(name, ** kwargs)

def _to_id_list(tokens):
    """
        Normalizes a (possibly nested / tensor) token sequence into a flat `list[int]`.

        Nested sequences are descended into, taking the first element at each level : the
        runtime streams yield `[batch][beam][ids]`, and the streaming consumers only ever
        follow the first beam of the first request (as `decode(out)[0][0]` already did).
    """
    if hasattr(tokens, 'tokens'):       tokens = process_model_output(tokens)
    if ops.is_tensor(tokens):           tokens = ops.convert_to_numpy(tokens)
    if isinstance(tokens, np.ndarray):  tokens = tokens.tolist()

    tokens = list(tokens)
    while tokens and isinstance(tokens[0], (list, tuple)):
        tokens = list(tokens[0])

    # called with the **cumulated** ids at each streaming step : the (Python-level) conversion is
    # skipped for the common case (`tolist()` / runtime outputs), which made streaming O(n²)
    if tokens and type(tokens[0]) is not int: tokens = [int(t) for t in tokens]
    return tokens

def _create_tokens_name(tokens):
    names = {}
    for token in tokens:
        name = ''.join([c for c in token.lower() if c.isalnum() or c == '_'])
        for suffix in ('_id', '_tag'):
            if name.endswith(suffix): name = name.replace(suffix, '_token')
        if 'token' not in name: name += '_token'
        
        if name in names:
            if 'start' in name or 'end' in name:
                raise RuntimeError('Unable to infer a unique name for token {}'.format(token))
            
            if '/' in token:
                name += '_end'
            else:
                name += '_start'
        
        names[name] = token
    
    return names

def _find_padding_end(ids, pad):
    """ Returns the index of the first `[pad, pad]` pair (the start of the padding), `len(ids)` if none """
    i = -1
    try:
        while True:
            i = ids.index(pad, i + 1)
            if i + 1 < len(ids) and ids[i + 1] == pad: return i
    except ValueError:
        return len(ids)

def _decode_utf8(data):
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        # The sequence ends in the middle of a multi-byte character, which is the normal case
        # for a still-generating (or truncated) output. The incremental decoder buffers that
        # incomplete tail internally, so it is simply dropped (invalid bytes in the middle are
        # replaced), instead of turning the **whole** string into mojibake.
        return codecs.getincrementaldecoder('utf-8')('replace').decode(data, False)

def _missing_tokenizer_files(path):
    """ `[]` if the snapshot `path` holds at least one tokenizer file (see `resolve_hf_snapshot`) """
    if any(os.path.exists(os.path.join(path, f)) for f in _tokenizer_files): return []
    return ['a tokenizer file ({})'.format(' / '.join(_tokenizer_files))]

def _load_transformers_tokenizer(name, ** kwargs):
    """
        Loads the `AutoProcessor` (whose `chat_template` may differ from the tokenizer one, e.g.
        for multimodal models like Mistral-Small-3.1), and falls back to `AutoTokenizer`

        A cached repo id is loaded from its local snapshot directory : given a repo id, every
        sub-loader of `transformers` checks each of its files against the hub (~50 requests for
        Qwen3.8), `local_files_only` not preventing the `model_info` call of `_patch_mistral_regex`
    """
    from transformers import AutoProcessor, AutoTokenizer

    name = resolve_hf_snapshot(name, validate = _missing_tokenizer_files, ** {
        k : kwargs[k] for k in ('revision', 'cache_dir') if kwargs.get(k, None)
    })
    try:
        return AutoProcessor.from_pretrained(name, ** kwargs)
    except Exception as e:
        logger.debug('`AutoProcessor` failed for {} ({}), using `AutoTokenizer`'.format(name, e))
        return AutoTokenizer.from_pretrained(name, ** kwargs)

def _get_backend_config(pretrained):
    """ Returns the serialized `tokenizers` pipeline (`tokenizer.json` content), `None` for slow tokenizers """
    backend = getattr(pretrained, 'backend_tokenizer', None)
    return json.loads(backend.to_str()) if backend is not None else None

def _get_vocab(pretrained):
    """ Returns the vocabulary (`list`) such that `vocab[i]` is the token with id `i` (added tokens included) """
    vocab = pretrained.get_vocab()
    id_to_token = {idx : token for token, idx in vocab.items()}
    if len(id_to_token) != len(vocab):
        logger.warning('Multiple tokens share the same id in the vocabulary : only the last one is kept')
    return [id_to_token.get(i, '<|unused_{}|>'.format(i)) for i in range(max(id_to_token) + 1)]

def _iter_components(component, key):
    """ Flattens a (possibly `Sequence`) pipeline component (`normalizer`, `pre_tokenizer`, ...) """
    if not component: return
    if component['type'] == 'Sequence':
        for sub in component.get(key, []): yield from _iter_components(sub, key)
    else:
        yield component

def _is_byte_level(data):
    return any(
        c['type'] == 'ByteLevel' for c in _iter_components(data.get('pre_tokenizer'), 'pretokenizers')
    ) or any(
        c['type'] == 'ByteLevel' for c in _iter_components(data.get('decoder'), 'decoders')
    )

def _get_sentencepiece_model(pretrained, data = None):
    """
        Returns the `(SentencePieceProcessor, offset)` of a sentencepiece-based `transformers`
        tokenizer, `(None, 0)` for any other one.

        Slow tokenizers (`transformers < 5`) expose it as `sp_model` (with the `fairseq_offset`).
        Otherwise, a Unigram (T5, XLM-R / BGE-M3, ...) or a non byte-level BPE (Llama-2, BARThez, ...)
        tokenizer keeps the original sentencepiece model as `vocab_file`. The offset between the
        sentencepiece and the tokenizer ids is then inferred from a regular piece.

        Note : the sentencepiece model is the reference, as `transformers >= 5` may not reproduce it
        when there is no `tokenizer.json` (e.g., the leading `▁` is lost for Llama-2 models)
    """
    sp_model = getattr(pretrained, 'sp_model', None)
    if sp_model is not None:
        return sp_model, getattr(pretrained, 'fairseq_offset', 0)

    if data is None or data['model']['type'] not in ('Unigram', 'BPE') or _is_byte_level(data):
        return None, 0

    vocab_file = getattr(pretrained, 'vocab_file', None)
    if not vocab_file or not vocab_file.endswith('.model') or not os.path.exists(vocab_file):
        return None, 0

    import sentencepiece

    sp_model = sentencepiece.SentencePieceProcessor()
    sp_model.Load(vocab_file)

    piece_id = min(10, sp_model.get_piece_size() - 1)
    offset   = pretrained.convert_tokens_to_ids(sp_model.id_to_piece(piece_id)) - piece_id
    return sp_model, offset

def _get_template_tokens(post_processor):
    """
        Returns the `(prefix, suffix)` special tokens added around a single sequence by the
        post-processor, or `None` if it is not supported
    """
    if not post_processor: return [], []

    kind = post_processor['type']
    if kind == 'Sequence':
        prefix, suffix = [], []
        for processor in post_processor.get('processors', []):
            tokens = _get_template_tokens(processor)
            if tokens is None: return None
            prefix, suffix = prefix + tokens[0], tokens[1] + suffix
        return prefix, suffix
    elif kind == 'ByteLevel':
        return [], []
    elif kind in ('RobertaProcessing', 'BertProcessing'):
        return [post_processor['cls'][0]], [post_processor['sep'][0]]
    elif kind == 'TemplateProcessing':
        prefix, suffix = [], []
        current = prefix
        for item in post_processor['single']:
            if 'Sequence' in item:
                current = suffix
            else:
                special = item['SpecialToken']['id']
                current.extend(post_processor['special_tokens'].get(special, {}).get('tokens', [special]))
        return prefix, suffix
    return None

def _get_sos_and_eos_config(post_processor):
    """
        Infers the default `add_sos` / `add_eos` (and the SOS / EOS tokens) from the post-processor.
        If it cannot be represented by a single SOS / EOS (e.g., the Whisper `<|startoftranscript|><|notimestamps|>`
        prefix), the legacy behavior (`use_sos_and_eos`) is kept
    """
    tokens = _get_template_tokens(post_processor)
    if tokens is None or len(tokens[0]) > 1 or len(tokens[1]) > 1:
        logger.info('The post-processor cannot be converted : `use_sos_and_eos` is used')
        return {}

    prefix, suffix = tokens
    result = {'add_sos' : bool(prefix), 'add_eos' : bool(suffix)}
    if prefix: result['sos_token'] = prefix[0]
    if suffix: result['eos_token'] = suffix[0]
    return result

def _get_split_pattern(pre_tokenizer):
    """ Returns the regex used to split the text into words (`None` if no split) """
    patterns, unsupported = [], []
    for component in _iter_components(pre_tokenizer, 'pretokenizers'):
        if component['type'] == 'Split':
            pattern = component['pattern']
            if component.get('invert') or component.get('behavior', 'Isolated') != 'Isolated':
                unsupported.append('Split({})'.format(component.get('behavior')))
            patterns.append(pattern['Regex'] if 'Regex' in pattern else re.escape(pattern['String']))
        elif component['type'] == 'ByteLevel':
            if component.get('add_prefix_space'): unsupported.append('ByteLevel(add_prefix_space)')
            if component.get('use_regex', True):  patterns.append(_gpt_pattern)
        else:
            unsupported.append(component['type'])

    if unsupported or len(patterns) > 1:
        logger.warning(
            'The pre-tokenizer is not fully supported ({}) : the tokenization may slightly differ'.format(
                ', '.join(unsupported) if unsupported else 'multiple splits'
            )
        )
    return patterns[0] if patterns else None

def _get_normalizer_cleaners(normalizer):
    """ Converts the normalizer into `cleaners` """
    cleaners, unsupported = [], []
    for component in _iter_components(normalizer, 'normalizers'):
        kind = component['type']
        if kind in ('NFC', 'NFKC', 'NFD', 'NFKD'):
            cleaners.append({'name' : 'normalize_unicode', 'form' : kind})
        elif kind == 'Lowercase':
            cleaners.append('lowercase')
        elif kind == 'StripAccents':
            cleaners.append('remove_accents')
        elif kind == 'BertNormalizer':
            # `clean_text` (control characters removal) is intentionally skipped : it reduces
            # performances by 30% for really rare occurences
            strip_accents = component.get('strip_accents')
            if strip_accents is None: strip_accents = component.get('lowercase', False)

            if component.get('handle_chinese_chars', False): cleaners.append('pad_cjk')
            if component.get('lowercase', False):            cleaners.append('lowercase')
            if strip_accents:                                cleaners.append('remove_accents')
        else:
            unsupported.append(kind)

    if unsupported:
        logger.warning('Unsupported normalizer(s) {} : they are ignored'.format(unsupported))
    return cleaners

def _get_bpe_config(data, chat_template):
    if not _is_byte_level(data):
        raise ValueError('Only byte-level BPE tokenizers are supported without sentencepiece model')

    model   = data['model']
    merges  = model['merges']
    if merges and isinstance(merges[0], str):
        merges = [tuple(pair.split(' ')) for pair in merges]

    if model.get('continuing_subword_prefix'):
        logger.warning('`continuing_subword_prefix` is not supported for BPE tokenizers')

    return {
        'lstrip'    : False,
        'rstrip'    : chat_template is None,
        'cleaners'  : _get_normalizer_cleaners(data.get('normalizer')),
        'split_pattern' : _get_split_pattern(data.get('pre_tokenizer')),
        'bpe_pairs'     : merges,
        'byte_encoder'  : bytes_to_unicode(),
        'bpe_end_of_word'   : model.get('end_of_word_suffix') or None,
        'ignore_merges'     : model.get('ignore_merges', False) or False
    }

def _get_wordpiece_config(data):
    return {
        'cleaners'  : _get_normalizer_cleaners(data.get('normalizer')) + [
            'split_punctuation', 'collapse_whitespace'
        ],
        'sub_word_prefix'   : data['model'].get('continuing_subword_prefix', '##')
    }

def pretty_print_template(template):
    if '\n' in template: return template
    
    indent = 0
    str_template = ''
    for part in template.split('{'):
        if not part: continue
        p_indent = indent
        if part[0] != '%':
            part = '{' + part
        elif part[2:].strip().startswith(('if', 'for')):
            indent += 1
        elif part[2:].strip().startswith(('elif', 'else')):
            p_indent -= 1
        elif part[2:].strip().startswith('end'):
            indent -= 1
            p_indent -= 1

        str_template += ' ' * p_indent * 4 + '{' + part + '\n'
    return str_template