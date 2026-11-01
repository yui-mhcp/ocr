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

"""
    `Translation` : a **multilingual text** (`{lang : text}`), resolved in a language only when it
    is rendered — e.g. the description of a tool, the summary a tool logs, an error message. The
    code producing a text never needs to know the language it will be read in.

    The texts are `str.format` templates : `format(* args, ** kwargs)` is **lazy** (it returns a new
    `Translation`, the arguments being applied at the resolution), so that a message can be built
    where its values are known and rendered later, in the language of the reader.

    The resolution of a language falls back on English, then on the first language given.

    ```python
    created = Translation(fr = '{name} : fiche créée', en = '{name}: card created')
    message = created.format(name = 'marc')
    message['fr']       # 'marc : fiche créée'
    message['de']       # 'marc: card created' (fallback)
    str(message)        # the English text
    (message + Translation(fr = ' ; ok', en = '; ok'))['fr']   # 'marc : fiche créée ; ok'
    ```
"""

import string

_DEFAULT_LANG   = 'en'

class Translation:
    """
        A multilingual `str.format` template (see the module docstring)
            - translations  : `{lang : text}` (a positional `dict` and / or the `lang = text` kwargs)
            - args / kwargs : the `format` arguments (`None` : a template, never formatted ; a
                              resolved text whose braces are literal, e.g. a concatenation)
    """
    __slots__ = ('translations', 'args', 'kwargs')

    def __init__(self, translations = None, /, *, _args = (), _kwargs = None, ** kwargs):
        translations = {** (translations or {}), ** kwargs}
        if not translations:
            raise ValueError('A `Translation` needs at least one language')
        elif not all(isinstance(text, str) for text in translations.values()):
            raise ValueError('The translations must be strings, got {}'.format(translations))

        self.translations   = translations
        self.args   = tuple(_args)
        self.kwargs = _kwargs

    @property
    def langs(self):
        return tuple(self.translations)

    @property
    def is_formatted(self):
        return self.kwargs is not None

    def template(self, lang = _DEFAULT_LANG):
        """ The raw text of `lang` (fallback : English, then the first language) """
        if lang in self.translations: return self.translations[lang]
        elif _DEFAULT_LANG in self.translations: return self.translations[_DEFAULT_LANG]
        return next(iter(self.translations.values()))

    def get(self, lang = _DEFAULT_LANG):
        """ The text of `lang`, formatted with the arguments of `format` """
        text = self.template(lang)
        if self.kwargs is None or (not self.args and not self.kwargs and '{' not in text):
            return text
        # the arguments may be `Translation`s themselves : resolved in the same language
        args    = [_resolve(arg, lang) for arg in self.args]
        kwargs  = {k : _resolve(v, lang) for k, v in self.kwargs.items()}
        return text.format(* args, ** kwargs)

    __getitem__ = get

    def format(self, * args, ** kwargs):
        """ A new `Translation` whose texts will be formatted with `args` / `kwargs` (lazy) """
        return Translation(self.translations, _args = args, _kwargs = kwargs)

    def fields(self, lang = _DEFAULT_LANG):
        """ The names of the replacement fields of the text of `lang` (`'{name!r}'` -> `'name'`) """
        return {
            name for _, name, _, _ in string.Formatter().parse(self.template(lang)) if name is not None
        }

    def to_dict(self):
        """ The raw texts `{lang : text}` (json-able : what defines it, e.g. for a prompt digest) """
        return dict(self.translations)

    def __add__(self, other):
        """ The concatenation, language by language (`other` : a `Translation` or a `str`) """
        if not isinstance(other, (Translation, str)): return NotImplemented
        langs = list(self.translations) + [
            lang for lang in getattr(other, 'translations', ()) if lang not in self.translations
        ]
        return Translation(
            {lang : self.get(lang) + get_translation(other, lang) for lang in langs}, _kwargs = None
        )

    def __radd__(self, other):
        if not isinstance(other, str): return NotImplemented
        return Translation({lang : other + self.get(lang) for lang in self.translations}, _kwargs = None)

    def __str__(self):
        return self.get(_DEFAULT_LANG)

    def __repr__(self):
        return 'Translation({})'.format(', '.join('{}={!r}'.format(k, v) for k, v in self.translations.items()))

    def __eq__(self, other):
        if not isinstance(other, Translation): return NotImplemented
        return (self.translations, self.args, self.kwargs) == (other.translations, other.args, other.kwargs)

    def __hash__(self):
        return hash(tuple(sorted(self.translations.items())))

def _resolve(value, lang):
    """ A `format` argument : a `Translation` in `lang`, any other value as is (its format spec applies) """
    return value.get(lang) if isinstance(value, Translation) else value

def get_translation(value, lang = _DEFAULT_LANG):
    """
        The text of `value` in `lang` :
            - a `Translation`   : resolved (see `Translation.get`)
            - a `str`           : returned as is
            - a `dict`          : `{lang : text}` (fallback : English, then the first language)
            - `None`            : `''`
            - any other value   : `str(value)`
    """
    if isinstance(value, Translation): return value.get(lang)
    elif isinstance(value, str): return value
    elif value is None: return ''
    elif isinstance(value, dict):
        if not value: return ''
        if lang in value: return value[lang]
        return value[_DEFAULT_LANG] if _DEFAULT_LANG in value else next(iter(value.values()))
    return str(value)
