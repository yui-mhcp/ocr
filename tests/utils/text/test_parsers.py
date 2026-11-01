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

""" Tests for ``utils.text.parsers.parse_document`` : the parsing of a file, with or without its cache """

import os
import shutil

from utils.text import parse_document

def test_parse_without_cache(data_dir, tmp_path):
    """ `cache_dir = None` (e.g. a document of an in-memory conversation) : parsed, nothing written """
    filename = str(tmp_path / 'test.txt')
    shutil.copy(os.path.join(data_dir, 'files', 'test.txt'), filename)

    paragraphs = parse_document(filename, cache_dir = None)
    assert paragraphs and paragraphs[0]['text'] == 'Hello World !'
    assert paragraphs[0]['filename'] == filename
    assert os.listdir(str(tmp_path)) == ['test.txt']

def test_parse_with_cache(data_dir, tmp_path):
    """ The parsed paragraphs are stored in `cache_dir` (`map.json` + one file per document), then
        reloaded from it """
    filename, cache_dir = os.path.join(data_dir, 'files', 'test.txt'), str(tmp_path / 'cache')

    paragraphs = parse_document(filename, cache_dir = cache_dir)
    assert 'map.json' in os.listdir(cache_dir) and len(os.listdir(cache_dir)) == 2
    assert parse_document(filename, cache_dir = cache_dir) == paragraphs
