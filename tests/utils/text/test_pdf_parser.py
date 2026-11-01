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

""" Tests for the layout analysis of ``utils.text.parsers.pdf_parser`` : the grouping of the blocks of
a page into lines, then of the lines into paragraphs, with non-text items (images) in between """

from utils.text.parsers.pdf_parser import group_blocks_in_lines, group_lines_in_paragraphs

def _text(text, y, x = 0.1, font_size = 10.):
    return {'text' : text, 'box' : [x, y, x + 0.3, y + 0.02], 'font_size' : font_size}

def _image(y):
    return {'type' : 'image', 'box' : [0.1, y, 0.6, y + 0.02]}

def test_an_image_is_a_line_of_its_own():
    """ An image overlapping the next text block vertically is not merged with it """
    blocks = [_text('Title', 0.1), _image(0.2), _text('first', 0.2, x = 0.6), _text('second', 0.2, x = 0.92)]
    lines  = group_blocks_in_lines(blocks)

    assert [line.get('type', 'text') for line in lines] == ['text', 'image', 'text']
    assert lines[2]['text'].split() == ['first', 'second']

def test_an_image_is_a_paragraph_of_its_own():
    """ An image between two lines of text splits them into two paragraphs, and stays alone """
    lines = [_text('A first line', 0.1), _image(0.13), _text('A second line', 0.16), _text('continued', 0.185)]
    paragraphs = group_lines_in_paragraphs(lines)

    assert [p.get('type', 'text') for p in paragraphs] == ['text', 'image', 'text']
    assert paragraphs[0]['text'] == 'A first line'
    assert paragraphs[2]['text'].split() == ['A', 'second', 'line', 'continued']
