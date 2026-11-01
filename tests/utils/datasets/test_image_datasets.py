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

""" Pure-``pytest`` tests for :mod:`utils.datasets.image_datasets`.

The ``Pascal VOC`` parser and the ``image_dataset_wrapper`` post-processing (``boxes`` as
``xywh`` arrays, ``nb_box``, ``pad_boxes``) are exercised on a tiny dataset written to the
``tmp_path`` fixture (2 images : one with 2 boxes, one with 1).
"""

import os
import numpy as np
import pytest

pd = pytest.importorskip('pandas')

from utils.datasets import get_dataset
from utils.datasets.loader import is_custom_dataset
from utils.datasets.image_datasets.voc import load_kangaroo, load_voc_annotations

_ANNOTATION = """<annotation>
    <filename>{filename}</filename>
    <size><width>{width}</width><height>{height}</height><depth>3</depth></size>
{objects}
</annotation>"""

_OBJECT = """    <object>
        <name>{label}</name>
        <bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox>
    </object>"""

_IMAGES = {
    '00001' : {'width' : 450, 'height' : 319, 'objects' : [('kangaroo', 233, 89, 386, 262), ('kangaroo', 134, 105, 341, 253)]},
    '00002' : {'width' : 300, 'height' : 200, 'objects' : [('kangaroo', 10, 20, 110, 220)]},
}


@pytest.fixture
def voc_dir(tmp_path):
    """ A `kangaroo`-like dataset (`images` / `annots` sub-directories) """
    os.makedirs(tmp_path / 'annots')
    for name, infos in _IMAGES.items():
        objects = '\n'.join(
            _OBJECT.format(label = label, x0 = x0, y0 = y0, x1 = x1, y1 = y1)
            for label, x0, y0, x1, y1 in infos['objects']
        )
        with open(tmp_path / 'annots' / '{}.xml'.format(name), 'w', encoding = 'utf-8') as f:
            # the extension is omitted for the 2nd image : the parser adds `.jpg`
            filename = name + '.jpg' if name == '00001' else name
            f.write(_ANNOTATION.format(
                filename = filename, width = infos['width'], height = infos['height'], objects = objects
            ))
    return str(tmp_path)


def test_datasets_are_registered():
    for name in ('voc', 'Pascal VOC', 'kangaroo', 'raccoon'):
        assert is_custom_dataset(name), name

def test_parse_voc_annotations(voc_dir):
    dataset = load_voc_annotations(voc_dir, 'images', 'annots')

    assert [row['filename'] for row in dataset] == [
        os.path.join(voc_dir, 'images', '00001.jpg'), os.path.join(voc_dir, 'images', '00002.jpg')
    ]
    assert [(row['height'], row['width']) for row in dataset] == [(319, 450), (200, 300)]
    assert dataset[0]['label'] == ['kangaroo', 'kangaroo']
    # `xyxy` in the xml -> `xywh`
    assert dataset[0]['boxes'] == [[233, 89, 153, 173], [134, 105, 207, 148]]
    assert dataset[1]['boxes'] == [[10, 20, 100, 200]]

def test_wrapper_converts_boxes(voc_dir):
    dataset = load_kangaroo(voc_dir)

    assert isinstance(dataset, pd.DataFrame)
    assert set(dataset.columns) == {'filename', 'label', 'height', 'width', 'boxes', 'nb_box'}
    assert dataset['nb_box'].tolist() == [2, 1]
    assert dataset['boxes'][1].dtype == np.int32 and dataset['boxes'][1].shape == (1, 4)

def test_wrapper_pad_boxes(voc_dir):
    dataset = load_kangaroo(voc_dir, pad_boxes = True)

    assert dataset['nb_box'].tolist() == [2, 1]   # the real number of boxes is kept
    assert [b.shape for b in dataset['boxes']] == [(2, 4), (2, 4)]
    np.testing.assert_array_equal(dataset['boxes'][1], [[10, 20, 100, 200], [0, 0, 0, 0]])
    assert dataset['label'][1] == ['kangaroo', '']

def test_get_dataset_with_directory(voc_dir):
    """ The `directory` argument overrides the default (`{DATASET_DIR}/kangaroo-master`) """
    dataset = get_dataset('kangaroo', directory = voc_dir, pad_boxes = True)

    assert len(dataset) == 2 and dataset['nb_box'].tolist() == [2, 1]
