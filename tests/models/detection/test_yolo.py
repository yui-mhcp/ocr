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

""" Pure-``pytest`` tests for the YOLO target encoding / decoding and its loss.

No model is built (the full backbone has ~50M parameters) : the module-level
``encode_output`` / ``decode_output`` are tested on synthetic numpy data, and ``YoloLoss``
(``keras``-marked) on targets produced by ``encode_output``, so that the encoder and the
loss are checked against the same convention.

Coverage :
  - ``encode_output`` : center cell, best anchor, grid-unit box, class index (``str`` /
    ``bytes`` / unknown label), padded and degenerated boxes, ``true_boxes`` ;
  - ``YoloLoss`` : the class loss reads the encoded class index, and the no-object loss
    ignores a prediction overlapping a ground-truth box (``best_ious``) ;
  - ``decode_output`` : thresholding, ``xyxy`` normalized boxes, per-class NMS, batches.
"""

import numpy as np
import pytest

from models.detection.yolo import decode_output, encode_output

GRID    = 13
IMAGE   = 416   # 32 pixels per grid cell
ANCHORS = np.array([[1., 1.], [3., 3.], [6., 6.]], dtype = np.float32)
LABELS  = ['a', 'b', 'c']


def encode(boxes, labels, nb_box = None, max_box_per_image = 4):
    boxes = np.array(boxes, dtype = np.float32).reshape(-1, 4)
    return encode_output(
        boxes,
        labels,
        len(boxes) if nb_box is None else nb_box,
        IMAGE,
        IMAGE,
        grid_h  = GRID,
        grid_w  = GRID,
        anchors = ANCHORS,
        label_to_idx    = {label : i for i, label in enumerate(LABELS)},
        max_box_per_image   = max_box_per_image
    )

# a 3x3-cells box centered on the middle of the cell (6, 6) -> best anchor = 1 (3x3)
BOX = [160, 160, 96, 96]


# --- encode_output -----------------------------------------------------------------

def test_encode_single_box():
    output, true_boxes = encode([BOX], ['c'])

    assert output.shape == (GRID, GRID, len(ANCHORS), 6)
    np.testing.assert_allclose(output[6, 6, 1], [6.5, 6.5, 3., 3., 1., 2.])
    assert np.count_nonzero(output[..., 4]) == 1

    np.testing.assert_allclose(true_boxes[0], [6.5, 6.5, 3., 3.])
    assert not true_boxes[1:].any()

@pytest.mark.parametrize('label, expected', [
    pytest.param('b',       1, id = 'str'),
    pytest.param(b'b',      1, id = 'bytes'),
    pytest.param('unknown', 0, id = 'unknown'),
])
def test_encode_class_index(label, expected):
    output, _ = encode([BOX], np.array([label], dtype = object))
    assert output[6, 6, 1, 5] == expected

def test_encode_ignores_padded_boxes():
    output, true_boxes = encode([BOX, [0, 0, 64, 64]], ['a', ''], nb_box = 1)
    assert np.count_nonzero(output[..., 4]) == 1
    assert np.count_nonzero(true_boxes.any(axis = -1)) == 1

def test_encode_skips_empty_box():
    output, true_boxes = encode([[10, 10, 0, 20]], ['a'])
    assert not output.any() and not true_boxes.any()

def test_encode_best_anchor_by_shape():
    # a 6x6-cells box (192 pixels) goes to the 6x6 anchor
    output, _ = encode([[112, 112, 192, 192]], ['a'])
    assert output[6, 6, 2, 4] == 1.


# --- YoloLoss ----------------------------------------------------------------------

def _perfect_prediction(output, nb_class = len(LABELS)):
    """ Builds a `y_pred` exactly matching the encoded `output` (class = its encoded index) """
    pred = np.zeros(output.shape[:-1] + (5 + nb_class, ), dtype = np.float32)
    cells = np.stack(np.meshgrid(np.arange(GRID), np.arange(GRID)), axis = -1)[:, :, None, :]
    pred[..., :2] = cells + 0.5
    pred[..., 2:4] = 1e-3
    pred[..., 5] = 1.

    obj = output[..., 4] == 1
    pred[obj, :5] = output[obj, :5]
    pred[obj, 5:] = np.eye(nb_class)[output[obj, 5].astype(np.int32)]
    return pred

def _loss(output, true_boxes, pred):
    from custom_train_objects.losses.yolo_loss import YoloLoss

    losses = YoloLoss()((output[None], true_boxes[None]), pred[None])
    return {k : float(v) for k, v in losses.items()}

@pytest.mark.keras
def test_loss_class_uses_encoded_index():
    output, true_boxes = encode([BOX], ['c'])
    pred = _perfect_prediction(output)

    assert _loss(output, true_boxes, pred)['loss_class'] < 1e-4

    # predicting the class 0 (i.e. the one-hot position of the class index) must be penalized
    pred[6, 6, 1, 5:] = [1., 0., 0.]
    assert _loss(output, true_boxes, pred)['loss_class'] > 1.

@pytest.mark.keras
def test_loss_perfect_prediction():
    output, true_boxes = encode([BOX], ['a'])
    losses = _loss(output, true_boxes, _perfect_prediction(output))
    for name in ('loss_xy', 'loss_wh', 'loss_conf', 'loss_class'):
        assert losses[name] < 1e-4, losses

@pytest.mark.keras
def test_loss_no_object_ignores_overlapping_prediction():
    output, true_boxes = encode([BOX], ['a'])
    pred = _perfect_prediction(output)

    # a confident prediction on another anchor, overlapping the ground-truth box (IoU = 1)
    overlapping = pred.copy()
    overlapping[6, 6, 0, :5] = [6.5, 6.5, 3., 3., 1.]
    assert _loss(output, true_boxes, overlapping)['loss_conf'] < 1e-4

    # the same confidence on a box far from any ground-truth box is penalized
    far = pred.copy()
    far[0, 0, 0, :5] = [0.5, 0.5, 1., 1., 1.]
    assert _loss(output, true_boxes, far)['loss_conf'] > 1e-4


# --- decode_output -----------------------------------------------------------------

def _raw_output(* boxes):
    """ `boxes` = `(grid_y, grid_x, anchor, xywh, confidence, class index)` tuples """
    output = np.zeros((GRID, GRID, len(ANCHORS), 5 + len(LABELS)), dtype = np.float32)
    for gy, gx, anchor, xywh, conf, label_idx in boxes:
        output[gy, gx, anchor, :4] = xywh
        output[gy, gx, anchor, 4]  = conf
        output[gy, gx, anchor, 5 + label_idx] = 1.
    return output

def test_decode_single_box():
    out = decode_output(
        _raw_output((6, 6, 1, [6.5, 6.5, 3., 3.], 0.9, 1)), obj_threshold = 0.5, labels = LABELS
    )
    assert out['format'] == 'xyxy'
    assert out['labels'] == ['b']
    np.testing.assert_allclose(out['scores'], [0.9])
    np.testing.assert_allclose(out['boxes'], [[5. / 13, 5. / 13, 8. / 13, 8. / 13]], rtol = 1e-5)

def test_decode_below_threshold():
    out = decode_output(
        _raw_output((6, 6, 1, [6.5, 6.5, 3., 3.], 0.3, 1)), obj_threshold = 0.5, labels = LABELS
    )
    assert len(out['boxes']) == 0

def test_decode_nms_is_per_class():
    box = [6.5, 6.5, 3., 3.]
    same_class = decode_output(
        _raw_output((6, 6, 1, box, 0.9, 0), (6, 6, 2, box, 0.8, 0)), obj_threshold = 0.5
    )
    assert len(same_class['boxes']) == 1
    np.testing.assert_allclose(same_class['scores'], [0.9])

    other_class = decode_output(
        _raw_output((6, 6, 1, box, 0.9, 0), (6, 6, 2, box, 0.8, 2)), obj_threshold = 0.5
    )
    assert len(other_class['boxes']) == 2

def test_decode_batch():
    output = _raw_output((6, 6, 1, [6.5, 6.5, 3., 3.], 0.9, 1))
    out = decode_output(np.stack([output, np.zeros_like(output)]), obj_threshold = 0.5)
    assert isinstance(out, list) and [len(o['boxes']) for o in out] == [1, 0]
