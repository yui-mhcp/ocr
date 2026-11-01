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

""" Shared, dependency-light helpers for the ``models.ocr.region_tracker`` test-suite.

The whole suite is **keras-free** (unmarked) : importing ``models.ocr`` triggers the
package auto-registration (``import_submodules``), but none of the OCR modules import a
deep-learning backend at load time (``utils.keras.ops`` dispatches ``np.ndarray`` to numpy,
``cv2`` / ``keras`` are pulled lazily). The tracker itself is pure numpy (IoU + Dice +
edit-distance), so its behaviour is exercised with plain ``np.ndarray`` frames.

The builders below mirror the tracker's real inputs :
  - :func:`box` / :func:`score_map` construct the relative-``xyxy`` boxes and the detector
    score-map (``(H, W)`` float) the tracker compares between frames ;
  - :func:`scan` replays one *scanned frame with detections* exactly like
    ``BaseOCR.stream_fn`` does around ``infer`` : ``set_frame`` (score map) then the
    ``BoxFilter.__call__`` pass (``start`` / ``filter`` / ``finish``), returning the indexes
    the tracker kept for OCR.
"""

import numpy as np

from models.ocr.region_tracker import RegionTracker


# --- builders ----------------------------------------------------------------------

def make_tracker(** kwargs):
    """ A :class:`RegionTracker` with the production defaults, overridable per-test. """
    defaults = dict(
        match_threshold    = 0.4,
        geometry_threshold = 0.75,
        n_stable   = 2,
        max_missing    = 3,
        stability_threshold    = 0.85,
        binarize_threshold     = 0.5,
        erosion    = 0.15,
        text_dist_threshold    = 0.2,
        memory_size    = 15,
    )
    defaults.update(kwargs)
    return RegionTracker(** defaults)


def box(x0, y0, x1, y1):
    """ A single relative-``xyxy`` box as a ``(4,)`` float32 array. """
    return np.array([x0, y0, x1, y1], dtype = 'float32')


def score_map(boxes = (), *, shape = (100, 100), value = 1.0):
    """ A ``(H, W)`` float32 detector score-map with ``value`` inside each relative box. """
    h, w = shape
    m = np.zeros((h, w), dtype = 'float32')
    for b in boxes:
        x0, y0, x1, y1 = b
        m[int(round(y0 * h)) : int(round(y1 * h)), int(round(x0 * w)) : int(round(x1 * w))] = value
    return m


# --- frame replay ------------------------------------------------------------------

def scan(tracker, boxes, *, score_map = None, rows = None):
    """ Replay one scanned frame : optionally register the score map, then run the
        ``BoxFilter`` pass. Returns the kept indexes (candidates selected for OCR).

        ``rows`` defaults to one row per box (``(1, 4)``), matching ``filter_boxes``.
    """
    boxes = np.asarray(boxes, dtype = 'float32')
    if rows is None:
        rows = [boxes[i : i + 1] for i in range(len(boxes))]
    indices = list(range(len(boxes)))

    if score_map is not None:
        tracker.set_frame(score_map)
    return np.asarray(tracker(boxes, indices, rows))
