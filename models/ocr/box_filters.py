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

import logging
import numpy as np

from dataclasses import dataclass

from utils.image import compute_iou, convert_box_format

logger  = logging.getLogger(__name__)

def filter_boxes(filters, boxes, indices = None, rows = None, ** kwargs):
    """
        Filter `boxes` by applying `filters` sequentially. A box is kept if it passes all the filters.

        Arguments :
            - filters   : a `list` of `callable`, the filters to apply
                          Should return the list of box indexes to keep
            - boxes     : `np.ndarray` of boxes (supposed to be in "xyxy" format)
            - indices   : `list` of box indexes (defaults to `range(len(boxes))`)
            - rows      : `list` of `np.ndarray` (defaults to 1 row per box)
        Return :
            - filtered_boxes    : `np.ndarray` with the boxes to keep
            - filtered_indices  : indices associated to `filtered_boxes`
            - filtered_rows     : list of rows associated to `filtered_boxes`
    """
    if callable(filters): filters = [filters]
    if indices is None: indices = list(range(len(boxes)))
    if rows is None:    rows = [boxes[i : i + 1] for i in range(len(boxes))]

    for f in filters:
        if len(boxes) == 0: return boxes, indices, rows
        keep_indexes    = f(boxes = boxes, indices = indices, rows = rows, ** kwargs)

        if len(keep_indexes) < len(boxes):
            boxes   = boxes[keep_indexes]
            rows    = [rows[idx] for idx in keep_indexes]
            indices = [indices[idx] for idx in keep_indexes]

    return boxes, indices, rows

class BoxFilter:
    """ Abstract class representing a box filtering strategy """
    def __call__(self, boxes, indices, rows, source = 'xyxy', ** kwargs):
        """
            Filtering method that takes the boxes / indices / rows (typically returned by `combine_boxes`) and returns the indices to keep

            Arguments :
                - boxes : 2-D `np.ndarray` with shape `(n_boxes, 4)`, the original boxes
                - indices   : list of combination indexes (mainly not used)
                - rows  : individual rows composing a box, `list` with `n_boxes` items, each item being a 2-D `np.ndarray` with shape `(n_rows, 4)`
            Return :
                - keep_indexes  : list of index(es) to keep

        """
        self.start()
        res = self.filter(boxes = boxes, indices = indices, rows = rows, source = source)
        self.finish()

        if isinstance(res, np.ndarray) and res.dtype == bool:
            res = np.where(res)[0]

        return res

    def start(self):    pass
    def finish(self):   pass
    def filter(self, boxes, indices, rows, ** kwargs):
        raise NotImplementedError()

@dataclass
class BoxItem:
    index      : int
    box        : np.ndarray
    rows       : np.ndarray
    indices    : list
    unseen     : int  = 0
    updated    : bool = True
    repetition : int  = 1

class RepetitionFilter(BoxFilter):
    def __init__(self, iou_threshold = 0.5, n_repeat = 2, max_unseen = 3, use_memory = False):
        """
            This filter keeps boxes that are repeated `n_repeat` consecutive frames at the same position (up to `iou_threshold` tolerance)
            This allows to ensure the box is repeated multiple times, and is a fiable prediction

            Arguments :
                - iou_threshold : the threshold to determine that 2 boxes are at the same position
                - n_repeat  : the number of times the box has to be emitted at the same position
                - max_unseen    : drops boxes that are not seen this number of frames
                - use_memory    : whether to filter boxes that match a previously emitted position
        """
        self.n_repeat       = n_repeat
        self.max_unseen     = max_unseen
        self.iou_threshold  = iou_threshold
        self.use_memory     = use_memory

        self.index     = 0

        self.memory    = []
        self.waiting   = []
        self.new_boxes = None

    def __len__(self):
        return len(self.waiting)

    def clear(self):
        self.waiting = []

    def start(self):
        self.index += 1
        self.new_boxes  = []
        for b in self.waiting: b.updated = False

    def finish(self):
        for b in self.waiting:
            b.unseen = 0 if b.updated else b.unseen + 1
        self.waiting = [b for b in self.waiting if b.unseen <= self.max_unseen] + self.new_boxes

    def select_candidates(self, candidates, indices, nested_check = False):
        """
            Selects the `candidates` (`BoxItem`) that have the same `indices` structure as the
            query box, i.e. the same number of rows (and, if `nested_check`, the same nested
            lengths). This guarantees that each selected candidate contributes the same number
            of rows, which `get_rows_iou` relies on to reshape the IoU matrix.

            Return : `(positions, rows)` where `positions` are indexes in `candidates` and
            `rows` is the concatenation of their `rows` (or `([], [])` if no candidate matches)
        """
        def _get_lengths(indices):
            if isinstance(indices, int): return 1
            return [len(idx) if isinstance(idx, list) else 1 for idx in indices]

        positions = [
            i for i, cand in enumerate(candidates)
            if len(cand.indices) == len(indices) and
            (not nested_check or _get_lengths(cand.indices) == _get_lengths(indices))
        ]
        if not positions: return [], []
        return positions, np.concatenate([candidates[i].rows for i in positions], axis = 0)

    def get_memory_boxes(self, indices):
        return self.select_candidates(self.memory, indices, nested_check = True)

    def get_waiting_boxes(self, indices):
        return self.select_candidates(self.waiting, indices, nested_check = True)

    def is_in_memory(self, box, rows, indices):
        positions, candidate_rows = self.get_memory_boxes(indices)
        matches = get_rows_iou(rows, candidate_rows, threshold = self.iou_threshold)
        if len(matches) and logger.isEnabledFor(logging.DEBUG):
            logger.debug('Index {} : box {} matches memorized {}'.format(
                self.index, box, self.memory[positions[matches[0]]]
            ))
        return len(matches) > 0

    def check_waiting(self, box, rows, indices):
        if self.n_repeat <= 1: return True

        candidate_idx, candidate_rows = self.get_waiting_boxes(indices)
        matches = get_rows_iou(rows, candidate_rows, threshold = self.iou_threshold)
        if len(matches) == 0:
            self.new_boxes.append(BoxItem(index = self.index, box = box, rows = rows, indices = indices))
            return False

        if len(matches) > 1:
            logger.warning(
                'Multiple matches detected ({}) ! Maybe your `iou_threshold` is too low ; only the best match is updated'.format(len(matches))
            )

        # `matches` is sorted by decreasing IoU : only the best match is updated
        idx  = candidate_idx[matches[0]]
        item = self.waiting[idx]
        item.repetition += 1
        item.updated    = True
        if item.repetition >= self.n_repeat:
            self.waiting.pop(idx)
            if self.use_memory: self.memory.append(item)
            return True
        return False

    def filter(self, boxes, indices, rows, ** _):
        if isinstance(rows, list):
            return np.array([
                self.filter(box, indices_i, rows_i)
                for box, indices_i, rows_i in zip(boxes, indices, rows)
            ], dtype = bool)

        # Filters out boxes that have already been emitted
        if self.use_memory and self.is_in_memory(boxes, rows, indices):
            return False
        # Check if the box was already seen at least `n_repeat - 1` times
        return self.check_waiting(boxes, rows, indices)

class RegionFilter(BoxFilter):
    """ Filter out boxes that are not in the given region (`[x_min, y_min, x_max, y_max]`) """
    def __init__(self, region, mode = 'overlap', ** kwargs):
        self.mode   = mode
        self.x_min, self.y_min, self.x_max, self.y_max = convert_box_format(
            region, target = 'xyxy', normalize_mode = 'relative', ** kwargs
        )[0]

    def filter(self, boxes, ** kwargs):
        if self.mode == 'overlap':
            return np.logical_and(
                np.logical_and(boxes[:, 0] < self.x_max, boxes[:, 2] > self.x_min),
                np.logical_and(boxes[:, 1] < self.y_max, boxes[:, 3] > self.y_min)
            )
        elif self.mode == 'center':
            center  = (boxes[:, :2] + boxes[:, 2:]) / 2.
            return np.logical_and(
                np.logical_and(self.x_min <= center[:, 0], center[:, 0] <= self.x_max),
                np.logical_and(self.y_min <= center[:, 1], center[:, 1] <= self.y_max)
            )
        else:
            raise ValueError('Unknown mode for region filtering : {} !'.format(self.mode))

class SizeFilter(BoxFilter):
    """ Filter out boxes that do not respect some size thresholds """
    def __init__(self,
                 min_h = None,
                 min_w = None,
                 max_h = None,
                 max_w = None,
                 min_area = None,
                 max_area = None,
                 ** kwargs
                ):
        self.min_h, self.max_h = min_h, max_h
        self.min_w, self.max_w = min_w, max_w
        self.min_area, self.max_area = min_area, max_area

    def filter(self, boxes, ** kwargs):
        valids = np.ones((len(boxes), ), dtype = bool)

        h = boxes[:, 3] - boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        for _min, _max, val in [
            (self.min_h, self.max_h, h),
            (self.min_w, self.max_w, w),
            (self.min_area, self.max_area, h * w)
        ]:
            if _min is not None: valids[val < _min] = False
            if _max is not None: valids[val >= _max] = False

        return valids

def get_rows_iou(rows, boxes, threshold = 0.8):
    """
        Matches `rows` (the rows of a single query box) against candidate boxes.

        `boxes` is the concatenation of the candidates' rows, where **each candidate
        contributes exactly `len(rows)` rows** (cf `RepetitionFilter.select_candidates`).
        A candidate matches if each of its rows has an IoU > `threshold` with a query row.

        Return : the matching candidate indexes, sorted by decreasing IoU
    """
    if len(boxes) == 0: return []

    # ious has shape `(len(rows), len(boxes))`
    ious = compute_iou(rows, boxes, as_matrix = True, source = 'xyxy')
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('IoU between rows and candidates :\n{}'.format(np.around(ious, decimals = 3)))
    # best IoU for each candidate row, reshaped to `(n_candidates, len(rows))`
    ious = np.reshape(np.max(ious, axis = 0), [-1, len(rows)])
    # a match requires that all rows from a candidate match a row in `rows`
    scores  = np.min(ious, axis = 1)
    matches = np.where(scores > threshold)[0]
    return matches[np.argsort(scores[matches])[::-1]]
