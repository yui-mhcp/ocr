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

from collections import deque
from dataclasses import dataclass, field

from utils import dice_coeff
from utils.keras import ops
from utils.text import edit_distance
from utils.image import compute_iou
from .box_filters import BoxFilter

logger = logging.getLogger(__name__)

@dataclass
class TrackedRegion:
    """
        A text *slot* tracked across frames by `RegionTracker`.

        The `box` (geometry) identifies the *location* of the slot, while `text` identifies
        its *content* : two consecutive dialogues typically occupy the same box (matched as
        the same slot), and are distinguished by their content only.
    """
    box     : np.ndarray            # union box, "xyxy" relative coordinates
    rows    : np.ndarray = field(repr = False)  # individual line boxes `(n_rows, 4)`

    text    : str   = None          # last OCR-ised text (`None` = never OCR-ised)
    scores  : np.ndarray    = field(default = None, repr = False)

    stale   : bool  = True          # whether the content changed since the last OCR
    stable_count    : int   = 1     # consecutive scanned frames with a stable content
    missing_count   : int   = 0     # consecutive scanned frames without detection
    updated : bool  = field(default = True, repr = False)   # transient per-frame flag

class RegionTracker(BoxFilter):
    """
        Persistent spatial memory associating text *slots* (positions) to their last
        OCR-ised *content*, replacing (in the streaming pipeline) the `RepetitionFilter`,
        the "already emitted" IoA filter and the emitted-texts sets.

        Used as the last element of the `BaseOCR.infer` box-filter chain : `filter` matches
        the combined candidate boxes against the tracked regions, and only keeps for OCR the
        regions that are *stable* and whose content is unknown or *stale*. The caller then
        reports the OCR-ised texts back via `register_text`, which decides the emission.

        Design principles (cf the streaming robustness analysis) :
            - **Slot vs content identity** : the geometric matching (IoU) identifies the
              *slot* only. A new dialogue re-using the dialogue box matches the same slot ;
              the content change is detected by the local score-map comparison, and the
              emission is decided by comparing the OCR-ised text to the slot's last text.
            - **Absence-based expiration** (no TTL) : a continuously detected region lives
              forever (at low fps, static HUD texts stay for hundreds of frames). A region
              only expires after `max_missing` consecutive *scanned* frames without
              detection (skipped frames never reach the tracker, so they do not count).
            - **Eroded local stability** : the per-region change detection compares the
              detector score-map crops of the region between consecutive scanned frames,
              with the box *eroded* by a margin so that the sub-pixel border jitter of the
              detection does not produce false "content changed" signals. The crops are
              binarized (`binarize_threshold`) so that a slightly moving background under a
              static text overlay does not perturb the comparison either.

        Arguments :
            - match_threshold   : minimal IoU for a candidate box to match a tracked slot
            - geometry_threshold    : minimal IoU between the slot's boxes on consecutive
                                      scanned frames for the region to be considered stable.
                                      The score-map Dice only covers the *current* box, so a
                                      truncated / growing detection (flicker, typewriter
                                      effect) can look stable — the geometry check catches
                                      it. `0` / `None` disables the check
            - n_stable  : number of consecutive stable scanned frames required before OCR
                          (replaces `RepetitionFilter.n_repeat`)
            - max_missing   : number of consecutive scanned frames a region survives
                              without being detected (replaces `RepetitionFilter.max_unseen`)
            - stability_threshold   : minimal local Dice between consecutive score-map crops
                                      for the region content to be considered stable
            - binarize_threshold    : threshold applied to the score-map crops before the
                                      Dice (`None` keeps the continuous maps)
            - erosion   : margin (fraction of the box height) removed on each side of the
                          crop before the comparison
            - text_dist_threshold   : maximal normalized edit distance for two texts to be
                                      considered the same content
            - memory_size   : size of the ring buffer of recently emitted texts, used as a
                              safety net against slight box shifts (a text re-appearing at a
                              *different* position after falling out of this buffer is
                              re-emitted, on purpose)
    """
    def __init__(self,
                 match_threshold    = 0.4,
                 geometry_threshold = 0.75,
                 n_stable   = 2,
                 max_missing    = 3,

                 stability_threshold    = 0.85,
                 binarize_threshold     = 0.5,
                 erosion    = 0.15,

                 text_dist_threshold    = 0.2,
                 memory_size    = 15
                ):
        self.match_threshold    = match_threshold
        self.geometry_threshold = geometry_threshold
        self.n_stable       = n_stable
        self.max_missing    = max_missing

        self.stability_threshold    = stability_threshold
        self.binarize_threshold     = binarize_threshold
        self.erosion    = erosion

        self.text_dist_threshold    = text_dist_threshold

        self.regions    = []
        self.recent_texts   = deque(maxlen = memory_size)

        self._cur_map   = None
        self._prev_map  = None
        self._pending   = []

    def __len__(self):
        return len(self.regions)

    def __repr__(self):
        return '<RegionTracker regions={} pending={}>'.format(len(self.regions), len(self._pending))

    def clear(self):
        self.regions    = []
        self.recent_texts.clear()
        self._cur_map, self._prev_map, self._pending = None, None, []

    def set_frame(self, score_map):
        """ Registers the current frame's detector score map (called before `infer`) """
        self._cur_map = score_map

    def observe_empty(self, score_map = None):
        """
            Registers a scanned frame without any detection : all the regions age (and
            possibly expire), and the score-map history advances
        """
        if score_map is not None: self.set_frame(score_map)
        self.start()
        self._update_regions(np.zeros((0, 4), dtype = 'float32'), [])
        self.finish()

    # `BoxFilter` interface

    def start(self):
        self._pending = []

    def finish(self):
        if self._cur_map is not None:
            self._prev_map  = self._cur_map
            self._cur_map   = None

    def filter(self, boxes, indices, rows, ** _):
        keep = self._update_regions(boxes, rows)
        return np.array(keep, dtype = 'int32')

    # OCR feedback

    def resolve(self, box):
        """ Returns the pending region associated to `box` (as returned by `infer`) """
        box = np.asarray(box).reshape(-1)
        for region in self._pending:
            if region.box.shape == box.shape and np.allclose(region.box, box):
                return region
        return None

    def register_text(self, region, text, scores = None, emittable = True):
        """
            Registers the OCR-ised `text` on `region`, and returns whether it should be
            emitted.

            The text is stored on the region even when not emitted (`emittable = False`, or
            same content as before) : this clears the `stale` flag, so a recurring false
            positive / rejected text is not re-OCR-ised every frame (same intent as the old
            `last_boxes` accumulation *before* the text filtering).
        """
        previous = region.text if region is not None else None
        if region is not None:
            region.text, region.scores, region.stale = text, scores, False

        if not emittable or not text: return False

        # same (or nearly same) content at the same slot : refresh only. This absorbs the
        # OCR drift (1-2 characters variations) that used to bypass the exact-match set.
        if previous is not None and edit_distance(previous, text, normalize = True) < self.text_dist_threshold:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Same content re-OCR-ised at {} : {}'.format(region.box, text))
            return False

        # safety net against slight box shifts (a "new" slot showing a recent text)
        for emitted in self.recent_texts:
            if edit_distance(emitted, text, normalize = True) < self.text_dist_threshold:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Text was recently emitted (at another slot) : {}'.format(text))
                return False

        self.recent_texts.append(text)
        return True

    # internals

    def _update_regions(self, boxes, rows):
        """
            Matches the candidate `boxes` / `rows` (combined by `combine_boxes`) against the
            tracked regions, creates the new regions, ages / expires the unmatched ones, and
            returns the indexes of the candidates to OCR (stable + unknown or stale content)
        """
        for region in self.regions: region.updated = False

        assignment = self._match(boxes)

        keep = []
        for b_idx in range(len(boxes)):
            r_idx = assignment.get(b_idx, None)
            if r_idx is None:
                region = TrackedRegion(
                    box = np.array(boxes[b_idx]), rows = np.asarray(rows[b_idx])
                )
                self.regions.append(region)
            else:
                region  = self.regions[r_idx]
                old_box = region.box
                region.box  = np.array(boxes[b_idx])
                region.rows = np.asarray(rows[b_idx])
                region.updated  = True
                region.missing_count    = 0
                self._update_stability(region, old_box)

            if self._should_ocr(region):
                self._pending.append(region)
                keep.append(b_idx)

        survivors = []
        for region in self.regions:
            if region.updated:
                survivors.append(region)
                continue
            # only *scanned* frames reach this point : a continuously detected (or simply
            # not re-scanned) region never expires, regardless of its age
            region.missing_count    += 1
            region.stable_count     = 0
            if region.missing_count <= self.max_missing:
                survivors.append(region)
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug('Region at {} expired (text : {})'.format(region.box, region.text))
        self.regions = survivors

        return keep

    def _match(self, boxes):
        """ Greedy 1-1 IoU matching between tracked regions and candidate boxes """
        if not self.regions or len(boxes) == 0: return {}

        region_boxes = np.stack([region.box for region in self.regions], axis = 0)
        iou = np.asarray(compute_iou(region_boxes, boxes, as_matrix = True, source = 'xyxy'))

        assignment, used_regions = {}, set()
        for flat in np.argsort(iou, axis = None)[::-1]:
            r_idx, b_idx = divmod(int(flat), iou.shape[1])
            if iou[r_idx, b_idx] < self.match_threshold: break
            if r_idx in used_regions or b_idx in assignment: continue
            assignment[b_idx] = r_idx
            used_regions.add(r_idx)

        return assignment

    def _update_stability(self, region, old_box = None):
        """ Updates `stable_count` / `stale` from the geometry + local score-map comparison """
        # a box whose geometry markedly changed (growing while the text appears, or a
        # flickering partial detection) is not stable, even when its score-map crop is :
        # the crop only covers the *current* box, so a missing part is invisible to the Dice
        if old_box is not None and self.geometry_threshold:
            geom_iou = float(np.asarray(compute_iou(
                old_box[None], region.box[None], as_matrix = False, source = 'xyxy'
            )).reshape(-1)[0])
            if geom_iou < self.geometry_threshold:
                region.stable_count = 1
                if region.text is not None: region.stale = True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Region at {} : geometry changed (IoU {:.3f}), stability reset'.format(
                        region.box, geom_iou
                    ))
                return

        if self._cur_map is None:
            # no score map provided : fall back to a pure repetition count
            region.stable_count += 1
            return

        if self._prev_map is None or self._prev_map.shape != self._cur_map.shape:
            region.stable_count = 1
            return

        dice = self._local_dice(
            self._crop(self._cur_map, region.box), self._crop(self._prev_map, region.box)
        )
        if dice >= self.stability_threshold:
            region.stable_count += 1
        else:
            region.stable_count = 1
            if region.text is not None: region.stale = True

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Region at {} : local dice {:.3f} (stable_count {}, stale {})'.format(
                region.box, dice, region.stable_count, region.stale
            ))

    def _should_ocr(self, region):
        return region.stable_count >= self.n_stable and (region.text is None or region.stale)

    def _crop(self, score_map, box):
        """ Crops `score_map` on `box` (relative "xyxy"), eroded by the `erosion` margin """
        h, w = score_map.shape[:2]
        x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h

        margin = (y1 - y0) * self.erosion
        # falls back to the un-eroded crop when the box is too small to erode
        if (y1 - y0) - 2. * margin < 1. or (x1 - x0) - 2. * margin < 1.: margin = 0.

        x0, y0 = max(int(round(x0 + margin)), 0), max(int(round(y0 + margin)), 0)
        x1  = min(max(int(round(x1 - margin)), x0 + 1), w)
        y1  = min(max(int(round(y1 - margin)), y0 + 1), h)
        if x0 >= x1 or y0 >= y1:    # degenerate box (fully outside the map)
            return np.zeros((1, 1), dtype = score_map.dtype)
        return score_map[y0 : y1, x0 : x1]

    def _local_dice(self, cur, prev):
        if self.binarize_threshold is not None:
            cur     = cur > self.binarize_threshold
            prev    = prev > self.binarize_threshold
        cur, prev = ops.convert_to_numpy(cur, 'float32'), ops.convert_to_numpy(prev, 'float32')

        # both crops empty : nothing on either frame → stable. `dice_coeff` (via `divide_no_nan`)
        # would return 0. here ("changed"), so this existence check keeps the guard without
        # recomputing the sums that `dice_coeff` already computes internally.
        if not cur.any() and not prev.any(): return 1.
        return float(dice_coeff(cur, prev))
