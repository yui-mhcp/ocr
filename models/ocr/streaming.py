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

from dataclasses import dataclass, field

from utils import dice_coeff, plot
from utils.keras import ops
from utils.text import edit_distance
from utils.image import compute_ioa, convert_box_format, draw_boxes, show_boxes
from .box_filters import SizeFilter, RegionFilter, RepetitionFilter
from .region_tracker import RegionTracker

logger = logging.getLogger(__name__)

@dataclass
class StreamState:
    """
        Mutable per-`stream` state shared between `BaseOCR.stream` and `BaseOCR.stream_fn`.

        It holds the temporal de-duplication counters (comparing consecutive detection maps),
        the set of already-emitted texts / boxes (to avoid re-emitting the same content), and a
        reference to the `RepetitionFilter` used by the box-filter chain (if any).

        Counters semantic :
            - n_duplicates  : number of *consecutive* frames matching the previous detection
                              map (drives the re-scan throttling) ; reset on scene change
            - n_skipped : number of consecutive frames without any emission (pure statistic,
                          it does not influence any decision) ; reset on emission
    """
    is_file : bool  = False

    emitted_texts   : set   = field(default_factory = set, repr = False)
    last_texts  : list  = field(default_factory = list)
    last_boxes  : np.ndarray    = field(default = None, repr = False)

    n_skipped   : int   = 0
    n_duplicates    : int   = 0
    prev_output : np.ndarray    = field(default = None, repr = False)

    repetition_filter   : RepetitionFilter   = None
    tracker : RegionTracker = None

    def set_new_frame(self, output):
        self.prev_output    = output
        self.last_boxes = None
        self.n_duplicates   = 0
        # NOTE : `last_texts` is intentionally *not* cleared here : it feeds the
        # edit-distance filter, which must survive a (possibly false) scene switch —
        # otherwise a slight OCR variation of an already-emitted background text passes
        # the exact-match `emitted_texts` check and is re-emitted right after the switch.
        # NOTE : the `repetition_filter` is intentionally *not* cleared on a scene change.
        # The scene-switch detection (Dice mismatch) is fragile, so a hard reset here would
        # too often drop legitimately-waiting boxes on a false switch.
        #if self.repetition_filter is not None: self.repetition_filter.clear()

    def should_skip_duplicate(self,
                              output,
                              *,
                              duplicate_threshold,
                              duplicate_tolerance,
                              rescan_every
                             ):
        """
            Compares `output` to the previous frame's detection map and decides whether the
            frame is a near-duplicate that can be skipped (throttling the OCR on a static
            scene). Updates `n_duplicates` / `prev_output` **in place**.

            The policy is :
                1. A new scene (no history, different shape, or Dice < `duplicate_threshold`)
                   resets the state and is always processed
                2. The first `duplicate_tolerance` duplicate frames are still processed, to
                   catch text that progressively appears on an otherwise stable scene
                3. Past the tolerance, only 1 frame every `rescan_every` is processed

            Return `(skip, is_duplicate, dice)` :
                - skip  : whether the caller should drop this frame
                - is_duplicate  : whether the frame matched the previous one
                - dice  : the Dice coefficient with the previous frame (`0.` if not comparable)
        """
        # A new frame (no history, or a different shape) resets the de-duplication state
        if self.prev_output is None or output.shape != self.prev_output.shape:
            self.set_new_frame(output)
            return False, False, 0.

        dice = dice_coeff(output, self.prev_output)
        if dice < duplicate_threshold:
            self.set_new_frame(output)
            return False, False, dice

        self.n_duplicates += 1
        if self.n_duplicates <= duplicate_tolerance:
            return False, True, dice

        skip = (self.n_duplicates - duplicate_tolerance) % rescan_every != 0
        return skip, True, dice


def make_emitted_box_filter(state, ioa_threshold):
    """
        Returns a box filter dropping detections that overlap (IoA >= `ioa_threshold`) a
        previously emitted box (`state.last_boxes`), so a region already OCR-ised is not
        re-processed on the following frames.
    """
    def _filter_emitted_boxes(boxes, ** _):
        if state.last_boxes is None: return list(range(len(boxes)))
        ioa = compute_ioa(boxes, state.last_boxes, as_matrix = True, source = 'xyxy')
        return np.where(np.all(ioa < ioa_threshold, axis = 1))[0]
    return _filter_emitted_boxes


def build_stream_filters(filters, state, *, region, n_repeat, ioa_threshold, tracker = None):
    """
        Builds the default streaming box-filter chain (unless `filters` is already provided),
        and registers the de-duplication component on `state` so `stream_fn` can use it :
            - `tracker` provided : the `RegionTracker` is appended as the last filter, and
              replaces both the `RepetitionFilter` and the IoA "already emitted" filter
            - otherwise (legacy pipeline) : the IoA filter is *prepended* (cheapest and most
              discriminant : already-emitted boxes must not pollute the `RepetitionFilter`
              state), and the `RepetitionFilter` (if any) is registered on `state`

        Return : the (possibly built) `filters` list.
    """
    if filters is None:
        filters = [SizeFilter(min_h = 0.025, min_w = 0.1)]
        if region:
            filters.append(RegionFilter(region = region, source = 'xyxy', mode = 'center'))
        if n_repeat and tracker is None:
            filters.append(RepetitionFilter(n_repeat = n_repeat, use_memory = False))
    elif not isinstance(filters, list):
        filters = [filters]

    if tracker is not None:
        filters.append(tracker)
        state.tracker = tracker
        return filters

    if ioa_threshold:
        filters.insert(0, make_emitted_box_filter(state, ioa_threshold))

    for f in filters:
        if isinstance(f, RepetitionFilter):
            state.repetition_filter = f
            break

    return filters


def filter_text_results(ocr_result,
                        reject,
                        last_emitted = None,
                        *,

                        text_threshold = 0.,
                        k   = 5,
                        max_dist   = 0.2,
                        skip_non_alpha = True,
                        skip_single_word   = True,

                        verbose    = True
                       ):
    """
        Streaming text filter : decides whether a single `ocr_result` should be emitted, and
        may mutate its `text` / `scores` in place (stripping `</s>`, dropping low-score parts).

        Rejects (returns `False`) when the text is empty, a single word / non-alphabetic (if the
        corresponding flags are set), already in `reject`, entirely below `text_threshold`, or
        too close (normalized edit distance `< max_dist`) to one of the last `k` emitted texts.
        In the latter case the text is also added to `reject` to suppress this recurring false
        positive on the next frames.
    """
    ocr_result['text'] = ocr_result['text'].replace('</s>', '').strip()
    text = ocr_result['text']
    if not text: return False
    elif skip_single_word and ' ' not in text: return False
    elif skip_non_alpha and not any(c.isalpha() for c in text): return False
    elif text in reject:
        if verbose: logger.info('Detected text was already emitted')
        return False

    scores = ocr_result['scores']
    if text_threshold != 0. and np.any(scores <= text_threshold):
        parts = text.split(' \n')
        if verbose:
            logger.info('Some scores are too low !\n{}'.format('\n'.join([
                '- {:.3f} : {}'.format(s, t) for s, t in zip(scores, parts)
            ])))
        if np.all(scores <= text_threshold): return False

        # only keeps the parts above the threshold, and keeps `scores` aligned with them
        kept    = [(p, s) for p, s in zip(parts, scores) if s > text_threshold]
        text    = ' \n'.join([p for p, _ in kept])
        ocr_result['text']   = text
        ocr_result['scores'] = np.array([s for _, s in kept])

    if last_emitted and k:
        for emitted in last_emitted[- k :]:
            dist = edit_distance(emitted, text, normalize = True)
            if dist < max_dist:
                if verbose:
                    logger.info('Rejecting text due to an edit distance of {:.3f}\n  Text : {}\n  Previously emitted : {}'.format(dist, text, emitted))
                reject.add(text)
                return False

    return True


def display_stream_results(image, results, detected = None, display = 1):
    """
        Displays the emitted `results` : an overview of the frame with all the emitted boxes,
        then (if `display > 1`) each individual box and its rows.
    """
    if detected is None:
        emitted_boxes = np.array([ocr_res['boxes'] for ocr_res in results])
        detected = draw_boxes(
            ops.convert_to_numpy(image, copy = True),
            emitted_boxes,
            source  = 'xyxy',
            color   = 'r',
            labels  = ['text'],
            show_text   = False
        )
    plot(detected, plot_type = 'imshow')

    if display > 1:
        for ocr_result in results:
            show_boxes(image, ocr_result['boxes'], source = ocr_result['format'])
            if len(ocr_result['rows']) > 1:
                show_boxes(
                    image,
                    ocr_result['rows'],
                    source = ocr_result['format'],
                    ncols    = len(ocr_result['rows'])
                )


def emit_stream_results(state, results, *, callback = None, verbose = True):
    """
        Registers each result as emitted on `state` (so the text / box de-duplication drops it
        on the next frames), logs it, and forwards its text to `callback` (e.g. the TTS queue).
    """
    for ocr_result in results:
        state.last_texts.append(ocr_result['text'])
        state.emitted_texts.add(ocr_result['text'])

        if verbose:
            logger.info('Box  : {}\nScores : {}\nText   : {}'.format(
                np.around(convert_box_format(
                    ocr_result['boxes'], target = 'xywh', source = ocr_result['format']
                ), decimals = 3),
                np.around(ocr_result['scores'], decimals = 3),
                ocr_result['text']
            ))

        if callback: callback(ocr_result['text'])
