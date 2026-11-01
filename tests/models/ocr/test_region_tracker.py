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

""" Pure-``pytest`` tests for :mod:`models.ocr.region_tracker`.

Keras-free (unmarked) : the tracker is pure numpy (IoU matching + local Dice on the
detector score-map + normalized edit-distance on the OCR texts). Frames are replayed with
the light builders in :mod:`._helpers`, mirroring how ``BaseOCR.stream_fn`` drives the
tracker around ``infer`` (``set_frame`` → ``BoxFilter`` pass → ``resolve`` / ``register_text``).

Coverage :
  - geometric matching : slot identity, ``match_threshold``, greedy 1-1 assignment ;
  - lifecycle : ``n_stable`` gating, ``max_missing`` expiration, continuous survival ;
  - content stability : score-map Dice (stable / changed), ``geometry_threshold`` override ;
  - the ``_local_dice`` / ``_crop`` internals (Dice reuse + the both-empty guard, erosion) ;
  - the OCR feedback : ``register_text`` emission rules and ``resolve`` ;
  - end-to-end : a dialogue slot whose content changes across frames.
"""

import numpy as np
import pytest

from utils.distances import dice_coeff
from models.ocr.box_filters import filter_boxes

from ._helpers import make_tracker, box, score_map, scan


# --- geometric matching ------------------------------------------------------------

def test_new_region_created_on_first_detection():
    t = make_tracker()
    scan(t, [box(0.1, 0.1, 0.4, 0.2)])
    assert len(t) == 1
    assert t.regions[0].stable_count == 1
    assert t.regions[0].text is None


def test_overlapping_box_matches_same_slot():
    t = make_tracker()
    scan(t, [box(0.10, 0.1, 0.50, 0.2)])
    scan(t, [box(0.11, 0.1, 0.51, 0.2)])   # IoU ~0.95 → same slot, box refreshed
    assert len(t) == 1
    np.testing.assert_allclose(t.regions[0].box, box(0.11, 0.1, 0.51, 0.2))


def test_iou_below_match_threshold_creates_new_slot():
    t = make_tracker(match_threshold = 0.5)
    scan(t, [box(0.10, 0.1, 0.30, 0.2)])
    scan(t, [box(0.25, 0.1, 0.45, 0.2)])   # IoU ~0.14 < 0.5 → distinct slot
    assert len(t) == 2


def test_greedy_one_to_one_matching_is_order_independent():
    t = make_tracker()
    a, b = box(0.1, 0.1, 0.3, 0.2), box(0.6, 0.1, 0.8, 0.2)
    scan(t, [a, b])
    assert len(t) == 2
    # both slightly moved and passed in swapped order : each matches its own slot, no new one
    scan(t, [box(0.61, 0.1, 0.81, 0.2), box(0.11, 0.1, 0.31, 0.2)])
    assert len(t) == 2


# --- n_stable gating ---------------------------------------------------------------

def test_region_not_ocred_before_n_stable():
    t = make_tracker(n_stable = 2)
    b = box(0.1, 0.1, 0.4, 0.2)
    assert list(scan(t, [b])) == []            # 1st frame : stable_count 1 < 2
    assert list(scan(t, [b])) == [0]           # 2nd frame : stable_count 2 → OCR
    assert t.regions[0].stable_count == 2


def test_n_stable_one_ocrs_immediately():
    t = make_tracker(n_stable = 1)
    assert list(scan(t, [box(0.1, 0.1, 0.4, 0.2)])) == [0]


# --- expiration (absence-based) ----------------------------------------------------

def test_region_expires_after_max_missing_scanned_frames():
    t = make_tracker(max_missing = 3)
    scan(t, [box(0.1, 0.1, 0.3, 0.2)])
    for i in range(3):
        t.observe_empty()
        assert len(t) == 1, 'expired too early (after {} missing frames)'.format(i + 1)
    t.observe_empty()                           # 4th consecutive miss → expires
    assert len(t) == 0


def test_continuously_detected_region_never_expires():
    t = make_tracker(max_missing = 3)
    b = box(0.1, 0.1, 0.3, 0.2)
    for _ in range(25):
        scan(t, [b])
    assert len(t) == 1
    assert t.regions[0].missing_count == 0


def test_observe_empty_ages_and_resets_stability():
    t = make_tracker()
    scan(t, [box(0.1, 0.1, 0.3, 0.2)])
    t.observe_empty()
    assert t.regions[0].missing_count == 1
    assert t.regions[0].stable_count == 0


# --- content stability (score-map Dice) --------------------------------------------

def test_stable_score_map_increments_stability():
    t = make_tracker(geometry_threshold = 0., erosion = 0., n_stable = 99)
    b = box(0.1, 0.1, 0.5, 0.4)
    m = score_map([b])
    scan(t, [b], score_map = m)                 # new : stable_count 1
    scan(t, [b], score_map = m)                 # identical crop → 2
    scan(t, [b], score_map = m)                 # → 3
    assert t.regions[0].stable_count == 3


def test_changed_score_map_resets_and_marks_stale():
    t = make_tracker(geometry_threshold = 0., erosion = 0., n_stable = 2)
    b     = box(0.1, 0.1, 0.5, 0.3)
    left  = score_map([box(0.1, 0.1, 0.3, 0.3)])
    right = score_map([box(0.3, 0.1, 0.5, 0.3)])
    scan(t, [b], score_map = left)
    scan(t, [b], score_map = left)
    reg = t.regions[0]
    reg.text, reg.stale = 'previous', False     # simulate an already-OCR-ised slot
    scan(t, [b], score_map = right)             # disjoint crop → Dice 0 < threshold
    assert reg.stable_count == 1
    assert reg.stale is True


def test_geometry_change_resets_stability_even_if_content_stable():
    # `stability_threshold = 0.` would make the score-map path always "stable" ; the
    # geometry check must still catch a box that markedly grows (partial / flickering detection)
    t = make_tracker(geometry_threshold = 0.75, stability_threshold = 0.)
    scan(t, [box(0.1, 0.1, 0.5, 0.2)])
    scan(t, [box(0.1, 0.1, 0.5, 0.2)])
    reg = t.regions[0]
    assert reg.stable_count == 2
    reg.text, reg.stale = 'x', False
    scan(t, [box(0.1, 0.1, 0.7, 0.2)])          # IoU ~0.67 < 0.75 → geometry reset
    assert reg.stable_count == 1
    assert reg.stale is True


# --- re-OCR decision (`_should_ocr`) -----------------------------------------------

def test_known_stable_content_is_not_reocred():
    t = make_tracker(geometry_threshold = 0., erosion = 0., n_stable = 2)
    b, m = box(0.1, 0.1, 0.5, 0.3), score_map([box(0.1, 0.1, 0.5, 0.3)])
    scan(t, [b], score_map = m)
    scan(t, [b], score_map = m)
    reg = t.resolve(b)
    t.register_text(reg, 'some text')           # text known, stale cleared
    assert list(scan(t, [b], score_map = m)) == []   # stable but known & fresh → skip


def test_content_change_triggers_reocr_after_restabilization():
    t = make_tracker(geometry_threshold = 0., erosion = 0., n_stable = 2)
    b     = box(0.2, 0.1, 0.6, 0.3)
    left  = score_map([box(0.2, 0.1, 0.4, 0.3)])
    right = score_map([box(0.4, 0.1, 0.6, 0.3)])
    scan(t, [b], score_map = left)
    scan(t, [b], score_map = left)
    reg = t.resolve(b)
    t.register_text(reg, 'first')
    # content changes : Dice drops → stale, but stability resets → must re-stabilize first
    assert list(scan(t, [b], score_map = right)) == []
    assert reg.stale is True and reg.stable_count == 1
    # new content held one more scanned frame → stable + stale → re-OCR
    assert list(scan(t, [b], score_map = right)) == [0]
    assert reg.stable_count == 2


# --- `_local_dice` (Dice reuse + both-empty guard) ---------------------------------

def test_local_dice_both_empty_returns_one():
    """ The guard preserved on top of `dice_coeff` : two empty crops = nothing changed. """
    t = make_tracker(binarize_threshold = 0.5)
    z = np.zeros((10, 10), dtype = 'float32')
    assert t._local_dice(z, z) == 1.0


def test_local_dice_disjoint_returns_zero():
    t = make_tracker(binarize_threshold = 0.5)
    a = np.zeros((10, 10), dtype = 'float32'); a[:5] = 1.0
    b = np.zeros((10, 10), dtype = 'float32'); b[5:] = 1.0
    assert t._local_dice(a, b) == pytest.approx(0.0)


def test_local_dice_matches_dice_coeff_on_binarized_crops(asserts):
    """ Away from the empty case, `_local_dice` must equal `utils.distances.dice_coeff`. """
    t = make_tracker(binarize_threshold = 0.5)
    a = np.zeros((10, 10), dtype = 'float32'); a[2:8, 2:8] = 1.0
    b = np.zeros((10, 10), dtype = 'float32'); b[4:10, 4:10] = 1.0
    expected = float(dice_coeff((a > 0.5).astype('float32'), (b > 0.5).astype('float32')))
    assert t._local_dice(a, b) == pytest.approx(expected)


def test_local_dice_binarization_ignores_subthreshold_noise():
    t = make_tracker(binarize_threshold = 0.5)
    ref   = np.zeros((10, 10), dtype = 'float32'); ref[2:8, 2:8] = 1.0
    noisy = ref.copy(); noisy[noisy == 0.] = 0.3      # background jitter below the threshold
    assert t._local_dice(ref, noisy) == pytest.approx(1.0)


# --- `_crop` (relative box → eroded pixel crop) ------------------------------------

def test_crop_relative_box_without_erosion():
    t = make_tracker(erosion = 0.)
    m = np.arange(100 * 100, dtype = 'float32').reshape(100, 100)
    np.testing.assert_array_equal(t._crop(m, box(0.1, 0.2, 0.5, 0.6)), m[20:60, 10:50])


def test_crop_erosion_shrinks_the_crop():
    t = make_tracker(erosion = 0.15)
    c = t._crop(np.ones((100, 100), dtype = 'float32'), box(0.0, 0.0, 1.0, 1.0))
    assert c.shape == (70, 70)                   # 15px margin removed on each side


def test_crop_degenerate_box_returns_empty():
    t = make_tracker(erosion = 0.)
    c = t._crop(np.ones((100, 100), dtype = 'float32'), box(1.1, 1.1, 1.2, 1.2))
    assert c.shape == (1, 1)
    assert float(c.sum()) == 0.


# --- OCR feedback : `resolve` / `register_text` ------------------------------------

def _pending_region(t, b):
    """ Stabilize a slot at `b` (n_stable = 2) so it is pending, and resolve it. """
    scan(t, [b]); scan(t, [b])
    reg = t.resolve(b)
    assert reg is not None
    return reg


def test_resolve_returns_the_pending_region():
    t = make_tracker(n_stable = 2)
    b = box(0.1, 0.1, 0.4, 0.2)
    reg = _pending_region(t, b)
    assert reg is t.regions[0]


def test_resolve_unknown_box_returns_none():
    t = make_tracker(n_stable = 2)
    _pending_region(t, box(0.1, 0.1, 0.4, 0.2))
    assert t.resolve(box(0.8, 0.8, 0.95, 0.9)) is None


def test_register_new_content_emits_and_is_remembered():
    t = make_tracker()
    reg = _pending_region(t, box(0.1, 0.1, 0.4, 0.2))
    assert t.register_text(reg, 'hello world') is True
    assert reg.text == 'hello world'
    assert reg.stale is False
    assert 'hello world' in list(t.recent_texts)


def test_register_same_content_refreshes_without_emitting():
    t = make_tracker(text_dist_threshold = 0.2)
    reg = _pending_region(t, box(0.1, 0.1, 0.4, 0.2))
    assert t.register_text(reg, 'hello world') is True
    assert t.register_text(reg, 'hello world.') is False   # 1 edit / 12 → 0.08 < 0.2
    assert reg.text == 'hello world.'                      # stored (refreshed)
    assert reg.stale is False


def test_register_recent_text_at_other_slot_is_suppressed():
    t = make_tracker(text_dist_threshold = 0.2)
    r1 = _pending_region(t, box(0.1, 0.1, 0.4, 0.2))
    assert t.register_text(r1, 'the quick brown fox') is True
    r2 = _pending_region(t, box(0.6, 0.6, 0.9, 0.7))
    assert r2 is not r1
    assert t.register_text(r2, 'the quick brown fox') is False   # recent-texts safety net


def test_register_non_emittable_stores_and_clears_stale():
    t = make_tracker()
    reg = _pending_region(t, box(0.1, 0.1, 0.4, 0.2))
    assert t.register_text(reg, 'rejected text', emittable = False) is False
    assert reg.text == 'rejected text'                     # memorized (won't be re-OCR-ised)
    assert reg.stale is False
    assert 'rejected text' not in list(t.recent_texts)


def test_register_empty_text_returns_false():
    t = make_tracker()
    reg = _pending_region(t, box(0.1, 0.1, 0.4, 0.2))
    assert t.register_text(reg, '') is False
    assert reg.stale is False


# --- BoxFilter contract / lifecycle ------------------------------------------------

def test_call_runs_start_filter_finish():
    t = make_tracker(n_stable = 1)
    b = box(0.1, 0.1, 0.4, 0.2)
    keep = t(np.asarray([b]), [0], [b[None]])
    assert list(keep) == [0]
    assert len(t) == 1


def test_integration_with_filter_boxes_chain():
    t = make_tracker(n_stable = 1)
    boxes = np.stack([box(0.1, 0.1, 0.4, 0.2), box(0.6, 0.6, 0.9, 0.7)])
    t.set_frame(score_map(boxes))
    fb, fi, fr = filter_boxes([t], boxes)
    assert len(fb) == 2                          # both fresh slots kept for OCR


def test_clear_resets_all_state():
    t = make_tracker()
    reg = _pending_region(t, box(0.1, 0.1, 0.4, 0.2))
    t.register_text(reg, 'hello world')
    t.clear()
    assert len(t) == 0
    assert list(t.recent_texts) == []
    assert t._pending == []
    assert t.resolve(box(0.1, 0.1, 0.4, 0.2)) is None


# --- end-to-end scenario -----------------------------------------------------------

def test_integration_dialogue_slot_content_change():
    """ A single subtitle slot whose text changes : each distinct content emitted once. """
    t = make_tracker(n_stable = 2, geometry_threshold = 0., erosion = 0.)
    slot  = box(0.2, 0.8, 0.8, 0.9)
    left  = score_map([box(0.2, 0.8, 0.5, 0.9)])
    right = score_map([box(0.5, 0.8, 0.8, 0.9)])

    emitted = []
    def feed(smap, text):
        if list(scan(t, [slot], score_map = smap)) == [0]:
            if t.register_text(t.resolve(slot), text):
                emitted.append(text)

    feed(left,  'good morning')     # frame 1 : stabilizing
    feed(left,  'good morning')     # frame 2 : stable → emit
    feed(left,  'good morning')     # frame 3 : known & fresh → skip
    feed(right, 'how are you')      # frame 4 : content change → reset
    feed(right, 'how are you')      # frame 5 : re-stabilized → emit

    assert emitted == ['good morning', 'how are you']
    assert len(t) == 1              # same slot throughout
