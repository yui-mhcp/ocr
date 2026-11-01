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

from loggers import timer
from .converter import box_converter_wrapper
from .metrics import compute_ioa

logger = logging.getLogger(__name__)

def _compute_groups_np(mask):
    edges = {i : set([i]) for i in range(len(mask))}
    for s, e in zip(* np.where(mask)):
        edges[s].add(e)
        edges[e].add(s)
    
    components, q, seen = [], [], set()
    for i in range(len(mask)):
        if i in seen: continue
        component = set()
        q.append(i)
        while q:
            idx = q.pop(0)
            if idx in seen: continue
            seen.add(idx)
            
            neighbors = edges[idx]

            component.update(neighbors)
            q.extend([n for n in neighbors if n not in seen])
        
        components.append(component)
    return components

def _compute_groups_networkx(mask):
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(zip(* np.where(mask)))
    return nx.connected_components(g)

compute_groups = _compute_groups_np

def _merge_lists(lists):
    if len(lists) == 1: return lists[0]
    union = []
    for l in lists: union.extend(l)
    return union

def _merge_rows(rows, indices):
    """
        Merge boxes in `rows` that are on the same line. This is useful to ensure that, within a paragraph, each box in `rows` correspond to the entire line
    """
    if isinstance(rows, list):
        return list(zip(* [_merge_rows(r, i) for r, i in zip(rows, indices)]))
    elif len(rows) < 2:
        return rows, indices
    
    overlap_y   = np.maximum(0., (
        np.minimum(rows[:, None, 3], rows[None, :, 3]) -
        np.maximum(rows[:, None, 1], rows[None, :, 1])
    )) / np.mean(rows[:, 3] - rows[:, 1])

    groups  = sorted([list(grp) for grp in compute_groups(overlap_y >= 0.5)], key = min)
    rows    = np.array([compute_union(rows[grp]) for grp in groups])
    indices = [
        _merge_lists([indices[idx] for idx in grp])
        for grp in groups
    ]

    return rows, indices

def _align_rows(rows, mode, center_threshold = 1e-2, ** _):
    """ Aligns `rows` on left/right border """
    if len(rows) == 1: return rows
    
    if mode == 'auto':
        x_center = (rows[:, 0] + rows[:, 2]) / 2.
        is_centered = np.all(np.abs(x_center - np.mean(x_center)) < center_threshold)
        if not is_centered: mode = 'left'
    
    if mode == 'left':
        rows[:, 0] = np.min(rows[:, 0])
    elif mode == 'right':
        rows[:, 2] = np.max(rows[:, 2])
    
    return rows

def _pairwise_heights(h, mode):
    """ Returns the `(n, n)` matrix of pairwise heights (`max` / `min` / `mean` of the pair) """
    if mode == 'max':       return np.maximum(h[:, None], h[None, :])
    elif mode == 'min':     return np.minimum(h[:, None], h[None, :])
    elif mode == 'mean':    return (h[:, None] + h[None, :]) / 2.
    raise ValueError('Unknown pairwise height mode : {} !'.format(mode))

def compute_union(boxes):
    if len(boxes) == 1: return boxes[0]
    return np.concatenate([
        np.min(boxes[:, :2], axis = 0), np.max(boxes[:, 2:], axis = 0)
    ], axis = 0)

def group_boxes(boxes,
                indices,
                groups,
                *,
                sort    = None,
                
                rows    = None,
                
                check_rows  = None,
                align_borders   = None,

                ** kwargs
               ):
    if rows is not None:
        if check_rows is None:      check_rows = True
        if align_borders is None:   align_borders = 'auto'
    
    res_boxes, res_indices, individuals = [], [], []
    for group in groups:
        group = list(group)
        if sort is not None and len(group) > 1:
            group = sorted(group, key = lambda idx: boxes[idx, sort])
        else:
            group = list(group)

        boxes_group = boxes[group]
        res_boxes.append(compute_union(boxes_group))
        
        if rows is None:
            res_indices.append([indices[idx] for idx in group])
            individuals.append(boxes_group)
        elif len(group) == 1:
            res_indices.append(indices[group[0]])
            individuals.append(rows[group[0]])
        else:
            res_indices.append(_merge_lists([indices[idx] for idx in group]))
            individuals.append(np.concatenate([rows[idx] for idx in group], axis = 0))
    
    if check_rows:
        individuals, res_indices = _merge_rows(individuals, res_indices)
    
    if align_borders:
        individuals = [_align_rows(rows, align_borders, ** kwargs) for rows in individuals]
    
    return np.array(res_boxes), res_indices, individuals

@timer
@box_converter_wrapper('xyxy', normalize_mode = 'relative', force_np = True, as_dict = False)
def combine_boxes_horizontal(boxes,
                             *,
                             
                             indices    = None,
                             
                             h_factor   = 1.5,
                             x_threshold    = None,
                             y_norm     = 'mean',
                             y_overlap_threshold    = 0.55,

                             ** kwargs
                            ):
    """
        Combines a list of boxes according to the following criteria :
            1. The distance between the right-side of box i and the left-side of box j is less than `x_threshold` (if the two boxes overlap on the x axis, the distance is set to 0)
            2. The y-overlap of box i and j divided by the pairwise height (cf `y_norm`) is greater than `y_overlap_threshold`

        The 1st criterion ensures proximity between the two boxes
        The 2nd criterion ensures that both boxes are on the same line with equivalent heights

        If `x_threshold` is not provided, it is defined *pairwise* as `mean(h_i, h_j) * h_factor`
        The intuition is that a standard space should be approximately equal to the height of
        the pair (a global threshold, e.g. based on the median height, is biased by unrelated
        boxes on the frame, such as smaller HUD texts)

        Arguments :
            - boxes : 2-D `np.ndarray` with shape `(n_boxes, 4)`
            - indices   : a list of indices (default to `range(n_boxes)`)
            - x_threshold   : the threshold for the distance between right/left sides
            - y_norm    : pairwise height used to normalize the y-overlap ('mean' / 'min' /
                          'max'). 'max' is the strictest (a single box with an inflated
                          height — a common detector noise — fails the criterion against its
                          own line) ; 'mean' (default) tolerates one outlier of the pair
            - overlap_threshold : fraction of the hgieht that should overlap between the 2 boxes
            - h_factor  : multiplies the pairwise height to define adaptive `x_threshold`
        Return :
            - combined_boxes    : `np.ndarray` with shape `(n_res_boxes, 4)`, the result of the combination
            - combined_indices  : a `list` with length `n_res_boxes`, where item `i` corresponds to values in `indices` used to create `combined_boxes[i]`
            - rows  : a `list` of `np.ndarray` where each item is the concatenation of `boxes` that have been combined together
        
        Example :
        ```python
        boxes = [
            [0, 0, 1, 1],
            [1, 0, 2, 1],
            [3, 3, 4, 4]
        ]
        combined_boxes, groups, rows = combine_boxes_horizontal(boxes)
        # the 1st box is the union of the 2 first boxes
        print(combined_boxes) # [[0, 0, 2, 1], [3, 3, 4, 4]]
        # the 1st groups contains [0, 1] as they have been combined
        print(groups) # [[0, 1], [2]]
        # the 1st group contains the 2 boxes that have been combined
        print(rows) # [array([[0, 0, 1, 1], [1, 0, 2, 1]]), array([[3, 3, 4, 4]])]
        ```
    """
    if indices is None: indices = list(range(len(boxes)))
    if len(boxes) <= 1: return boxes, [indices], [boxes]

    h = boxes[:, 3] - boxes[:, 1]
    if x_threshold is None:
        x_threshold = _pairwise_heights(h, 'mean') * h_factor
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('X threshold (pairwise) :\n{}'.format(np.around(x_threshold, decimals = 3)))

    norm_h  = _pairwise_heights(h, y_norm)
    diff_border = np.abs(boxes[:, None, 2] - boxes[None, :, 0])
    
    overlap_x   = (
        np.minimum(boxes[:, None, 2], boxes[None, :, 2]) -
        np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    ) > 0.
    diff_border = diff_border * ~overlap_x

    overlap_y   = np.maximum(0., (
        np.minimum(boxes[:, None, 3], boxes[None, :, 3]) -
        np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    ))

    should_combine_horizontal = np.logical_and(
        diff_border <= x_threshold,
        np.divide(
            overlap_y, norm_h, out = np.zeros_like(overlap_y), where = norm_h > 0
        ) >= y_overlap_threshold
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Horizontal combination :\nX distance :\n{}\n% y overlap :\n{}'.format(
            np.around(diff_border, decimals = 3),
            np.around(overlap_y / norm_h, decimals = 2)
        ))
    
    return group_boxes(boxes, indices, compute_groups(should_combine_horizontal), sort = 0)

@timer
@box_converter_wrapper('xyxy', normalize_mode = 'relative', force_np = True, as_dict = False)
def combine_boxes_vertical(boxes,
                           *,
                           
                           indices  = None,
                           
                           y_threshold  = None,
                           h_threshold  = 0.02,
                           x_overlap_threshold    = 0.,
                           shift_factor = 0.5,
                           
                           ** kwargs
                          ):
    """
        Combines a list of boxes according to the following criteria :
            1. The distance between the bottom-side of box i and the top-side of box j is less than `y_threshold` or the two boxes overlap on the y axis
            2. The difference in heights between boxes i and j is less than `h_threshold`
            3. The x-overlap of box i and j divided by the maximal width (of boxes i and j) is greater than `x_overlap_threshold` (default to 0, meaning that they simply have to overlap)
        
        The 1st criterion ensures proximity between the two boxes
        The 2nd criterion ensures that both boxes have approximately equivalent heights
        The 3rd criterion ensures that both boxes are overlapping on the x axis
        
        If `y_threshold` is not provided, it is defined as `median(heights) / 4`
        The intuition is that a standard space should be approximately equal to 25% of the median height
        
        The `shift_factor` allows to control in which fraction of the height a box should be
        The intuition is that lines in a paragraph are aligned on the left side
        Therefore, if box i overlaps with box j but is completely on the right side, they may not belong to the same paragraph
        
        Arguments :
            - boxes : 2-D `np.ndarray` with shape `(n_boxes, 4)`
            - indices   : a list of indices (default to `range(n_boxes)`)
            - y_threshold   : the threshold for the distance between bottom/top sides
            - h_threshold   : the threshold for the difference in heights
            - overlap_threshold : fraction of the (shifted) width that should overlap between the 2 boxes
            - shift_factor  : shift the right sode to the left before computing overlap
        Return :
            - combined_boxes    : `np.ndarray` with shape `(n_res_boxes, 4)`, the result of the combination
            - combined_indices  : a `list` with length `n_res_boxes`, where item `i` corresponds to values in `indices` used to create `combined_boxes[i]`
            - rows  : a `list` of `np.ndarray` where each item is the concatenation of `boxes` that have been combined together
        
        Example :
        ```python
        boxes = [
            [0, 0, 1, 1],
            [0, 1, 1, 2],
            [3, 3, 4, 4]
        ]
        combined_boxes, groups, rows = combine_boxes_horizontal(boxes)
        # the 1st box is the union of the 2 first boxes
        print(combined_boxes) # [[0, 0, 1, 2], [3, 3, 4, 4]]
        # the 1st groups contains [0, 1] as they have been combined
        print(groups) # [[0, 1], [2]]
        # the 1st group contains the 2 boxes that have been combined
        print(rows) # [array([[0, 0, 1, 1], [0, 1, 1, 2]]), array([[3, 3, 4, 4]])]
        ```
    """
    if indices is None: indices = list(range(len(boxes)))
    if len(boxes) <= 1: return boxes, [indices], [boxes]
    
    h = boxes[:, 3] - boxes[:, 1]
    if y_threshold is None: y_threshold = max(np.median(h) / 4., 1e-2)
    
    h_diff  = np.abs(h[:, None] - h[None, :])
    diff_border = np.abs(boxes[:, None, 3] - boxes[None, :, 1])

    overlap_y   = (
        np.minimum(boxes[:, None, 3], boxes[None, :, 3]) -
        np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    ) > 0
    diff_border = diff_border * ~overlap_y

    shifted_w   = (boxes[:, 2] - boxes[:, 0]) * shift_factor
    min_w       = np.minimum(shifted_w[:, None], shifted_w[None, :])
    
    shifted_x_max   = boxes[:, 2] - shifted_w
    overlap_x       = (
        np.minimum(shifted_x_max[:, None], shifted_x_max[None, :]) -
        np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    )
    if x_overlap_threshold == 0.:
        x_overlap_mask = overlap_x > 0.
    else:
        overlap_x = overlap_x / np.minimum(shifted_w[:, None], shifted_w[None, :])
        x_overlap_mask = overlap_x > x_overlap_threshold

    should_combine_vertical = np.logical_and(
        np.logical_and(diff_border <= y_threshold, h_diff <= h_threshold),
        x_overlap_mask
    )

    return group_boxes(boxes, indices, compute_groups(should_combine_vertical), sort = 1)

@timer
@box_converter_wrapper('xyxy', normalize_mode = 'relative', force_np = True, as_dict = False)
def combine_boxes_overlap(boxes, indices = None, overlap_threshold = 0.5, ** kwargs):
    is_overlapping = compute_ioa(boxes, source = 'xyxy', as_matrix = True) > overlap_threshold
    return group_boxes(boxes, indices, compute_groups(is_overlapping), sort = 1, ** kwargs)

@timer
def combine_line_continuations(boxes,
                               indices,
                               rows,
                               *,

                               line_gap_factor  = 2.,
                               y_norm   = 'mean',
                               y_overlap_threshold  = 0.55,

                               ** kwargs
                              ):
    """
        Post-combination pass merging groups that are *continuations of the same line* :
        groups i and j are merged if a row of i and a row of j share the same y-band
        (y-overlap normalized by the `y_norm` pairwise height >= `y_overlap_threshold`) and
        are horizontally within `line_gap_factor * mean(h)` of each other.

        This is the targeted exception to the `shift_factor` veto of
        `combine_boxes_vertical` : a right-side fragment of a line failed to join its
        paragraph vertically (it is right-shifted by design), but it *continues an existing
        row* — membership is decided at the row level, never from the paragraph geometry, so
        genuinely side-by-side texts (on distinct lines) are not merged.

        `line_gap_factor` controls the aggressiveness (a standard space ~= the row height) ;
        `0` / `None` disables the pass.

        Arguments / return : same contract as `combine_boxes_overlap` (`boxes` is the
        `(n_groups, 4)` union boxes in relative "xyxy", `rows` the per-group row arrays)
    """
    if not line_gap_factor or len(boxes) <= 1: return boxes, indices, rows

    all_rows    = np.concatenate(rows, axis = 0)
    group_ids   = np.concatenate([np.full((len(r), ), i) for i, r in enumerate(rows)])

    h = all_rows[:, 3] - all_rows[:, 1]

    overlap_y   = np.maximum(0., (
        np.minimum(all_rows[:, None, 3], all_rows[None, :, 3]) -
        np.maximum(all_rows[:, None, 1], all_rows[None, :, 1])
    ))
    norm_h      = _pairwise_heights(h, y_norm)
    same_line   = np.divide(
        overlap_y, norm_h, out = np.zeros_like(overlap_y), where = norm_h > 0
    ) >= y_overlap_threshold

    diff_border = np.abs(all_rows[:, None, 2] - all_rows[None, :, 0])
    overlap_x   = (
        np.minimum(all_rows[:, None, 2], all_rows[None, :, 2]) -
        np.maximum(all_rows[:, None, 0], all_rows[None, :, 0])
    ) > 0.
    diff_border = diff_border * ~overlap_x

    is_continuation = np.logical_and(
        same_line, diff_border <= _pairwise_heights(h, 'mean') * line_gap_factor
    )
    # only cross-group links are relevant
    is_continuation = np.logical_and(
        is_continuation, group_ids[:, None] != group_ids[None, :]
    )
    if not np.any(is_continuation): return boxes, indices, rows

    # lift the row-level links to a group-level adjacency matrix
    adjacency = np.zeros((len(boxes), len(boxes)), dtype = bool)
    src, dst  = np.where(is_continuation)
    adjacency[group_ids[src], group_ids[dst]] = True

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Line continuations detected between groups :\n{}'.format(
            np.stack([group_ids[src], group_ids[dst]], axis = 1)
        ))

    return group_boxes(
        boxes, indices, compute_groups(adjacency), sort = 1, rows = rows, ** kwargs
    )

def _flatten_indices(indices):
    """ Recursively flattens a (possibly nested) index structure to a flat `list` of ids """
    if isinstance(indices, (int, np.integer)): return [indices]
    flat = []
    for idx in indices: flat.extend(_flatten_indices(idx))
    return flat

@timer
def sanitize_line_rows(boxes,
                       indices,
                       rows,
                       word_boxes,
                       *,

                       word_indices     = None,
                       sanitize_lines   = False,
                       line_split_factor    = 0.6,
                       align_borders    = 'auto',

                       ** kwargs
                      ):
    """
        Final sanitization pass : rebuilds the rows of each group from the *original word
        boxes*, guaranteeing that the rows of a group are vertically disjoint lines
        (`bottom_i <= top_{i+1}`), so that no OCR crop mixes two lines of text.

        Rationale : rows are unions of word boxes, and the detector sometimes produces word
        boxes with an inflated height (~2 lines). Such a box makes its row bleed onto the
        neighboring line, and `_merge_rows` (transitive overlap-chaining) can even merge two
        genuine lines *through* it, yielding a single multi-line row (unreadable crop). The
        y-*center* of a word — even an inflated one — stays on the side of its true line,
        so re-clustering the word centers recovers the correct line structure :
            1. the group's words are clustered into lines (1-D clustering of the y-centers,
               split when the gap between consecutive centers exceeds
               `line_split_factor * median(word heights)`)
            2. each line keeps its *union* band (no glyph clipping on sane paragraphs) ;
               only where two adjacent lines overlap, they are clamped at the boundary
               halfway between `median(bottoms_i)` and `median(tops_i+1)` (medians ignore
               the bleeding outliers). A pair whose *median* bands already overlap is left
               untouched (no confident sanitization possible)

        Arguments :
            - boxes / indices / rows    : the final output of the combination chain
            - word_boxes    : the original `(n_words, 4)` boxes given to `combine_boxes`
            - word_indices  : index values associated to `word_boxes` (defaults to positions)
            - sanitize_lines    : disables the pass when `False`
            - line_split_factor : y-center gap (in fraction of the median word height) that
                                  separates two lines
        Return : `(boxes, indices, rows)`, same contract as `combine_boxes_overlap`
    """
    if not sanitize_lines or not line_split_factor or len(boxes) == 0:
        return boxes, indices, rows

    id_map = None
    if word_indices is not None:
        id_map = {idx : i for i, idx in enumerate(word_indices)}

    res_boxes, res_indices, res_rows = [], [], []
    for box, group_indices, group_rows in zip(boxes, indices, rows):
        flat = _flatten_indices(group_indices)
        if len(flat) < 2:
            res_boxes.append(box), res_indices.append(group_indices), res_rows.append(group_rows)
            continue

        positions   = [id_map[idx] for idx in flat] if id_map is not None else flat
        words       = word_boxes[np.array(positions, dtype = 'int64')]

        centers = (words[:, 1] + words[:, 3]) / 2.
        order   = np.argsort(centers)
        h_ref   = np.median(words[:, 3] - words[:, 1])

        # 1-D clustering of the word y-centers into lines (top to bottom)
        clusters = [[order[0]]]
        for prev, cur in zip(order[:-1], order[1:]):
            if centers[cur] - centers[prev] > line_split_factor * h_ref:
                clusters.append([])
            clusters[-1].append(cur)

        # per line : [x_min, union_top, x_max, union_bottom, median_top, median_bottom]
        lines = []
        for cluster in clusters:
            wc = words[cluster]
            lines.append([
                np.min(wc[:, 0]), np.min(wc[:, 1]), np.max(wc[:, 2]), np.max(wc[:, 3]),
                np.median(wc[:, 1]), np.median(wc[:, 3])
            ])

        # conflict-only clamping of adjacent lines at the median boundary
        for i in range(len(lines) - 1):
            if lines[i][3] <= lines[i + 1][1]:  continue    # already disjoint
            if lines[i][5] >= lines[i + 1][4]:  continue    # median bands overlap : skip
            boundary = (lines[i][5] + lines[i + 1][4]) / 2.
            lines[i][3]     = min(lines[i][3], boundary)
            lines[i + 1][1] = max(lines[i + 1][1], boundary)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Lines {} / {} of a group clamped at y = {:.3f}'.format(
                    i, i + 1, boundary
                ))

        new_rows = np.array([l[:4] for l in lines], dtype = word_boxes.dtype)
        if align_borders:
            new_rows = _align_rows(new_rows, align_borders, ** kwargs)

        res_boxes.append(compute_union(new_rows))
        res_indices.append([[flat[j] for j in cluster] for cluster in clusters])
        res_rows.append(new_rows)

    return np.array(res_boxes), res_indices, res_rows

@timer
@box_converter_wrapper('xyxy', normalize_mode = 'relative', force_np = True, as_dict = False)
def combine_boxes(boxes, indices = None, ** kwargs):
    """
        Combines `boxes` (list of e.g., single-word boxes) by creating horizontal then vertical combinations.
        This enables, as an example, to combine the individual words detected by `EAST` to sentences (horizontal lines), then paragraphs (vertically grouping lines)
        
        This method calls sequentially
            1) {combine_boxes_horizontal}
            2) {combine_boxes_vertical}
            3) {combine_boxes_overlap}
            4) {combine_line_continuations}
            5) {sanitize_line_rows}
    """
    if indices is None: indices = list(range(len(boxes)))

    combined_boxes, combined_indices, _ = combine_boxes_horizontal(
        boxes, indices = indices, source = 'xyxy', ** kwargs
    )
    combined_boxes, combined_indices, rows = combine_boxes_vertical(
        combined_boxes, indices = combined_indices, source = 'xyxy', ** kwargs
    )
    combined_boxes, combined_indices, rows = combine_boxes_overlap(
        combined_boxes, indices = combined_indices, rows = rows, source = 'xyxy', ** kwargs
    )
    combined_boxes, combined_indices, rows = combine_line_continuations(
        combined_boxes, combined_indices, rows, ** kwargs
    )
    combined_boxes, combined_indices, rows = sanitize_line_rows(
        combined_boxes, combined_indices, rows, boxes, word_indices = indices, ** kwargs
    )

    return combined_boxes, combined_indices, rows