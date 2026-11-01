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

import os
import glob
import time
import logging
import numpy as np

from functools import partial

from loggers import Timer, timer
from utils.keras import TensorSpec, ops
from utils import contains_index_format, format_path_index, load_json, pad_batch, plot_multiple
from utils.image import HTTPScreenMirror, combine_boxes
from utils.callbacks import ImageSaver, JSONSaver, FunctionCallback, OCRDisplayer, QueueCallback, apply_callbacks
from ..core import BaseModel, TextModelMixin, ImageModelMixin
from .box_filters import filter_boxes
from .region_tracker import RegionTracker
from .streaming import (
    StreamState, build_stream_filters, display_stream_results, emit_stream_results,
    filter_text_results
)

logger  = logging.getLogger(__name__)

class BaseOCR(ImageModelMixin, TextModelMixin, BaseModel):
    _directories    = {
        ** BaseModel._directories, 'stream_dir' : '{root}/{self.name}/stream'
    }

    output_signature    = TextModelMixin.text_signature

    prepare_input   = ImageModelMixin.get_image_with_box
    prepare_output  = TextModelMixin.encode_text
    
    def __init__(self, lang = 'multi', input_size = None, max_output_length = None, ** kwargs):
        self._init_text(lang = lang, ** kwargs)
        self._init_image(input_size = input_size, ** kwargs)
        
        self.max_output_length  = max_output_length

        super().__init__(** kwargs)
        
        if self.runtime == 'keras':
            from architectures import set_cudnn_lstm
            set_cudnn_lstm(self.model.engine)
        
        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)

    @property
    def input_signature(self):
        if self.is_encoder_decoder:
            return (self.image_signature, self.text_signature)
        return self.image_signature
    
    @property
    def default_loss_config(self):
        return {
            'pad_value'     : self.blank_token_idx,
            'eos_value'     : self.eos_token_idx,
            'from_logits'   : True
        }
    
    @property
    def default_metrics_config(self):
        return self.default_loss_config
    
    def __str__(self):
        return super().__str__() + self._str_image() + self._str_text()
    
    @timer(name = 'inference')
    def infer(self,
              data,
              *,
              
              detector  = 'east',
              detector_kwargs = {'merge_method' : 'union'},

              combine   = True,
              
              box_filters   = None,
              
              dezoom_factor = 1.,
              
              method    = 'beam_search',
              num_beams = 10,
              num_sentences = 1,
              length_power  = .25,
              
              threshold = 0.,
              
              box_callback  = None,
              
              callbacks = None,
              predicted = None,
              overwrite = False,
              return_output = True,
              
              ** kwargs
             ):
        """
            Arguments :
                - inputs    : `Tensor` of rank 3 or 4 (single / batched image(s))
                - kwargs    : forwarded to `decode_output` if `get_boxes = True`
            Return :
                - output    : `dict` containing the following entries
                              - filename    : `str`, the input image file (if provided)
                              - image       : `np.ndarray`, the raw loaded unprocessed image
                              - boxes       : `dict` with the `boxes`, `format` and `scores` entries
                              - output  : `Tensor`, the model output
            
            Note : depending whether `image` is a filename already in `predicted`, some keys in the output may be missing (notably `output` and `image`)
        """
        if predicted and not overwrite and isinstance(data, str) and data in predicted:
            return self._finalize_cached_prediction(
                data, callbacks = callbacks, predicted = predicted
            )
        
        filename, image = self.get_image_data(data)
        if not isinstance(data, dict) or 'boxes' not in data:
            if isinstance(detector, str):
                from models import get_pretrained
                detector = get_pretrained(detector, ** detector_kwargs)
            
            data = detector.infer(image)
        
        if combine:
            boxes, indices, rows = combine_boxes(
                data['boxes'], image = image, ** kwargs
            )
        else:
            boxes = data['boxes']
            if isinstance(boxes, dict): boxes = boxes['boxes']
            indices = list(range(len(boxes)))
            rows    = [boxes[i : i + 1] for i in range(len(boxes))]

        if box_filters:
            boxes, indices, rows = filter_boxes(box_filters, boxes, indices, rows)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Start OCR on {} ({} boxes)'.format(filename or 'raw image', len(boxes)))
        
        ocr_results = []
        for i, rows_i in enumerate(rows):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('- Rows : {}'.format(np.around(rows_i, decimals = 3)))
            
            outputs, scores = [], []
            with Timer('OCR'):
                for row in rows_i:
                    inp = self.get_input(
                        {'filename' : image, 'boxes' : {'boxes' : row, 'format' : 'xyxy'}},
                        dezoom_factor = dezoom_factor
                    )
                    if any(s == 0 for s in inp.shape):
                        logger.warning('Invalid input encountered : {}'.format(inp.shape))
                        continue
                    
                    out = self.compiled_infer(
                        inp[None],
                        method  = method,
                        num_beams   = num_beams,
                        num_sentences   = 1,
                        length_power    = length_power,
                        ** kwargs
                    )

                    if self.is_encoder_decoder:
                        scores.append(np.squeeze(
                            ops.convert_to_numpy(out.scores) / ops.convert_to_numpy(out.lengths)
                        ))
                        out = self.decode_output(out)
                        if isinstance(out[0], list): out = out[0]
                    outputs.append(out[0])

                if not outputs:
                    continue
                elif self.is_encoder_decoder:
                    texts   = outputs
                    scores  = np.array(scores)
                else:
                    with Timer('CTC Beam search'):
                        lengths = np.array([len(out) for out in outputs], dtype = 'int32')
                        outputs = pad_batch(outputs, pad_value = 0., dtype = 'float32')

                        texts, scores = self.decode_output(
                            outputs,
                            lengths = lengths,
                            method  = method,
                            num_beams   = num_beams,
                            return_scores   = True,
                            ** kwargs
                        )
                        if isinstance(texts[0], list): texts = [txt[0] for txt in texts]
                        scores = ops.convert_to_numpy(scores).reshape(-1)

            if threshold != 0. and np.any(scores <= threshold):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('- Prediction score is too low, skipping this prediction')
                continue

            box_infos = {
                'boxes' : boxes[i],
                'rows'  : rows_i,
                'text'  : ' \n'.join(texts),
                'format'    : 'xyxy',
                'scores'    : scores
            }
            ocr_results.append(box_infos)

            if box_callback is not None:
                box_callback(box_infos, image = image)
            
        
        output = {
            'filename'  : filename,
            'boxes'     : data.get('boxes', None) if isinstance(data, dict) else None,
            'ocr'       : ocr_results,
            'image'     : image
        }
        return self._finalize_predictions(
            output, callbacks = callbacks, predicted = predicted, return_output = return_output
        )

    def filter_prediction_output(self, output):
        return {k : v for k, v in output.items() if 'image' not in k}

    
    def compile(self, *, loss = None, metrics = None, ** kwargs):
        if not loss:
            loss = 'TextLoss' if self.is_encoder_decoder else 'CTCLoss'
        if not metrics:
            metrics = ['TextAccuracy'] if self.is_encoder_decoder else []
        
        return super().compile(loss = loss, metrics = metrics, ** kwargs)

    def decode_output(self, output, ** kwargs):
        if self.is_encoder_decoder:
            return self.decode_text(output, ** kwargs)
        return self.ctc_decode_text(output, ** kwargs)

    def prepare_data(self, data):
        image   = self.prepare_input(data)
        tokens  = self.prepare_output(data, key = 'label')

        if self.is_encoder_decoder:
            return (image, tokens[:-1]), tokens[1:]
        return image, tokens
    
    def filter_input(self, inputs):
        image = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        return self.filter_image(image)
    
    def filter_output(self, output):
        all_positive = ops.all(ops.shape(output) > 0)
        if self.max_output_length is None: return all_positive
        return ops.logical_and(all_positive, ops.shape(output)[-1] <= self.max_output_length)

    def augment_input(self, inputs):
        if self.is_encoder_decoder:
            return (self.augment_image(inputs[0]), inputs[1])
        return self.augment_image(inputs)
    
    def process_input(self, inputs, ** kwargs):
        if isinstance(inputs, (list, tuple)):
            return (self.process_image(inputs[0], ** kwargs), inputs[1])
        return self.process_image(inputs, ** kwargs)
    
    def get_dataset_config(self, * args, ** kwargs):
        inp_padding = (0., self.blank_token_idx) if self.is_encoder_decoder else 0.
        kwargs.update({
            'pad_kwargs'    : {
                'padding_values'    : (
                    inp_padding, self.blank_token_idx
                )
            }
        })
        
        return super().get_dataset_config(* args, ** kwargs)
    
    def get_inference_callbacks(self,
                                *,

                                save    = True,
                                save_if_raw    = None,
                                 
                                directory  = None,
                                raw_img_dir    = None,
                                 
                                filename   = 'image_{}.jpg',
                                # Verbosity config
                                verbose = 1,
                                display = None,
                                
                                post_processing = None,
                                save_in_parallel = False,

                                ** kwargs
                               ):
        if save and save_if_raw is None: save_if_raw = True
        elif save_if_raw: save = True
        if display is None: display = not save

        if directory is None: directory = self.pred_dir
        
        predicted   = {}
        callbacks   = []
        if save_if_raw:
            if raw_img_dir is None: raw_img_dir = os.path.join(directory, 'images')
            callbacks.append(ImageSaver(
                cond    = lambda filename = None, ** _: not isinstance(filename, str),
                file_format = os.path.join(raw_img_dir, filename),
                index_key   = 'frame_index',
                save_in_parallel    = save_in_parallel
            ))
            
        if save:
            map_file    = os.path.join(directory, 'map.json')
            predicted   = load_json(map_file, {})
        
            callbacks.append(JSONSaver(
                data    = predicted,
                filename    = map_file,
                primary_key = 'filename',
                save_in_parallel    = save_in_parallel
            ))
        
        if display:
            callbacks.append(OCRDisplayer())
        
        if post_processing is not None:
            if not isinstance(post_processing, list): post_processing = [post_processing]
            for fn in post_processing:
                if callable(fn):
                    callbacks.append(FunctionCallback(fn))
                elif hasattr(fn, 'put'):
                    callbacks.append(QueueCallback(fn))

        return predicted, callbacks

    def _normalize_prediction_inputs(self, inputs):
        if ops.is_array(inputs) and len(inputs.shape) == 3: return [inputs]
        return super()._normalize_prediction_inputs(inputs)

    def stream_fn(self,
                  *,
                  image,
                  boxes,
                  output,
                  frame_index,
                  state,
                  
                  detected  = None,

                  display   = True,
                  verbose   = True,
                  filters   = None,
                  callback  = None,
                  callbacks = None,

                  duplicate_threshold  = 0.9,
                  duplicate_tolerance  = 5,
                  duplicate_wait_time  = 0.,
                  rescan_every = None,

                  k    = 15,
                  max_dist     = 0.2,
                  text_threshold   = -0.2,
                  skip_non_alpha   = True,
                  skip_single_word = True,

                  wait_factor  = 0.,

                  ** kwargs
                 ):
        """
            Per-frame streaming callback (cf `stream`) : de-duplicates the frame against the
            previous detection map, OCR-ises the (box-filtered) new regions, filters the texts,
            then emits the genuinely new results (`state`, logs, `callback` and `callbacks`).

            Two de-duplication pipelines coexist (selected by `stream(track_regions = ...)`) :
                - `state.tracker` set : the `RegionTracker` (inside the box-filter chain)
                  handles the spatial + textual de-duplication (cf `region_tracker.py`)
                - legacy : `last_boxes` IoA filter + `RepetitionFilter` + emitted-texts sets

            `kwargs` are forwarded to `self.infer` (generation / box-combination config).
        """
        if verbose: logger.info('Frame #{} ({} boxes)'.format(frame_index, len(boxes['boxes'])))

        output = output[:, :, 0]
        if len(boxes['boxes']) == 0:
            # an empty frame still counts as *scanned* for the tracker : regions age here
            if state.tracker is not None: state.tracker.observe_empty(output)
            state.set_new_frame(output)
            state.n_skipped += 1
            return

        if rescan_every is None: rescan_every = duplicate_tolerance
        prev_output = state.prev_output
        skip, is_duplicate, dice = state.should_skip_duplicate(
            output,
            duplicate_threshold = duplicate_threshold,
            duplicate_tolerance = duplicate_tolerance,
            rescan_every    = rescan_every
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('- Is duplicate : {} ({:.3f}) - waiting : {}'.format(
                is_duplicate, dice, len(state.repetition_filter) if state.repetition_filter is not None else '?'
            ))

        if not is_duplicate and display == 3 and prev_output is not None:
            plot_multiple(
                image = image, output = output, prev = prev_output, plot_type = 'imshow', ncols = 3
            )

        if skip:
            state.n_skipped += 1
            # throttles the frame consumption on a live stream showing a static scene
            if duplicate_wait_time and not state.is_file:
                time.sleep(duplicate_wait_time)
            return

        if state.tracker is not None: state.tracker.set_frame(output)

        res = self.infer(
            {'image' : image, 'boxes' : boxes}, combine = True, box_filters = filters, ** kwargs
        ).get('ocr', [])

        if len(res) == 0:
            state.n_skipped += 1
            return

        if state.tracker is not None:
            # region-tracking pipeline : each OCR-ised box is reported back to its slot,
            # which decides the emission (new slot / changed content). The generic guards of
            # `filter_text_results` (strip `</s>`, single word, low-score parts, ...) still
            # apply, but its emitted-texts / edit-distance de-duplication is disabled
            # (`reject = set()`, `k = 0`) : that is now the tracker's job.
            kept = []
            for ocr_res in res:
                emittable = filter_text_results(
                    ocr_res,
                    set(),
                    text_threshold  = text_threshold,
                    k   = 0,
                    skip_non_alpha  = skip_non_alpha,
                    skip_single_word    = skip_single_word,
                    verbose = verbose
                )
                if state.tracker.register_text(
                    state.tracker.resolve(ocr_res['boxes']),
                    ocr_res['text'],
                    ocr_res['scores'],
                    emittable   = emittable
                ):
                    kept.append(ocr_res)
            res = kept
        else:
            # NOTE : `last_boxes` accumulates *all* OCR-ised boxes, on purpose *before* the text
            # filtering below : a region whose text is rejected (e.g. a recurring false positive)
            # is still memorized so `make_emitted_box_filter` stops re-OCR-ising it every frame.
            emitted_boxes = np.array([ocr_res['boxes'] for ocr_res in res])
            if state.last_boxes is None:
                if verbose: logger.info('- 1st emission : {}'.format(state))
                state.last_boxes = emitted_boxes
            else:
                state.last_boxes = np.concatenate([state.last_boxes, emitted_boxes], axis = 0)

            res = [ocr_res for ocr_res in res if filter_text_results(
                ocr_res,
                state.emitted_texts,
                state.last_texts,
                text_threshold  = text_threshold,
                k   = k,
                max_dist    = max_dist,
                skip_non_alpha  = skip_non_alpha,
                skip_single_word    = skip_single_word,
                verbose = verbose
            )]

        if len(res) == 0:
            if verbose:
                logger.info('[SKIP] Frame #{} : all results have been filtered !'.format(frame_index))

            state.n_skipped += 1
            return

        state.n_skipped = 0
        
        if display:
            display_stream_results(image, res, detected = detected, display = display)

        emit_stream_results(state, res, callback = callback, verbose = verbose)

        if callbacks:
            # the emitted (filtered) results are applied manually here, and *not* inside
            # `self.infer` : callbacks must only run on genuinely new text (see the
            # duplicate / repetition filtering above)
            out = {'image' : image, 'frame_index' : frame_index, 'boxes' : boxes, 'ocr' : res}
            apply_callbacks(callbacks, self.filter_prediction_output(out), out)

        if wait_factor:
            # leaves time to the TTS to read the emitted text before the next prediction
            # (hypothesis : the text stays on screen at least the time to read it)
            time.sleep(1. + wait_factor * max([len(ocr_res['text']) for ocr_res in res]))

    def stream(self,
               stream,
               *,
               
               detector = 'east',
               detector_kwargs  = {},

               verbose  = True,
               display  = False,
               
               callback = None,
               post_processing  = None,
               
               filters  = None,
               region   = [0.2, 0.05, 0.6, 0.95],
               n_repeat = 2,
               ioa_threshold    = 0.9,

               track_regions    = False,
               tracker_config   = {},

               save = True,
               save_stream  = None,
               save_frames  = None,
               save_transformed = False,
               save_in_parallel = True,

               directory    = None,
               filename     = 'frame-{}.jpg',
               stream_name  = None,

               fps  = None,
               show = False,
               max_time = None,
               buffer_size  = 1,
               
               output_shape = None,

               ** kwargs
              ):
        """
            Runs the OCR streaming pipeline : `detector.stream(...)` is called with
            `post_processing = self.stream_fn`, which is applied on each frame (with the frame
            and its detected boxes).

            kwargs routing :
                - `kwargs` are forwarded to `stream_fn` **only** (i.e. to `self.infer` :
                  generation / box-combination / text-filtering config)
                - `detector_kwargs` are forwarded to `detector.stream(...)` **only** (detector
                  inference + `stream_camera` config, e.g. `use_xla`, `nb_frames`, ...)

            De-duplication pipeline :
                - `track_regions = True` : the `RegionTracker` (configured by `tracker_config`,
                  cf `region_tracker.py`) replaces the `RepetitionFilter`, the IoA
                  "already emitted" filter and the emitted-texts sets (`n_repeat` /
                  `ioa_threshold` are then ignored)
                - `track_regions = False` (legacy) : `n_repeat` / `ioa_threshold` /
                  emitted-texts de-duplication, as before
        """
        state = StreamState(is_file = isinstance(stream, str) and os.path.isfile(stream))

        if isinstance(detector, str):
            from models import get_pretrained
            detector = get_pretrained(detector)
        
        if post_processing is not None: callback = post_processing
        if callback is not None and not callable(callback):
            assert hasattr(callback, 'put'), '`callback` must be callable or have a put method !'
            callback = callback.put
        
        _is_file_stream = state.is_file
        
        if save_stream is None: save_stream = save and not _is_file_stream
        if save_frames is None: save_frames = save
        if max_time is None:    max_time = 600 if not _is_file_stream else 0
        
        if directory is None: directory = self.stream_dir
        if not stream_name:
            stream_name = 'stream-{}' if not _is_file_stream else os.path.basename(stream).split('.')[0]
        
        stream_dir = os.path.join(directory, stream_name)
        if contains_index_format(stream_dir):
            stream_dir  = format_path_index(stream_dir)
            stream_name = os.path.basename(stream_dir)
        
        _, callbacks = self.get_inference_callbacks(
            save    = save_frames,
            display = False,
            save_if_raw = save_frames,
            directory   = stream_dir,
            filename    = filename,
            save_in_parallel  = save_in_parallel
        )
        
        # Initializes the box-filter chain (also registers the `RepetitionFilter` or the
        # `RegionTracker` on the state)
        filters = build_stream_filters(
            filters, state,
            region  = region,
            n_repeat    = n_repeat,
            ioa_threshold   = ioa_threshold,
            tracker = RegionTracker(** tracker_config) if track_regions else None
        )

        logits_filter = ops.convert_to_tensor([self.ukn_token_idx], 'int32')
        
        camera = stream
        if not _is_file_stream and isinstance(stream, str) and stream.startswith('http://192'):
            camera, frame = HTTPScreenMirror.build(stream)
            output_shape  = frame.shape[:2]

        try:
            return detector.stream(
                camera,
                fps = fps,
                show    = show,
                max_time    = max_time,
                buffer_size  = buffer_size,

                save    = False,
                display = False,
                directory   = directory,
                stream_name = stream_name,
                save_stream = save_stream,
                save_transformed    = save_transformed,

                output_shape = output_shape,

                return_output   = True,
                post_processing = partial(
                    self.stream_fn,
                    state   = state,
                    verbose = verbose,
                    display = display,

                    filters = filters,
                    callback    = callback,
                    callbacks   = callbacks,
                    logits_filter   = logits_filter,
                    ** kwargs
                ),

                ** detector_kwargs
            )
        finally:
            for callback in callbacks: callback.join()

    def get_config(self):
        return {
            ** super().get_config(),
            ** self.get_config_image(),
            ** self.get_config_text(),
            'max_output_length' : self.max_output_length
        }