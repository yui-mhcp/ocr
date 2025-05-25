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
from dataclasses import dataclass, field

from loggers import Timer, timer
from utils.text import edit_distance
from utils.keras import TensorSpec, ops
from utils import contains_index_format, dice_coeff, format_path_index, load_json, pad_batch, plot, plot_multiple
from utils.image import HTTPScreenMirror, SizeFilter, RegionFilter, RepetitionFilter, _image_formats, combine_boxes, compute_ioa, convert_box_format, draw_boxes, filter_boxes, show_boxes
from utils.callbacks import ImageSaver, JSONSaver, FunctionCallback, OCRDisplayer, QueueCallback, apply_callbacks
from ..interfaces.base_text_model import BaseTextModel
from ..interfaces.base_image_model import BaseImageModel

logger  = logging.getLogger(__name__)

@dataclass
class StreamState:
    is_file : bool  = False
    
    emitted_texts   : set   = field(default_factory = set, repr = False)
    last_texts  : list  = field(default_factory = list)
    last_boxes  : np.ndarray    = field(default = None, repr = False)
    
    n_skipped   : int   = 0
    n_duplicates    : int   = 0
    prev_output : np.ndarray    = field(default = None, repr = False)
    
    repetition_filter   : RepetitionFilter   = None
    
    def set_new_frame(self, output):
        self.prev_output    = output
        self.last_texts = []
        self.last_boxes = None
        self.n_duplicates   = 0
        #if self.repetition_filter is not None: self.repetition_filter.clear()
    
class BaseOCR(BaseImageModel, BaseTextModel):
    _directories    = {
        ** BaseImageModel._directories, 'stream_dir' : '{root}/{self.name}/stream'
    }

    output_signature    = BaseTextModel.text_signature

    prepare_input   = BaseImageModel.get_image_with_box
    prepare_output  = BaseTextModel.encode_text
    
    def __init__(self, lang = 'multi', input_size = None, max_output_length = None, ** kwargs):
        self._init_text(lang = lang, ** kwargs)
        self._init_image(input_size = input_size, ** kwargs)
        
        self.max_output_length  = max_output_length

        super().__init__(** kwargs)
        
        if self.runtime == 'keras':
            from architectures import set_cudnn_lstm
            set_cudnn_lstm(self.model)
        
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
            if callbacks: apply_callbacks(callbacks, predicted[data], {}, save = False)
            return predicted[data]
        
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

            if threshold < 0. and np.any(scores < threshold):
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
        
        entry = None
        if callbacks:
            infos = {k : v for k, v in output.items() if 'image' not in k}

            entry = apply_callbacks(callbacks, infos, output, save = True)
        
        return output if return_output else predicted.get(entry, {})

    
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

    @timer
    def predict(self, inputs, ** kwargs):
        if (isinstance(inputs, (str, dict))) or (ops.is_array(inputs) and len(inputs.shape) == 3):
            inputs = [inputs]
        
        return super().predict(inputs, ** kwargs)

    def stream_fn(self,
                  *,
                  image,
                  boxes,
                  output,
                  frame_index,
                  detected  = None,
                  
                  display   = True,
                  verbose   = True,
                  filters   = None,
                  callback  = None,
                  callbacks = None,

                  duplicate_threshold  = 0.9,
                  duplicate_tolerance  = 5,
                  duplicate_wait_time  = 0.,

                  threshold    = -0.2,
                  wait_factor  = 0.,
                  
                  ** kwargs
                 ):
        if verbose: logger.info('Frame #{} ({} boxes)'.format(frame_index, len(boxes['boxes'])))

        output = output[:, :, 0]
        if len(boxes['boxes']) == 0:
            self._stream_state.set_new_frame(output)
            self._stream_state.n_skipped += 1
            return

        is_duplicate, dice = False, 0.
        if self._stream_state.prev_output is not None:
            dice = dice_coeff(output, self._stream_state.prev_output)
            is_duplicate = dice >= duplicate_threshold

        if verbose:
            logger.info('- Is duplicte : {} ({:.3f}) - waiting : {}'.format(
                is_duplicate, dice, len(self._stream_state.repetition_filter) if self._stream_state.repetition_filter is not None else '?'
            ))

        if is_duplicate:
            self._stream_state.n_duplicates += 1
            if self._stream_state.n_duplicates > duplicate_tolerance:
                if not self._stream_state.is_file and self._stream_state.n_duplicates % 2 == 0:
                    time.sleep(duplicate_wait_time)

                if self._stream_state.n_skipped % duplicate_tolerance < 2:
                    self._stream_state.n_skipped += 1
                    return

        else:
            if display == 3 and self._stream_state.prev_output is not None:
                plot_multiple(
                    image = image, output = output, prev = self._stream_state.prev_output,
                    plot_type = 'imshow', ncols = 3
                )
            self._stream_state.set_new_frame(output)

        res = self.infer(
            {'image' : image, 'boxes' : boxes}, combine = True, box_filters = filters, ** kwargs
        ).get('ocr', [])

        is_empty    = len(res) == 0
        if not is_empty:
            emitted_boxes = np.array([ocr_res['boxes'] for ocr_res in res])
            if self._stream_state.last_boxes is None:
                if verbose: logger.info('- 1st emission : {}'.format(self._stream_state))
                self._stream_state.last_boxes = emitted_boxes
            else:
                self._stream_state.last_boxes = np.concatenate([
                    self._stream_state.last_boxes, emitted_boxes
                ], axis = 0)
            
            res = [ocr_res for ocr_res in res if _filter_text_results(
                ocr_res,
                self._stream_state.emitted_texts,
                self._stream_state.last_texts,
                threshold = threshold,
                verbose = verbose,
                ** kwargs
            )]

        if len(res) == 0:
            self._stream_state.n_skipped += 1

            if not is_empty and verbose:
                logger.info('[SKIP] All results have been filtered !'.format(frame_index))

            return
        else:
            self._stream_state.n_skipped = 0

        if display:
            emitted_boxes = np.array([ocr_res['boxes'] for ocr_res in res])
            if detected is None:
                detected = draw_boxes(
                    ops.convert_to_numpy(image, copy = True),
                    emitted_boxes,
                    source  = 'xyxy',
                    color   = 'r',
                    labels  = ['text'],
                    show_text   = False
                )
            plot(detected, plot_type = 'imshow')

        for ocr_result in res:
            self._stream_state.last_texts.append(ocr_result['text'])
            self._stream_state.emitted_texts.add(ocr_result['text'])

            if verbose:
                logger.info('Box  : {}\nScores : {}\nText   : {}'.format(
                    np.around(convert_box_format(
                        ocr_result['boxes'], target = 'xywh', source = ocr_result['format']
                    ), decimals = 3),
                    np.around(ocr_result['scores'], decimals = 3),
                    ocr_result['text']
                ))

            if callback: callback(ocr_result['text'])

            if display > 1:
                show_boxes(image, ocr_result['boxes'], source = ocr_result['format'])
                if len(ocr_result['rows']) > 1:
                    show_boxes(
                        image,
                        ocr_result['rows'],
                        source = ocr_result['format'],
                        ncols    = len(ocr_result['rows'])
                    )

        if callbacks:
            apply_callbacks(callbacks, {}, {
                'image' : image, 'frame_index' : frame_index, 'boxes' : boxes, 'ocr' : res
            })

        if wait_factor:
            time.sleep(
                1. + wait_factor * max([len(ocr_res['text']) for ocr_res in res])
            )

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
        ####################
        #    OCR method    #
        ####################
        
        self._stream_state = StreamState(is_file = isinstance(stream, str) and os.path.isfile(stream))
        
        def _filter_emitted_boxes(boxes, ** _):
            if self._stream_state.last_boxes is None: return list(range(len(boxes)))
            ioa = compute_ioa(boxes, self._stream_state.last_boxes, as_matrix = True, source = 'xyxy')
            return np.where(np.all(ioa < ioa_threshold, axis = 1))[0]


        ####################
        #  Initialization  #
        ####################
        
        if isinstance(detector, str):
            from models import get_pretrained
            detector = get_pretrained(detector)
        
        if post_processing is not None: callback = post_processing
        if callback is not None and not callable(callback):
            assert hasattr(callback, 'put'), '`callback` must be callable or have a put method !'
            callback = callback.put
        
        _is_file_stream = self._stream_state.is_file
        
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
        
        # Initializes filters
        if filters is None:
            filters = [SizeFilter(min_h = 0.025, min_w = 0.1)]
            if region:
                filters.append(RegionFilter(region = region, source = 'xyxy', mode = 'center'))
            if n_repeat:
                filters.append(RepetitionFilter(n_repeat = n_repeat, use_memory = False))
        elif not isinstance(filters, list):
            filters = [filters]
        
        if ioa_threshold:
            filters.append(_filter_emitted_boxes)
        
        repet_filter = None
        if any(isinstance(f, RepetitionFilter) for f in filters):
            self._stream_state.repetition_filter = [
                f for f in filters if isinstance(f, RepetitionFilter)
            ][0]
        
        logits_filter = ops.convert_to_tensor([self.ukn_token_idx], 'int32')
        
        camera = stream
        if not _is_file_stream and isinstance(stream, str) and stream.startswith('http://192'):
            for _ in range(5):
                camera = HTTPScreenMirror(stream)
                ret, frame  = camera.read()
                if not ret:
                    time.sleep(1)
                else:
                    output_shape    = frame.shape[:2]
                    break
            if not ret: raise RuntimeError('Unable to connect to {} !'.format(stream))

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

                output_shape = output_shape,

                return_output   = True,
                post_processing = partial(
                    self.stream_fn,
                    verbose = verbose,
                    display = display,
                    
                    filters = filters,
                    callback    = callback,
                    callbacks   = callbacks,
                    logits_filter   = logits_filter,
                    ** kwargs
                ),

                ** {** kwargs, ** detector_kwargs}
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

def _filter_text_results(ocr_result,
                         reject,
                         last_emitted   = [],
                         threshold      = 0.,
                         verbose    = True,
                         *,
                         
                         k   = 5,
                         max_dist   = 0.2,
                         skip_non_alpha = True,
                         skip_single_word   = True,
                         
                         ** kwargs
                        ):
    ocr_result['text'] = ocr_result['text'].replace('</s>', '').strip()
    text = ocr_result['text']
    if not text: return False
    elif skip_single_word and ' ' not in text: return False
    elif skip_non_alpha and not any(c.isalpha() for c in text): return False
    elif text in reject:
        if verbose: logger.info('Detected text was already emitted')
        return False
    
    if threshold != 0. and np.any(ocr_result['scores'] <= threshold):
        parts = text.split(' \n')
        if verbose:
            logger.info('Some scores are too low !\n{}'.format('\n'.join([
                '- {:.3f} : {}'.format(s, t) for s, t in zip(ocr_result['scores'], parts)
            ])))
        if np.all(ocr_result['scores'] <= threshold): return False

        if verbose: logger.info('Parts : {}'.format(parts))
        text = ' \n'.join([
            p if s > threshold else '' for p, s in zip(parts, ocr_result['scores'])
        ]).strip()
        ocr_result['text'] = text
    
    if last_emitted and k:
        for emitted in last_emitted[- k :]:
            dist = edit_distance(emitted, text, normalize = True)
            if dist < max_dist:
                if verbose:
                    logger.info('Rejecting text due to an edit distance of {:.3f}\n  Text : {}\n  Previously emitted : {}'.format(dist, text, emitted))
                reject.add(text)
                return False

    return True
