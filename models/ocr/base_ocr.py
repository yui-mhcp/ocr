# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
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

from utils import *
from utils.image import *
from models.utils import *
from utils.callbacks import *
from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, ops
from utils.image import _image_formats, HTTPScreenMirror, save_image, load_image
from utils.image.bounding_box import *

from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_image_model import BaseImageModel
from custom_architectures.current_blocks import set_cudnn_lstm

logger  = logging.getLogger(__name__)

class BaseOCR(BaseImageModel, BaseTextModel):
    _directories    = {
        ** BaseImageModel._directories, 'stream_dir' : '{root}/{self.name}/stream'
    }

    output_signature    = BaseTextModel.text_signature

    augment_raw_data    = BaseImageModel.augment_box
    prepare_input   = BaseImageModel.get_image_with_box
    prepare_output  = BaseTextModel.encode_text
    
    def __init__(self, lang = 'multi', input_size = None, max_output_length = None, ** kwargs):
        self._init_text(lang = lang, ** kwargs)
        self._init_image(input_size = input_size, ** kwargs)
        
        self.max_output_length  = max_output_length

        super().__init__(** kwargs)
        
        set_cudnn_lstm(self.model)
                
        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)

    @property
    def input_signature(self):
        if self.is_encoder_decoder:
            return (self.image_signature, self.text_signature)
        return self.image_signature

    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_image)
    
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
    def infer(self, inputs, training = False, ** kwargs):
        if ops.rank(inputs) == 3: inputs = ops.expand_dims(inputs, axis = 0)
        
        return self.compiled_infer(inputs, training = training, ** kwargs)
    
    def compile(self, loss = None, metrics = None, ** kwargs):
        if not loss:
            loss = 'TextLoss' if self.is_encoder_decoder else 'CTCLoss'
        if not metrics:
            metrics = ['TextAccuracy'] if self.is_encoder_decoder else []
        
        return super().compile(loss = loss, metrics = metrics, ** kwargs)

    def decode_output(self, output, remove_tokens = True, ** kwargs):
        if self.is_encoder_decoder:
            return self.decode_text(output, remove_tokens = remove_tokens, ** kwargs)
        return self.ctc_decode_text(output, remove_tokens = remove_tokens, ** kwargs)

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
    
    def get_prediction_callbacks(self,
                                 *,

                                 save    = True,
                                 save_if_raw    = None,
                                 
                                 directory  = None,
                                 raw_img_dir    = None,
                                 
                                 filename   = 'image_{}.jpg',
                                 # Verbosity config
                                 verbose = 1,
                                 display = None,
                                 
                                 use_multithreading = False,

                                 ** kwargs
                                ):
        if save and save_if_raw is None: save_if_raw = True
        elif save_if_raw: save = True
        if display is None: display = not save

        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        
        predicted   = {}
        callbacks   = []
        if save_if_raw:
            if raw_img_dir is None: raw_img_dir = os.path.join(directory, 'images')
            callbacks.append(ImageSaver(
                key = 'filename',
                name    = 'saving raw',
                cond    = lambda filename = None, ** _: not isinstance(filename, str),
                data_key    = 'image',
                file_format = os.path.join(raw_img_dir, filename),
                index_key   = 'frame_index',
                use_multithreading  = use_multithreading
            ))
            
        if save:
            predicted   = load_json(map_file, {})
        
            callbacks.append(JSonSaver(
                data    = predicted,
                filename    = map_file,
                force_keys  = {'boxes', 'ocr'},
                primary_key = 'filename',
                use_multithreading = use_multithreading
            ))
        
        if display:
            callbacks.append(OCRDisplayer())
        
        return predicted, ['ocr'], callbacks

    @timer
    def predict(self,
                images,
                batch_size = 1,
                *,
                
                detector   = None,
                detector_kwargs = {'merge_method' : 'union'},
                
                overwrite   = False,
                
                combine     = True,
                box_filters = None,
                combine_config  = {},
                
                dezoom_factor   = 1.,
                
                method  = 'beam',
                num_beams   = 10,
                num_sentences   = 1,
                length_power    = 0.25,
                threshold   = 0.,
                
                box_processing  = None,
                
                predicted   = None,
                _callbacks  = None,
                required_keys   = None,
                
                ** kwargs
               ):
        """
            Performs Optical Character Recognition (OCR) on the givan `images` (either filename / raw)
            
            Arguments :
                - images  : the image(s) to perform OCR on
                    - str   : the filename of the image
                    - dict  : informations about the image
                        - must contain at least `filename` or `image` + (optional) `boxes`
                    - np.ndarray / Tensor    : the raw image
                    
                    - list / pd.DataFrame   : an iterable of the above types
                - batch_size    : the number of prediction to perform in parallel (currently, only `1` has been tested)
                
                - detector  : an `BaseDetector` instance (e.g. `EAST`) to first detect text boxes before performing OCR
                - detector_kwargs : kwargs for the `detector.predict` call (see note for example)
                
                - save    : whether to save result or not
                - directory   : where to save result (saved in `{directory}/map.json`)
                - overwrite   : whether to overwrite already predicted images
                - timestamp   : the request timestamp (i.e. do not overwrite prediction performed after this timestamp)
                - save_if_raw : whether to save raw image
                
                - combine     : whether to combine boxes in lines / paragraphs
                                /!\ OCR is currently only performed on lines and not on paragraphs
                                as the default model works with CTC decoding 
                - box_filter  : callable with the following signature : `box_filter(boxes, indices, rows) -> keep_indexes`
                                it should return the indexes of the boxes to keep (useful to filter out too small boxes)
                - combine_threshold : the `threshold` argument to the `combine_boxes` call
                - threshold   : the text log-likelihood threshold (i.e. all texts with a score lower than this threshold are skipped)
                
                - display : whether to display the result or not
                - box_processing  : callable with the following signature : `box_processing(box_infos, image)`
                                    this may be useful to perform an action directly after performing OCR on a single box
                - post_processing : callable with the following signature : `post_processing(infos, image)`
                                    the difference with `box_processing` is that this function is called after performing OCR on
                                    an image, and `infos` therefore contains the information about all boxes in the image
                
                - kwargs    : forwarded to `self.infer`
            Returns :
                - result    : a list of tuple (image, result)
                    - image     : either the filename (if any), either the raw image
                    - result    : a `dict` with (at least) keys
                        - timestamp : the timestamp at which the prediction has been performed
                        - ocr       : a `list` of `dict` containing the OCR information about the boxes
                                      {boxes:, rows:, box_mode:, text:, scores:}
                        - filename  : (if not raw image)
            
            Note : the `detector` argument can be used to perform detection + OCR in a single line of code
            ```python
            # Example in 1 line
            result = ocr_model.predict(
                filename, detector = 'east_en', combine = True
            )
            
            # Equivalent to
            
            ## Performs text detection with `detector_model`, returning a list of 3-elements tuple `(image, detected, infos)`
            detected = detector_model.predict(filename, save = False, dicsplay = False)
            
            ## Only the last element `infos` is expected by the OCR model
            detected = [d[-1] for d in detected]
            
            ## performs OCR on the detection result
            result   = ocr_model.predict(detected, combine = True)
            ``` 
        """
        ########################################
        #     Initial detection (optional)     #
        ########################################
        
        now = time.time()
        
        if detector is not None:
            if isinstance(detector, str):
                from models import get_pretrained
                detector = get_pretrained(detector)
            
            for k in ('save', 'display'): detector_kwargs.setdefault(k, False)
            images = [out[-1] for out in detector.predict(images, ** detector_kwargs)]
        
        ####################
        #  Pre-processing  #
        ####################
        
        with time_logger.timer('initialization'):
            join_callbacks = _callbacks is None
            if _callbacks is None:
                predicted, required_keys, _callbacks = self.get_prediction_callbacks(** kwargs)

            results, inputs, indexes, files, duplicates, filtered = prepare_prediction_results(
                images,
                predicted,
                
                rank    = 3,
                primary_key = 'filename',
                expand_files    = False,
                normalize_entry = path_to_unix,
                
                overwrite   = overwrite,
                required_keys   = required_keys,
                
                filters = lambda f: not f.endswith(_image_formats)
            )
            
            if filtered:
                logger.info('Skip files with unsupported extensions : {}'.format(filtered))
        
        ####################
        #  Inference loop  #
        ####################
        
        show_idx = apply_callbacks(results, 0, _callbacks)
        
        for idx, file, data in zip(indexes, files, inputs):
            assert isinstance(data, dict) and 'boxes' in data
            
            ##############################
            #  Image/boxes preparation   #
            ##############################

            image = self.get_image_data(data)

            infos = data.copy()
            infos['timestamp'] = now

            image = self.get_image_data(data)
            if combine:
                boxes, indices, rows = combine_boxes(
                    infos['boxes'], image = image, ** combine_config
                )
            else:
                boxes = convert_box_format(
                    infos['boxes'], BoxFormat.XYXY, image = image
                )
                if isinstance(boxes, dict): boxes = boxes['boxes']
                indices = list(range(len(boxes)))
                rows    = [boxes[i : i + 1] for i in range(len(boxes))]

            if box_filters:
                boxes, indices, rows = filter_boxes(box_filters, boxes, indices, rows)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Start OCR on {}'.format(file if file else 'raw image'))
            
            ####################
            #     OCR loop     #
            ####################

            for i, rows_i in enumerate(rows):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('- Rows : {}'.format(np.around(rows_i, decimals = 3)))
                
                outputs, scores = [], []
                with time_logger.timer('OCR'):
                    for row in rows_i:
                        inp = self.get_input(
                            {'filename' : image, 'boxes' : row, 'source' : BoxFormat.XYXY},
                            dezoom_factor = dezoom_factor
                        )
                        if any(s == 0 for s in inp.shape):
                            logger.info('Invalid input encountered : {}'.format(inp.shape))
                            continue

                        out = self.infer(
                            inp,
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

                    if not outputs: continue

                    if not self.is_encoder_decoder:
                        with time_logger.timer('CTC Beam search'):
                            lengths = np.array([len(out) for out in outputs], dtype = 'int32')
                            outputs = stack_batch(
                                outputs, pad_value = 0., dtype = 'float32', maybe_pad = True
                            )

                            texts, scores = self.decode_output(
                                outputs,
                                lengths = lengths,
                                method  = method,
                                return_scores   = True,
                                ** kwargs
                            )
                            if isinstance(texts[0], list): texts = [txt[0] for txt in texts]
                            scores = ops.convert_to_numpy(scores).reshape(-1)
                    else:
                        texts   = outputs
                        scores  = np.array(scores)

                if threshold < 0. and np.any(scores < threshold):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('- Prediction score is too low, skipping this prediction')
                    continue

                box_infos = {
                    'box' : boxes[i],
                    'rows'  : rows_i,
                    'text'  : ' \n'.join(texts),
                    'source'    : BoxFormat.XYXY,
                    'scores'    : scores
                }
                infos.setdefault('ocr', []).append(box_infos)

                if box_processing is not None:
                    box_processing(box_infos, image = image)

            if file:
                for duplicate_idx in duplicates[file]:
                    results[duplicate_idx] = (predicted.get(file, {}), infos)
            else:
                results[idx] = ({}, infos)

            show_idx = apply_callbacks(results, show_idx, _callbacks)

        if join_callbacks:
            for callback in _callbacks: callback.join()

        return [
            (stored.get('filename', None), out if out else stored)
            for stored, out in results
        ]

    def stream(self,
               stream,
               *,
               
               show = False,
               display  = False,
               max_time = None,
               buffer_size  = 1,
               
               fps  = None,
               low_fps  = None,
               high_fps = None,
               n_fps_change = 5,
               
               detector = 'east',
               
               save = True,
               save_stream  = None,
               save_frames  = None,
               save_transformed = False,
               
               directory    = None,
               filename     = 'frame-{}.jpg',
               stream_name  = None,
               
               output_shape = None,

               callback = None,

               wait_factor  = 0.,
               duplicate_threshold  = 0.9,
               duplicate_warmup     = 5,
               duplicate_tolerance  = 5,
               duplicate_wait_time  = 0.,

               filters  = None,
               region   = [0.2, 0.1, 0.6, 0.95],
               n_repeat = 2,
               ioa_threshold    = 0.9,
               
               threshold    = -0.2,
               
               ** kwargs
              ):
        ####################
        #    OCR method    #
        ####################
        
        _state  = {
            'prev_output'   : None,
            'emitted_texts' : [],
            'emitted_boxes' : (),
            'n_skipped'     : 0,
            'n_duplicates'  : 0,
            'warmup_frames' : 0,
            'high_fps'  : -1,
            'warmup'    : False
        }
        last_emitted    = []
        emitted_texts   = set()
        
        def _maybe_change_fps_mode(index, low = None, high = None):
            if low_fps is None or high_fps is None: return
            
            if low and _state['high_fps'] != -1 and index >= _state['high_fps'] - n_fps_change:
                set_fps(low_fps)
                _state['high_fps'] = -1
                logger.info('Frame #{} : set low fps mode'.format(index))
            elif high:
                if _state['high_fps'] == -1:
                    set_fps(high_fps)
                    logger.info('Frame #{} : set high fps mode'.format(index))
                _state['high_fps'] = index
            
        def _filter_emitted_boxes(boxes, ** _):
            if len(_state['emitted_boxes']) == 0: return list(range(len(boxes)))
            ioa = compute_ioa(boxes, _state['emitted_boxes'], as_matrix = True)
            return np.where(np.any(ioa >= ioa_threshold, axis = 1))[0]

        def perform_ocr(image, boxes, output, frame_index, detected = None, ** _):
            if len(boxes['boxes']) == 0:
                _state.update({
                    'warmup' : False, 'n_skipped' : _state['n_skipped'] + 1
                })
                _maybe_change_fps_mode(frame_index, low = True)
                return
            
            if ops.rank(output) == 3: output = output[:, :, 0]
            
            is_duplicate = False
            if _state['prev_output'] is not None:
                dice = dice_coeff(output, _state['prev_output'])
                is_duplicate = dice >= duplicate_threshold
            
            logger.info('Frame #{} : duplicte = {} - high fps : {} - waiting : {}'.format(
                frame_index, is_duplicate, _state['high_fps'], len(repet_filter)
            ))
            if is_duplicate:
                if _state['warmup']:
                    if repet_filter is None or len(repet_filter):
                        _maybe_change_fps_mode(frame_index, high = True)
                    _state['warmup_frames'] += 1
                    if _state['warmup_frames'] > duplicate_warmup: _state['warmup'] = False
                else:
                    _state['n_duplicates'] += 1
                    if not _is_file_stream and _state['n_duplicates'] % 2 == 0:
                        time.sleep(duplicate_wait_time)

                    if _state['n_duplicates'] > duplicate_tolerance and _state['n_skipped'] > duplicate_tolerance:
                        _state['n_skipped'] += 1
                        _maybe_change_fps_mode(frame_index, low = True)
                        return
            
            else:
                if _state['emitted_texts']:
                    _maybe_change_fps_mode(frame_index, high = True)
                
                _state.update({
                    'prev_output'   : output,
                    'n_duplicates'  : 0,
                    'warmup_frames' : 1,
                    'emitted_texts' : [],
                    'emitted_boxes' : (),
                    'warmup'    : True
                })
            
            res = self.predict(
                {'image' : image, 'boxes' : boxes},

                combine = True,
                box_filters = filters,
                
                logits_filter     = logits_filter,

                predicted   = {},
                _callbacks  = [],
                required_keys   = [],
                
                ** kwargs
            )[0][-1]

            is_empty    = len(res.get('ocr', [])) == 0
            if not is_empty:
                res['ocr']  = [
                    ocr_res for ocr_res in res['ocr']
                    if _filter_results(
                        ocr_res, emitted_texts, last_emitted, threshold = threshold, ** kwargs
                    )
                ]
            
            if len(res.get('ocr', [])) == 0:
                _state['n_skipped'] += 1
                
                if not is_empty:
                    logger.info('[SKIP #{}] All results have been filtered !'.format(frame_index))

                return

            logger.info('\nFrame #{}'.format(frame_index))
            emitted_boxes = np.array([ocr_res['box'] for ocr_res in res['ocr']])
            if not _state['emitted_texts']:
                logger.info('- 1st emission ({} warmup - {} duplicates - {} skipped)'.format(
                    _state['warmup_frames'], _state['n_duplicates'], _state['n_skipped']
                ))
                _state['emitted_boxes'] = emitted_boxes
            else:
                _state['emitted_boxes'] = np.concatenate([
                    _state['emitted_boxes'], emitted_boxes
                ], axis = 0)

            _state['n_skipped'] = 0

            if display:
                if detected is None:
                    detected = draw_boxes(ops.convert_to_numpy(image, copy = True), boxes)
                plot(detected, plot_type = 'imshow')
            
            for ocr_result in res['ocr']:
                last_emitted.append(ocr_result['text'])
                _state['emitted_texts'].append(ocr_result['text'])
                emitted_texts.add(ocr_result['text'])

                logger.info('Box  : {}\nScores : {}\nText   : {}'.format(
                    np.around(convert_box_format(
                        ocr_result['box'], target = BoxFormat.XYWH, source = ocr_result['source']
                    ), decimals = 3),
                    np.around(ocr_result['scores'], decimals = 3),
                    ocr_result['text']
                ))

                if callback: callback(ocr_result['text'])

                if display > 1:
                    show_boxes(image, ocr_result['box'], source = ocr_result['source'])
                    if len(ocr_result['rows']) > 1:
                        show_boxes(
                            image,
                            ocr_result['rows'],
                            source = ocr_result['source'],
                            ncols    = len(ocr_result['rows'])
                        )

            if save_frames:
                apply_callbacks_raw({
                    'image' : image, 'frame_index' : frame_index, 'boxes' : boxes, ** res
                }, callbacks)

            if wait_factor:
                time.sleep(
                    1. + wait_factor * max([len(ocr_res['text']) for ocr_res in res['ocr']])
                )


        ####################
        #  Initialization  #
        ####################
        
        if isinstance(detector, str):
            from models import get_pretrained
            detector = get_pretrained(detector)
        
        if callback is not None and not callable(callback):
            assert hasattr(callback, 'put'), '`callback` must be callable or have a put method !'
            callback = callback.put
        
        _is_file_stream = isinstance(stream, str) and os.path.isfile(stream)
        
        if save_stream is None: save_stream = save and not _is_file_stream
        if save_frames is None: save_frames = save
        if max_time is None:    max_time = 600 if not _is_file_stream else 0
        if low_fps:     fps = low_fps
        
        if directory is None: directory = self.stream_dir
        if not stream_name:
            stream_name = 'stream-{}' if not _is_file_stream else os.path.basename(stream).split('.')[0]
        
        stream_dir = os.path.join(directory, stream_name)
        if contains_index_format(stream_dir):
            stream_dir  = format_path_index(stream_dir)
            stream_name = os.path.basename(stream_dir)
        
        _, _, callbacks = self.get_prediction_callbacks(
            save    = save_frames,
            display = False,
            save_if_raw = save_frames,
            directory   = stream_dir,
            filename    = filename,
            use_multithreading  = kwargs.get('use_multithreading', True)
        )
        
        # Initializes filters
        if filters is None:
            filters = [SizeFilter(min_h = 0.025, min_w = 0.1)]
            if region:
                filters.append(RegionFilter(region = region, mode = 'center'))
            if n_repeat:
                filters.append(RepetitionFilter(n_repeat = n_repeat, use_memory = False))
        elif not isinstance(filters, list):
            filters = [filters]
        
        if ioa_threshold:
            filters.append(_filter_emitted_boxes)
        
        repet_filter = None
        if any(isinstance(f, RepetitionFilter) for f in filters):
            repet_filter = [f for f in filters if isinstance(f, RepetitionFilter)][0]
        
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
                post_processing = perform_ocr,

                ** kwargs
            )
        finally:
            for callback in callbacks: callback.join()

    stream_video = stream
    
    def get_config(self):
        config = super().get_config()
        config.update({
            ** self.get_config_image(),
            ** self.get_config_text(),
            'max_output_length' : self.max_output_length
        })
        return config

def _filter_results(ocr_result,
                    reject,
                    last_emitted    = [],
                    threshold       = 0.,
                    k   = 5,
                    max_dist    = 0.2,
                    skip_non_alpha  = True,
                    skip_single_word    = True,
                    ** kwargs
                   ):
    ocr_result['text'] = ocr_result['text'].replace('</s>', '')
    text = ocr_result['text']
    if not text: return False
    elif skip_single_word and ' ' not in text: return False
    elif skip_non_alpha and not any(c.isalpha() for c in text): return False
    elif text in reject:
        logger.info('Detected text was already emitted')
        return False
    

    ocr_result['scores'] = np.array(ocr_result['scores'])
    if threshold != 0. and np.any(ocr_result['scores'] <= threshold):
        parts = text.split(' \n')
        logger.info('Some scores are too low !\n{}'.format('\n'.join([
            '- {:.3f} : {}'.format(s, t) for s, t in zip(ocr_result['scores'], parts)
        ])))
        if np.all(ocr_result['scores'] <= threshold): return False

        logger.info('Parts : {}'.format(parts))
        text = ' \n'.join([
            p if s > threshold else '' for p, s in zip(parts, ocr_result['scores'])
        ]).strip()
        ocr_result['text'] = text
    
    if k > 0 and last_emitted:
        for emitted in last_emitted[- k :]:
            dist = edit_distance(emitted, text, normalize = True)
            if dist < max_dist:
                logger.info('Rejecting text due to an edit distance of {:.3f}\n  Text : {}\n  Previously emitted : {}'.format(dist, text, emitted))
                reject.add(text)
                return False

    return True
