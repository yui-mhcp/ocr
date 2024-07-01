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
import pandas as pd

from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, ops
from utils import plot, load_json, dump_json, get_filename, should_predict, pad_batch
from utils.distance import dice_coeff, edit_distance
from utils.text import build_masking_filter
from utils.image import HTTPScreenMirror, save_image, load_image
from utils.image.bounding_box import *
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_image_model import BaseImageModel
from custom_architectures.current_blocks import set_cudnn_lstm

logger  = logging.getLogger(__name__)

class BaseOCR(BaseImageModel, BaseTextModel):
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
    def stream_dir(self):
        return os.path.join(self.folder, "stream")

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
        if len(ops.shape(inputs)) == 3: inputs = inputs[None]
        if hasattr(self.model, 'infer'):
            kwargs.setdefault('max_length', self.max_output_length)
        
            return self.model.infer(inputs, training = training, ** kwargs)
        return self(inputs, training = training, ** kwargs)
    
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
    
    @timer
    def predict(self,
                images,
                batch_size = 1,
                
                detector   = None,
                detector_kwargs = {'merge_method' : 'union'},
                
                save    = True,
                directory   = None,
                overwrite   = False,
                timestamp   = -1,
                save_if_raw = True,
                
                combine     = True,
                box_filter  = None,
                combine_config  = {},
                
                dezoom_factor   = 1.,
                
                method  = 'beam',
                num_beams   = 10,
                num_sentences   = 1,
                length_power    = 0.25,
                threshold   = 0.,
                
                display = None,
                box_processing  = None,
                post_processing = None,
                
                ** kwargs
               ):
        """
            Performs Optical Character Recognition (OCR) on the givan `images` (either filename / raw)
            
            Arguments :
                - images  : the image(s) to perform OCR on
                    - str   : the filename of the image
                    - dict / pd.Series  : informations about the image
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
        ####################
        # helping function #
        ####################
        
        def save_raw_image(image):
            os.makedirs(raw_img_dir, exist_ok = True)
            filename = os.path.join(raw_img_dir, 'image_{}.png'.format(len(os.listdir(raw_img_dir))))
            save_image(image = image, filename = filename)
            return filename
        
        now = time.time()
        
        if detector is not None:
            if isinstance(detector, str):
                from models import get_pretrained
                detector = get_pretrained(detector)
            
            for k in ('save', 'display'): detector_kwargs.setdefault(k, False)
            detected = detector.predict(images, ** detector_kwargs)
            images = [detect[-1] for detect in detected]

        if display is None: display = not save
        if box_filter is not None: box_filter = combine_box_filters(box_filter)
        
        if not isinstance(images, list):
            if isinstance(images, pd.DataFrame):    images = images.to_dict('records')
            elif isinstance(images, (str, dict)):   images = [images]
            elif not ops.is_array(images):
                raise ValueError(
                    'Unsupported `images` type : {}\n{}'.format(type(images), images)
                )
            elif len(images.shape) == 3: images = images[None]

        
        ##############################
        #   Saving initialization    #
        ##############################
        
        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        raw_img_dir = os.path.join(directory, 'images')
        
        predicted   = load_json(map_file, default = {})
        
        ####################
        #  Pre-processing  #
        ####################
        
        results     = [None] * len(images)
        duplicatas  = {}
        requested   = [(get_filename(img, keys = ('filename', 'image')), img) for img in images]
        
        inputs  = []
        for i, (file, img) in enumerate(requested):
            if not should_predict(predicted, file, overwrite = overwrite, timestamp = timestamp, required_keys = ['ocr']):
                results[i] = (file, predicted[file])
                continue

            if isinstance(file, str):
                duplicatas.setdefault(file, []).append(i)
                if len(duplicatas[file]) > 1: continue
            
            inputs.append((i, file, img))
        
        ####################
        #  Inference loop  #
        ####################
        
        show_idx = post_process(results, 0, display, post_processing)
        
        if len(inputs) > 0:
            for idx, file, data in inputs:
                infos = data.copy() if isinstance(data, dict) else {}
                infos['timestamp'] = now
                
                image = load_image(file if file is not None else data['image'])
                if 'boxes' not in infos:
                    raise NotImplementedError(
                        'Currently, only data with `boxes` key are supported'
                    )

                if combine:
                    boxes, indices, rows = combine_boxes(
                        data['boxes'], image = image, ** combine_config
                    )
                else:
                    boxes = convert_box_format(
                        data['boxes'], BoxFormat.XYXY, image = file
                    )
                    if isinstance(boxes, dict): boxes = boxes['boxes']
                    indices = list(range(len(boxes)))
                    rows    = [boxes[i : i + 1] for i in range(len(boxes))]

                if box_filter is not None:
                    keep_indexes = box_filter(boxes = boxes, indices = indices, rows = rows)
                    boxes, rows  = boxes[keep_indexes], [rows[index] for index in keep_indexes]

                for i, rows_i in enumerate(rows):
                    logger.debug('Rows : {}'.format(np.around(rows_i, decimals = 3)))
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
                                num_sentences   = num_sentences,
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
                            lengths = [len(out) for out in outputs]
                            outputs = pad_batch(outputs, pad_value = 0., dtype = np.float32)

                            with time_logger.timer('CTC Beam search'):
                                texts, scores = self.decode_output(
                                    outputs,
                                    lengths = lengths,
                                    method  = method,
                                    return_scores   = True
                                )
                                if isinstance(texts[0], list): texts = [txt[0] for txt in texts]
                                scores = ops.convert_to_numpy(scores).reshape(-1)
                        else:
                            texts   = outputs
                            scores  = np.array(scores)

                    if threshold < 0. and np.any(scores < threshold):
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
                
                if file is None: file = data
                
                if isinstance(file, str):
                    infos['filename']   = file
                    predicted[file] = infos
                    
                    for duplicate_idx in duplicatas[file]:
                        results[duplicate_idx] = (file, infos)
                    
                    if save:
                        dump_json(map_file, predicted, indent = 4)
                else:
                    results[idx] = (file if file is not None else img, infos)
                
                show_idx = post_process(results, show_idx, display, post_processing)

        return results

    def stream_video(self,
                     filename   = None,
                     url    = None,
                     detector   = 'east',
                     max_time   = None,
                     
                     method = 'beam',
                     dezoom_factor  = 1.1,
                     
                     save   = True,
                     show   = True,
                     save_frames    = True,
                     directory  = None,
                     output_file    = 'stream-{}.mp4',
                     output_shape   = None,
                     
                     callback   = None,
                     buffer_size    = 1,
                     wait_factor    = 0.,
                     duplicate_tolerance    = 2,
                     duplicate_wait_time    = 0.2,
                     
                     threshold  = -0.2,
                     region     = [0.25, 0.15, 0.6, 0.95],
                     n_repeat   = 2,
                     filters    = None,
                     ** kwargs
                    ):
        ####################
        #    OCR method    #
        ####################
        
        _state  = {'prev_output' : None, 'n_skipped' : 0, 'n_duplicates' : 0}
        emitted_texts, last_emitted = set(), []
        
        def perform_ocr(detected, image, infos):
            if len(infos.get('boxes', {}).get('width', ())) == 0: return
            
            is_duplicate = False
            if _state['prev_output'] is not None:
                dice = dice_coeff(infos['output'], _state['prev_output'])
                if dice >= 0.9:
                    is_duplicate = True
                    
                    _state['n_skipped'] += 1
                    _state['n_duplicates'] += 1
                    if not filename and _state['n_duplicates'] % 2 == 0:
                        time.sleep(duplicate_wait_time)
                    
                    if _state['n_duplicates'] > duplicate_tolerance:
                        return
            
            res = self.predict(
                {** infos, 'image' : image},

                method = method,
                num_sentences     = 1,
                logits_filter     = logits_filter,
                dezoom_factor     = dezoom_factor,

                save       = False,
                display    = False,
                combine    = True,
                box_filter = filters,
                ** kwargs
            )[0][-1]

            is_empty    = len(res.get('ocr', [])) == 0
            if not is_empty:
                res['ocr']  = [
                    ocr_res for ocr_res in res.get('ocr', [])
                    if _filter_results(
                        ocr_res, emitted_texts, last_emitted, threshold = threshold, ** kwargs
                    )
                ]

            if len(res.get('ocr', [])) == 0:
                if not is_duplicate:
                    _state['n_duplicate'] = 0
                    _state['n_skipped'] += 1
                if not is_empty:
                    logger.info('[SKIP #{}] All results have been filtered !'.format(
                        infos['frame_index']
                    ))
                return

            if not is_duplicate:
                if _state['n_skipped']:
                    logger.info('{} frames skipped ({} duplicates)'.format(
                        _state['n_skipped'], _state['n_duplicates']
                    ))
                _state.update({
                    'n_skipped' : 0, 'n_duplicates' : 0, 'prev_output' : infos['output']
                })

            logger.info('\nFrame #{}'.format(infos['frame_index']))
            if show: plot(detected)
            for ocr_result in res['ocr']:
                last_emitted.append(ocr_result['text'])
                emitted_texts.add(ocr_result['text'])

                logger.info('Box  : {}\nScores : {}\nText   : {}'.format(
                    np.around(convert_box_format(
                        ocr_result['box'], BoxFormat.XYWH, box_mode = BoxFormat.CORNERS
                    ), decimals = 3),
                    np.around(ocr_result['scores'], decimals = 3),
                    ocr_result['text']
                ))

                if callback: callback(ocr_result['text'])

                if show:
                    show_boxes(image, ocr_result['box'], box_mode = ocr_result['box_mode'])
                    if len(ocr_result['rows']) > 1:
                        show_boxes(
                            image,
                            ocr_result['rows'],
                            box_mode = ocr_result['box_mode'],
                            ncols    = len(ocr_result['rows'])
                        )

            if save_frames:
                save_image(image = image, filename = os.path.join(
                    frames_dir, 'frame_{}.jpg'.format(infos['frame_index'])
                ))

            if wait_factor > 0:
                time.sleep(
                    1. + wait_factor * max([len(ocr_res['text']) for ocr_res in res['ocr']])
                )


        ####################
        #  Initialization  #
        ####################
        
        if isinstance(detector, str):
            from models import get_pretrained
            detector = get_pretrained(detector)
        
        if not callable(callback):
            assert hasattr(callback, 'put'), '`callback` must be callable or have a put method !'
            callback = callback.put
        
        if max_time is None: max_time = 600 if not filename else 0
        
        if not save: save_frames = False
        if directory is None: directory = self.stream_dir
        output_file = os.path.join(directory, output_file)
        if '{}' in output_file:
            output_file = output_file.format(len(glob.glob(output_file.replace('{}', '*'))))
        
        basename    = os.path.basename(output_file).split('.')[0]
        frames_dir  = os.path.join(directory, basename) if save_frames else None
        if save_frames: os.makedirs(frames_dir, exist_ok = True)
        
        # Initializes filters
        if filters is None:
            filters = [SizeFilter(min_h = 0.025, min_w = 0.1)]
            if region:
                filters.append(RegionFilter(region = region, mode = 'center'))
            if n_repeat:
                filters.append(RepetitionFilter(n_repeat = n_repeat, filter_memory = False))

        logits_filter = build_masking_filter(indices = self.ukn_token_idx)
        
        cam = filename
        if not filename:
            for _ in range(5):
                cam = HTTPScreenMirror(url)
                ret, frame  = cam.read()
                if not ret:
                    time.sleep(1)
                else:
                    output_shape    = frame.shape[:2]
                    break
            if not ret: raise RuntimeError('Unable to connect to {} !'.format(url))

        if not save: output_file = None
        detector.stream(
            cam_id   = cam,
            show     = False,
            max_time    = max_time,
            buffer_size  = buffer_size,

            output_file  = output_file,
            output_shape = output_shape,

            save      = False,
            verbose   = False,
            display   = False,

            return_output   = True,
            post_processing = perform_ocr,
            ** kwargs
        )
        logger.info('Stream finished and saved to {} !'.format(output_file))
        return output_file

    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_image(),
            ** self.get_config_text(),
            'max_output_length' : self.max_output_length
        })
        return config

@timer
def post_process(results, idx, display, post_processing):
    while idx < len(results) and results[idx] is not None:
        file, infos = results[idx]
        
        if display:
            image = load_image(file).numpy()
            plot(draw_boxes(image.copy(), infos['boxes'], show_text = False))
            for box_infos in infos['ocr']:
                logger.info('Text (score {}) : {}'.format(
                    np.around(box_infos['scores'], decimals = 3), box_infos['text']
                ))
                plot(load_image(
                    image, boxes = box_infos['box'], source = BoxFormat.CORNERS
                ))
        
        if post_processing is not None:
            try:
                post_processing(infos, image = file)
            except Exception as e:
                logger.error('An error occured in the `post_processing` function !\n  {}'.format(e))
        
        idx += 1
        
    return idx

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
    elif text in reject: return False
    

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
