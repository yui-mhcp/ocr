
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils import plot, load_json, dump_json, pad_batch
from utils.image import save_image, load_image, plot
from utils.image.box_utils import BoxFormat, convert_box_format, combine_boxes, draw_boxes
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_image_model import BaseImageModel

logger  = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

DEFAULT_MAX_TEXT_LENGTH = 32

class BaseOCR(BaseImageModel, BaseTextModel):
    output_signature    = BaseTextModel.text_signature

    get_input   = BaseImageModel.get_image_with_box
    get_output  = BaseTextModel.tf_encode_text
    augment_input   = BaseImageModel.augment_image
    preprocess_input    = BaseImageModel.preprocess_image
    
    def __init__(self,
                 lang   = 'multi',
                 input_size = None,
                 
                 max_output_length = DEFAULT_MAX_TEXT_LENGTH,
                 ** kwargs
                ):
        self._init_text(lang = lang, ** kwargs)
        self._init_image(input_size = input_size, ** kwargs)
        
        self.max_output_length  = max_output_length

        super().__init__(** kwargs)
        
        if hasattr(self.model, '_build'): self.model._build()
        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)
    
    @property
    def input_signature(self):
        if self.is_encoder_decoder:
            return (self.image_signature, self.text_signature)
        return self.image_signature
    
    @property
    def default_image_augmentation(self):
        augment_methods = ['random_rotate']
        if self.has_variable_input_size: augment_methods += ['random_resize']
        return super().default_image_augmentation + augment_methods

    @property
    def training_hparams(self):
        image_hparams = self.training_hparams_image
        return super().training_hparams(
            ** image_hparams,
            ** self.get_image_augmentation_config(image_hparams),
            max_output_length   = None
        )

    def __str__(self):
        return super().__str__() + self._str_image() + self._str_text()
    
    @timer(name = 'inference')
    def infer(self, * args, ** kwargs):
        if hasattr(self.model, 'infer'):
            kwargs.setdefault('max_length', self.max_output_length)
        
            return self.model.infer(* args, ** kwargs)
        return self(* args, ** kwargs)
    
    def encode_data(self, data):
        image   = self.get_input(data)
        tokens  = self.get_output(data, default_key = 'label')

        if self.is_encoder_decoder:
            return (image, tokens[:-1]), tokens[1:]
        return image, tokens
    
    def filter_input(self, inputs):
        image = inputs[0] if self.is_encoder_decoder else inputs
        return tf.reduce_all(tf.shape(image) > 0)
    
    def filter_output(self, output):
        return tf.logical_and(
            tf.shape(output)[0] > 0, tf.shape(output)[-1] <= self.max_output_length
        )

    def filter_data(self, inputs, outputs):
        return tf.logical_and(
            self.filter_input(inputs), self.filter_output(outputs)
        )
    
    def augment_data(self, inputs, outputs):
        if self.is_encoder_decoder:
            (image, tokens) = inputs
            return (self.augment_input(image), tokens), outputs
        return self.augment_input(inputs), outputs
    
    def preprocess_data(self, inputs, outputs):
        if self.is_encoder_decoder:
            (image, tokens) = inputs
            return (self.preprocess_input(image), tokens), outputs
        return self.preprocess_input(inputs), outputs
    
    def get_dataset_config(self, ** kwargs):
        inp_padding = (0., self.blank_token_idx) if self.is_encoder_decoder else 0.
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'  : True,
            'pad_kwargs'    : {
                'padding_values'    : (
                    inp_padding, self.blank_token_idx
                )
            }
        })
        
        return super().get_dataset_config(** kwargs)
    
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
                combine_threshold = 0.0125,
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
                    - np.ndarray / tf.Tensor    : the raw image
                    
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
        
        @timer
        def post_process(idx):
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
                            image, bbox = box_infos['box'], box_mode = BoxFormat.CORNERS
                        ))
                
                if post_processing is not None:
                    post_processing(infos, image = file)
                
                idx += 1
            
            return idx
        
        def save_raw_image(image):
            os.makedirs(raw_img_dir, exist_ok = True)
            filename = os.path.join(raw_img_dir, 'image_{}.png'.format(len(os.listdir(raw_img_dir))))
            save_image(image = image, filename = filename)
            return filename
        
        def should_predict(image):
            if isinstance(image, (dict, pd.Series)) and 'filename' in image: image = image['filename']
            if isinstance(image, str) and image in predicted:
                if not overwrite or (timestamp != -1 and timestamp <= predicted[image].get('timestamp', -1)):
                    return False
            return True
        
        def get_filename(image):
            if isinstance(image, (dict, pd.Series)):
                image = image['filename' if 'filename' in image else 'image']
            if isinstance(image, (np.ndarray, tf.Tensor)):
                if not save_raw_image or len(image.shape) != 3: return None
                image = save_raw_image(image)
            elif not isinstance(image, str):
                raise ValueError('Unknown image type ({}) : {}'.format(type(image), image))
            return image.replace(os.path.sep, '/')
        
        now = time.time()
        
        if detector is not None:
            if isinstance(detector, str):
                from models import get_pretrained
                detector = get_pretrained(detector)
            
            detected = detector.predict(
                images, save = False, display = False, ** detector_kwargs
            )
            images = [detect[-1] for detect in detected]

        if display is None: display = not save
        
        if isinstance(images, pd.DataFrame): images = images.to_dict('records')
        if not isinstance(images, (list, tuple, np.ndarray, tf.Tensor)): images = [images]
        elif isinstance(images, (np.ndarray, tf.Tensor)) and len(images.shape) == 3:
            images = tf.expand_dims(images, axis = 0)
        
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
        requested   = [(get_filename(img), img) for img in images]
        
        inputs  = []
        for i, (file, img) in enumerate(requested):
            if not should_predict(file):
                results[i] = (file, predicted[file])
            else:
                if isinstance(file, str):
                    duplicatas.setdefault(file, []).append(i)
                    if len(duplicatas[file]) > 1: continue
                
                inputs.append((i, file, img))
        
        ####################
        #  Inference loop  #
        ####################
        
        show_idx = post_process(0)
        
        if len(inputs) > 0:
            for idx, file, data in inputs:
                infos = data.copy() if isinstance(data, dict) else {}
                infos['timestamp'] = now
                
                image = load_image(file)
                if 'boxes' in data:
                    if combine:
                        boxes, indices, rows = combine_boxes(
                            data['boxes'],
                            image = image,
                            box_mode = data.get('box_mode', BoxFormat.DEFAULT),
                            threshold = combine_threshold
                        )
                    else:
                        boxes = convert_box_format(
                            data['boxes'], BoxFormat.CORNERS, image = file,
                        )
                        if len(boxes.shape) == 1: boxes = np.expand_dims(boxes, axis = 0)
                        indices = list(range(len(boxes)))
                        rows    = [boxes[i : i + 1] for i in range(len(boxes))]
                    
                    if box_filter is not None:
                        keep_indexes = box_filter(boxes = boxes, indices = indices, rows = rows)
                        boxes, rows  = boxes[keep_indexes], [rows[index] for index in keep_indexes]
                    
                    for i, rows_i in enumerate(rows):
                        time_logger.start_timer('OCR')
                        outputs = []
                        for row in rows_i:
                            inp = self.get_input(image, bbox = row, box_mode = BoxFormat.CORNERS)

                            inp = self.preprocess_input(tf.expand_dims(inp, axis = 0))
                            out = self.infer(inp)[0]
                            outputs.append(out)

                        lengths = tf.cast([len(out) for out in outputs], tf.int32)
                        outputs = pad_batch(outputs, pad_value = 0., dtype = np.float32)

                        time_logger.start_timer('CTC Beam search')
                        texts, scores = self.decode_output(outputs, lengths = lengths, method = 'beam', return_scores = True)
                        time_logger.stop_timer('CTC Beam search')

                        time_logger.stop_timer('OCR')
                        
                        if threshold < 0. and np.any(scores < threshold):
                            continue
                        
                        box_infos = {
                            'box' : boxes[i], 'box_mode' : BoxFormat.CORNERS, 'rows' : rows_i, 'text' : '\n'.join(texts), 'scores' : scores
                        }
                        infos.setdefault('ocr', []).append(box_infos)
                        
                        if box_processing is not None:
                            box_processing(box_infos, image = image)
                else:
                    raise NotImplementedError()
                
                if file is None: file = data
                
                if isinstance(file, str):
                    infos['filename']   = file
                    should_save     = True
                    predicted[file] = infos
                    
                    for duplicate_idx in duplicatas[file]:
                        results[duplicate_idx] = (file, infos)
                    
                    if save:
                        dump_json(map_file, predicted, indent = 4)
                else:
                    results[idx] = (file if file else img, infos)
                
                show_idx = post_process(show_idx)
        
        return results

    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_image(),
            ** self.get_config_text(),
            'max_output_length' : self.max_output_length
        })
        return config
