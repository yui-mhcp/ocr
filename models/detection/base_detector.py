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
import time
import shutil
import logging
import numpy as np

from utils import *
from utils.image import *
from utils.keras_utils import ops
from loggers import timer, time_logger
from utils.image import _video_formats, _image_formats

from models.utils import prepare_prediction_results
from models.interfaces.base_model import BaseModel
from models.interfaces.base_image_model import BaseImageModel
from models.interfaces.base_classification_model import BaseClassificationModel

logger      = logging.getLogger(__name__)

class BaseDetector(BaseClassificationModel, BaseImageModel):
    _directories    = {
        ** BaseModel._directories, 'stream_dir' : '{root}/{self.name}/stream'
    }
    
    prepare_input   = BaseImageModel.get_image
    augment_input   = BaseImageModel.augment_image
    process_input   = BaseImageModel.process_image
    
    def decode_output(self, model_output, ** kwargs):   raise NotImplementedError()
    def prepare_output(self, data, ** kwargs):          raise NotImplementedError()
    
    def __init__(self, labels = None, *, obj_threshold  = 0.35, nms_threshold  = 0.2, ** kwargs):
        self._init_image(** kwargs)
        self._init_labels(labels if labels else ['object'], ** kwargs)

        self.obj_threshold  = obj_threshold
        self.nms_threshold  = nms_threshold
        
        super(BaseDetector, self).__init__(** kwargs)

    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_image)
    
    def __str__(self):
        return super().__str__() + self._str_image() + self._str_labels()

    @timer(name = 'inference', log_if_root = False)
    def detect(self, inputs, get_boxes = False, return_output = False, ** kwargs):
        """
            Performs prediction on `image` and returns either the model's output either the boxes (if `get_boxes = True`)
            
            Arguments :
                - inputs    : `Tensor` of rank 3 or 4 (single / batched image(s))
                - get_boxes : bool, whether to decode the model's output or not
                - training  : whether to make prediction in training mode
                - kwargs    : forwarded to `decode_output` if `get_boxes = True`
            Return :
                if `get_boxes == False` :
                    model's output of shape (B, grid_h, grid_w, nb_box, 5 + nb_class)
                else:
                    list of boxes (where boxes is the list of BoundingBox for detected objects)
                
        """
        inputs = ops.convert_to_tensor(inputs, 'float32')
        if ops.rank(inputs) == 3:    inputs = inputs[None]
        
        outputs = self(inputs, ** kwargs)
        if not get_boxes: return outputs
        
        boxes = self.decode_output(outputs, inputs = inputs, ** kwargs)
        return boxes if not return_output else zip(outputs, boxes)
    
    @timer(name = 'drawing')
    def draw_prediction(self, image, boxes, labels = None, as_mask = False, ** kwargs):
        """ Calls `draw_boxes` or `mask_boxes` depending on `as_mask` and returns the result """
        if len(boxes['boxes'] if isinstance(boxes, dict) else boxes) == 0:
            return image
        
        if as_mask:
            return mask_boxes(image, boxes, ** kwargs)
        
        if not labels: labels = self.labels
        kwargs.setdefault('use_label', True)

        return draw_boxes(image, boxes, labels = labels, ** kwargs)
    
    def get_prediction_callbacks(self,
                                 *,

                                 save    = True,
                                 save_empty = False,
                                 save_detected  = None,
                                 save_boxes     = False,
                                 
                                 directory  = None,
                                 raw_img_dir    = None,
                                 detected_dir   = None,
                                 boxes_dir      = None,
                                 
                                 filename   = 'image_{}.jpg',
                                 detected_filename  = '{basename}-detected.jpg',
                                 boxes_filename     = '{basename}-box-{box_index}.jpg',
                                 # Verbosity config
                                 verbose = 1,
                                 display = None,
                                 
                                 post_processing    = None,
                                 
                                 use_multithreading = False,

                                 ** kwargs
                                ):
        """
            Return a list of `utils.callbacks.Callback` instances that handle data saving/display
            
            Arguments :
                - save  : whether to save detection results
                          Set to `True` if `save_boxes` or `save_detected` is True
                - save_empty    : whether to save raw images if no object has been detected
                - save_detected : whether to save the image with detected objects
                - save_boxes    : whether to save boxes as individual images (not supported yet)
                
                - directory : root directory for saving (see below for the complete tree)
                - raw_img_dir   : where to save raw images (default `{directory}/images`)
                - detected_dir  : where to save images with detection (default `{directory}/detected`)
                - boxes_dir     : where to save individual boxes (not supported yet)
                
                - filename  : raw image file format
                - detected_filename : image with detection file format
                - boxes_filename    : individual boxes file format
                
                - display   : whether to display image with detection
                              If `None`, set to `True` if `save == False`
                - verbose   : verbosity level (cumulative, i.e., level 2 includes level 1)
                              - 1 : displays the image with detection
                              - 2 : displays the individual boxes
                              - 3 : logs the boxes position
                                 
                - post_processing   : callback function applied on the results
                                      Takes as input all kwargs returned by `self.predict`
                                      - image   : the raw original image (`ndarray / Tensor`)
                                      - boxes   : the detected objects (`dict`)
                                      * filename    : the image file (`str`)
                                      * detected    : the image with detection (`ndarray`)
                                      * output      : raw model output (`Tensor`)
                                      * frame_index : the frame index in a stream (`int`)
                                      Entries with "*" are conditionned and are not always provided
                
                - use_multithreading    : whether to multi-thread the saving callbacks
                
                - kwargs    : mainly ignored
            Return : (predicted, required_keys, callbacks)
                - predicted : the mapping `{filename : infos}` stored in `{directory}/map.json`
                - required_keys : expected keys to save (see `models.utils.should_predict`)
                - callbacks : the list of `Callback` to be applied on each prediction
        """
        if save_detected or save_boxes: save = True
        if save_detected is None:       save_detected = save
        if display is None: display = not save
        if save is None:    save = not display
        
        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        
        predicted   = {}
        callbacks   = []
        required_keys   = ['boxes']
        if save:
            predicted   = load_json(map_file, {})
            
            required_keys.append('filename')
            if raw_img_dir is None: raw_img_dir = os.path.join(directory, 'images')
            callbacks.append(ImageSaver(
                key = 'filename',
                name    = 'saving raw',
                cond    = lambda boxes, filename = None, ** _: (not isinstance(filename, str)) and (save_empty or len(boxes['boxes'] if isinstance(boxes, dict) else boxes)),
                data_key    = 'image',
                file_format = os.path.join(raw_img_dir, filename),
                index_key   = 'frame_index',
                use_multithreading  = use_multithreading
            ))
        
            if save_detected:
                if detected_dir is None: detected_dir = os.path.join(directory, 'detected')
                required_keys.append('detected')
                callbacks.append(ImageSaver(
                    key = 'detected',
                    name = 'saving detected',
                    cond    = lambda boxes, ** _: save_empty or len(
                        boxes['boxes'] if isinstance(boxes, dict) else boxes),
                    data_key    = 'detected',
                    file_format = os.path.join(detected_dir, detected_filename),
                    initializers    = {'detected' : partial(self.draw_prediction, ** kwargs)},
                    use_multithreading  = use_multithreading
                ))

            if save_boxes:
                raise NotImplementedError()
        
            callbacks.append(JSonSaver(
                data    = predicted,
                filename    = map_file,
                force_keys  = {'boxes'},
                primary_key = 'filename',
                use_multithreading = use_multithreading
            ))
        
        if display:
            if verbose > 1:
                callbacks.append(BoxesDisplayer(
                    max_display = display,
                    print_boxes = verbose == 3,
                    labels  = kwargs.get('labels', self.labels)
                ))
            
            callbacks.append(ImageDisplayer(
                data_key    = 'detected',
                max_display = display,
                initializers    = {'detected' : partial(self.draw_prediction, ** kwargs)},
            ))
        
        if post_processing is not None:
            callbacks.append(FunctionCallback(post_processing))
        
        return predicted, required_keys, callbacks
    
    @timer
    def stream(self,
               stream,
               *,
               
               save = None,
               filename = 'frame-{}.jpg',
               directory    = None,
               stream_name  = None,
               save_detected    = False,

               show = True,
               display  = False,
               
               save_stream  = None,
               output_file  = None,
               
               save_transformed = None,
               transformed_file = None,
               
               use_multithreading   = True,
               
               ** kwargs
              ):
        """
            Perform live object detection on `stream` (camera ID, video file or any other valid camera)
            
            Arguments :
                - stream    : the stream camera ID / file / ..., see the `cam_id` argument of `utils.image.stream_camera`
                
                - save  : whether to save the detection result or not
                - filename  : the raw frame file format (formatted with the frame index)
                - directory : where to save the stream results (default to `self.stream_dir`)
                - stream_name   : the specific stream name, saved in `{directory/{stream_name}/...`
                - save_detected : whether to save raw frame with detected objects
                
                - show  : whether to show the stream with `cv2.imshow`
                - display   : whether to display the frames with detection with `plot`
                
                - save_stream   : whether to save raw stream or not
                - output_file   : the filename of the raw stream (`{stream_dir}/{output_file}`)
                
                - save_transformed  : whether to save the stream with bounding boxes or not
                - transformed_file  : where to save the stream with bounding boxes
               
               - use_multithreading : whether to multi-thread everything (recommanded !)
               
               - kwargs : forwarded to `self.predict` and `stream_camera`
            
            By default, the function simply displays (with `cv2.imshow`) the transformed stream (`show = True`)
            If `save == True` or `show == False` without any other configuration, it will save the raw stream (if `stream` is not a video file), and raw frames + detection information
            
            The default `stream_name` is the stream basename (if video file) or `stream-{}`
            The default `output_file` is `stream.mp4`
            The default `transformed_file` is `{basename(output_file)}-transformed.mp4`
        """
        if save is None: save = not show
        if save_stream is None:
            save_stream = (save or output_file is not None) and not isinstance(stream, str)
        if save_transformed is None:
            save_transformed = transformed_file is not None
        
        if save_stream:
            if output_file is None: output_file = 'stream.mp4'
        
        if save_transformed and transformed_file is None:
            if output_file:                 basename = output_file
            elif isinstance(stream, str):   basename = os.path.basename(stream)
            else:                           basename = 'stream.mp4'
            basename, ext = os.path.splitext(basename)
            transformed_file = '{}-transformed{}'.format(basename, ext)
        
        if directory is None: directory = self.stream_dir
        if not stream_name:
            stream_name = 'stream-{}' if not isinstance(stream, str) else os.path.basename(stream).split('.')[0]
        
        stream_dir = os.path.join(directory, stream_name)
        if contains_index_format(stream_dir):
            stream_dir = format_path_index(stream_dir)
        
        kwargs.update({
            'cam_id'    : stream,
            
            'show'  : show,
            'display'   : display,
            
            'save'  : save,
            'save_detected' : save_detected,
            
            'filename'  : filename,
            'directory' : stream_dir,
            'raw_img_dir'   : os.path.join(stream_dir, 'frames'),
            'detected_dir'  : os.path.join(stream_dir, 'detected'),
            'boxes_dir'     : os.path.join(stream_dir, 'boxes'),
            
            'use_multithreading'    : use_multithreading
        })
        if save_stream:
            kwargs['output_file'] = os.path.join(stream_dir, output_file)
        if save_transformed:
            kwargs['transformed_file'] = os.path.join(stream_dir, transformed_file)
        
        # for tensorflow-graph compilation (the 1st call is much slower than the next ones)
        input_size = [s if s is not None else 128 for s in self.input_size]
        self.detect(ops.zeros(input_size, dtype = 'float32'))

        predicted, required_keys, callbacks = self.get_prediction_callbacks(** kwargs)
        
        def detection(img):
            return self.predict(
                img,
                
                force_draw  = show or save_transformed,
                
                predicted   = predicted,
                _callbacks  = callbacks,
                required_keys   = required_keys,
                
                ** kwargs
            )[0][1]
        
        stream_camera(
            transform_fn     = detection,
            add_copy    = True,
            add_index   = True,
            name        = 'frame transform',
            ** kwargs
        )
        
        for callback in callbacks: callback.join()
        
        return stream_dir
    
    @timer
    def predict(self,
                images,
                batch_size = 16,
                return_output   = False,
                *,
                
                overwrite   = False,
                force_draw  = False,
                
                _callbacks  = None,
                predicted   = None,
                required_keys   = None,
                
                ** kwargs
               ):
        ####################
        #  Initialization  #
        ####################

        now = time.time()
        with time_logger.timer('initialization'):
            join_callbacks = _callbacks is None
            if _callbacks is None:
                predicted, required_keys, _callbacks = self.get_prediction_callbacks(** kwargs)
        
            results, inputs, indexes, files, duplicates, filtered = prepare_prediction_results(
                images,
                predicted,
                
                rank    = 3,
                primary_key = 'filename',
                expand_files    = True,
                normalize_entry = path_to_unix,
                
                overwrite   = overwrite,
                required_keys   = required_keys,
                
                filters = {
                    'video' : lambda f: f.endswith(_video_formats),
                    'invalid'   : lambda f: not f.endswith(_image_formats)
                }
            )
            
            videos = filtered.pop('video', [])
            if 'invalid' in filtered:
                logger.info('Skip files with unsupported extensions : {}'.format(
                    filtered['invalid']
                ))
        
        ####################
        #  Inference loop  #
        ####################
        
        show_idx = apply_callbacks(results, 0, _callbacks)
        
        for start in range(0, len(inputs), batch_size):
            with time_logger.timer('batch processing'):
                batch_images    = inputs[start : start + batch_size]
                if isinstance(batch_images, list):
                    batch_images = [self.get_image_data(inp) for inp in batch_images]

                    batch   = stack_batch([
                        self.get_input(image) for image in batch_images
                    ], pad_value = 0., dtype = 'float32', maybe_pad = self.has_variable_input_size)
                else:
                    batch = self.get_input(batch_images)

            # Computes detection + output decoding
            boxes   = self.detect(
                batch, get_boxes = True, return_output = return_output, ** kwargs
            )

            for idx, file, data, image, box in zip(
                indexes[start : start + batch_size],
                files[start : start + batch_size],
                inputs[start : start + batch_size],
                batch_images,
                boxes):
                
                output, box = (None, box) if not return_output else box

                infos = {} if not isinstance(data, dict) else data.copy()
                infos.update({'image' : image, 'boxes' : box, 'timestamp' : now})
                
                if force_draw:
                    if 'image_copy' in infos:
                        detected = infos['image_copy']
                    else:
                        detected = infos['image']
                        if isinstance(detected, np.ndarray): detected = detected.copy()
                    infos['detected'] = convert_to_uint8(self.draw_prediction(
                        detected, box, ** kwargs
                    ))
                
                if return_output: infos['output'] = output
                # Sets result at the (multiple) index(es)
                if file:
                    for duplicate_idx in duplicates[file]:
                        results[duplicate_idx] = (predicted.get(file, {}), infos)
                else:
                    results[idx] = ({}, infos)

            show_idx = apply_callbacks(results, show_idx, _callbacks)

        if join_callbacks:
            for callback in _callbacks: callback.join()
        
        return [(
            stored['filename'] if 'filename' in stored else output.get('filename', output['image']),
            output['detected'] if output and 'detected' in output else stored.get('detected', None),
            output if output else stored
        ) for (stored, output) in results]

    def get_config(self):
        config = super(BaseDetector, self).get_config()
        config.update({
            ** self.get_config_image(),
            ** self.get_config_labels(),
            'obj_threshold' : self.obj_threshold,
            'nms_threshold' : self.nms_threshold
        })
        
        return config

