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
import numpy as np

from functools import partial

from utils import Stream, contains_index_format, format_path_index
from loggers import timer
from utils.keras import ops
from utils.callbacks import *
from utils.image import stream_camera, draw_boxes
from ..interfaces.base_model import BaseModel
from ..interfaces.base_image_model import BaseImageModel
from ..interfaces.base_classification_model import BaseClassificationModel

class BaseDetector(BaseClassificationModel, BaseImageModel):
    _directories    = {
        ** BaseModel._directories, 'stream_dir' : '{root}/{self.name}/stream'
    }
    
    input_signature = BaseImageModel.image_signature
    
    prepare_input   = BaseImageModel.get_image
    process_input   = BaseImageModel.process_image
    
    def decode_output(self, model_output, ** kwargs):   raise NotImplementedError()
    def prepare_output(self, data, ** kwargs):          raise NotImplementedError()
    
    def __init__(self, labels = None, *, obj_threshold  = 0.35, nms_threshold  = 0.2, ** kwargs):
        self._init_image(** kwargs)
        self._init_labels(labels if labels else ['object'], ** kwargs)

        self.obj_threshold  = obj_threshold
        self.nms_threshold  = nms_threshold
        
        super(BaseDetector, self).__init__(** kwargs)
    
    def __str__(self):
        return super().__str__() + self._str_image() + self._str_labels()

    @timer(name = 'inference')
    def infer(self, image, *, predicted = None, overwrite = False, ** kwargs):
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
        if predicted and not overwrite and isinstance(image, str) and image in predicted:
            return predicted[image]

        infos = image.copy() if isinstance(image, dict) else {}
        
        filename, image = self.get_image_data(image)
        inputs  = self.get_input(image)
        if ops.rank(inputs) == 3:    inputs = inputs[None]
        
        output = self.compiled_infer(inputs, ** kwargs)[0]
        
        boxes = self.decode_output(output, inputs = inputs, ** kwargs)
        return {
            ** infos, 'filename' : filename, 'boxes' : boxes, 'image' : image, 'output' : output
        }
    
    @timer(name = 'drawing')
    def draw_prediction(self, *, boxes, image = None, filename = None, as_mask = False, ** kwargs):
        """ Calls `draw_boxes` or `mask_boxes` depending on `as_mask` and returns the result """
        if image is None: image = filename
        
        if len(boxes['boxes'] if isinstance(boxes, dict) else boxes) == 0:
            return image
        
        if as_mask:
            return mask_boxes(image, boxes, ** kwargs)
        
        
        kwargs.setdefault('labels', self.labels)
        kwargs.setdefault('use_label', True)

        return draw_boxes(image, boxes, ** kwargs)
    
    def get_inference_callbacks(self,
                                 *,

                                 save    = True,
                                 save_empty = False,
                                 save_detected  = None,
                                 
                                 directory  = None,
                                 raw_img_dir    = None,
                                 detected_dir   = None,
                                 
                                 filename   = 'image_{}.jpg',
                                 detected_filename  = '{basename}-detected.jpg',
                                 # Verbosity config
                                 verbose = 1,
                                 display = None,
                                 
                                 post_processing    = None,
                                 
                                 save_in_parallel   = False,

                                 ** kwargs
                                ):
        """
            Return a list of `utils.callbacks.Callback` instances that handle data saving/display
            
            Arguments :
                - save  : whether to save detection results
                          Set to `True` if `save_boxes` or `save_detected` is True
                - save_empty    : whether to save raw images if no object has been detected
                - save_detected : whether to save the image with detected objects
                
                - directory : root directory for saving (see below for the complete tree)
                - raw_img_dir   : where to save raw images (default `{directory}/images`)
                - detected_dir  : where to save images with detection (default `{directory}/detected`)
                
                - filename  : raw image file format
                - detected_filename : image with detection file format
                
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
        if display is None:         display = not save and not save_detected
        if save_detected is None:   save_detected = save
        elif save_detected:         save = True
        elif save is None:          save = not display
        
        predicted, callbacks = {}, []
        if save:
            if directory is None: directory = self.pred_dir
            map_file    = os.path.join(directory, 'map.json')

            predicted   = load_json(map_file, default = {})
            
            if raw_img_dir is None: raw_img_dir = os.path.join(directory, 'images')
            callbacks.append(ImageSaver(
                cond    = lambda boxes, filename = None, ** _: (not isinstance(filename, str)) and (
                    save_empty or len(boxes['boxes'] if isinstance(boxes, dict) else boxes)
                ),
                index_key   = 'frame_index',
                file_format = os.path.join(raw_img_dir, filename),
                save_in_parallel    = save_in_parallel
            ))
        
            if save_detected:
                if detected_dir is None: detected_dir = os.path.join(directory, 'detected')
                callbacks.append(ImageSaver(
                    key = 'detected',
                    name = 'saving detected',
                    cond    = lambda boxes, ** _: save_empty or len(
                        boxes['boxes'] if isinstance(boxes, dict) else boxes
                    ),
                    initializer = {
                        'detected'  : partial(self.draw_prediction, ** kwargs)
                    },
                    data_key    = 'detected',
                    file_format = os.path.join(detected_dir, detected_filename),
                    save_in_parallel  = save_in_parallel
                ))
        
            callbacks.append(JSONSaver(
                data    = predicted,
                filename    = map_file,
                primary_key = 'filename',
                save_in_parallel = save_in_parallel
            ))
        
        if display:
            callbacks.append(ImageDisplayer(
                data_key    = 'detected',
                max_display = display,
                initializer = {
                    'detected'  : partial(self.draw_prediction, ** kwargs)
                }
            ))
            
            if verbose > 1:
                callbacks.append(BoxesDisplayer(
                    max_display = display,
                    print_boxes = verbose == 3,
                    labels  = kwargs.get('labels', self.labels)
                ))

        
        if post_processing is not None:
            if not isinstance(post_processing, list): post_processing = [post_processing]
            for fn in post_processing:
                if callable(fn):
                    callbacks.append(FunctionCallback(fn))
                elif hasattr(fn, 'put'):
                    callbacks.append(QueueCallback(fn))
        
        return predicted, callbacks
    
    @timer
    def predict(self,
                inputs,
                *,
                
                predicted = None,
                callbacks = None,
                
                return_results  = True,
                return_output   = None,
                
                ** kwargs
               ):
        if (isinstance(inputs, (str, dict))) or (ops.is_array(inputs) and len(inputs.shape) == 3):
            inputs = [inputs]
        
        join_callbacks = predicted is None
        if predicted is None:
            predicted, callbacks = self.get_inference_callbacks(** kwargs)
        
        if return_output is None:
            return_output = not any(isinstance(callback, JSONSaver) for callback in callbacks)
        
        results = []
        for image, output in Stream(self.infer, inputs, predicted = predicted, ** kwargs).items():
            infos = {k : v for k, v in output.items() if 'image' not in k and k != 'output'}
            
            entry = apply_callbacks(callbacks, infos, output, save = 'image' in output)
            if entry is None: entry = output.get('filename', None)
            
            if return_results:
                results.append(
                    output if return_output or entry is None else predicted.get(entry, None)
                )
        
        if join_callbacks:
            for callback in callbacks: callback.join()
        
        return results

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
               
               save_in_parallel   = True,
               
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
        
        if save_stream and output_file is None:
            output_file = 'stream.mp4'
        
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
            
            'save_in_parallel'    : save_in_parallel
        })
        if save_stream:
            kwargs['output_file'] = os.path.join(stream_dir, output_file)
        if save_transformed:
            kwargs['transformed_file'] = os.path.join(stream_dir, transformed_file)
        
        # for tensorflow-graph compilation (the 1st call is much slower than the next ones)
        input_size = [s if s is not None else 128 for s in self.input_size]
        self.infer(ops.zeros(input_size, dtype = 'float32'))

        predicted, callbacks = self.get_inference_callbacks(** kwargs)
        
        def detection(img):
            output = self.infer(
                img, predicted = predicted, ** kwargs
            )
            infos = {k : v for k, v in output.items() if 'image' not in k and k != 'output'}
            
            apply_callbacks(callbacks, infos, output, save = True)

            if not show and not save_transformed:
                return None
            elif 'detected' in output: return output['detected']
            elif 'image_copy' in output: output['image'] = output['image_copy']
            elif isinstance(output['image'], np.ndarray): output['image'] = output['image'].copy()
            
            return self.draw_prediction(** output)
        
        try:
            stream_camera(
                transform_fn = detection, add_copy = True, add_index = True, ** kwargs
            )
        finally:
            for callback in callbacks: callback.join()
        
        return stream_dir

    def get_config(self):
        return {
            ** super().get_config(),
            ** self.get_config_image(),
            ** self.get_config_labels(),
            
            'obj_threshold' : self.obj_threshold,
            'nms_threshold' : self.nms_threshold
        }

