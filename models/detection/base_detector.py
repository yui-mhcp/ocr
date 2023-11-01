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
import glob
import time
import shutil
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils.thread_utils import Consumer
from custom_architectures import get_architecture
from models.interfaces.base_image_model import BaseImageModel
from utils import load_json, dump_json, normalize_filename, should_predict, get_filename, plot, pad_batch
from utils.image import _video_formats, _image_formats, load_image, save_image, stream_camera, BASE_COLORS, get_video_infos
from utils.image.box_utils import *

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')


class BaseDetector(BaseImageModel):
    get_input   = BaseImageModel.get_image
    augment_input   = BaseImageModel.augment_image
    preprocess_input    = BaseImageModel.preprocess_image
    
    def decode_output(self, model_output, ** kwargs):
        raise NotImplementedError()
        
    def get_output(self, data, ** kwargs):
        raise NotImplementedError()
    
    def __init__(self,
                 labels = None,
                 nb_class   = None,
                 input_size = None,
                 
                 obj_threshold  = 0.35,
                 nms_threshold  = 0.2,

                 ** kwargs
                ):
        self._init_image(input_size = input_size, ** kwargs)

        if labels is None: labels = ['object']
        self.labels   = list(labels) if not isinstance(labels, str) else [labels]
        self.nb_class = max(len(self.labels), nb_class if nb_class is not None else 1)
        if self.nb_class > len(self.labels):
            self.labels += [''] * (self.nb_class - len(self.labels))

        self.obj_threshold  = obj_threshold
        self.nms_threshold  = nms_threshold
        
        self.label_to_idx   = {label : i for i, label in enumerate(self.labels)}
        
        super(BaseDetector, self).__init__(** kwargs)
    
    @property
    def stream_dir(self):
        return os.path.join(self.folder, "stream")
    
    @property
    def use_labels(self):
        return False if self.nb_class == 1 else True

    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_image)
    
    def __str__(self):
        des = super().__str__()
        des += self._str_image()
        des += "- Labels (n = {}) : {}\n".format(len(self.labels), self.labels)
        return des

    @timer(name = 'inference', log_if_root = False)
    def detect(self, image, get_boxes = False, training = False, return_output = False, ** kwargs):
        """
            Performs prediction on `image` and returns either the model's output either the boxes (if `get_boxes = True`)
            
            Arguments :
                - image : tf.Tensor of rank 3 or 4 (single / batched image(s))
                - get_boxes : bool, whether to decode the model's output or not
                - training  : whether to make prediction in training mode
                - kwargs    : forwarded to `decode_output` if `get_boxes = True`
            Return :
                if `get_boxes == False` :
                    model's output of shape (B, grid_h, grid_w, nb_box, 5 + nb_class)
                else:
                    list of boxes (where boxes is the list of BoundingBox for detected objects)
                
        """
        if not isinstance(image, tf.Tensor): image = tf.cast(image, tf.float32)
        if len(image.shape) == 3:            image = tf.expand_dims(image, axis = 0)
        
        outputs = self(image, training = training)
        
        if not get_boxes: return outputs
        
        boxes = self.decode_output(outputs, ** kwargs)
        return boxes if not return_output else zip(outputs, boxes)
    
    def encode_data(self, data):
        return self.get_input(data), self.get_output(data)
    
    def augment_data(self, image, output):
        return self.augment_input(image), output
    
    def preprocess_data(self, image, output):
        return self.preprocess_input(image), output
    
    @timer(name = 'drawing')
    def draw_prediction(self, image, boxes, labels = None, as_mask = False, ** kwargs):
        """ Calls `draw_boxes` or `mask_boxes` depending on `as_mask` and returns the result """
        if len(boxes) == 0: return image
        
        if as_mask:
            return mask_boxes(image, boxes, ** kwargs)
        
        kwargs.setdefault('color', BASE_COLORS)
        kwargs.setdefault('use_label', True)
        kwargs.setdefault('labels', labels if labels is not None else self.labels)

        return draw_boxes(image, boxes, ** kwargs)
    
    @timer
    def show_result(self, image, detected, boxes, verbose = 1, ** kwargs):
        if not verbose: return
        if verbose > 1:
            if verbose == 3:
                logger.info("{} boxes found :\n{}".format(
                    len(boxes), '\n'.join(str(b) for b in boxes)
                ))

            show_boxes(image, boxes, labels = kwargs.pop('labels', self.labels), ** kwargs)
        # Show original image with drawed boxes
        plot(detected, title = '{} object(s) detected'.format(len(boxes)), ** kwargs)

    @timer
    def save_image(self, image, directory = None, filename = 'image_{}.jpg', ** kwargs):
        """ Saves `image` to `directory` (default `{self.pred_dir}/images/`) and returns its path """
        if isinstance(image, str): return image
        if directory is None: directory = os.path.join(self.pred_dir, 'images')
        
        if not filename.startswith(directory): filename = os.path.join(directory, filename)
        if '{}' in filename:
            filename = filename.format(len(glob.glob(filename.replace('{}', '*'))))
        
        save_image(filename = filename, image = image)
        
        return filename
    
    @timer
    def save_detected(self,
                      detected  = None,
                      image = None,
                      boxes = None,
                      dezoom_factor = 1.,
                      
                      directory = None,
                      filename  = '{}_detected.jpg',
                      original_filename = None,
                      
                      ** kwargs
                     ):
        """
            Saves the image with drawn detection in `directory` (default `{self.pred_dir}/detected`)
            
            Arguments :
                - detected  : the image with already drawn predictions
                - image / boxes / dezoom_factor : information required to produce `detected` if not provided. This information is required if `detected` is not provided, ignored otherwise
                
                - directory : where to save the image
                - filename  : the filename format of the image
                - original_filename : the filename of the original image
                
                - kwargs    : forwarded to `self.draw_prediction`
            Return :
                - filename  : the filename of the saved image
        """
        if detected is None:
            assert boxes is not None and image is not None, 'You must provide either `detected` either `image + boxes`'
            
            image_h, image_w = get_image_size(image)
            normalized_boxes = [get_box_pos(
                box, image_h = image_h, image_w = image_w, labels = labels,
                extended = True, dezoom_factor = dezoom_factor,
                normalize_mode = NORMALIZE_WH
            )[:5] for box in boxes]
            
            detected = self.draw_prediction(image, boxes, ** kwargs)

        if directory is None: directory = os.path.join(directory, 'detected')
        if not filename.startswith(directory): filename = os.path.join(directory, filename)
        
        if '{}' in filename:
            if isinstance(original_filename, str):
                img_name    = os.path.splitext(os.path.basename(original_filename))[0]
            else:
                img_name    = 'image_{}'.format(len(glob.glob(filename.replace('{}', '*'))))
            
            filename = filename.format(img_name)
        
        save_image(filename = filename, image = detected)
        
        return filename

    def _get_saving_functions(self, max_workers = -1, ** kwargs):
        fake_fn = lambda * args, ** kwargs: None
        
        saving_functions    = [
            kwargs.get('show_result_fn',  None),
            kwargs.get('save_json_fn',    dump_json),
            kwargs.get('save_image_fn',   self.save_image),
            kwargs.get('save_detected_fn', self.save_detected),
            kwargs.get('save_boxes_fn',   extract_boxes)
        ]
        
        if max_workers >= 0:
            for i in range(len(saving_functions)):
                if saving_functions[i] is not None:
                    saving_functions[i] = Consumer(
                        saving_functions[i], max_workers = max_workers if i > 1 else 0
                    )
                    saving_functions[i].start()
        
        return [
            fn if fn is not None else fake_fn for fn in saving_functions
        ]
    
    @timer
    def stream(self, stream_name = 'stream_{}', save = False, max_workers = 0, ** kwargs):
        """
            Performs streaming either on camera (default) or on filename (by specifying the `cam_id` kwarg)
        """
        kwargs.update({'save' : save})
        if stream_name:
            directory = kwargs.get('directory', self.stream_dir)
            if '{}' in stream_name:
                stream_name = stream_name.format(len(
                    glob.glob(os.path.join(directory, stream_name.replace('{}', '*')))
                ))
            
            kwargs.update({
                'directory' : os.path.join(directory, stream_name),
                'raw_img_dir'   : os.path.join(directory, stream_name, 'frames'),
                'detected_dir'  : os.path.join(directory, stream_name, 'detected'),
                'boxes_dir'     : os.path.join(directory, stream_name, 'boxes')
            })
        else:
            kwargs.setdefault('directory', self.stream_dir)
        
        # for tensorflow-graph compilation (the 1st call is much slower than the next ones)
        input_size = [s if s is not None else 128 for s in self.input_size]
        self.detect(tf.random.uniform(input_size))

        saving_functions    = self._get_saving_functions(
            max_workers = max_workers, show_result_fn = None, save_json_fn = None
        )
        
        map_file    = os.path.join(kwargs['directory'], 'map.json')
        predicted   = load_json(map_file, default = {})
        
        stream_camera(
            transform_fn     = lambda img: self.predict(
                img,
                force_draw  = True,
                predicted   = predicted,
                create_dirs = img['frame_index'] == 0,
                saving_functions    = saving_functions,
                ** kwargs
            )[0][1],
            max_workers = max_workers,
            add_copy    = True,
            add_index   = True,
            name        = 'frame transform',
            ** kwargs
        )
        
        for fn in saving_functions:
            if isinstance(fn, Consumer): fn.join()
        
        if predicted: dump_json(map_file, predicted, indent = 4)
        
        return map_file
    
    @timer
    def predict(self,
                images,
                batch_size = 16,
                return_output   = False,
                # general saving config
                directory   = None,
                overwrite   = False,
                timestamp   = -1,
                max_workers = -1,
                create_dirs = True,
                saving_functions    = None,
                # Saving mapping + raw images
                save    = True,
                predicted   = None,
                img_num = -1,
                save_empty  = False,
                raw_img_dir = None,
                filename    = 'image_{}.jpg',
                # Saving images with drawn detection
                force_draw      = False,
                save_detected   = False,
                detected_dir    = None,
                detected_filename   = 'detected_{}.jpg',
                # Saving boxes as individual images
                save_boxes  = False,
                boxes_dir   = None,
                boxes_filename  = '{}_box_{}.jpg',
                # Verbosity config
                verbose = 1,
                display = None,
                plot_kwargs = {},
                
                post_processing = None,
                
                ** kwargs
               ):
        """
            Performs image object detection on the givan `images` (either filename / embeddings / raw)
            
            Arguments :
                - images  : the image(s) to detect objects on
                    - str   : the filename of the image
                    - dict / pd.Series  : informations about the image
                        - must contain at least `filename` or `embedding` or `image_embedding`
                    - np.ndarray / tf.Tensor    : the embedding for the image
                    
                    - list / pd.DataFrame   : an iterable of the above types
                - batch_size    : the number of prediction to perform in parallel
                
                - directory : where to save the mapping file / folders for the saved results
                              if not provided, default to `self.pred_dir`
                - overwrite : whether to overwrite already predicted files (ignored for raw images)
                - timestamp : the timestamp of the request (if `overwrite = True` but this timestamp is lower than the timestamp of the prediction, the image is not overwritte), may be useful for versioning
                - max_workers   : the number of saving workers
                    - -1    : the saving functions are applied sequentially in the main thread
                    - 0     : each saving function is called in a separated thread
                    - >0    : each saving function is called in `max_workers` parallel threads
                - create_dirs   : whether to create sub-folder or not (for streaming optimization)
                - saving_functions  : tuple of 5 elements, the saving functions (for streaming)
                
                - save  : whether to save the results or not (set to `True` if any other `save_*` is True)
                - predicted : the saved mapping (for streaming optimization)
                - img_num   : the image number to use for raw images (for streaming optimization)
                - save_empty    : whether to save result for images without any detected object
                - filename  : filename format for raw images
                
                - force_draw    : whether to force drawing boxes or not (drawing is done by default if `save_detected` or `verbose` or `post_processing` is provided)
                - save_detected : whether to save images with drawn boxes
                - detected_dir  : where to save the detections (default to `{directory}/detected`)
                - detected_filename : filename format for the images with detection
                
                - save_boxes    : whether to save each box as an image
                - boxes_dir     : where to save the extracted boxes (default to `{directory}/boxes`)
                - boxes_filename    : the box filename format
                
                - verbose   : the verbosity level
                    - 0 : silent (no display)
                    - 1 : plots the detected image
                    - 2 : plots the detected image + each individual box
                    - 3 : plots the detected image + each individual box + logs their information
                - display   : number of images to plots (cf `verbose`), if `True` or `-1`, displays all images
                - plot_kwargs   : kwargs for the `plot` calls (ignored if silent mode)
                
                - post_processing   : a callable that takes the detected image as 1st arg + `image` and `infos` as kwargs
                
                - kwargs    : forwarded to `self.infer` (and thus to `self.decode_output`)
            Returns :
                - result    : a list of tuple (image, detected, result)
                    - image     : either the filename (if any), either the original image
                    - detected  : the image with drawn boxes (can be either the numpy array, either its filename, either None)
                    - result    : a `dict` with (at least) keys
                        - boxes     : list of boxes, the predicted positions of the objects
                        - timestamp : the timestamp at which the prediction has been performed
                        - filename (if `save` or filename)  : the original image filename
                        - detected (if `save_detected`)     : the filename of the image with boxes
                        - filename_boxes (if `save_boxes`)  : the filenames for the boxes images
            
            Note : videos are supported by this function but are simply forwarded to `self.predict_videos`. This distinction is important because the keys in `result` are different, and thus, to avoid any confusion, the mapping file is not the same (`map.json` vs `map_videos.json`).
            In this case, `kwargs` are forwarded to `self.predict_videos`
        """
        ####################
        #  Initialization  #
        ####################

        time_logger.start_timer('initialization')

        stop_saving_functions = False
        if saving_functions is None:
            stop_saving_functions   = True
            saving_functions    = self._get_saving_functions(
                max_workers = max_workers, show_result_fn = self.show_result
            )

        if len(saving_functions) != 5:
            raise ValueError('`saving_functions` must be of length 5 !')
        
        show_result_fn, save_json_fn, save_image_fn, save_detected_fn, save_boxes_fn = saving_functions

        now = time.time()
        
        if isinstance(images, pd.DataFrame): images = images.to_dict('records')
        if not isinstance(images, (list, tuple, np.ndarray, tf.Tensor)): images = [images]
        elif isinstance(images, (np.ndarray, tf.Tensor)) and len(images.shape) == 3:
            images = np.expand_dims(images, axis = 0)

        if save_detected or save_boxes: save = True
        if display is None:         display = True if not save else False
        if display in (-1, True):   display = len(images)
        if display:                 verbose = max(verbose, 1)
        
        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        
        required_keys   = ['boxes']
        if save:
            if raw_img_dir is None: raw_img_dir = os.path.join(directory, 'images')
            if create_dirs: os.makedirs(raw_img_dir, exist_ok = True)
            required_keys.append('filename')
        
        if save_detected:
            if detected_dir is None: detected_dir = os.path.join(directory, 'detected')
            if create_dirs: os.makedirs(detected_dir, exist_ok = True)
            required_keys.append('detected')

        if save_boxes:
            if boxes_dir is None: boxes_dir = os.path.join(directory, 'boxes')
            if create_dirs: os.makedirs(boxes_dir, exist_ok = True)
            required_keys.append('filename_boxes')
        
        if predicted is None:
            predicted   = load_json(map_file, default = {})
        
        time_logger.stop_timer('initialization')

        ####################
        #  Pre-processing  #
        ####################
        
        time_logger.start_timer('pre-processing')
        
        results     = [None] * len(images)
        duplicatas  = {}
        requested   = [(get_filename(img, keys = ('filename', )), img) for img in images]
        
        videos, inputs = [], []
        for i, (file, img) in enumerate(requested):
            if not should_predict(predicted, file, overwrite = overwrite, timestamp = timestamp, required_keys = required_keys):
                results[i] = (file, predicted[file].get('detected', None), predicted[file])
                continue

            if isinstance(file, str):
                duplicatas.setdefault(file, []).append(i)
                
                if file.endswith(_video_formats):
                    videos.append(file)
                    continue
                elif len(duplicatas[file]) > 1:
                    continue
            
            inputs.append((i, file, img))
        
        time_logger.stop_timer('pre-processing')

        ####################
        #  Inference loop  #
        ####################
        
        show_idx    = post_process(
            results, 0, verbose, display, show_result_fn, post_processing, ** plot_kwargs
        )
        
        if len(inputs) > 0:
            for start in range(0, len(inputs), batch_size):
                time_logger.start_timer('batch processing')
                
                batch_inputs    = inputs[start : start + batch_size]
                batch_images    = [_get_image(file, data) for _, file, data in batch_inputs]
                
                batch   = [self.get_input(image) for image in batch_images]
                if len(batch) == 1:
                    batch = tf.expand_dims(batch[0], axis = 0)
                elif self.has_variable_input_size:
                    batch = tf.cast(pad_batch(batch, dtype = np.float32), tf.float32)
                else:
                    batch = tf.stack(batch, axis = 0)
                
                batch = self.preprocess_input(batch)
                
                time_logger.stop_timer('batch processing')

                # Computes detection + output decoding
                boxes   = self.detect(
                    batch, get_boxes = True, return_output = return_output, ** kwargs
                )
                
                should_save = False
                for (idx, file, data), image, box in zip(batch_inputs, batch_images, boxes):
                    output, box = (None, box) if not return_output else box
                    
                    infos = data if isinstance(data, dict) else {}
                    infos = {
                        k : v for k, v in infos.items()
                        if 'image' not in k or not isinstance(v, (np.ndarray, tf.Tensor))
                    }
                    if file is None:
                        if isinstance(data, (np.ndarray, tf.Tensor)):
                            file = data
                        elif isinstance(data, (dict, pd.Series)) and 'image' in data:
                            file = data['image']
                        else:
                            file = image
                    # Maybe skips the image if nothing has been detected
                    if not isinstance(file, str) and not save_empty and len(box) == 0:
                        results[idx] = (file, file, infos)
                        continue
                    
                    time_logger.start_timer('post processing')

                    basename, detected = None, None
                    if save_detected or verbose or post_processing is not None or force_draw:
                        time_logger.start_timer('drawing boxes')
                        
                        if isinstance(data, (dict, pd.Series)) and 'image_copy' in data:
                            detected = data['image_copy']
                        elif isinstance(file, np.ndarray):
                            detected = image.copy() if save else file
                        elif isinstance(file, tf.Tensor):
                            detected = file.numpy()
                        else:
                            detected = image
                        
                        detected    = self.draw_prediction(detected, box, ** kwargs)
                        time_logger.stop_timer('drawing boxes')

                    infos.update({'boxes' : box, 'timestamp' : now})
                    if save:
                        time_logger.start_timer('saving frame')
                        # Saves the raw image (i.e. if it is not already a filename)
                        # The filename for the raw image is pre-computed here, even if `self.save_image` may do it, in order to allow multi-threading
                        if not isinstance(file, str):
                            if isinstance(data, (dict, pd.Series)) and 'frame_index' in data:
                                img_num = data['frame_index']
                            elif img_num == -1:
                                img_num = len(glob.glob(os.path.join(
                                    raw_img_dir, filename.replace('{}', '*')
                                )))
                            img_file = os.path.join(raw_img_dir, filename.format(img_num))
                            img_num += 1

                            save_image_fn(file, filename = img_file, directory = raw_img_dir)
                            file    = img_file
                        
                        basename    = os.path.splitext(os.path.basename(file))[0]
                        should_save     = True
                        predicted[file] = infos
                        
                        time_logger.stop_timer('saving frame')

                    if isinstance(file, str):
                        infos['filename']   = file
                    
                    if save_detected:
                        time_logger.start_timer('saving detected')
                        detected_file = os.path.join(
                            detected_dir, detected_filename.format(basename)
                        )
                        save_detected_fn(
                            detected,
                            directory   = detected_dir,
                            filename    = detected_file
                        )
                        infos['detected'] = detected_file
                        time_logger.stop_timer('saving detected')

                    if save_boxes:
                        time_logger.start_timer('saving boxes')
                        box_files = [os.path.join(
                            boxes_dir, boxes_filename.format(basename, i)
                        ) for i in range(len(box))]
                        save_boxes_fn(
                            file,
                            image   = image,
                            boxes   = box,
                            labels  = self.labels,
                            directory   = boxes_dir,
                            file_format = box_files,
                            ** kwargs
                        )
                        infos['filename_boxes'] = {
                            box_file : get_box_infos(b, labels = self.labels, ** kwargs)
                            for box_file, b in zip(box_files, box)
                        }
                        time_logger.stop_timer('saving boxes')

                    # This ensures that the `infos` saved will not be modified by `post_process`
                    infos = infos.copy()
                    if return_output: infos['output'] = output
                    # Sets result at the (multiple) index(es)
                    if isinstance(file, str) and file in duplicatas:
                        for duplicate_idx in duplicatas[file]:
                            results[duplicate_idx] = (file, detected, infos)
                    else:
                        results[idx] = (file, detected, infos)
                    
                    time_logger.stop_timer('post processing')
                
                if save and should_save:
                    time_logger.start_timer('saving json')
                    save_json_fn(map_file, data = predicted.copy(), indent = 4)
                    time_logger.stop_timer('saving json')
                
                show_idx    = post_process(
                    results, show_idx, verbose, display, show_result_fn, post_processing, ** plot_kwargs
                )

        if stop_saving_functions:
            for fn in saving_functions:
                if isinstance(fn, Consumer): fn.join()

        if videos:
            kwargs.setdefault('save_frames', save)
            video_results = self.predict_video(
                videos,
                save    = save,
                save_detected   = save_detected,
                save_boxes      = save_boxes,
                
                directory   = directory,
                overwrite   = overwrite,
                max_workers = max_workers,
                
                ** kwargs
            )
            
            for file, infos in video_results:
                for duplicate_idx in duplicatas[file]:
                    results[duplicate_idx] = (file, infos.get('detected', None), infos)
        
        return results

    @timer
    def predict_video(self,
                      videos,
                      save  = True,
                      save_video = True,
                      save_frames   = False,
                      
                      directory = None,
                      overwrite = False,

                      tqdm  = lambda x: x,
                      
                      ** kwargs
                     ):
        """
            Perform prediction on `videos` (with `self.stream` method)
            
            Arguments :
                - videos    : (list of) video filenames
                - save      : whether to save the mapping file
                - save_video    : whether to save the result video with drawn boxes
                - save_frames   : whether to save each frame individually (see `save` in `self.predict`)
                
                - directory : where to save the mapping
                - overwrite : whether to overwrite (or not) the already predicted videos
                
                - tqdm  : progress bar
                
                - kwargs    : forwarded to `self.stream`
            Return :
                - list of tuple [(path, infos), ...]
                    - path  : original video path
                    - infos : general information on the prediction with keys
                        - width / height / fps / nb_frames  : general information on the video
                        - frames (if `save_frames`)     : filename for the frames mapping file (i.e. the output of `self.predict`)
                        - detected (if `save_video`)    : the path to the output video
        """
        if not save_frames: kwargs.update({'save_detected' : False, 'save_boxes' : False})
        kwargs.setdefault('show', False)
        kwargs.setdefault('max_time', -1)
        
        videos = normalize_filename(videos)
        if not isinstance(videos, (list, tuple)): videos = [videos]
        
        # get saving directory
        if directory is None: directory = self.pred_dir
        
        map_file    = os.path.join(directory, 'map_videos.json')
        infos_videos    = load_json(map_file, default = {})
        
        video_dir = os.path.join(directory, 'videos')
        
        # Filters files that do not end with a valid video extension
        videos = [video for video in videos if video.endswith(_video_formats)]
        
        for path in tqdm(set(videos)):
            video_name, ext  = os.path.splitext(os.path.basename(path))
            # Maybe skip because already predicted
            if not overwrite and path in infos_videos:
                if not save_frames or (save_frames and infos_videos[path]['frames'] is not None):
                    if not save_video or (save_video and infos_videos[path]['detected'] is not None):
                        continue
            
            save_dir    = os.path.join(video_dir, video_name)
            out_file = None if not save_video else os.path.join(
                save_dir, '{}_detected{}'.format(video_name, ext)
            )
            map_frames = os.path.join(save_dir, 'map.json') if save_frames else None
            
            if os.path.exists(save_dir): shutil.rmtree(save_dir)
            if out_file and os.path.exists(out_file): os.remove(out_file)
            if out_file: os.makedirs(save_dir, exist_ok = True)
            
            self.stream(
                cam_id  = path,
                save    = save_frames,
                directory   = save_dir,
                output_file = out_file,
                
                ** kwargs
            )
            
            infos   = get_video_infos(path).__dict__
            if out_file:    infos['detected'] = out_file
            if save_frames: infos['frames'] = os.path.join(save_dir, 'map.json')

            infos_videos[path] = infos
        
            dump_json(map_file, infos_videos, indent = 4)
        
        return [(video, infos_videos[video]) for video in videos]

    def evaluate(self, 
                 generator, 
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        if True:
            raise NotImplementedError('This method is deprecated and has to be updated to be used !')
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """    
        return -1.
        # gather all detections and annotations
        generator.batch_size = 1
        all_detections     = [[None for i in range(len(self.labels))] for j in range(len(generator))]
        all_annotations    = [[None for i in range(len(self.labels))] for j in range(len(generator))]

        for i in range(len(generator)):
            inputs, _ = generator.__getitem__(i)
            raw_image, _ = inputs
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self._predict(inputs, get_boxes=True)
            
            score = np.array([box.get_score() for box in pred_boxes])
            pred_labels = np.array([box.get_label() for box in pred_boxes])        
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])  
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]
            
            # copy detections to all_detections
            for label in range(len(self.labels)):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)
            
            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

        return average_precisions
                                
    def get_config(self, * args, ** kwargs):
        config = super(BaseDetector, self).get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_image(),
            'labels'    : self.labels,
            'nb_class'  : self.nb_class,
            
            'obj_threshold' : self.obj_threshold,
            'nms_threshold' : self.nms_threshold
        })
        
        return config

def _get_image(file, data):
    if isinstance(data, dict):
        if 'tf_image' in data:
            return data['tf_image']
        for k in ('image_copy', 'image', 'filename'):
            if k in data: return load_image(data[k])
    return load_image(file)

@timer
def post_process(results, idx, verbose, max_display, show_result_fn, post_processing_fn, ** kwargs):
    while idx < len(results) and results[idx] is not None:
        image, detected, infos = results[idx]
        
        if verbose and idx < max_display:
            if isinstance(detected, str): detected = load_image(detected)
            show_result_fn(
                image,
                boxes   = infos.get('boxes', []),
                detected    = detected,
                verbose = verbose,
                ** kwargs
            )
        
        if post_processing_fn is not None:
            post_processing_fn(detected, image = image, infos = infos)
        
        idx += 1

    return idx

