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
import shutil
import logging
import numpy as np
import pandas as pd

from utils import *
from utils.image import *
from utils.callbacks import *
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
        self._init_labels(labels if labels is not None else ['object'], ** kwargs)

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
        
        kwargs.setdefault('color', BASE_COLORS)
        kwargs.setdefault('use_label', True)
        kwargs.setdefault('labels', labels if labels is not None else self.labels)

        return draw_boxes(image, boxes, ** kwargs)
    
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
            
            By default, the function simply displays (wit `cv2.imshow`) the transformed stream
            If `save == True` or `show == False` without any other configuration, it will save the raw stream (if `stream` is not a video file), and raw frames + detection information
            
            The default `stream_name` is the stream basename (if video file) or `stream-{}`
            The default `output_file` is `stream.mp4`
            The default `transformed_file` is `{output_file_basename}-transformed.mp4`
        """
        if save is None: save = not show
        if save_stream is None:
            save_stream = (save or output_file is not None) and not isinstance(stream, str)
        if save_transformed is None:
            save_transformed = transformed_file is not None
        
        if save_stream:
            if output_file is None: output_file = 'stream.mp4'
            save = True
        
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

        predicted, required_keys, callbacks    = self.get_prediction_callbacks(** kwargs)
        
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

                infos = {'image' : image, 'boxes' : box, 'timestamp' : now}
                if isinstance(data, dict):
                    infos.update(data)
                elif file:
                    infos['filename'] = file
                
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
            for callback in _callbacks:
                if isinstance(callback, Callback): callback.join()

        if videos: raise NotImplementedError()
        
        return [(
            stored['filename'] if 'filename' in stored else output.get('filename', output['image']),
            output['detected'] if output and 'detected' in output else stored.get('detected', None),
            output if output else stored
        ) for (stored, output) in results]

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
                                
    def get_config(self):
        config = super(BaseDetector, self).get_config()
        config.update({
            ** self.get_config_image(),
            ** self.get_config_labels(),
            'obj_threshold' : self.obj_threshold,
            'nms_threshold' : self.nms_threshold
        })
        
        return config

