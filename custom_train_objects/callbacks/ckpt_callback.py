
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

import time
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

MIN_MODE    = 0
MAX_MODE    = 0

class CkptCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint, directory, save_best_only = False,
                 monitor = 'val_loss', mode = MIN_MODE, max_to_keep = 1, verbose = True,
                 save_every_hour = True, **kwargs):
        super(CkptCallback, self).__init__(**kwargs)
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, directory = directory, 
                                                       max_to_keep = max_to_keep)
        self.save_best_only = save_best_only
        self.monitor    = monitor
        self.prev_val   = None
        self.verbose    = verbose
        self.save_every_hour    = save_every_hour
        self.last_saving_time   = time.time()
        
        self.compare = lambda x_prev, x: x < x_prev if mode == MIN_MODE else lambda x_prev, x: x > x_prev
        
    def on_train_begin(self, *args):
        self.last_saving_time = time.time()
        
    def on_train_batch_end(self, *args, **kwargs):
        if self.save_every_hour and time.time() - self.last_saving_time > 3600:
            self.save()
        
    def on_epoch_end(self, epoch, logs):
        if self.save_best_only and self.monitor in logs:            
            new_val = logs[self.monitor]
            if self.prev_val is None or self.compare(self.prev_val, new_val):
                self.save(epoch + 1)
                self.prev_val = new_val
        else:
            self.save(epoch + 1)
        
    def save(self, epoch = None):
        if self.verbose:
            if epoch:
                logger.info("\nSaving at epoch {} !".format(epoch))
            else:
                logger.info("\nSaving after 1 hour training !")
        self.ckpt_manager.save()
        self.last_saving_time = time.time()
        