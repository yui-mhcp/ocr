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
import sys
import logging

from logging.handlers import SMTPHandler

from .time_logging import TIME_LEVEL, TIME_DEBUG_LEVEL, timer, time_logger
from .telegram_handler import TelegramHandler
from .utils import get_object, partial, dispatch_wrapper

logger  = logging.getLogger(__name__)

DEV_LEVEL   = 11
RETRACING_LEVEL = 18

_styles = {
    'basic' : '{message}',
    'extended'  : '{asctime} : {levelname} : {message}',
    'dev'       : '{asctime} : {levelname} : {module} ({funcName}, {lineno}) : {message}'
}
_levels = {
    'debug' : logging.DEBUG,
    'dev'   : DEV_LEVEL,
    'time_debug'    : TIME_DEBUG_LEVEL,
    'time'  : TIME_LEVEL,
    'retracing' : RETRACING_LEVEL,
    'info'      : logging.INFO,
    'warning'   : logging.WARNING,
    'error'     : logging.ERROR,
    'critical'  : logging.CRITICAL
}
_handlers   = {
    'stream'    : logging.StreamHandler,
    'file'      : logging.FileHandler,
    'smtp'      : SMTPHandler,
    'telegram'  : TelegramHandler
}

_default_style  = os.environ.get('LOG_STYLE', 'basic').lower()
_default_level  = os.environ.get('LOG_LEVEL', 'info').lower()
_default_format = _styles.get(_default_style, _default_style)

logging.basicConfig(
    level   = _levels.get(_default_level, _default_level),
    stream  = sys.stdout,
    format  = _default_format,
    style   = '%' if '%' in _default_format else '{'
)

def add_level(value, name):
    """
        Adds a new level to the logging module
        
        Arguments :
            - value : the log level value (e.g., logging.DEBUG = 10, logging.INFO = 20, ...)
            - name  : the level name
        
        Example :
        ```python
        # add a 'dev' level just above the debug level
        add_level('dev', 11)
        # Now it is possible to set the level with the `set_level` method
        set_level('dev')
        # log a message with the new `.dev` method
        logging.dev('This is a test !')
        # logging.getLogger(__name__).dev('This will also work !')
        ```
    """
    name = name.lower()
    if name not in _levels:
        set_level.dispatch(name, value)

    logging.addLevelName(value, name.upper())
    if not hasattr(logging, name.upper()):
        setattr(logging, name.upper(), value)
    if not hasattr(logging, name):
        setattr(logging, name, partial(logging.log, value))
    if not hasattr(logging.Logger, name):
        setattr(logging.Logger, name, partial(logging.Logger.log, value))

def set_style(style, logger = None):
    """ Sets the logging style to `logger` (root logger if None) """
    global _default_style
    _default_style = style
    
    formatter = get_formatter(style)
    
    for handler in logging.getLogger(logger).handlers:
        handler.setFormatter(formatter)

@dispatch_wrapper(_levels, 'level')
def set_level(level, logger = None):
    """ Sets the `logger` level to `level` """
    if isinstance(level, str):  level = level.lower()
    if isinstance(logger, str) or logger is None: logger = logging.getLogger(logger)
    logger.setLevel(_levels.get(level, level))

def get_formatter(format = _default_style, style = None, datefmt = None, ** kwargs):
    if isinstance(format, str): format = {'fmt' : _styles.get(format, format)}
    if isinstance(format, dict): format.setdefault('style', '%' if '%' in format['fmt'] else '{')
    
    return format if not isinstance(format, dict) else logging.Formatter(** format)

@dispatch_wrapper(_handlers, 'handler')
def add_handler(handler, * args, logger = None, level = None, add_formatter = True, ** kwargs):
    global _default_style
    
    if logger is None:              logger = logging.getLogger()
    elif isinstance(logger, str):   logger = logging.getLogger(logger)
    
    if isinstance(level, str): level = _levels[level.lower()]
    
    fmt = kwargs.pop('format', _default_style)
    
    if isinstance(handler, str):
        handler = get_object(_handlers, handler, * args, ** kwargs)
    
    if isinstance(handler, str) or handler is None: return
    if level is not None: handler.setLevel(level)
    
    if add_formatter and fmt is not None:
        formatter = get_formatter(fmt)
        
        handler.setFormatter(formatter)
    
    if level is not None and logger.level > handler.level: logger.setLevel(level)

    logger.addHandler(handler)
    return handler

add_basic_handler   = partial(
    add_handler, 'stream', sys.stdout, format = 'basic'
)
add_file_handler    = partial(
    add_handler, 'file', filename = 'logs.log', encoding = 'utf-8', format = 'extended'
)

@add_handler.dispatch('tts')
def try_tts_handler(* args, ** kwargs):
    try:
        from loggers.tts_handler import TTSHandler
        return TTSHandler(* args, ** kwargs)
    except ImportError as e:
        logger.error("Error when adding TTSHandler : {}".format(e))
        return None

for name, val in _levels.items(): add_level(val, name)
