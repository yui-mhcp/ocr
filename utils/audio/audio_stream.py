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

import time
import logging
import inspect

from threading import RLock, Event

from ..threading import run_in_thread

logger = logging.getLogger(__name__)

class AudioStream:
    def __init__(self, rate, format, channels = 1, fps = 10, name = None,
                 keep_alive = False, ** kwargs):
        self.fps    = fps
        self.name   = name
        self.rate   = rate
        self.format = format
        self.channels   = channels
        # whether to keep the `PyAudio` instance alive between playbacks (see `_ensure_audio`).
        # `True` for the reused player singleton ; `False` for throwaway streams (e.g. the
        # per-call `AudioRecorder`), which are fully torn down when they finish.
        self.keep_alive = keep_alive
        self.kwargs = kwargs
        
        self.mutex  = RLock()
        self._audio = None
        self._stream    = None
        self._finished  = Event()

    @property
    def chunk_size(self):
        return self.rate // self.fps
    
    @property
    def finished(self):
        return self._finished.is_set()
    
    def __repr__(self):
        return '<{}{} rate={} channels={}>'.format(
            self.__class__.__name__,
            '' if not self.name else ' name={}'.format(self.name),
            self.rate,
            self.channels
        )
    
    def start(self):
        with self.mutex:
            if self._stream is not None and not self.finished: return self
            elif self._stream is not None: self._close_stream()

            self._open_stream()

            self._finished.clear()
            self.start_finalizer()

        return self

    def _ensure_audio(self):
        """
            Lazily creates the `pyaudio.PyAudio` instance and keeps it alive across playbacks.

            Creating a `PyAudio` runs `Pa_Initialize`, which enumerates every ALSA/JACK device
            (the noisy `cards.pcm.*` / `jack server is not running` messages, printed on the C
            stderr and thus *not* maskable via `logging`) and costs a non-negligible amount of
            time. Doing it once per process instead of once per playback removes both the spam
            repetition and the per-call latency. Only an *open stream* holds the device, so keeping
            this instance alive between playbacks does not lock the speakers/microphone.
        """
        import pyaudio

        if self._audio is None:
            self._audio = pyaudio.PyAudio()
        return self._audio

    def _open_stream(self):
        import pyaudio

        if 'input_device_index' in self.kwargs:
            infos = self._ensure_audio().get_device_info_by_host_api_device_index(
                0, self.kwargs['input_device_index']
            )
            self.name = infos['name']
            if self.rate is None:
                self.rate = int(infos['defaultSampleRate' if self.input else 'maxOutputChannels'])

            if self.channels is None:
                self.channels = infos['maxInputChannels' if self.input else 'maxOutputChannels']

        def _open():
            return self._ensure_audio().open(
                rate    = self.rate,
                format  = getattr(pyaudio, 'pa' + self.format.capitalize()),
                output  = self.output,
                input   = self.input,
                channels    = self.channels,
                stream_callback = self.stream_callback,
                frames_per_buffer   = self.chunk_size,
                ** self.kwargs
            )

        try:
            self._stream = _open()
        except Exception as e:
            # the persistent `PyAudio` handle may have gone stale (device unplugged, audio server
            # restarted, ...) : tear it down and retry once with a fresh instance
            logger.warning('{} failed to open its stream ({}) : retrying with a fresh PyAudio'.format(
                self.__class__.__name__, e
            ))
            self.terminate()
            self._stream = _open()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('{} is started'.format(self.__class__.__name__))

    def join(self):
        return self._finished.wait()

    def _close_stream(self):
        """
            Closes the audio stream (releasing the device) while keeping the persistent `PyAudio`
            instance alive for the next playback. Idempotent and mutex-protected : it may race with
            `start_finalizer` (both may call it) and with a concurrent `start`.
        """
        with self.mutex:
            if self._stream is None: return
            self._stream.close()
            self._stream = None

    def terminate(self, force = True):
        """
            Full teardown : closes the stream *and* destroys the persistent `PyAudio` instance.
            Used by `pause` and registered via `atexit` (best-effort : `atexit` does not run on
            `SIGKILL`, nor on the default `SIGTERM`, but the OS reclaims every audio handle on
            process death anyway, so this is a courtesy, never a correctness requirement).
        """
        with self.mutex:
            if not force and self._stream is not None and not self._stream.is_stopped(): return

            self._close_stream()
            if self._audio is not None:
                self._audio.terminate()
                self._audio = None

        logger.debug('{} is finished'.format(self.__class__.__name__))

    @run_in_thread(daemon = True)
    def start_finalizer(self):
        self.join()
        if self.keep_alive: self._close_stream()
        else:               self.terminate(False)