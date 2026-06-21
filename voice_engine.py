import numpy as np

from audio_capture import AudioCapture
from audio_network import AudioNetwork
from audio_playback import AudioPlayback
from dsp_engine import DSPEngine

from config import (
    BLOCK_SIZE,
    INPUT_CHANNELS,
    MAP_WIDTH,
    MIC_THRESHOLD,
)


class VoiceEngine:

    def __init__(
        self,
        local_ip,
        local_port,
        remote_ip,
        remote_port,
    ):

        # Audio modules

        self.capture = AudioCapture()

        self.network = AudioNetwork(
            local_ip=local_ip,
            local_port=local_port,
            remote_ip=remote_ip,
            remote_port=remote_port,
        )

        self.playback = AudioPlayback()

        self.dsp = DSPEngine()

        # Runtime state

        self.listener_position = (0.0, 0.0)
        self.source_position = (0.0, 0.0)

        self.current_volume = 0.0

    # =====================================

    # Lifecycle

    # =====================================

    def start(self):

        self.capture.start()
        self.playback.start()

    def stop(self):

        self.capture.stop()
        self.playback.stop()

    # =====================================

    # Position

    # =====================================

    def set_listener_position(
        self,
        position,
    ):

        self.listener_position = position

    def set_source_position(
        self,
        position,
    ):

        self.source_position = position

    # =====================================

    # RMS

    # =====================================

    def calculate_volume(
        self,
        audio_buffer,
    ):

        rms = np.sqrt(
            np.mean(audio_buffer ** 2)
        )

        return float(rms)

    # =====================================

    # Current Volume

    # =====================================

    def get_current_volume(self):

        return self.current_volume

    # =====================================

    # Capture + Voice Activity Detection
    # + UDP Send
    # =====================================

    def capture_and_send(self):

        frame = self.capture.read()

        volume = self.calculate_volume(
            frame
        )

        self.current_volume = min(
            max(volume, 0.0),
            1.0,
        )

        making_noise = (
            self.current_volume >= MIC_THRESHOLD
        )

        if making_noise:

            self.network.send_audio(
                frame
            )

        return making_noise

    # =====================================

    # Receive + DSP + Playback

    # =====================================

    def receive_and_play(self):

        frame = self.network.receive_audio(

            block_size=BLOCK_SIZE,

            channels=INPUT_CHANNELS,

        )

        if frame is None:

            return

        stereo = self.dsp.process(

            audio_buffer=frame,

            listener_position=self.listener_position,

            source_position=self.source_position,

            map_width=MAP_WIDTH,

        )

        self.playback.play(
            stereo
        )