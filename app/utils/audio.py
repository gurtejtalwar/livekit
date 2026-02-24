from livekit import rtc, agents
from pyspeex_noise import AudioProcessor

class SpeexAudioInput(agents.voice.io.AudioInput):
    def __init__(self, original_input: agents.voice.io.AudioInput):
        super().__init__()
        self._original_input = original_input
        
        # 160 samples = 10ms at 16kHz
        self._frame_size = 160 
        
        # auto_gain: 0 to 32768, noise_suppression: dB (e.g. -30)
        self._processor = AudioProcessor(auto_gain=2000, noise_suppression=-30)
        self._buffer = bytearray()

    async def __anext__(self) -> rtc.AudioFrame:
        # We need exactly 320 bytes (160 samples * 2 bytes for int16)
        while len(self._buffer) < (self._frame_size * 2):
            upstream_frame = await self._original_input.__anext__()
            self._buffer.extend(upstream_frame.data)

        # Slice out the 10ms chunk
        to_process = self._buffer[:self._frame_size * 2]
        self._buffer = self._buffer[self._frame_size * 2:]

        # Apply the Speex DSP magic
        # process_10ms returns the cleaned bytes
        clean_bytes = self._processor.process_10ms(bytes(to_process))

        return rtc.AudioFrame(
            data=clean_bytes,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=self._frame_size
        )

    async def aclose(self) -> None:
        await self._original_input.aclose()