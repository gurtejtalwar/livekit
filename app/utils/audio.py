from livekit.agents.voice import io as voice_io
from livekit import rtc
from pyspeex_noise import AudioProcessor

class SpeexAudioInput(voice_io.AudioInput):
    def __init__(self, original_input: voice_io.AudioInput):
        # 1. 'label' is a required keyword-only argument in newer Agent SDKs
        super().__init__(label="speex_noise_filter")
        
        self._original_input = original_input
        
        # 2. Speex configuration
        self._sample_rate = 16000
        self._frame_size = 160  # 10ms at 16kHz
        self._processor = AudioProcessor(auto_gain=2000, noise_suppression=-30)
        self._buffer = bytearray()
        
        # 3. AudioResampler uses positional args: (input_rate, output_rate)
        # Assuming your LiveKit room is 48kHz (standard)
        self._resampler = rtc.AudioResampler(48000, 16000)

    async def __anext__(self) -> rtc.AudioFrame:
        # Fill buffer until we have 10ms of 16kHz audio (320 bytes)
        while len(self._buffer) < (self._frame_size * 2):
            upstream_frame = await self._original_input.__anext__()
            
            # Resample 48kHz -> 16kHz
            resampled_frames = self._resampler.push(upstream_frame)
            for f in resampled_frames:
                self._buffer.extend(f.data)

        # Slice exactly 10ms
        to_process = self._buffer[:self._frame_size * 2]
        self._buffer = self._buffer[self._frame_size * 2:]

        # Noise suppression
        clean_bytes = self._processor.process_10ms(bytes(to_process))

        return rtc.AudioFrame(
            data=clean_bytes,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=self._frame_size
        )

    async def aclose(self) -> None:
        await self._original_input.aclose()