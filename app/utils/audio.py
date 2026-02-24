from livekit.agents.voice import io as voice_io
from livekit import rtc
from pyspeex_noise import AudioProcessor

class SpeexAudioInput(voice_io.AudioInput):
    def __init__(self, original_input: voice_io.AudioInput):
        # The 'label' is now a required keyword-only argument in newer SDKs
        super().__init__(label="speex_noise_filter")
        
        self._original_input = original_input
        
        # Speex setup
        self._sample_rate = 16000
        self._frame_size = 160  # 10ms at 16kHz
        self._processor = AudioProcessor(auto_gain=2000, noise_suppression=-30)
        self._buffer = bytearray()
        
        # Resampler: Most rooms are 48kHz, Speex needs 16kHz
        self._resampler = rtc.AudioResampler(
            source_sample_rate=48000, 
            target_sample_rate=16000
        )

    async def __anext__(self) -> rtc.AudioFrame:
        while len(self._buffer) < (self._frame_size * 2):
            upstream_frame = await self._original_input.__anext__()
            
            # Convert to 16kHz
            resampled_frames = self._resampler.push(upstream_frame)
            for f in resampled_frames:
                self._buffer.extend(f.data)

        # Slice 10ms
        to_process = self._buffer[:self._frame_size * 2]
        self._buffer = self._buffer[self._frame_size * 2:]

        # Clean with Speex
        clean_bytes = self._processor.process_10ms(bytes(to_process))

        return rtc.AudioFrame(
            data=clean_bytes,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=self._frame_size
        )

    async def aclose(self) -> None:
        await self._original_input.aclose()