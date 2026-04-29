import logging
import torch
import numpy as np
from livekit import rtc
from df.enhance import init_df, enhance

logger = logging.getLogger(__name__)

# DeepFilterNet v3 operates natively at 48kHz
_DF_SAMPLE_RATE = 48000
# Frame size is typically 20ms (960 samples at 48kHz)
_DF_FRAME_SIZE_MS = 20
_DF_EXPECTED_SAMPLES = (_DF_SAMPLE_RATE * _DF_FRAME_SIZE_MS) // 1000

class DeepFilterNoiseSuppressor(rtc.FrameProcessor[rtc.AudioFrame]):
    """In-process DeepFilterNet noise suppressor for LiveKit.
    
    This implementation handles the upsampling to 48kHz required by the model
    and maintains the internal LSTM/Convolutional state across audio frames.
    """

    def __init__(self, strength: float = 1.0, attenuation_limit_db: float = 100.0) -> None:
            # Initialize model and state
            self._model, self._df_state, _ = init_df()
            
            self._input_queue = np.zeros(0, dtype=np.float32)
            self._output_queue = np.zeros(0, dtype=np.float32)
            self._strength = max(0.0, min(1.0, strength))
            self._attenuation_limit = attenuation_limit_db
            
            self._downsampler: rtc.AudioResampler | None = None
            self._upsampler: rtc.AudioResampler | None = None
            self._native_rate: int = 0
            self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled:
            return frame

        # 1. Setup Resamplers for 48kHz requirement
        if frame.sample_rate != self._native_rate:
            self._native_rate = frame.sample_rate
            if frame.sample_rate != _DF_SAMPLE_RATE:
                self._downsampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=_DF_SAMPLE_RATE,
                    num_channels=1,
                )
                self._upsampler = rtc.AudioResampler(
                    input_rate=_DF_SAMPLE_RATE,
                    output_rate=frame.sample_rate,
                    num_channels=1,
                )
            else:
                self._downsampler = None
                self._upsampler = None

        # 2. Convert to Float32 Mono
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1)

        mono_int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        mono_frame = rtc.AudioFrame(
            data=mono_int16.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=len(mono_int16),
        )

        # 3. Upsample to 48kHz
        if self._downsampler is not None:
            frames_48k = self._downsampler.push(mono_frame)
        else:
            frames_48k = [mono_frame]

        if not frames_48k:
            return frame

        samples_48k = np.concatenate([
            np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
            for f in frames_48k
        ])

        self._input_queue = np.concatenate([self._input_queue, samples_48k])

        # 4. Process in 20ms Blocks (960 samples)
        while len(self._input_queue) >= _DF_EXPECTED_SAMPLES:
            chunk = self._input_queue[:_DF_EXPECTED_SAMPLES]
            self._input_queue = self._input_queue[_DF_EXPECTED_SAMPLES:]

            # DeepFilterNet processing
            # 1. Convert NumPy to Torch Tensor (shared memory)
            chunk_tensor = torch.from_numpy(chunk)
            
            # 2. Process with DeepFilterNet
            # Returns a torch.Tensor
            enhanced_tensor = enhance(
                self._model, 
                self._df_state, 
                chunk_tensor, 
                atten_lim_db=self._attenuation_limit
            )
            
            # 3. Convert back to NumPy for the LiveKit output queue
            # .detach().cpu() is good practice though it's already on CPU here
            enhanced_chunk = enhanced_tensor.detach().cpu().numpy().flatten()

            # Apply strength (Wet/Dry Blend)
            if self._strength < 1.0:
                enhanced_chunk = (self._strength * enhanced_chunk) + ((1.0 - self._strength) * chunk)

            self._output_queue = np.concatenate([self._output_queue, enhanced_chunk])

        # 5. Drain and Downsample back to native rate
        # We drain exactly the amount of samples that entered the 48kHz stage
        n_48k = len(samples_48k)
        if len(self._output_queue) < n_48k:
            return frame

        out_48k = self._output_queue[:n_48k]
        self._output_queue = self._output_queue[n_48k:]

        out_int16_48k = (np.clip(out_48k, -1.0, 1.0) * 32767.0).astype(np.int16)
        out_frame_48k = rtc.AudioFrame(
            data=out_int16_48k.tobytes(),
            sample_rate=_DF_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=len(out_int16_48k),
        )

        if self._upsampler is not None:
            out_frames = self._upsampler.push(out_frame_48k)
        else:
            out_frames = [out_frame_48k]

        if not out_frames:
            return frame

        out_samples = np.concatenate([
            np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
            for f in out_frames
        ])

        # Match exact input frame length and channel count
        target = frame.samples_per_channel
        out_samples = out_samples[:target] if len(out_samples) > target else np.pad(out_samples, (0, max(0, target - len(out_samples))))

        if frame.num_channels > 1:
            out_samples = np.repeat(out_samples, frame.num_channels)

        return rtc.AudioFrame(
            data=(np.clip(out_samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )

    def _close(self) -> None:
        self._enabled = False