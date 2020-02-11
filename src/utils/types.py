from dataclasses import dataclass


@dataclass
class Config:
    audio_processing = {
        'frame_length': 0.20,
        'frame_stride': 0.1,
        'use_temp_derrivatives': True,
    }
