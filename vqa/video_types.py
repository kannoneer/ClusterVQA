from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    blockw = 4
    blockh = 4
    num_blocks = 3000
    dither_strength = 4.0
    quantize_before_block_search = True
    max_keyframe_distance = 30
    min_keyframe_distance = 15
    scene_cut_error_limit = 3.0
    block_replacement_error = 2.0
    max_vectors_to_fit = 200000

    def vector_size(self):
        return self.blockh * self.blockw * 3


@dataclass
class VideoHeader:
    version = 1
    width: int
    height: int
    numframes: int
    fps: float
    cfg: Config


@dataclass
class RawFrame:
    yuv: np.ndarray
    blocksx: int
    blocksy: int

    def blocks_per_frame(self):
        return self.blocksy * self.blocksx


@dataclass
class Codebook:
    codes: np.ndarray


@dataclass
class EncodedFrame:
    indices: np.ndarray
    deltamask: np.ndarray
