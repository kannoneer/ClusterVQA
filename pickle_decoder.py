"""
Decodes videos saved as Python's .pickle files.

Convert output to lossless video with

    ffmpeg -framerate 30 -i "output/frame%04d".png -crf 0 out.mp4
    
"""

import sys
import os
import pickle
import time
import numpy as np
from PIL import Image

from vqa.video_types import *
from vqa.block_ops import merge_frame, upscale_deltamask
from vqa.conversions import perceptual2rgb


def decode_pickled_video(f, outdir):
    header = pickle.load(f)
    print(header)

    decoded_yuv = np.zeros((header.height, header.width, 3))
    codebook = None
    frame_idx = 0

    old_yuv = np.zeros((header.height, header.width, 3), dtype=np.float32)

    while frame_idx < header.numframes:
        obj = pickle.load(f)
        print(f"[{frame_idx+1}/{header.numframes}] {type(obj).__name__}")

        if isinstance(obj, Codebook):
            codebook = obj
            continue

        if isinstance(obj, EncodedFrame):
            merge_start = time.time()
            merge_frame(
                obj.indices, codebook, header.cfg.blockw, header.cfg.blockh, decoded_yuv
            )

            mask = upscale_deltamask(
                obj.deltamask, header.cfg.blockw, header.cfg.blockh
            )
            old_yuv[mask] = decoded_yuv[mask]

            merge_end = time.time()
            decoded_rgb = perceptual2rgb(old_yuv)
            decode_end = time.time()
            outpath = os.path.join(outdir, f"frame{frame_idx+1:04d}.png")
            Image.fromarray(np.clip(decoded_rgb * 255, 0, 255).astype(np.uint8)).save(
                outpath, optimize=False
            )
            write_end = time.time()

            print(
                f"[{frame_idx}] merge: {(merge_end - merge_start)*1000:.2f} ms, conversion: {(decode_end - merge_end)*1000:.2f} ms, writing:  {(write_end - decode_end)*1000:.2f}"
            )
            frame_idx += 1
        else:
            raise Exception(f"Unknown object {obj}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        videofile = sys.argv[1]
        outdir = sys.argv[2]
    else:
        videofile = "video.pickle"
        outdir = "output"

    print(f"Decoding {videofile} into {outdir}")

    with open(videofile, "rb") as f:
        decode_pickled_video(f, outdir)
