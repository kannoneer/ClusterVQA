import os.path
import sys
import time
import glob
import io
import vqa
from vqa import quantizer
from vqa.encoders import PickleEncoder, HiColorVQAEncoder, VizEncoder
from vqa.video_types import *
from vqa.audiotrack import load_raw_s16le_file
from pickle_decoder import decode_pickled_video

datadir = "testdata/logo"
tempdir = "testoutput"
frame_paths = list(sorted(glob.glob(os.path.join(datadir, "*.png"))))
test_fps = 15
cfg = Config()
cfg.num_blocks = 100
cfg.max_vectors_to_fit = 10000
cfg.min_keyframe_distance = 30
cfg.max_keyframe_distance = 30
cfg.scene_cut_error_limit = 1e9  # disable scene cut detection


def pickle_end_to_end_test():
    class MemoryPickler:
        def __init__(self):
            self.data = bytearray()

        def write(self, stuff):
            self.data += stuff

    memory = MemoryPickler()
    encoder = PickleEncoder(memory, None)

    start = time.time()
    quantizer.encode_video(frame_paths, test_fps, cfg, encoder)
    took = time.time() - start
    print(f"Got {len(memory.data)} bytes")
    assert len(memory.data) > 1000
    assert encoder.frame_idx == 35

    if not os.path.isdir(tempdir):
        os.mkdir(tempdir)

    inputstream = io.BytesIO(memory.data)
    decode_pickled_video(inputstream, tempdir)

    for name in ["frame0001.png", "frame0012.png", "frame0025.png", "frame0035.png"]:
        source = quantizer.load_float_rgb(os.path.join(datadir, name))
        result = quantizer.load_float_rgb(os.path.join(tempdir, name))
        error = ((source - result) ** 2).mean()
        print(f"{name} MSE: {error}")
        assert error < 0.001

    print(f"Encoding took {took:.3f} s, {took/len(frame_paths):.4f} s/frame")
    print("Pickle end to end test OK")


def visual_inspection_test():
    """Not an automated test. It just shows the test encoding results."""

    encoder = VizEncoder(None, framerange=range(len(frame_paths)))
    quantizer.encode_video(frame_paths, test_fps, cfg, encoder)


def vqa_smoke_test():
    if not os.path.isdir(tempdir):
        os.mkdir(tempdir)

    videofile = os.path.join(tempdir, "logo.vqa")
    track = load_raw_s16le_file(
        "testdata/bleeps_22050_pcm_s16le.raw", rate=22050, is_stereo=True
    )

    if os.path.exists(videofile):
        os.remove(videofile)

    with open(videofile, "w") as _:
        pass

    with open(videofile, "rb+") as f:
        start = time.time()
        quantizer.encode_video(frame_paths, test_fps, cfg, HiColorVQAEncoder(f, track))
        took = time.time() - start

    assert os.path.exists(videofile), "VQA file should exist"
    assert (
        os.path.getsize(videofile) > 1024 * 10
    ), "VQA file should be larger than 10 KiB"

    print("VQA Smoke Test OK")


suites = ["default", "all", "visual"]


def print_usage():
    print("run_tests.py [SUITE]")
    print("Where SUITE is one of ", suites)


if __name__ == "__main__":
    suite = "default"
    if len(sys.argv) > 1:
        suite = sys.argv[1]

    if suite not in suites:
        print_usage()
        sys.exit(1)

    if suite == "default" or suite == "all":
        pickle_end_to_end_test()
        vqa_smoke_test()

    if suite == "all":
        vqa.block_ops.test_block_ops()
        vqa.vptr_format.test_vptr_encoding()
        vqa.lcw.test_compression()
        vqa.lcw.test_decompression()
        vqa.audiotrack.test_pcm_loading()

    if suite == "visual":
        visual_inspection_test()

    print(f"OK")
