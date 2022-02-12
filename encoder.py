import vqa.quantizer as quantizer
import glob
import os
import argparse

import vqa.runtime_config
from vqa.video_types import *
from vqa.encoders import PickleEncoder, VizEncoder, HiColorVQAEncoder
from vqa.audiotrack import load_raw_s16le_file

tempcfg = Config()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("framedir", help="PNG image sequence location")
parser.add_argument("output", help="Output file name")
parser.add_argument(
    "--encoder",
    default="vqa",
    choices=["vqa", "pickle", "viz"],
    help="The format to encode into. 'vqa' writes Hi-Color VQA movie files, 'pickle' writes a Python pickled debug file, 'viz' presents encoding results visually for debugging.",
)
parser.add_argument(
    "--fps", default=15, help="Video speed in frames per second", type=int
)
parser.add_argument(
    "--blocks",
    default=tempcfg.num_blocks,
    help="How many codes per codebook to allocate. Reasonable values are 500-8000. Higher numbers increase image quality at the expense of encoding time.",
    type=int,
)
parser.add_argument(
    "--blocksize",
    default="4x4",
    choices=["4x4", "4x2"],
    help="Block size. 4x2 yields higher quality at the expense of file size.",
)
parser.add_argument(
    "--audiorate", "-ar", default=44100, help="Audio sampling rate in Hz", type=int
)
parser.add_argument(
    "--audio",
    help="Soundtrack in raw pcm_s16le format. Use 'ffmpeg video.mp4 -f s16le -c:a pcm_s16le -ar 44100 audio.raw' to convert into this format.",
)
parser.add_argument(
    "--min-keyframe-distance",
    default=tempcfg.min_keyframe_distance,
    type=int,
    help="Never encode two codebooks closer than this many frames.",
)
parser.add_argument(
    "--max-keyframe-distance",
    default=tempcfg.max_keyframe_distance,
    type=int,
    help="Always encode a codebook if this many frames have passed.",
)
parser.add_argument(
    "--block-replacement-error",
    default=tempcfg.block_replacement_error,
    type=float,
    help="How much accumulated error should a block have to get re-encoded. Set to 0.0 to re-encode every frame. Reasonable values 0-100.",
)
parser.add_argument(
    "--scene-cut-error-limit",
    default=tempcfg.scene_cut_error_limit,
    type=float,
    help="How much should subsequent frames differ to trigger a new codebook. Lower values are more sensitive to changes. Reasonable values 1-10",
)
parser.add_argument(
    "--dither-strength",
    default=tempcfg.dither_strength,
    type=float,
    help="Ordered dithering strength. Values 0-10.",
)
parser.add_argument(
    "--max-vectors-to-fit",
    default=tempcfg.max_vectors_to_fit,
    type=int,
    help="Higher values give better output quality at the expense of encoding time. Reasonable values 10 000-300 000.",
)
parser.add_argument(
    "--quantize-before-block-search",
    default=tempcfg.quantize_before_block_search,
    type=bool,
    help="Quantize colors before codebook fitting. Should give higher quality if enabled.",
)

parser.add_argument(
    "--verbose", action="store_true", help="Prints more detailed logging messages."
)

args = parser.parse_args()

vqa.runtime_config.verbose = args.verbose

if args.min_keyframe_distance < 15:
    print(f"Warning: The set minimum keyframe distance {args.min_keyframe_distance} is very short may produce a video that doesn't play in game")

track = None
cfg = Config()
cfg.min_keyframe_distance = args.min_keyframe_distance
cfg.max_keyframe_distance = args.max_keyframe_distance
cfg.scene_cut_error_limit = args.scene_cut_error_limit
cfg.block_replacement_error = args.block_replacement_error
cfg.num_blocks = args.blocks
cfg.max_vectors_to_fit = args.max_vectors_to_fit
cfg.dither_strength = args.dither_strength

if args.blocksize == "4x4":
    cfg.blockw = 4
    cfg.blockh = 4
elif args.blocksize == "4x2":
    cfg.blockw = 4
    cfg.blockh = 2
else:
    raise Exception(f"Unknown block size {args.blocksize}")

if args.audio is not None:
    track = load_raw_s16le_file(args.audio, rate=args.audiorate, is_stereo=True)
    print(
        f"Loaded audio {args.audio}: {track.rate} Hz, Channels: {track.num_channels()}, {track.bits_per_channel()} bps, {track.length():.2f} s"
    )

frame_paths = list(sorted(glob.glob(os.path.join(args.framedir, "*.png"))))
print(f"Encoding {len(frame_paths)} frames of {args.framedir} into {args.output}")

# Create the file if it's missing
with open(args.output, "w") as f:
    pass

# Open for read & write because encoder also needs to read it
with open(args.output, "rb+") as f:
    if args.encoder == "pickle":
        encoder = PickleEncoder(f, track)
    elif args.encoder == "viz":
        encoder = VizEncoder(track, framerange=range(len(frame_paths)))
    else:
        encoder = HiColorVQAEncoder(f, track)
    quantizer.encode_video(frame_paths, args.fps, cfg, encoder)
