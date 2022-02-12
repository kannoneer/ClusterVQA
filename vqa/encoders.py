import pickle
import struct
import os

from .video_types import *
from .block_ops import merge_frame, upscale_deltamask
from .lcw import compress_lcw
from .conversions import perceptual2rgb, round_rgb_float_to_rgb555
from .vptr_format import compress_vptr
from .audiotrack import AudioTrack


class Encoder:
    def write_header(self, header: VideoHeader, track: AudioTrack):
        raise NotImplementedError()

    def write_codebook(self, codebook: Codebook):
        raise NotImplementedError()

    def write_frame(self, indices: np.array, deltamask: np.array):
        raise NotImplementedError()

    def finalize(self):
        pass


class PickleEncoder(Encoder):
    def __init__(self, file, track: AudioTrack):
        self.file = file
        self.frame_idx = 0

    def write_header(self, header: VideoHeader):
        pickle.dump(header, self.file)

    def write_codebook(self, codebook: Codebook):
        pickle.dump(codebook, self.file)

    def write_frame(self, indices: np.array, deltamask: np.array):
        frame = EncodedFrame(indices, deltamask)
        pickle.dump(frame, self.file)
        self.frame_idx += 1


import importlib


class VizEncoder(Encoder):
    def __init__(self, track: AudioTrack, framerange=None):
        self.plt = importlib.import_module("matplotlib.pyplot")
        self.framerange = framerange
        self.frame_idx = 0
        self.codebook = None

    def write_header(self, header: VideoHeader):
        print(header)
        self.header = header
        self.old_yuv = np.zeros((header.height, header.width, 3), dtype=np.float32)

    def write_codebook(self, codebook: Codebook):
        self.codebook = codebook
        self.codebook_frame = self.frame_idx
        print(f"[{self.frame_idx}] Got codebook")

    def write_frame(self, indices: np.array, deltamask: np.array):
        numvalid = np.sum(deltamask)
        print(f"[{self.frame_idx}] Got a frame with {numvalid}/{deltamask.size} blocks")

        hdr = self.header
        decoded_yuv = np.zeros_like(self.old_yuv)
        merge_frame(indices, self.codebook, hdr.cfg.blockw, hdr.cfg.blockh, decoded_yuv)
        mask = upscale_deltamask(deltamask, hdr.cfg.blockw, hdr.cfg.blockh)
        self.old_yuv[mask] = decoded_yuv[mask]

        if self.framerange and self.frame_idx in self.framerange:
            # Visualize the codebook blocks as an image
            columns = 32
            book = self.codebook.codes
            codeimg = np.zeros(
                (columns * ((book.shape[0] + columns - 1) // columns), book.shape[1])
            )
            codeimg[: book.shape[0], :] = book
            codeimg = codeimg.reshape(
                codeimg.shape[0] // columns,
                columns,
                self.header.cfg.blockh,
                hdr.cfg.blockw,
                3,
            )
            codeimg = codeimg.transpose(0, 2, 1, 3, 4)
            codeimg = codeimg.reshape(
                codeimg.shape[0] * codeimg.shape[1],
                codeimg.shape[2] * codeimg.shape[3],
                3,
            )
            codeimg = perceptual2rgb(codeimg)

            rgb = perceptual2rgb(self.old_yuv)

            self.fig, self.ax = self.plt.subplots(1, 3, figsize=(16, 6))
            self.ax[0].imshow(rgb)
            self.ax[0].set_title("Reconstructed frame")
            self.ax[1].imshow(rgb)
            self.ax[1].imshow(mask, alpha=0.5)
            self.ax[1].set_title("Updated blocks")
            self.ax[2].imshow(codeimg)
            self.ax[2].set_title(f"Codebook set on frame {self.codebook_frame}")
            self.fig.suptitle(f"Frame {self.frame_idx}")
            self.plt.show()

        self.frame_idx += 1

    def finalize(self):
        print(f"Done")


# RGB channel bit shifts to "0rrrrrgg gggbbbbb" bit positions
rgb555_shifts = np.array([[[10, 5, 0]]])


def encode_codebook(codebook: Codebook, cfg: Config):
    # From 1D vectors to three RGB vectors
    yuv = codebook.codes.reshape(-1, cfg.blockh * cfg.blockw, 3)
    # Convert to RGB and round to 5-bit range
    rgb_rounded = round_rgb_float_to_rgb555(perceptual2rgb(yuv))
    # Shift R and G channels leftwards
    rgb_rounded = np.left_shift(rgb_rounded, rgb555_shifts)
    # and combine each RGB triplet to a single 16-bit value with bitwise OR
    rgb555 = np.bitwise_or.reduce(rgb_rounded, axis=-1)
    rgb555 = rgb555.astype(np.uint16)
    packed = rgb555.tobytes(order="C")
    assert len(packed) == codebook.codes.shape[0] * cfg.blockh * cfg.blockw * 2
    return packed


from contextlib import contextmanager


class HiColorVQAEncoder(Encoder):
    @contextmanager
    def Chunk(name, file):
        file.write(name)
        file.write(b"xxxx")
        data_ofs = file.tell()

        yield data_ofs

        current = file.tell()
        size = current - data_ofs
        file.seek(data_ofs - 4, os.SEEK_SET)
        file.write(struct.pack(">L", size))
        file.seek(current, os.SEEK_SET)
        if file.tell() % 2 == 1:
            file.write(b"\0")  # Next chunk needs to be 16-bit aligned

    def __init__(self, file, track: AudioTrack):
        self.file = file
        self.track = track
        self.header = None
        self.codebook = None
        self.new_codebook = None
        self.framenum = 0
        self.frame_data_offsets = []
        self.audio_sample_pos = 0.0
        self.cbfz_chunk_infos = []  # A list of (framenum, ofs, size) tuples

    def write_u8(self, x):
        self.file.write(struct.pack("B", x))

    def write_u16(self, x):
        self.file.write(struct.pack("<H", x))

    def write_u32(self, x):
        self.file.write(struct.pack("<L", x))

    def write_u32be(self, x):
        self.file.write(struct.pack(">L", x))

    def write_header(self, header: VideoHeader):
        # Header gets written in finalize() after all the video stream data
        # because we need to know the number of keyframes at that point.
        self.header = header
        self.file.truncate()  # Files opened in the 'rb+' mode don't get truncated by default.

    def write_codebook(self, codebook: Codebook):
        self.new_codebook = codebook

    def write_cbfz(self, codebook: Codebook):
        assert codebook.codes.shape[0] <= self.header.cfg.num_blocks

        with HiColorVQAEncoder.Chunk(b"CBFZ", self.file) as chunk_start_offset:
            data = encode_codebook(codebook, self.header.cfg)
            cbfz_data = compress_lcw(data, newformat=True)
            self.file.write(cbfz_data)
            size = self.file.tell() - chunk_start_offset
            self.cbfz_chunk_infos.append((self.framenum, chunk_start_offset, size))

    def write_snd0(self):
        assert (
            len(self.track.data) == self.track.uncomp_size
        ), "audio track must be uncompressed PCM"
        assert self.track.bits_per_channel() == 16

        samples_per_frame = self.track.rate / self.header.fps
        start = int(round(self.audio_sample_pos))
        self.audio_sample_pos += samples_per_frame
        end = int(round(self.audio_sample_pos))
        framesize = self.track.bytes_per_frame()
        block = self.track.data[(framesize * start) : (framesize * end)]
        with HiColorVQAEncoder.Chunk(b"SND0", self.file) as chunk_start_offset:
            self.file.write(block)

    def write_frame(self, indices: np.array, deltamask: np.array):
        write_new_codebook = self.new_codebook is not None
        finf_offset = self.file.tell()

        if write_new_codebook:
            self.codebook = self.new_codebook
            self.new_codebook = None

        self.frame_data_offsets.append(finf_offset)

        if self.framenum > 0 and write_new_codebook:
            assert write_new_codebook
            with HiColorVQAEncoder.Chunk(b"VQFL", self.file) as chunk_start_offset:
                self.write_cbfz(self.codebook)

        # Sound chunks must be written after VQFL because otherwise codebook updates won't work ingame.
        if self.track is not None:
            self.write_snd0()

        with HiColorVQAEncoder.Chunk(b"VQFR", self.file) as chunk_start_offset:
            if self.framenum == 0 and write_new_codebook:
                self.write_cbfz(self.codebook)

            with HiColorVQAEncoder.Chunk(b"VPRZ", self.file) as chunk_start_offset:
                raw_vptr = compress_vptr(indices, deltamask)
                vprz = compress_lcw(raw_vptr, newformat=True)
                self.file.write(vprz)

        self.framenum += 1

    def finalize(self):
        # Write headers to the beginning of the VQA file. This means we have to do a little dance and read back the
        # data written so far ('streamdata' below), write the headers, and then append the old data at the end.
        filesize = self.file.tell()
        self.file.seek(0, os.SEEK_SET)
        streamdata = self.file.read(filesize)
        self.file.seek(0, os.SEEK_SET)

        header = self.header

        self.file.write(b"FORMxxxxWVQA")
        self.file.write(b"VQHD")
        self.write_u32be(42)

        if float(int(header.fps)) != header.fps:
            print(f"Non-integral FPS {header.fps} will be truncated")

        self.write_u16(3)  # version
        self.write_u16(0x1C if self.track is None else 0x1D)
        self.write_u16(header.numframes)
        self.write_u16(header.width)
        self.write_u16(header.height)
        self.write_u8(header.cfg.blockw)
        self.write_u8(header.cfg.blockh)
        self.write_u8(int(header.fps))
        self.write_u8(0)  # Codebook parts, this is zero in Hi-Color VQAs
        self.write_u16(0)  # Colors is always 0 in Hi-Color VQAs
        self.write_u16(header.cfg.num_blocks)  # MaxBlocks
        self.write_u32(0)  # Unknown1
        self.write_u16(32765)  # Max frame size, no idea what it is

        if self.track is None:
            self.write_u16(0)  # Freq
            self.write_u8(0)  # Channels
            self.write_u8(0)  # Sound resolution
        else:
            self.write_u16(self.track.rate)
            self.write_u8(self.track.num_channels())
            self.write_u8(self.track.bits_per_channel())

        self.write_u32(0)  # Unknown3
        self.write_u16(4)  # Unknown4, always 4

        # Compute and store the largest  CBFZ chunk size in this file
        max_cbfz_size = 0
        for _, _, size in self.cbfz_chunk_infos:
            max_cbfz_size = max(max_cbfz_size, size)
        self.write_u32(max_cbfz_size)
        self.write_u32(0)  # Unknown 5

        cbnum = len(self.cbfz_chunk_infos)

        # Loop information header (LINH) and data (LIND)
        with HiColorVQAEncoder.Chunk(b"LINF", self.file) as _:
            with HiColorVQAEncoder.Chunk(b"LINH", self.file) as _:
                self.write_u16(cbnum)
                self.write_u16(2)
                self.write_u16(0)

            with HiColorVQAEncoder.Chunk(b"LIND", self.file) as _:
                for framenum, ofs, size in self.cbfz_chunk_infos:
                    self.write_u16(framenum)
                    self.write_u16(header.numframes - 1)

        # The 'spread' number seems to be the minimum number of frames between codebook updates.
        spread = header.numframes
        if cbnum > 1:
            for i in range(0, len(self.cbfz_chunk_infos) - 1):
                spread = min(
                    spread,
                    self.cbfz_chunk_infos[i + 1][0] - self.cbfz_chunk_infos[i][0],
                )

        # Codebook information header (CINH) and data (CIND)
        with HiColorVQAEncoder.Chunk(b"CINF", self.file) as _:
            with HiColorVQAEncoder.Chunk(b"CINH", self.file) as _:
                self.write_u16(cbnum)
                self.write_u16(spread)
                self.write_u32(0)

            # Write sizes of all codebook chunks.
            with HiColorVQAEncoder.Chunk(b"CIND", self.file) as _:
                for framenum, ofs, size in self.cbfz_chunk_infos:
                    self.write_u16(framenum)
                    self.write_u32(size)

        # The frame info chunk has absolute offsets to each frame's data.
        # First write a placeholder, measure its size and then seek back and write over the real offsets.
        with HiColorVQAEncoder.Chunk(b"FINF", self.file) as finf_data_start:
            for _ in self.frame_data_offsets:
                self.write_u32(0)

        total_headers_size = self.file.tell()

        # Write adjusted offsets because frames have been moved forward because we wrote the header at the beginning.
        # Frame offsets in FINF are always 16-bit aligned so the last bit gets dropped here
        self.file.seek(finf_data_start, os.SEEK_SET)
        for ofs in self.frame_data_offsets:
            finf_addr = ofs + total_headers_size
            assert finf_addr % 2 == 0
            self.write_u32(finf_addr // 2)

        self.file.seek(total_headers_size, os.SEEK_SET)

        # Rewrite the actual video data after the header
        self.file.write(streamdata)

        filesize = self.file.tell()

        # Write total file size after the FORM chunk
        self.file.seek(4, os.SEEK_SET)
        self.write_u32be(filesize - 8)
        self.file.seek(filesize, os.SEEK_SET)
