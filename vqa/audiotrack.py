import os
from dataclasses import dataclass


@dataclass
class AudioTrack:
    rate = 4800
    uncomp_size = 0
    data: bytes = None

    def num_channels(self):
        raise NotImplementedError()

    def bits_per_channel(self):
        raise NotImplementedError()

    def bytes_per_frame(self):
        return (2 if self.bits_per_channel() == 16 else 1) * self.num_channels()

    def length(self):
        "Clip length in seconds"
        return (self.uncomp_size / self.bytes_per_frame()) / self.rate


@dataclass
class RawS16LETrack(AudioTrack):
    rate = 4800
    uncomp_size = 0
    data: bytes = None
    is_stereo = True

    def num_channels(self):
        return 2 if self.is_stereo else 1

    def bits_per_channel(self):
        return 16


def load_raw_s16le_file(path: str, rate, is_stereo) -> RawS16LETrack:
    track = RawS16LETrack()

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0, os.SEEK_SET)
        track.data = f.read(size)
        track.uncomp_size = len(track.data)
        track.rate = rate
        track.is_stereo = is_stereo

        return track


def test_pcm_loading():
    raw = load_raw_s16le_file(
        "testdata/bleeps_22050_pcm_s16le.raw", rate=22050, is_stereo=True
    )
    print(f"raw length: {raw.length()} s")
    assert raw.length() > 3.0 and raw.length() < 3.1
    assert len(raw.data) == 269800
    print("Audio track test OK")


if __name__ == "__main__":
    test_pcm_loading()
