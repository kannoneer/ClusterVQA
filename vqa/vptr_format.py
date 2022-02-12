"""
VPTR block map run length encoding (RLE) routines.

TODO use all commands during compression for higher efficiency
"""

import numpy as np


def decompress_vptr(buf, output: np.ndarray):
    sp = 0

    def read():
        nonlocal sp
        x = buf[sp]
        sp += 1
        return x

    N = len(buf)
    A = output.reshape(-1)

    cur = 0

    while sp < N:
        val = read()
        val |= read() << 8

        action = (val & 0xE000) >> 13
        if action == 0:
            count = val & 0x1FFF
            cur += count
        elif action == 1:
            block = val & 0xFF
            count = (((val >> 8) & 0x1F) + 1) << 1
            for i in range(count):
                A[cur] = block
                cur += 1
        elif action == 2:
            block = val & 0xFF
            count = (((val >> 8) & 0x1F) + 1) << 1
            A[cur] = block
            cur += 1
            for i in range(count):
                A[cur] = read()
                cur += 1
        elif action == 3 or action == 4:
            A[cur] = val & 0x1FFF
            cur += 1
        elif action == 5 or action == 6:
            count = read()
            for i in range(count):
                A[cur] = val & 0x1FFF
                cur += 1


def compress_vptr(buf: np.ndarray, validmask: np.ndarray):
    assert len(buf.shape) == 2
    assert buf.shape == validmask.shape
    assert buf.dtype == np.uint16 or buf.dtype == np.int32 or buf.dtype == np.int64
    assert validmask.dtype == bool
    output = bytearray()

    def write_uint(x):
        output.append(x & 0xFF)
        output.append(x >> 8)

    yblocks, xblocks = buf.shape

    for y in range(yblocks):
        x = 0
        while x < xblocks:
            skiplen = 0

            while x < xblocks and not validmask[y, x]:
                skiplen += 1
                x += 1

            if skiplen > 0:
                assert skiplen <= 0x1FFF
                write_uint(skiplen)

            runx = x
            while runx < xblocks and validmask[y, runx]:
                runx += 1

            runlen = runx - x
            if runlen > 0:
                for i in range(runlen):
                    write_uint(0x6000 | (buf[y, x] & 0x1FFF))
                    x += 1

    return bytes(output)


def test_vptr_encoding():
    out = np.zeros(5, np.uint16)
    decompress_vptr(bytearray([4, 0, 1, 0x60]), out)
    assert out[4] == 1
    assert out[0:3].max() == 0
    print(out)

    buf = np.array([[-1, -1, -1, 1, 2, 0], [0, 0, 3, 4, 5, -1]], dtype=np.int32)
    mask = np.array(
        [[False, False, False, True, True, True], [True, True, True, True, True, False]]
    )

    compressed = compress_vptr(buf, mask)
    output_buf = -np.ones(buf.shape, dtype=np.int32)
    decompress_vptr(compressed, output_buf)
    print(output_buf)
    diff = np.sum(np.abs(output_buf[mask] - buf[mask]))
    assert diff == 0

    from numpy.random import default_rng

    rng = default_rng(seed=0)
    for i in range(10):
        h, w = (1 + i * 20, 2 + i * 40)
        buf = rng.integers(0, 0x1000, (h, w), np.int32)
        mask = rng.integers(0, 2, (h, w)).astype(bool)
        buf[np.bitwise_not(mask)] = -1

        compressed = compress_vptr(buf, mask)
        output_buf = -np.ones(buf.shape, dtype=np.int32)
        decompress_vptr(compressed, output_buf)
        diff = np.sum(np.abs(output_buf[mask] - buf[mask]))
        print(f"[{i}] {w}x{h} diff: {diff}")
        assert diff == 0

    print("OK")


if __name__ == "__main__":
    test_vptr_encoding()
