"""
Prints offsets and names of top level VQA chunks.
"""
import struct
import os
import sys

frame = 0


def print_linf(data):
    def read(n):
        nonlocal data
        x = data[:n]
        data = data[n:]
        return x

    print("LINF chunk contents:")

    assert read(4) == b"LINH"
    linhsize = struct.unpack(">L", read(4))[0]
    assert linhsize == 6
    cbnum = struct.unpack("<H", read(2))[0]
    val1 = struct.unpack("<H", read(2))[0]
    read(2)
    print("  cbnum:", cbnum)
    print("  val1:", val1)
    assert read(4) == b"LIND"
    lindsize = struct.unpack(">L", read(4))[0]
    for i in range(cbnum):
        a = struct.unpack("<H", read(2))[0]
        b = struct.unpack("<H", read(2))[0]
        print(f"  {i}: {a}, {b}")


def print_cinf(data):
    def read(n):
        nonlocal data
        x = data[:n]
        data = data[n:]
        return x

    print("CINF chunk contents:")

    assert read(4) == b"CINH"
    linhsize = struct.unpack(">L", read(4))[0]
    assert linhsize == 8
    cbnum = struct.unpack("<H", read(2))[0]
    spread = struct.unpack("<H", read(2))[0]
    read(4)
    print("  cbnum:", cbnum)
    print("  spread:", spread)
    assert read(4) == b"CIND"
    cindsize = struct.unpack(">L", read(4))[0]
    for i in range(cbnum):
        a = struct.unpack("<H", read(2))[0]
        b = struct.unpack("<L", read(4))[0]
        print(f"  {i}: {a}, {b}")


def print_finf(data):
    def read(n):
        nonlocal data
        x = data[:n]
        data = data[n:]
        return x

    print("FINF chunk contents:")

    frame = 0
    while len(data) > 0:
        addr = 2 * struct.unpack("<L", read(4))[0]
        print(f"  {frame}: 0x{addr:08x}")
        frame += 1


def print_header(data):
    def read(n):
        nonlocal data
        x = data[:n]
        data = data[n:]
        return x

    def read_u8():
        return struct.unpack("B", read(1))[0]

    def read_u16():
        return struct.unpack("<H", read(2))[0]

    def read_u32():
        return struct.unpack("<L", read(4))[0]

    def read_u32be():
        return struct.unpack(">L", read(4))[0]

    print("Header:")

    print(f"  Version: {read_u16()}")
    print(f"  Flags: {read_u16()}")
    print(f"  Num frames:\t{read_u16()}")
    print(f"  Image width:\t{read_u16()}")
    print(f"  Image height:\t{read_u16()}")
    print(f"  Block width:\t{read_u8()}")
    print(f"  Block height:\t{read_u8()}")
    print(f"  FPS:\t{read_u8()}")
    print(f"  Group size:\t{read_u8()}")
    print(f"  Colors:\t{read_u16()}")
    print(f"  Max blocks:\t{read_u16()}")
    print(f"  X pos:\t{read_u16()}")
    print(f"  Y pos:\t{read_u16()}")
    print(f"  Max framesize:\t{read_u16()}")

    print(f"  Sound sample rate:\t{read_u16()}")
    print(f"  Sound channels:\t{read_u8()}")
    print(f"  Sound bits per sample:\t{read_u8()}")

    print(f"  Alt sample rate:\t{read_u16()}")
    print(f"  Alt channels:\t{read_u8()}")
    print(f"  Alt bits per sample:\t{read_u8()}")
    print(f"  Color mode:\t{read_u8()}")
    print(f"  Unknown6:\t{read_u8()}")
    print(f"  Largest CBFZ:\t{read_u32()}")
    print(f"  Unknown7:\t{read_u32()}")


have_subchunks = set([b"VQFR", b"VQFL"])
handlers = {
    b"LINF": print_linf,
    b"CINF": print_cinf,
    b"FINF": print_finf,
    b"VQHD": print_header,
}


def print_chunks(f, endpoint, level=0):
    global frame
    ofs = f.tell()
    if ofs >= endpoint:
        return False
    chunkid = f.read(4)
    if len(chunkid) < 4:
        return False
    chunksize = struct.unpack(">L", f.read(4))[0]
    print(
        f"{ofs:08x} [{frame: 3d}] {'    ' * level} {chunkid.decode('ascii')} ({chunksize} bytes)"
    )

    if chunkid == b"VPRZ" or chunkid == b"VPTR":
        frame += 1

    if chunkid in have_subchunks:
        while print_chunks(f, ofs + 8 + chunksize, level + 1):
            pass
    else:
        data = f.read(chunksize)
        if chunkid in handlers:
            handlers[chunkid](data)

        if chunksize % 2 == 1:
            f.seek(1, os.SEEK_CUR)

    return True


def print_file(path):
    global frame
    frame = 0

    with open(path, "rb") as f:
        form = f.read(4)
        assert form == b"FORM"
        size = struct.unpack(">L", f.read(4))[0]
        wvqa = f.read(4)
        assert wvqa == b"WVQA"

        while print_chunks(f, f.tell() + size):
            pass


if __name__ == "__main__":
    print_file(sys.argv[1])
