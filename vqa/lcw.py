"""
Routines for Westwood LCW ("format80") compression.
See the description at https://moddingwiki.shikadi.net/wiki/Westwood_LCW
"""
import time


def decompress_lcw(buf, expected_output=None):
    """
    Decompresses an LCW compressed buffer.

    'expected_output' is a debugging feature to verify compression output
    """

    newformat = buf[0] == 0
    input_size = len(buf)

    start = 1 if newformat else 0
    inpos = start
    output = bytearray()
    last_dp = 0

    def read():
        nonlocal inpos
        x = buf[inpos]
        inpos += 1
        return x

    while inpos - start < input_size:
        a = read()

        if a == 0x80:
            break

        if (a & 0x80) == 0:  # Command 2
            count = (a >> 4) + 3
            relpos = ((a & 0x0F) << 8) | read()
            src = len(output) - relpos
            for i in range(src, src + count):
                output.append(output[i])
        else:
            if a == 0xFE:  # Command 4
                count = read()
                count |= read() << 8
                value = read()
                for i in range(count):
                    output.append(value)
            elif a == 0xFF:  # Command 5
                count = read()
                count |= read() << 8
                src = read()
                src |= read() << 8
                if newformat:
                    src = len(output) - src
                for i in range(src, src + count):
                    output.append(output[i])
            elif (a >> 6) == 2:  # Command 1
                count = a & 0x3F
                for i in range(count):
                    output.append(read())
            elif (a >> 6) == 3:  # Command 3
                count = (a & 0x3F) + 3
                src = read()
                src |= read() << 8
                if newformat:
                    src = len(output) - src
                for i in range(src, src + count):
                    output.append(output[i])
            else:
                raise Exception(f"format80 command {a} not handled")

        if expected_output is not None:
            for i in range(last_dp, len(output)):
                assert output[i] == expected_output[i]

        last_dp = len(output)

    return output


def compress_lcw(A, newformat=False):
    N = len(A)
    output = bytearray()
    literal_run = bytearray()

    if newformat:
        output.append(0)

    def flush_literals():
        nonlocal output
        nonlocal literal_run

        if len(literal_run) == 0:
            return

        # Emit command 1
        count = len(literal_run)
        assert count <= 0x3F
        output.append(0x80 | count)
        output += literal_run
        literal_run = bytearray()

    def emit_literal(x):
        literal_run.append(x)
        if len(literal_run) == 0x3F:
            flush_literals()

    def emit_ref(sp, start, count):
        flush_literals()
        relofs = sp - start
        if newformat:
            # Actually a relative offset if encoding using new format
            absofs = sp - start
        else:
            absofs = start

        if count <= 10 and relofs <= 0xFFF:
            # Emit command 2
            a = ((count - 3) << 4) | ((relofs >> 8) & 0x0F)
            b = relofs & 0xFF
            output.append(a)
            output.append(b)
        elif count <= 64 and absofs <= 0xFFFF:
            # Emit command 3
            assert count - 3 <= 0x3F
            a = 0xC0 | ((count - 3) & 0x3F)
            assert (
                a != 0xFF and a != 0xFE
            )  # commands 0xff and 0xfe are reserved for commands 4 and 5
            output.append(a)
            output.append(absofs & 0xFF)
            output.append(absofs >> 8)
        else:
            raise Exception(f"run ({relofs}, {count}) couldn't be encoded")

    MAX_REF_DISTANCE = 4095
    SEARCH_WINDOW = 8
    MAX_RUN_LENGTH = 64
    sp = 0

    # TODO non-greedy match search
    # TODO accelerating search like in LZ4
    # TODO implement commands 4 and 5

    seen = {}  # maps bytes -> start offset

    while sp < N:
        if sp < N - 4:
            part = A[sp : sp + 4]
            if part in seen:
                ofs = seen[part]
                seen[part] = sp
                if sp - ofs <= MAX_REF_DISTANCE or not newformat:
                    cur = sp + len(part)
                    j = ofs + len(part)
                    while cur < N and (j - ofs) < MAX_RUN_LENGTH and A[cur] == A[j]:
                        cur += 1
                        j += 1
                    count = j - ofs

                    # print(f"[{sp}] found a cached match ({ofs}, {count})")

                    emit_ref(sp, ofs, count)
                    sp += count
                    continue
            else:
                seen[part] = sp

        for i in range(sp - 1, max(0, sp - SEARCH_WINDOW) - 1, -1):
            cur = sp
            j = i
            while cur < N and (j - i) < MAX_RUN_LENGTH and A[cur] == A[j]:
                cur += 1
                j += 1
            count = j - i
            if count >= 3:
                emit_ref(sp, i, count)
                sp += count
                break
        else:
            emit_literal(A[sp])
            sp += 1

    flush_literals()
    output.append(0x80)

    return bytes(output)


def test_decompression():
    cases = [
        "testdata/chunks/cbfz1.bin",
        "testdata/chunks/relative1.bin",
        "testdata/chunks/vprz1.bin",
        "testdata/chunks/vprz2.bin",
    ]

    for path in cases:
        print(path)
        with open(path, "rb") as f:
            compressed = f.read()
        with open(path + ".expected", "rb") as f:
            expected = f.read()

        decompressed = decompress_lcw(compressed, expected)

        assert len(decompressed) == len(expected)
        for i in range(len(expected)):
            assert decompressed[i] == expected[i]

        print("OK")


def test_compression():
    cases = [
        b"..x.xx.xx.xx.xx.xx",
        b"abaaaaa",
        open("testdata/chunks/vprz1.bin.expected", "rb").read(),
        open("testdata/chunks/vprz2.bin.expected", "rb").read(),
        open("testdata/chunks/cbfz1.bin.expected", "rb").read(),
    ]

    for idx, case in enumerate(cases):
        print(f"Case {idx}")
        start = time.time()
        compressed = compress_lcw(case, len(case) > 0xFFFF)
        after_compression = time.time()
        decompressed = decompress_lcw(compressed, case)
        after_decompression = time.time()

        assert len(case) == len(decompressed)
        for i in range(len(case)):
            assert decompressed[i] == case[i]

        ratio = len(compressed) / len(case)
        compression_speed = len(case) / max(1e-6, after_compression - start)
        decompression_speed = len(compressed) / max(
            1e-6, after_decompression - after_compression
        )
        print(
            f"{ratio*100:.2f} % of original, Comp: {compression_speed/1024.:.2f} KiB/sec, decomp: {decompression_speed/1024.:.2f} KiB/s"
        )


if __name__ == "__main__":
    test_decompression()
    test_compression()
