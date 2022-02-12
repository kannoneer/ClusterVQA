import numpy as np
from .video_types import *


def crop_to_blocks(img, c: Config):
    h, w, _ = img.shape
    blocksy = h // c.blockh
    blocksx = w // c.blockw
    return img[: blocksy * c.blockh, : blocksx * c.blockw], blocksy, blocksx


def block_slice(x: int, y: int, blockw: int, blockh: int):
    return slice(y * blockh, y * blockh + blockh), slice(
        x * blockw, x * blockw + blockw
    )


def split_frame(frame: np.ndarray, blockw: int, blockh: int, out_X: np.ndarray):
    """
    Reorders 'frame' into an [HxW, num_vectors] array.
    This function's implementation is equivalent to the below loop.

        for y in range(blocksy):
            for x in range(blocksx):
                vec = frame[block_slice(x,y,c)]
                vec = vec.reshape(-1)
                out_X[y * blocksx + x, :] = vec

    """

    h, w, _ = frame.shape
    blocksy = h // blockh
    blocksx = w // blockw

    vector_size = blockh * blockw * 3

    f = frame.reshape(blocksy, blockh, blocksx, blockw, 3).transpose(0, 2, 1, 3, 4)
    vectors = f.reshape(f.shape[0] * f.shape[1], f.shape[2] * f.shape[3] * f.shape[4])
    assert vectors.shape == (blocksy * blocksx, vector_size)
    assert vectors.shape == out_X.shape
    np.copyto(out_X, vectors)


def merge_frame(
    indices: np.array,
    codebook: Codebook,
    blockw: int,
    blockh: int,
    out_decoded: np.ndarray,
):
    """
    This function's implementation is equivalent to the below loop but 16x faster.

        for y in range(blocksy):
            for x in range(blocksx):
                code = indices[y, x]
                vec = codebook.codes[code]
                vec = vec.reshape((c.blockh, c.blockw, 3))
                out_decoded[block_slice(x,y,c)] = vec

    """

    blocksy, blocksx = indices.shape
    # shape [blocksy x blocksx x blockh x blockw x 3]
    frame_blocks = codebook.codes[indices.reshape(-1)].reshape(
        blocksy, blocksx, blockh, blockw, 3
    )
    # shape [blocksy x blockh x blocksx x blockw x 3]
    frame = frame_blocks.transpose(0, 2, 1, 3, 4)
    fs = frame.shape
    # shape [width x height x 3]

    frame = frame.reshape(fs[0] * fs[1], fs[2] * fs[3], 3)
    assert frame.shape == out_decoded.shape
    np.copyto(out_decoded, frame)


def upscale_deltamask(mask, blockw, blockh):
    """
    Nearest neigbhor upscaling of the 'mask' 2D array with 'blockh' and 'blockw' in the respective vertical and horizontal directions.
    Returns a per-pixel mask with the shape (mask.shape[0] * blockh,  mask.shape[1] * blockw)
    """
    blocksy, blocksx = mask.shape
    repeated = np.tile(mask.reshape((*mask.shape, 1)), (1, 1, blockh * blockw))
    return (
        repeated.reshape(blocksy, blocksx, blockh, blockw)
        .transpose(0, 2, 1, 3)
        .reshape(blocksy * blockh, blocksx * blockw)
    )


def test_split(rng, blockw, blockh):
    """
    Test that 'split_frame()' is equivalent to the pseudocode in its comment.
    """

    frame = rng.uniform(low=0.0, high=1.0, size=(40, 20, 3))
    blocksy = frame.shape[0] // blockh
    blocksx = frame.shape[1] // blockw
    X = np.zeros((blocksy * blocksx, blockh * blockw * 3))
    split_frame(frame, blockw, blockh, X)

    X2 = np.zeros_like(X)

    for y in range(blocksy):
        for x in range(blocksx):
            vec = frame[block_slice(x, y, blockw, blockh)]
            vec = vec.reshape(-1)
            X2[y * blocksx + x, :] = vec

    assert (X == X2).all()


def test_block_ops():
    from numpy.random import default_rng

    c = Config()
    c.blockw = 4
    c.blockh = 2
    rng = default_rng(seed=0)
    test_split(rng, c.blockw, c.blockh)

    mask = upscale_deltamask(np.array([[1, 2], [3, 4]]), 2, 2)
    assert (
        mask == np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    ).all()
    mask = upscale_deltamask(np.array([[1, 2], [3, 4]]), 1, 2)
    assert (mask == np.array([[1, 2], [1, 2], [3, 4], [3, 4]])).all()

    print("Block ops tests OK")
