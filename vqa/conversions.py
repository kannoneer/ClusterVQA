import numpy as np

USE_YUV_COLORSPACE = True

if USE_YUV_COLORSPACE:
    from skimage.color import rgb2yuv, yuv2rgb

# Weight channels via rec601 luma formula to get some kind of perceptual distance computations.
RGB_WEIGHTS = np.array([[[0.299, 0.587, 0.114]]])


def rgb2perceptual(rgb: np.ndarray):
    assert len(rgb.shape) == 3
    if USE_YUV_COLORSPACE:
        return rgb2yuv(rgb) * 100
    else:
        return rgb * RGB_WEIGHTS * 100


def perceptual2rgb(perceptual: np.ndarray):
    assert len(perceptual.shape) == 3
    if USE_YUV_COLORSPACE:
        return yuv2rgb(perceptual) / 100
    else:
        return perceptual / 100 / RGB_WEIGHTS


def round_rgb_float_to_rgb555_slow(rgbfloat: np.ndarray):
    return np.clip(np.round(rgbfloat * 31.5).astype(int), 0, 31)


def round_rgb_float_to_rgb555(rgbfloat: np.ndarray):
    return np.clip((rgbfloat * 31.5 + 0.5).astype(int), 0, 31)


def expand_rgb555_to_rgb888(rgb555: np.ndarray):
    # Repeat the the value in the lower bits: ___ABCDE -> ABCDEABC
    return np.bitwise_or(np.right_shift(rgb555, 2), np.left_shift(rgb555, 3))


# Threshold maps for ordered dithering.
# From Wikipedia: https://en.wikipedia.org/w/index.php?title=Ordered_dithering&oldid=1062432580#Threshold_map
threshold_map_2x2 = -0.5 + (1.0 / 4) * np.array([[0, 2], [3, 1]])
threshold_map_8x8 = -0.5 + (1.0 / 16.0) * np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
)


def make_2x2_dither_table(output_shape):
    h, w = output_shape
    tiled = np.tile(threshold_map_2x2, ((h + 1) // 2, (w + 1) // 2))
    return tiled[:h, :w]


def make_4x4_dither_table(output_shape):
    h, w = output_shape
    tiled = np.tile(threshold_map_8x8, ((h + 3) // 4, (w + 3) // 4))
    return tiled[:h, :w]


if __name__ == "__main__":
    values = np.array([0, 64, 127, 128, 254, 255])
    floats = values / 255.0
    quant_ref = round_rgb_float_to_rgb555_slow(floats)
    quant = round_rgb_float_to_rgb555(floats)
    print(values)
    print(quant)
    expanded_ref = expand_rgb555_to_rgb888(quant_ref)
    expanded = expand_rgb555_to_rgb888(quant)
    error = np.abs(expanded - values)
    error_ref = np.abs(expanded_ref - values)

    print("error    :", error)
    print("error_ref:", error_ref)
    assert (error == error_ref).all()

    assert make_2x2_dither_table((5, 4)).shape == (5, 4)
    assert make_4x4_dither_table((5, 4)).shape == (5, 4)

    import matplotlib.pyplot as plt

    plt.plot(values, label="Input")
    plt.plot(expanded_ref, label="Output (reference)")
    plt.plot(expanded, label="Output (fast rounding)")
    plt.plot(error, label="Error")
    plt.suptitle("Integer quantization and expansion test")
    plt.legend()
    plt.show()
