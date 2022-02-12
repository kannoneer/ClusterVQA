"""
A Vector Quantization Example.

Run in the 'docs' directory:

    python how_it_works.py

This script compresses a single image into 4x2 tiles, or "vectors", and then decodes it.
The tile codebook is found with Mini Batch K-Means clustering. Read more at
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html

This is the same technique that 'quantizer.py' uses but without the YUV conversion,
delta compression and image sequence handling boilerplate. And only for a single frame.
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

if len(sys.argv) == 2:
    path = sys.argv[1]
else:
    path = "../testdata/rodents.png"

num_codes = 20  # Size of the codebook
tilew = 4  # Tile width in pixels
tileh = 2  # Tile height in pixels


def load_float_rgb(path):
    img = Image.open(path)
    img_int = np.array(img)[:, :, :3]
    img_float = img_int.astype(float) / 255
    return img_float


img_rgb = load_float_rgb(path)


def crop_to_tiles(img):
    h, w, _ = img.shape
    tilesy = h // tileh
    tilesx = w // tilew
    return img_rgb[: tilesy * tileh, : tilesx * tilew], tilesy, tilesx


# Round down the image size to a multiple of tile size
img_rgb_crop, tilesy, tilesx = crop_to_tiles(img_rgb)
croph, cropw, _ = img_rgb_crop.shape

# Build a big matrix 'X' of shape [num_tiles, pixels_per_tile * 3] that represents our input image for
# the clustering algorithm. Each row of the matrix is one unrolled 4x2 RGB tile as a 24-element vector.

blocks_per_frame = tilesy * tilesx

vector_size = tileh * tilew * 3
X = np.zeros((blocks_per_frame, vector_size), np.float32)

for y in range(tilesy):
    for x in range(tilesx):
        vec = img_rgb_crop[
            (y * tileh) : (y * tileh + tileh), (x * tilew) : (x * tilew + tilew), :
        ]
        vec = vec.reshape(-1)
        assert vec.size == vector_size
        X[y * tilesx + x, :] = vec

# Find the codebook with K-means.

kmeans = MiniBatchKMeans(n_clusters=num_codes, random_state=0).fit(X)
codebook = kmeans.cluster_centers_  # Representatives
codes = kmeans.predict(X)  # Tile -> codebook assignments

# Decode the codebook+codes combo into an RGB image for display.

decoded = np.zeros_like(img_rgb_crop)
for y in range(tilesy):
    for x in range(tilesx):
        tile_idx = y * tilesx + x
        code = codes[tile_idx]
        vec = codebook[code]
        assert vec.size == vector_size
        vec = vec.reshape((tileh, tilew, 3))
        decoded[
            (y * tileh) : (y * tileh + tileh), (x * tilew) : (x * tilew + tilew), :
        ] = vec

fig, ax = plt.subplots(1, 1)
ax.imshow(codes.reshape((tilesy, tilesx)))
ax.set_title("Tile assignments")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
axes = ax.reshape(-1)
axes[0].imshow(img_rgb_crop)
axes[1].imshow(decoded)
axes[0].set_title("Input")
axes[1].set_title(f"Quantized to {num_codes} {tilew}x{tileh} tiles")
plt.tight_layout()

plt.show()
