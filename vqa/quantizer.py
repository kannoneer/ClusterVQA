"""
A clustering vector quantizer.

Reads some frames in a buffer, splits them into blocks, converts those blocks to an array of
vectors 'X', finds a codebook by clustering that matrix 'X' with K-means, replaces each block with
a matching index into the codebook and finally passes the frame and the codebook to an Encoder
that actually writes to a file.
"""

import time
from typing import List
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

import vqa.runtime_config
from .video_types import *
from .block_ops import *
from .encoders import Encoder
from .conversions import (
    rgb2perceptual,
    make_4x4_dither_table,
    round_rgb_float_to_rgb555,
    expand_rgb555_to_rgb888,
)


def log_verbose(msg: str):
    if vqa.runtime_config.verbose:
        print(msg)


def load_float_rgb(path):
    img = Image.open(path)
    img_int = np.array(img)[:, :, :3]
    return img_int.astype(float) / 255


def load_and_prepare_frame(path: str, cfg: Config, convert_to_yuv=True):
    img_rgb = load_float_rgb(path)

    dither = make_4x4_dither_table(img_rgb.shape[:2])
    dither = np.expand_dims(dither, 2)

    if cfg.dither_strength > 0.0:
        img_rgb = np.clip(img_rgb + cfg.dither_strength * dither / 256.0, 0, 1)

    if cfg.quantize_before_block_search:
        img_rgb = expand_rgb555_to_rgb888(round_rgb_float_to_rgb555(img_rgb)) / 255.0

    if convert_to_yuv:
        img_adjusted = rgb2perceptual(img_rgb)
        # print(f"input min/max {img_rgb.min()} {img_rgb.max()} adjusted min/max {img_adjusted.min()} {img_adjusted.max()}")
    else:
        img_adjusted = img_rgb
    img_in, blocksy, blocksx = crop_to_blocks(img_adjusted, cfg)
    return RawFrame(img_in, blocksx, blocksy)


def fit_clusters(
    X: np.ndarray, num_clusters: int, max_vectors_to_fit: int, deduplicate=True, seed=0
):
    counts = None
    if deduplicate:
        orig_size = X.shape[0]
        X, counts = np.unique(X, axis=0, return_counts=True)
        log_verbose(f"Deduped {orig_size} vectors into {X.shape[0]}")

    if X.shape[0] > max_vectors_to_fit:
        sorted_indices = np.argsort(counts)
        keep_these = sorted_indices[-max_vectors_to_fit:]
        Xsample = X[keep_these]
        counts = counts[keep_these]
        print(
            f"Subsampled clustering input to {(Xsample.shape[0]/X.shape[0])*100:.3f} % most common vectors"
        )
    else:
        Xsample = X

    kmeans = MiniBatchKMeans(
        n_clusters=min(num_clusters - 1, Xsample.shape[0]),
        n_init=3,
        init="random",
        reassignment_ratio=0.02,
        random_state=seed,
        compute_labels=False,
    ).fit(Xsample, None, counts)
    log_verbose(
        f"K-Means ran for {kmeans.n_iter_} iterations and {kmeans.n_steps_} minibatches"
    )

    # Add a black block because it's often needed even if the clustering didn't produce it exactly.
    # The (0, 0, 0) triplet means black also in YUV.
    clusters = kmeans.cluster_centers_
    clusters = np.resize(clusters, (clusters.shape[0] + 1, clusters.shape[1]))
    clusters[-1, :] = 0
    return clusters


def sort_and_prune_clusters(X: np.ndarray, clusters: np.ndarray):
    """Sorts most used clusters to the front and drops unused ones."""
    bigtree = KDTree(clusters, leaf_size=30, metric="euclidean")
    labels = bigtree.query(X, k=1, return_distance=False)
    label_indices, blocks_per_code = np.unique(labels, return_counts=True)
    in_use = np.flip(label_indices[np.argsort(blocks_per_code)])
    log_verbose(
        f"Codebook size after pruning: {in_use.shape[0]} entries ({in_use.shape[0]/float(clusters.shape[0])*100:.2f} % of original)"
    )
    return clusters[in_use]


def build_cluster_tree(clusters: np.ndarray):
    return KDTree(clusters, leaf_size=30, metric="euclidean")


def encode_video(frame_paths: List[str], fps: float, cfg: Config, encoder: Encoder):
    first_test_frame = load_and_prepare_frame(frame_paths[0], cfg)
    blocks_per_frame = first_test_frame.blocks_per_frame()
    num_frames = len(frame_paths)

    header = VideoHeader(
        first_test_frame.yuv.shape[1],
        first_test_frame.yuv.shape[0],
        num_frames,
        fps,
        cfg,
    )

    del first_test_frame

    blocksy = header.height // cfg.blockh
    blocksx = header.width // cfg.blockw

    codebook = None

    encoder.write_header(header)

    encode_start_time = time.time()

    frame_idx = 0
    frames_written = 0

    error_history = []
    input_queue = {}  # frame_idx -> raw

    def log_info(msg):
        print(f"[{frames_written}/{len(frame_paths)}, {frame_idx} read] {msg}")

    while frames_written < len(frame_paths):
        # The input queue should hold the last frame, this one and future ones but not anything older than that.
        input_queue = {
            idx: raw for idx, raw in input_queue.items() if idx >= frame_idx - 1
        }

        log_info(f"Reading source images")

        output_queue = []

        while len(output_queue) < cfg.max_keyframe_distance and frame_idx < len(
            frame_paths
        ):
            if frame_idx in input_queue:
                raw = input_queue[frame_idx]
            else:
                raw = load_and_prepare_frame(frame_paths[frame_idx], cfg)
                input_queue[frame_idx] = raw

            if (frame_idx - 1) in input_queue:
                last_raw = input_queue[frame_idx - 1]
                mse_yuv = np.mean((last_raw.yuv - raw.yuv) ** 2)
                error_history.append(mse_yuv)

                # Allow a scene cut to happen only if "min keyframe distance" frames have passed since the last codebook.
                if len(output_queue) >= cfg.min_keyframe_distance:
                    error_history = error_history[1:]
                    running_error = max(1e-6, sum(error_history) / len(error_history))
                else:
                    running_error = 1e9

                mse_ratio = mse_yuv / running_error
                log_verbose(
                    f"MSE to last = {mse_yuv:.3f}, running MSE: {running_error:.3f}, ratio: {mse_ratio:.4f}"
                )

                is_first_frame_of_window = len(output_queue) == 0
                if (
                    mse_ratio > cfg.scene_cut_error_limit
                    and not is_first_frame_of_window
                ):
                    log_info(f"Scene cut {mse_yuv:.4f} > {running_error:.4f}!")
                    break

            Xi = np.zeros((blocks_per_frame, cfg.vector_size()), np.float32)
            split_frame(raw.yuv, cfg.blockw, cfg.blockh, Xi)
            output_queue.append((raw, Xi))

            frame_idx += 1

        def to_frame_shape(Xi):
            return Xi.reshape(blocksy, blocksx, cfg.vector_size())

        X = np.zeros(
            (blocks_per_frame * len(output_queue), cfg.vector_size()), np.float32
        )

        _, X0 = output_queue[0]
        X0 = to_frame_shape(X0)

        framebuffer = X0.copy()
        accum_frame_error = np.zeros((blocksy, blocksx))
        frame_update_masks = np.zeros(
            (len(output_queue), X0.shape[0], X0.shape[1]), dtype=np.bool
        )
        # always write the first frame
        X[0:blocks_per_frame] = X0.reshape(-1, cfg.vector_size())
        frame_update_masks[0] = True
        num_visible_blocks = X0.shape[0] * X0.shape[1]

        for i in range(1, len(output_queue)):
            _, Xi = output_queue[i]
            Xi = to_frame_shape(Xi)
            per_block_error = ((Xi - framebuffer) ** 2).mean(2)
            accum_frame_error += per_block_error

            stale_blocks = accum_frame_error > cfg.block_replacement_error

            newblocks = Xi[stale_blocks].reshape(-1, cfg.vector_size())
            X[
                num_visible_blocks : (num_visible_blocks + newblocks.shape[0])
            ] = newblocks
            num_visible_blocks += newblocks.shape[0]
            frame_update_masks[i] = stale_blocks
            framebuffer[stale_blocks] = Xi[stale_blocks]
            accum_frame_error[stale_blocks] = 0

        X = X[:num_visible_blocks]

        log_info(f"Fitting {cfg.num_blocks} clusters to {X.shape} array")
        start = time.time()
        clusters = fit_clusters(
            X,
            num_clusters=cfg.num_blocks,
            max_vectors_to_fit=cfg.max_vectors_to_fit,
            deduplicate=True,
            seed=frames_written,
        )
        log_info(f"Took {time.time() - start:.4} s")
        clusters = sort_and_prune_clusters(X, clusters)

        # We need a separate KDTree for block-to-code assignment because we rearranged clusters so the K-means object isn't valid anymore.
        tree = build_cluster_tree(clusters)

        codebook = Codebook(clusters.astype(np.float32))
        encoder.write_codebook(codebook)

        for i, (raw, Xi) in enumerate(output_queue):

            indices = tree.query(Xi, k=1, return_distance=False)
            indices = indices.reshape(raw.blocksy, raw.blocksx)

            stale_blocks = frame_update_masks[i]
            # wipe clear blocks that our encoder shouldn't be seeing
            indices[np.bitwise_not(stale_blocks)] = -1

            num_stale = np.sum(stale_blocks)
            log_info(
                f"Writing a frame with {num_stale} ({(num_stale / float(indices.size))*100:.2f} %) changed blocks"
            )
            encoder.write_frame(indices, stale_blocks)

            elapsed = time.time() - encode_start_time
            frames_written += 1

        secs_per_frame = elapsed / frames_written
        remaining = secs_per_frame * (num_frames - frames_written)
        log_info(f"{int(remaining/60):02d}:{int(remaining)%60:02d} remaining")

    encoder.finalize()

    total_took = time.time() - encode_start_time
    print(
        f"Encoding {num_frames} frames took {int(total_took/60)} m {int(total_took)%60} s, ",
        end="",
    )
    print(f"{total_took/num_frames:.3f} s per frame on average")
