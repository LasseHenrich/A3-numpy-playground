"""
Movement ops — reshape, ravel, transpose, axis permutation.

Movement ops change *how* a flat memory buffer is interpreted — shape,
stride, axis order — without copying any data.  They are O(1).

Key operations:
  x.reshape(new_shape)         change shape; total elements must match
  x.reshape(-1)  / x.ravel()  flatten to 1-D
  x.reshape(n, -1)             keep first axis, infer second
  x.T                          reverse all axes (shorthand for transpose)
  x.transpose(axes)            permute axes in the given order

The problems below require you to figure out *which* reshape / transpose
to apply — there is often exactly one shape that makes things work.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np


def batch_flatten(X: np.ndarray) -> np.ndarray:
    """
    Flattens each sample in a batch into a 1-D vector.

    The first axis is the batch axis (size B).  All remaining axes are the
    per-sample shape.  After flattening, each sample occupies one row.

    Example:
        X.shape == (32, 28, 28)   →   output.shape == (32, 784)
        X.shape == (4, 3, 8, 8)   →   output.shape == (4, 192)

    Arguments:
        X -- np.ndarray with ndim >= 2, shape (B, ...)

    Returns:
        np.ndarray of shape (B, D) where D = product of all non-batch dims

    Raises:
        ValueError -- if X has fewer than 2 dimensions
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def split_into_patches(X: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    """
    Divides a single 2-D image into a grid of non-overlapping patches using
    only reshape (no loops, no slicing).

    Input  shape: (H, W)
    Output shape: (H // patch_h, W // patch_w, patch_h, patch_w)

    The output[i, j] is the patch at grid position (i, j), as a 2-D array of
    shape (patch_h, patch_w) containing the original pixel values in row-major
    order.

    Hint: reshape to (nh, patch_h, nw, patch_w) first, then transpose.

    Arguments:
        X       -- np.ndarray of shape (H, W)
        patch_h -- height of each patch; must evenly divide H
        patch_w -- width  of each patch; must evenly divide W

    Returns:
        np.ndarray of shape (H // patch_h, W // patch_w, patch_h, patch_w)

    Raises:
        ValueError -- if X is not 2-D, or patch dimensions don't divide evenly
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def channels_first_to_last(X: np.ndarray) -> np.ndarray:
    """
    Converts an image batch from channels-first to channels-last format.

    PyTorch stores images as (B, C, H, W).
    TensorFlow / matplotlib expect   (B, H, W, C).

    Use only reshape / transpose — not np.moveaxis or np.rollaxis.

    Arguments:
        X -- np.ndarray of shape (B, C, H, W)

    Returns:
        np.ndarray of shape (B, H, W, C)

    Raises:
        ValueError -- if X does not have exactly 4 dimensions
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def interleave_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Given two matrices of the same shape (n, d), produces a (2n, d) matrix
    whose rows alternate: A[0], B[0], A[1], B[1], ..., A[n-1], B[n-1].

    Implement using stack + reshape — no loops, no concatenate.

    Arguments:
        A -- np.ndarray of shape (n, d)
        B -- np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (2*n, d)

    Raises:
        ValueError -- if A and B do not have the same shape or are not 2-D
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def tile_vector(v: np.ndarray, n: int) -> np.ndarray:
    """
    Repeats a 1-D vector v as the rows of a 2-D matrix n times,
    using only reshape and broadcast (no np.tile, no np.repeat).

    Output[i, :] == v  for all i in 0..n-1

    Arguments:
        v -- np.ndarray of shape (d,)
        n -- positive integer, number of rows

    Returns:
        np.ndarray of shape (n, d)

    Raises:
        ValueError -- if v is not 1-D or n < 1
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


if __name__ == "__main__":
    print("=== Movement Ops ===\n")

    X = np.arange(32 * 28 * 28).reshape(32, 28, 28)
    print("batch_flatten (32,28,28) →", batch_flatten(X).shape)

    img = np.arange(16).reshape(4, 4)
    patches = split_into_patches(img, 2, 2)
    print("split_into_patches 4×4 into 2×2 →", patches.shape)
    print("patch [0,0]:", patches[0, 0])
    print("patch [0,1]:", patches[0, 1])

    imgs = np.zeros((4, 3, 8, 8))
    print("\nchannels_first_to_last (4,3,8,8) →", channels_first_to_last(imgs).shape)

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[10.0, 20.0], [30.0, 40.0]])
    print("\ninterleave_rows:")
    print(interleave_rows(A, B))

    v = np.array([1.0, 2.0, 3.0])
    print("\ntile_vector([1,2,3], 4):")
    print(tile_vector(v, 4))
