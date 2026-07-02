"""
Broadcasting — basic rules.

Broadcasting lets NumPy apply operations between arrays of different but
compatible shapes without copying data.  The framework "stretches" size-1
axes to match the other array.

Rules (applied right-to-left on the shape tuples):
  1. Prepend 1s to the shorter shape until both have the same ndim.
  2. For each axis: sizes must be equal, or one of them must be 1.
  3. The output size along each axis is the maximum of the two.
  4. If sizes are incompatible, a ValueError is raised.

Examples:
    (3,)    + (3,)    → (3,)       same shape
    (3, 1)  + (1, 4)  → (3, 4)    both axes stretched
    (2, 3)  + (3,)    → (2, 3)    (3,) treated as (1, 3), row broadcast

A common pitfall — this fails:
    (2, 3) + (2,)   →  ERROR     because right-to-left: 2 vs 3 mismatch

The fix is to reshape (2,) → (2, 1) so it aligns as a column:
    (2, 3) + (2, 1) → (2, 3)   ✓

In these problems you will need to figure out the right reshape yourself.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np


def add_bias(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Adds a bias vector to every row of a matrix.

    Arguments:
        X -- np.ndarray of shape (n, d)
        b -- np.ndarray of shape (d,)

    Returns:
        np.ndarray of shape (n, d) where output[i] = X[i] + b
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def scale_columns(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Multiplies each column j of X by the scalar s[j].

    Arguments:
        X -- np.ndarray of shape (n, d)
        s -- np.ndarray of shape (d,)

    Returns:
        np.ndarray of shape (n, d) where output[:, j] = X[:, j] * s[j]
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def scale_rows(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Multiplies each row i of X by the scalar s[i].

    Arguments:
        X -- np.ndarray of shape (n, d)
        s -- np.ndarray of shape (n,)

    Returns:
        np.ndarray of shape (n, d) where output[i] = X[i] * s[i]

    Hint: s has shape (n,) but X has shape (n, d).  A bare multiply will
          fail.  What reshape makes s broadcast correctly against X?
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def standardize(X: np.ndarray) -> np.ndarray:
    """
    Z-score standardises each column of X:

        Z[i, j] = (X[i, j] - mean_j) / std_j

    where mean_j and std_j are computed column-wise (ddof=0).
    Columns with zero standard deviation are left unchanged (divide by 1).

    Arguments:
        X -- np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (n, d)

    Raises:
        ValueError -- if X is not 2-D
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def pairwise_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the matrix of all pairwise differences D[i, j] = x[i] - y[j].

    Arguments:
        x -- np.ndarray of shape (m,)
        y -- np.ndarray of shape (n,)

    Returns:
        np.ndarray of shape (m, n)

    Raises:
        ValueError -- if x or y is not 1-D
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


if __name__ == "__main__":
    print("=== Broadcasting ===\n")

    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.array([10.0, 20.0, 30.0])
    print("add_bias:")
    print(add_bias(X, b))

    s_col = np.array([0.0, 1.0, 2.0])
    print("\nscale_columns (s=[0,1,2]):")
    print(scale_columns(X, s_col))

    s_row = np.array([2.0, 0.5])
    print("\nscale_rows (s=[2,0.5]):")
    print(scale_rows(X, s_row))

    print("\nstandardize:")
    print(standardize(X))

    x1d = np.array([0.0, 1.0, 2.0])
    y1d = np.array([0.0, 10.0])
    print("\npairwise_differences([0,1,2], [0,10]):")
    print(pairwise_differences(x1d, y1d))
