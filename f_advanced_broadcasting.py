"""
Advanced broadcasting — inserting axes with np.newaxis.

Sometimes arrays don't align for broadcasting automatically.  The fix is
to insert a size-1 axis with np.newaxis (alias: None) to make shapes line
up right-to-left.

    x.shape == (m,)   →   x[:, np.newaxis].shape == (m, 1)
    x.shape == (m,)   →   x[np.newaxis, :].shape == (1, m)

Example — scaling rows of a matrix:
    X.shape == (m, d),  s.shape == (m,)
    X * s[:, None]   # (m, d) * (m, 1) → (m, d)  ✓
    X * s            # (m, d) * (m,)   → error    ✗  (aligns right-to-left)

The key skill is: draw the shapes, align them right-to-left, and insert
newaxis dimensions where needed.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np


def outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the outer product of two 1-D arrays using broadcasting.
    Do NOT use np.outer.

    output[i, j] = a[i] * b[j]

    Arguments:
        a -- np.ndarray of shape (m,)
        b -- np.ndarray of shape (n,)

    Returns:
        np.ndarray of shape (m, n)

    Raises:
        ValueError -- if a or b is not 1-D
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def pairwise_squared_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Computes the matrix of squared Euclidean distances between every pair of
    rows, using broadcasting (no loops).

    D[i, j] = ||X[i] - Y[j]||^2 = sum_k (X[i,k] - Y[j,k])^2

    Arguments:
        X -- np.ndarray of shape (m, d)
        Y -- np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (m, n)

    Raises:
        ValueError -- if X and Y are not both 2-D or have different d
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def cosine_similarity_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Computes the cosine similarity between every pair of rows from X and Y.

    S[i, j] = (X[i] · Y[j]) / (||X[i]||_2 * ||Y[j]||_2)

    The result is a matrix of shape (m, n).

    Implement using matrix multiplication and broadcasting — no loops.

    Arguments:
        X -- np.ndarray of shape (m, d), rows are non-zero vectors
        Y -- np.ndarray of shape (n, d), rows are non-zero vectors

    Returns:
        np.ndarray of shape (m, n), values in [-1, 1]

    Raises:
        ValueError -- if X and Y are not 2-D or have different d
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Applies softmax row-wise to a 2-D matrix (numerically stable).

        softmax(Z)[i, j] = exp(Z[i,j]) / sum_k exp(Z[i,k])

    Use the max-subtraction trick to prevent overflow.

    Arguments:
        Z -- np.ndarray of shape (n, C)

    Returns:
        np.ndarray of shape (n, C), each row sums to 1.0

    Raises:
        ValueError -- if Z is not 2-D
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def broadcast_add_explain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Adds x and y, where x has shape (m, d) and y has shape (m,).

    A naive `x + y` will fail because NumPy aligns shapes right-to-left:
        x: (m, d)
        y:    (m,)
    The rightmost sizes are d vs m — a mismatch unless d == m.

    Reshape y so it broadcasts as a column vector instead of a row vector,
    then add.

    Arguments:
        x -- np.ndarray of shape (m, d)
        y -- np.ndarray of shape (m,)

    Returns:
        np.ndarray of shape (m, d) where output[i, :] = x[i, :] + y[i]

    Raises:
        ValueError -- if x.shape[0] != y.shape[0]
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


if __name__ == "__main__":
    print("=== Advanced Broadcasting ===\n")

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    print("outer_product([1,2,3], [10,20]):")
    print(outer_product(a, b))

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])
    print("\npairwise_squared_distances:")
    print(pairwise_squared_distances(X, Y))

    print("\ncosine_similarity_matrix:")
    print(cosine_similarity_matrix(X, Y))

    Z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    P = softmax(Z)
    print("\nsoftmax row sums:", P.sum(axis=1))

    x2d = np.ones((3, 4))
    y1d = np.array([1.0, 2.0, 3.0])
    print("\nbroadcast_add_explain (3,4) + (3,) as column:")
    print(broadcast_add_explain(x2d, y1d))
