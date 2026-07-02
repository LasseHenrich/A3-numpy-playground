"""
Transpose and axis permutation.

The transpose of a matrix swaps rows and columns:

    A.shape == (m, n)   →   A.T.shape == (n, m)
    A.T[i, j] == A[j, i]

For higher-dimensional arrays, np.transpose(A, axes) permutes axes:

    A.shape == (2, 3, 4)
    A.transpose(2, 0, 1).shape == (4, 2, 3)

`.T` is a shorthand that reverses all axes.

Transpose is NOT covered in the lecture — but the building blocks are:
it is a movement op (O(1), no data copy) that reorders how indices map
to memory.  The problems below require you to work out the right axis
order yourself.

A common mistake: .T on a 1-D array does nothing (a vector has only one
axis to reverse).  Use reshape(-1, 1) to turn a 1-D array into a column.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np


def gram_matrix(X: np.ndarray) -> np.ndarray:
    """
    Computes the Gram matrix G where G[i, j] is the inner product of
    rows i and j of X.

    If X has shape (n, d), the result has shape (n, n).

    Arguments:
        X -- np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (n, n), symmetric positive semi-definite

    Raises:
        ValueError -- if X is not 2-D
    """

    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    
    return X @ X.T


def scatter_matrix(X: np.ndarray) -> np.ndarray:
    """
    Computes the scatter matrix (unnormalised covariance):

        S = (X - mean).T @ (X - mean)

    where mean is computed column-wise (i.e. the mean of each feature).
    S has shape (d, d) and is proportional to the sample covariance matrix
    (divide by n-1 for Bessel-corrected covariance).

    Arguments:
        X -- np.ndarray of shape (n, d), n samples, d features

    Returns:
        np.ndarray of shape (d, d)

    Raises:
        ValueError -- if X is not 2-D
    """

    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    
    mean = np.mean(X, axis=0)
    return (X - mean).T @ (X - mean)


def batch_transpose(A: np.ndarray) -> np.ndarray:
    """
    Transposes a batch of matrices.

    Each matrix in the batch is transposed independently:
        output[b] = A[b].T

    Input  shape: (B, m, n)
    Output shape: (B, n, m)

    Arguments:
        A -- np.ndarray of shape (B, m, n)

    Returns:
        np.ndarray of shape (B, n, m)

    Raises:
        ValueError -- if A does not have exactly 3 dimensions
    """

    if A.ndim != 3:
        raise ValueError("A must be 3-D")
    
    return A.transpose(0, 2, 1)


def is_symmetric(A: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Returns True if A is symmetric (A == A.T) within tolerance tol.

    Arguments:
        A   -- np.ndarray of shape (n, n)
        tol -- absolute tolerance for element-wise comparison

    Returns:
        bool

    Raises:
        ValueError -- if A is not 2-D or not square
    """

    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    
    return np.allclose(A, A.T, atol=tol)


def column_outer_products(X: np.ndarray) -> np.ndarray:
    """
    Given a matrix X of shape (n, d), computes the sum of outer products of
    its rows:

        M = sum_{i=0}^{n-1} X[i, :, None] @ X[i, None, :]

    This is the d×d matrix of pairwise feature dot-products — exactly the
    normal-equations matrix used in least-squares regression.

    Implement this as a single matrix expression (no loops).

    Arguments:
        X -- np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (d, d)

    Raises:
        ValueError -- if X is not 2-D
    """

    if X.ndim != 2:
        raise ValueError("X is not 2-D")
    
    return X.T @ X


if __name__ == "__main__":
    print("=== Transpose ===\n")

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    G = gram_matrix(X)
    print("gram_matrix (3×2) → (3×3):")
    print(G)
    print("symmetric:", is_symmetric(G))

    print("\nscatter_matrix (3×2) → (2×2):")
    print(scatter_matrix(X))

    A = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    print("\nbatch_transpose (2,3,4) →", batch_transpose(A).shape)
    print("A[0].T matches batch_transpose(A)[0]:", np.allclose(A[0].T, batch_transpose(A)[0]))

    print("\ncolumn_outer_products (3×2) → (2×2):")
    print(column_outer_products(X))
