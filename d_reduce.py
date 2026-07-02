"""
Reduce ops — collapsing an array along one or more axes.

A reduce op walks the entire buffer and collapses a dimension to a scalar
(or a smaller array).  It is O(n).

NumPy reduce ops always take an `axis` argument.  The result has the same
number of dimensions as the input but the reduced axis has size 1 (if
keepdims=True) or is removed (if keepdims=False, the default).

Key operations:
    x.sum(axis=k)       sum along axis k
    x.max(axis=k)       maximum along axis k
    x.argmax(axis=k)    index of the maximum along axis k
    x.cumsum(axis=k)    cumulative sum along axis k

In these problems you are NOT allowed to use np.mean, np.var, np.std,
np.median, or np.percentile — compute them from their definitions using
sum, cumsum, max, and elementwise ops.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np


def row_mean(X: np.ndarray) -> np.ndarray:
    """
    Computes the mean of each row of a 2-D matrix without using np.mean.

    Arguments:
        X -- np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (n,)

    Raises:
        ValueError -- if X is not 2-D
    """
    if X.ndim != 2:
        raise ValueError("X must be 2-D")

    n, d = X.shape
    return np.sum(X, axis=1) / d

def column_variance(X: np.ndarray) -> np.ndarray:
    """
    Computes the Bessel-corrected sample variance of each column (ddof=1)
    without using np.var or np.std.

    Var_j = (1 / (n-1)) * sum_i (X[i,j] - mean_j)^2

    Arguments:
        X -- np.ndarray of shape (n, d), n >= 2

    Returns:
        np.ndarray of shape (d,)

    Raises:
        ValueError -- if X is not 2-D or has fewer than 2 rows
    """

    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    
    n, d = X.shape
    
    if n < 2:
        raise ValueError("X must have at least 2 rows")
    
    col_mean = np.sum(X, axis=0) / n
    return (1 / (n-1)) * np.sum((X - col_mean) ** 2, axis=0)


def log_sum_exp_rows(X: np.ndarray) -> np.ndarray:
    """
    Computes log(sum_j exp(X[i, j])) for each row i, stably.

    Use the max-subtraction trick to avoid overflow:
        LSE(row) = max(row) + log(sum_j exp(row_j - max(row)))

    Arguments:
        X -- np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (n,)

    Raises:
        ValueError -- if X is not 2-D
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def cumulative_row_mean(X: np.ndarray) -> np.ndarray:
    """
    For each row of X, computes the running (cumulative) mean from left to
    right.

    output[i, j] = mean(X[i, 0], X[i, 1], ..., X[i, j])

    Use cumsum — no loops.

    Arguments:
        X -- np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (n, d)

    Raises:
        ValueError -- if X is not 2-D
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def top_k_mask(x: np.ndarray, k: int) -> np.ndarray:
    """
    Returns a boolean mask of the same shape as x where True marks the k
    largest values (ties broken by position — leftmost wins).

    Example:
        x = [3, 1, 4, 1, 5, 9, 2, 6]  k=3  →  [F,F,F,F,F,T,F,T] ... top 3 are 9,6,5

    Use argsort or argpartition — no loops.

    Arguments:
        x -- np.ndarray of shape (n,), 1-D
        k -- int, 1 <= k <= n

    Returns:
        np.ndarray of shape (n,), dtype bool

    Raises:
        ValueError -- if x is not 1-D or k is out of range
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


if __name__ == "__main__":
    print("=== Reduce Ops ===\n")

    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("row_mean:")
    print(row_mean(X))

    rng = np.random.default_rng(0)
    X2 = rng.standard_normal((5, 4))
    print("\ncolumn_variance (vs np.var ddof=1):")
    print(column_variance(X2))
    print(X2.var(axis=0, ddof=1))

    Z = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    print("\nlog_sum_exp_rows (stable for large values):")
    print(log_sum_exp_rows(Z))

    X3 = np.array([[1.0, 2.0, 3.0, 4.0]])
    print("\ncumulative_row_mean:")
    print(cumulative_row_mean(X3))

    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    print("\ntop_k_mask(x, 3):", top_k_mask(x, 3))
    print("top 3 values:    ", x[top_k_mask(x, 3)])
