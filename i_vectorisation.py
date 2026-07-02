"""
Python → NumPy: vectorising a k-NN classifier.

This file gives you three "slow" reference implementations written in
plain Python, and asks you to replace each one with a NumPy equivalent
that uses no explicit for-loops or list comprehensions.

Each pair tests a different combination of the three primitive categories:

  npdistance  — elementwise + reduce          (1-D squared Euclidean)
  npnearest   — movement + elementwise + reduce  (1-NN lookup for one query)
  npbatch     — movement + elementwise + reduce  (k-NN for a batch of queries)

The Python versions are correct and are the specification.  Your NumPy
versions must return the same results (or equivalent types) for any input.

Notation:
  d  — number of features
  n  — number of training points
  m  — number of query points

DO NOT MODIFY THE FUNCTION SIGNATURES OR THE PYTHON REFERENCE IMPLEMENTATIONS.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Reference implementations (do NOT change these)
# ---------------------------------------------------------------------------

def pydistance(x1: list, x2: list) -> float:
    """Squared Euclidean distance between two feature vectors (pure Python)."""
    return sum((a - b) ** 2 for a, b in zip(x1, x2))


def pynearest(u: list, X: list[list], Y: list) -> object:
    """1-nearest-neighbour label for query u from training set (X, Y)."""
    dists = [pydistance(u, x) for x in X]
    nearest_idx = dists.index(min(dists))
    return Y[nearest_idx]


def pybatch(U: list[list], X: list[list], Y: list, k: int = 1) -> list:
    """
    k-nearest-neighbour labels for every query in U.

    For each query u in U, finds the k nearest rows of X and returns
    the most common label among them (majority vote; ties broken by
    lower label value via min).
    """
    results = []
    for u in U:
        dists = [pydistance(u, x) for x in X]
        sorted_indices = sorted(range(len(dists)), key=lambda i: dists[i])
        k_labels = [Y[i] for i in sorted_indices[:k]]
        results.append(max(set(k_labels), key=k_labels.count))
    return results


# ---------------------------------------------------------------------------
# Your NumPy implementations (fill these in)
# ---------------------------------------------------------------------------

def npdistance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Squared Euclidean distance between two 1-D feature vectors.

    Equivalent to pydistance but using NumPy elementwise ops and a reduce.

    Arguments:
        x1 -- np.ndarray of shape (d,)
        x2 -- np.ndarray of shape (d,)

    Returns:
        float -- sum of squared element-wise differences

    Raises:
        ValueError -- if x1 and x2 are not 1-D or have different lengths
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def npnearest(u: np.ndarray, X: np.ndarray, Y: np.ndarray) -> object:
    """
    1-nearest-neighbour label for a single query point u.

    Computes the squared Euclidean distance from u to every row of X,
    then returns the label in Y at the index of the minimum distance.

    No explicit loops — use broadcasting to compute all distances at once.

    Hint: X has shape (n, d) and u has shape (d,).  Subtract u from X
    (broadcasting handles the alignment), square elementwise, then reduce
    along axis=1 to get a distance per row.

    Arguments:
        u -- np.ndarray of shape (d,), the query point
        X -- np.ndarray of shape (n, d), training features
        Y -- np.ndarray of shape (n,), training labels

    Returns:
        The label Y[i] where i = argmin of squared distances

    Raises:
        ValueError -- if shapes are incompatible
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def npbatch(U: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int = 1) -> np.ndarray:
    """
    k-nearest-neighbour labels for a batch of query points.

    For each row u of U, finds the k nearest rows of X and returns the
    majority-vote label (ties broken by the smallest label value).

    No explicit loops over queries — compute the full (m, n) distance
    matrix at once using broadcasting, then use argsort to rank neighbours.

    Hint: reshape U to (m, 1, d) and X to (1, n, d).  Subtracting gives
    a (m, n, d) difference tensor; sum-of-squares along axis=2 gives the
    (m, n) distance matrix.

    Arguments:
        U -- np.ndarray of shape (m, d), query points
        X -- np.ndarray of shape (n, d), training features
        Y -- np.ndarray of shape (n,), training labels
        k -- int, number of neighbours (default 1)

    Returns:
        np.ndarray of shape (m,) — predicted label for each query

    Raises:
        ValueError -- if shapes are incompatible or k is out of range
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


if __name__ == "__main__":
    print("=== k-NN Vectorisation ===\n")

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((20, 4))
    Y_train = rng.integers(0, 3, size=20)

    x1, x2 = X_train[0], X_train[1]
    print("pydistance:", pydistance(x1.tolist(), x2.tolist()))
    print("npdistance:", npdistance(x1, x2))

    u = rng.standard_normal(4)
    print("\npynearest:", pynearest(u.tolist(), X_train.tolist(), Y_train.tolist()))
    print("npnearest:", npnearest(u, X_train, Y_train))

    U = rng.standard_normal((5, 4))
    py_preds = pybatch(U.tolist(), X_train.tolist(), Y_train.tolist(), k=3)
    np_preds = npbatch(U, X_train, Y_train, k=3)
    print("\npybatch (k=3):", py_preds)
    print("npbatch (k=3):", np_preds.tolist())
    print("Match:", py_preds == np_preds.tolist())
