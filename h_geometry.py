"""
Formula translation — geometry and distances.

These problems require combining broadcasting, reduce ops, and formula
translation.  None of them use ML concepts — they are pure geometry.

Formulae:

  Euclidean distance:
      d(x, y) = sqrt(Σ_k (x_k − y_k)²)

  Cosine distance:
      d_cos(x, y) = 1 − (x · y) / (||x|| · ||y||)

  Haversine distance (great-circle distance on a sphere):
    Given two points (lat1, lon1) and (lat2, lon2) in radians:
      a = sin²((lat2−lat1)/2) + cos(lat1)·cos(lat2)·sin²((lon2−lon1)/2)
      d = 2·R·arcsin(sqrt(a))
    where R ≈ 6371 km is the Earth's mean radius.

  k-NN:
    For each query row in X, find the indices of the k nearest rows in Y
    (by squared Euclidean distance).

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np


def euclidean_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Computes the matrix of Euclidean distances between every pair of rows.

    D[i, j] = ||X[i] - Y[j]||_2

    Implement using broadcasting — no loops, no scipy.

    Arguments:
        X -- np.ndarray of shape (m, d)
        Y -- np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (m, n), values >= 0

    Raises:
        ValueError -- if X and Y are not 2-D or have different d
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def knn_indices(X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """
    For each row in X, returns the indices of the k nearest rows in Y
    (by Euclidean distance).

    The result is sorted: nearest first.

    Arguments:
        X -- np.ndarray of shape (m, d), query points
        Y -- np.ndarray of shape (n, d), database points
        k -- int, number of neighbours, 1 <= k <= n

    Returns:
        np.ndarray of shape (m, k), dtype int — index into Y for each query

    Raises:
        ValueError -- if k is out of range or shapes are incompatible
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def haversine(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    R: float = 6371.0,
) -> np.ndarray:
    """
    Computes the great-circle (haversine) distance in kilometres between
    pairs of GPS coordinates.

    Inputs are in degrees; convert to radians inside the function.

    Formula:
        a = sin²((lat2−lat1)/2) + cos(lat1)·cos(lat2)·sin²((lon2−lon1)/2)
        d = 2·R·arcsin(sqrt(a))

    All four input arrays must have the same shape; the output has the same
    shape.

    Arguments:
        lat1, lon1 -- np.ndarray of any shape, origin coordinates (degrees)
        lat2, lon2 -- np.ndarray of same shape, destination coordinates (degrees)
        R          -- Earth radius in km (default 6371.0)

    Returns:
        np.ndarray of same shape, distances in kilometres

    Raises:
        ValueError -- if input shapes are not all equal
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def pairwise_haversine(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Computes the haversine distance between every pair of GPS points.

    Arguments:
        coords1 -- np.ndarray of shape (m, 2), columns are [latitude, longitude] in degrees
        coords2 -- np.ndarray of shape (n, 2), columns are [latitude, longitude] in degrees

    Returns:
        np.ndarray of shape (m, n), distances in kilometres

    Raises:
        ValueError -- if inputs are not 2-D or don't have exactly 2 columns
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


if __name__ == "__main__":
    print("=== Geometry ===\n")

    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    Y = np.array([[0.0, 0.0], [1.0, 0.0]])
    print("euclidean_distances:")
    print(euclidean_distances(X, Y))

    rng = np.random.default_rng(0)
    db = rng.standard_normal((10, 3))
    queries = rng.standard_normal((3, 3))
    print("\nknn_indices (k=2):")
    print(knn_indices(queries, db, k=2))

    # Berlin → Munich  (≈ 504 km)
    lat1, lon1 = np.array([52.52]), np.array([13.41])
    lat2, lon2 = np.array([48.14]), np.array([11.58])
    print(f"\nBerlin → Munich haversine: {haversine(lat1, lon1, lat2, lon2)[0]:.1f} km")

    cities = np.array([[52.52, 13.41], [48.14, 11.58], [53.55, 10.00]])  # Berlin, Munich, Hamburg
    print("\npairwise_haversine (Berlin, Munich, Hamburg):")
    print(pairwise_haversine(cities, cities).round(0))
