import numpy as np
import pytest
from h_geometry import euclidean_distances, knn_indices, haversine, pairwise_haversine


# ---------------------------------------------------------------------------
# euclidean_distances
# ---------------------------------------------------------------------------

def test_euclidean_distances_shape():
    pass

def test_euclidean_distances_pythagorean():
    pass

def test_euclidean_distances_self_zero():
    pass

def test_euclidean_distances_non_negative():
    pass

def test_euclidean_distances_raises():
    pass


# ---------------------------------------------------------------------------
# knn_indices
# ---------------------------------------------------------------------------

def test_knn_indices_shape():
    pass

def test_knn_indices_nearest_is_self():
    pass

def test_knn_indices_sorted_by_distance():
    pass

def test_knn_indices_raises_k_out_of_range():
    pass


# ---------------------------------------------------------------------------
# haversine
# ---------------------------------------------------------------------------

def test_haversine_berlin_munich():
    # Berlin (52.52°N, 13.41°E) to Munich (48.14°N, 11.58°E) ≈ 504 km
    pass

def test_haversine_same_point_is_zero():
    pass

def test_haversine_non_negative():
    pass

def test_haversine_raises_shape_mismatch():
    pass


# ---------------------------------------------------------------------------
# pairwise_haversine
# ---------------------------------------------------------------------------

def test_pairwise_haversine_shape():
    pass

def test_pairwise_haversine_diagonal_zero():
    pass

def test_pairwise_haversine_symmetric():
    pass

def test_pairwise_haversine_raises_wrong_cols():
    pass
