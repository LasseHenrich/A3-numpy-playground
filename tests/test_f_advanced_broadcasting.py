import numpy as np
import pytest
from f_advanced_broadcasting import (
    outer_product,
    pairwise_squared_distances,
    cosine_similarity_matrix,
    softmax,
    broadcast_add_explain,
)


# ---------------------------------------------------------------------------
# outer_product
# ---------------------------------------------------------------------------

def test_outer_product_matches_np_outer():
    pass

def test_outer_product_shape():
    pass

def test_outer_product_raises_non_1d():
    pass


# ---------------------------------------------------------------------------
# pairwise_squared_distances
# ---------------------------------------------------------------------------

def test_pairwise_squared_distances_shape():
    pass

def test_pairwise_squared_distances_self_zero():
    pass

def test_pairwise_squared_distances_known():
    pass

def test_pairwise_squared_distances_non_negative():
    pass

def test_pairwise_squared_distances_raises():
    pass


# ---------------------------------------------------------------------------
# cosine_similarity_matrix
# ---------------------------------------------------------------------------

def test_cosine_similarity_matrix_shape():
    pass

def test_cosine_similarity_matrix_self_is_one():
    pass

def test_cosine_similarity_matrix_range():
    pass

def test_cosine_similarity_matrix_raises():
    pass


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one():
    pass

def test_softmax_uniform_input():
    pass

def test_softmax_stable_large_values():
    pass

def test_softmax_raises_non_2d():
    pass


# ---------------------------------------------------------------------------
# broadcast_add_explain
# ---------------------------------------------------------------------------

def test_broadcast_add_explain_values():
    pass

def test_broadcast_add_explain_shape():
    pass

def test_broadcast_add_explain_raises():
    pass
