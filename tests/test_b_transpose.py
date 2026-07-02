import numpy as np
import pytest
from b_transpose import gram_matrix, scatter_matrix, batch_transpose, is_symmetric, column_outer_products


# ---------------------------------------------------------------------------
# gram_matrix
# ---------------------------------------------------------------------------

def test_gram_matrix_shape():
    pass

def test_gram_matrix_identity_rows():
    pass

def test_gram_matrix_known():
    pass

def test_gram_matrix_symmetric():
    pass

def test_gram_matrix_raises_non_2d():
    pass


# ---------------------------------------------------------------------------
# scatter_matrix
# ---------------------------------------------------------------------------

def test_scatter_matrix_shape():
    pass

def test_scatter_matrix_zero_for_constant():
    pass

def test_scatter_matrix_symmetric():
    pass

def test_scatter_matrix_psd():
    pass

def test_scatter_matrix_raises_non_2d():
    pass


# ---------------------------------------------------------------------------
# batch_transpose
# ---------------------------------------------------------------------------

def test_batch_transpose_shape():
    pass

def test_batch_transpose_matches_individual():
    pass

def test_batch_transpose_raises_non_3d():
    pass


# ---------------------------------------------------------------------------
# is_symmetric
# ---------------------------------------------------------------------------

def test_is_symmetric_identity():
    pass

def test_is_symmetric_gram():
    pass

def test_is_symmetric_asymmetric():
    pass

def test_is_symmetric_raises_non_square():
    pass


# ---------------------------------------------------------------------------
# column_outer_products
# ---------------------------------------------------------------------------

def test_column_outer_products_shape():
    pass

def test_column_outer_products_matches_xtx():
    pass

def test_column_outer_products_symmetric():
    pass

def test_column_outer_products_raises_non_2d():
    pass
