import numpy as np
import pytest
from b_transpose import gram_matrix, scatter_matrix, batch_transpose, is_symmetric, column_outer_products


# ---------------------------------------------------------------------------
# gram_matrix
# ---------------------------------------------------------------------------

def test_gram_matrix_shape():
    X = np.random.randn(5, 3)
    G = gram_matrix(X)
    assert G.shape == (5, 5), f"Expected shape (5, 5), got {G.shape}"

def test_gram_matrix_identity_rows():
    # Orthonormal rows should give identity Gram matrix
    X = np.eye(3)
    G = gram_matrix(X)
    expected = np.eye(3)
    np.testing.assert_allclose(G, expected, atol=1e-9)

def test_gram_matrix_known():
    # Known test case: [[1, 0], [0, 1], [1, 1]]
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    G = gram_matrix(X)
    # Gram[i,j] = dot(X[i], X[j])
    expected = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 2.0]
    ])
    np.testing.assert_allclose(G, expected, atol=1e-9)

def test_gram_matrix_symmetric():
    X = np.random.randn(4, 3)
    G = gram_matrix(X)
    np.testing.assert_allclose(G, G.T, atol=1e-9)

def test_gram_matrix_raises_non_2d():
    X_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        gram_matrix(X_1d)

    X_3d = np.random.randn(2, 3, 4)
    with pytest.raises(ValueError):
        gram_matrix(X_3d)


# ---------------------------------------------------------------------------
# scatter_matrix
# ---------------------------------------------------------------------------

def test_scatter_matrix_shape():
    X = np.random.randn(10, 5)
    S = scatter_matrix(X)
    assert S.shape == (5, 5), f"Expected shape (5, 5), got {S.shape}"

def test_scatter_matrix_zero_for_constant():
    # If all rows are identical, centered data is zero → scatter is zero
    X = np.ones((5, 3)) * 2.5  # All rows identical
    S = scatter_matrix(X)
    np.testing.assert_allclose(S, np.zeros((3, 3)), atol=1e-9)

def test_scatter_matrix_symmetric():
    X = np.random.randn(6, 4)
    S = scatter_matrix(X)
    np.testing.assert_allclose(S, S.T, atol=1e-9)

def test_scatter_matrix_psd():
    # Scatter matrix is positive semi-definite: all eigenvalues >= 0
    X = np.random.randn(8, 3)
    S = scatter_matrix(X)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals >= -1e-9), f"Negative eigenvalues found: {eigvals}"

def test_scatter_matrix_raises_non_2d():
    X_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        scatter_matrix(X_1d)

    X_3d = np.random.randn(2, 3, 4)
    with pytest.raises(ValueError):
        scatter_matrix(X_3d)


# ---------------------------------------------------------------------------
# batch_transpose
# ---------------------------------------------------------------------------

def test_batch_transpose_shape():
    A = np.random.randn(3, 4, 5)
    result = batch_transpose(A)
    assert result.shape == (3, 5, 4), f"Expected shape (3, 5, 4), got {result.shape}"

def test_batch_transpose_matches_individual():
    # Each matrix in batch should be transposed independently
    A = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    result = batch_transpose(A)

    for b in range(A.shape[0]):
        np.testing.assert_allclose(result[b], A[b].T, atol=1e-9)

def test_batch_transpose_raises_non_3d():
    A_2d = np.random.randn(3, 4)
    with pytest.raises(ValueError):
        batch_transpose(A_2d)

    A_4d = np.random.randn(2, 3, 4, 5)
    with pytest.raises(ValueError):
        batch_transpose(A_4d)


# ---------------------------------------------------------------------------
# is_symmetric
# ---------------------------------------------------------------------------

def test_is_symmetric_identity():
    A = np.eye(4)
    assert is_symmetric(A) is True

def test_is_symmetric_gram():
    # Gram matrix is always symmetric
    X = np.random.randn(5, 3)
    G = gram_matrix(X)
    assert is_symmetric(G) is True

def test_is_symmetric_asymmetric():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert is_symmetric(A) is False

def test_is_symmetric_raises_non_square():
    A_rect = np.random.randn(3, 4)
    with pytest.raises(ValueError):
        is_symmetric(A_rect)

# def test_is_symmetric_tolerance():
#     # Test tolerance parameter
#     A = np.array([[1.0, 1e-10], [1e-10, 1.0]])
#     assert is_symmetric(A, tol=1e-8) is True
#     assert is_symmetric(A, tol=1e-12) is False


# ---------------------------------------------------------------------------
# column_outer_products
# ---------------------------------------------------------------------------

def test_column_outer_products_shape():
    X = np.random.randn(7, 5)
    M = column_outer_products(X)
    assert M.shape == (5, 5), f"Expected shape (5, 5), got {M.shape}"

def test_column_outer_products_matches_xtx():
    # column_outer_products should equal X.T @ X
    X = np.random.randn(6, 4)
    M = column_outer_products(X)
    expected = X.T @ X
    np.testing.assert_allclose(M, expected, atol=1e-9)

def test_column_outer_products_symmetric():
    # X.T @ X is always symmetric
    X = np.random.randn(8, 5)
    M = column_outer_products(X)
    np.testing.assert_allclose(M, M.T, atol=1e-9)

def test_column_outer_products_raises_non_2d():
    X_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        column_outer_products(X_1d)

    X_3d = np.random.randn(2, 3, 4)
    with pytest.raises(ValueError):
        column_outer_products(X_3d)

def test_column_outer_products_known():
    # Test with known values
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    M = column_outer_products(X)
    expected = np.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(M, expected, atol=1e-9)
