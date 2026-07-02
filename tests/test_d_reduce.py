import numpy as np
import pytest
from d_reduce import row_mean, column_variance, log_sum_exp_rows, cumulative_row_mean, top_k_mask


# ---------------------------------------------------------------------------
# row_mean
# ---------------------------------------------------------------------------

# def test_row_mean_basic():
#     X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#     result = row_mean(X)
#     expected = np.array([2.0, 5.0])
#     np.testing.assert_allclose(result, expected, atol=1e-9)

# def test_row_mean_matches_numpy():
#     X = np.random.randn(10, 5)
#     result = row_mean(X)
#     expected = X.mean(axis=1)
#     np.testing.assert_allclose(result, expected, atol=1e-9)

def test_row_mean_shape():
    X = np.random.randn(7, 4)
    result = row_mean(X)
    assert result.shape == (7,), f"Expected shape (7,), got {result.shape}"

def test_row_mean_raises_non_2d():
    X_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        row_mean(X_1d)

    X_3d = np.random.randn(2, 3, 4)
    with pytest.raises(ValueError):
        row_mean(X_3d)


# ---------------------------------------------------------------------------
# column_variance
# ---------------------------------------------------------------------------

def test_column_variance_matches_numpy():
    X = np.random.randn(10, 5)
    result = column_variance(X)
    expected = X.var(axis=0, ddof=1)
    np.testing.assert_allclose(result, expected, atol=1e-9)

def test_column_variance_constant_column():
    # If a column is constant, variance should be 0
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    result = column_variance(X)
    expected = np.array([0.0, 1.0])  # First column constant (var=0), second has variance
    np.testing.assert_allclose(result, expected, atol=1e-9)

def test_column_variance_raises_too_few_rows():
    X_single = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        column_variance(X_single)

def test_column_variance_raises_non_2d():
    X_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        column_variance(X_1d)

    X_3d = np.random.randn(2, 3, 4)
    with pytest.raises(ValueError):
        column_variance(X_3d)

def test_column_variance_non_negative():
    X = np.random.randn(20, 5)
    result = column_variance(X)
    assert np.all(result >= 0.0), "Variance must be non-negative"


# ---------------------------------------------------------------------------
# log_sum_exp_rows
# ---------------------------------------------------------------------------

def test_log_sum_exp_rows_shape():
    X = np.random.randn(5, 4)
    result = log_sum_exp_rows(X)
    assert result.shape == (5,), f"Expected shape (5,), got {result.shape}"

def test_log_sum_exp_rows_uniform():
    # For uniform row [a, a, a, a], LSE = a + log(d) where d is number of columns
    X = np.array([[2.0, 2.0, 2.0]])
    result = log_sum_exp_rows(X)
    expected = 2.0 + np.log(3.0)
    np.testing.assert_allclose(result, expected, atol=1e-9)

def test_log_sum_exp_rows_stable():
    # Test numerical stability with large values
    X = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]])
    result = log_sum_exp_rows(X)
    assert not np.any(np.isnan(result)), "log_sum_exp_rows must not produce NaN"
    assert not np.any(np.isinf(result)), "log_sum_exp_rows must not produce Inf"
    # First row LSE should be close to 1002
    assert result[0] > 1000.0, "Result should be large but finite"

def test_log_sum_exp_rows_raises_non_2d():
    X_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        log_sum_exp_rows(X_1d)

    X_3d = np.random.randn(2, 3, 4)
    with pytest.raises(ValueError):
        log_sum_exp_rows(X_3d)


# ---------------------------------------------------------------------------
# cumulative_row_mean
# ---------------------------------------------------------------------------

def test_cumulative_row_mean_shape():
    X = np.random.randn(5, 6)
    result = cumulative_row_mean(X)
    assert result.shape == X.shape, f"Expected shape {X.shape}, got {result.shape}"

def test_cumulative_row_mean_values():
    # Test with known values: [1, 2, 3, 4]
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    result = cumulative_row_mean(X)
    # cumsum = [1, 3, 6, 10], divide by [1, 2, 3, 4]
    expected = np.array([[1.0, 1.5, 2.0, 2.5]])
    np.testing.assert_allclose(result, expected, atol=1e-9)

def test_cumulative_row_mean_last_col_is_row_mean():
    X = np.random.randn(8, 5)
    result = cumulative_row_mean(X)
    row_means = X.mean(axis=1)
    # Last column of result should equal row mean
    np.testing.assert_allclose(result[:, -1], row_means, atol=1e-9)

def test_cumulative_row_mean_raises_non_2d():
    X_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        cumulative_row_mean(X_1d)

    X_3d = np.random.randn(2, 3, 4)
    with pytest.raises(ValueError):
        cumulative_row_mean(X_3d)


# ---------------------------------------------------------------------------
# top_k_mask
# ---------------------------------------------------------------------------

def test_top_k_mask_basic():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    mask = top_k_mask(x, 3)
    # Top 3 are: 9 (index 5), 6 (index 7), 5 (index 4)
    assert mask.sum() == 3, f"Expected 3 True values, got {mask.sum()}"
    # Check that the 3 largest values are marked as True
    top_values = np.sort(x)[-3:]
    x_top_k = x[mask]
    assert np.all(np.isin(x_top_k, top_values)), "Marked values should be in top k"

def test_top_k_mask_all():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = top_k_mask(x, 5)
    assert mask.sum() == 5, "k=n should mark all as True"
    assert np.all(mask), "All elements should be True when k=n"

def test_top_k_mask_single():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    mask = top_k_mask(x, 1)
    assert mask.sum() == 1, "k=1 should mark exactly 1 as True"
    assert x[mask][0] == 9.0, "Single marked value should be the maximum"

def test_top_k_mask_shape():
    x = np.random.randn(100)
    mask = top_k_mask(x, 10)
    assert mask.shape == x.shape, f"Mask shape {mask.shape} should match input shape {x.shape}"
    assert mask.dtype == bool, "Mask should be boolean"

def test_top_k_mask_raises_non_1d():
    x_2d = np.random.randn(3, 4)
    with pytest.raises(ValueError):
        top_k_mask(x_2d, 2)

    x_3d = np.random.randn(2, 3, 4)
    with pytest.raises(ValueError):
        top_k_mask(x_3d, 2)

def test_top_k_mask_raises_k_out_of_range():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(ValueError):
        top_k_mask(x, 0)  # k too small

    with pytest.raises(ValueError):
        top_k_mask(x, 6)  # k > n
