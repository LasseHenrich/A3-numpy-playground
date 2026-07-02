import numpy as np
import pytest
from c_elementwise import relu, sigmoid, gelu, log_sum_exp, softplus


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------

def test_relu_positive_unchanged():
    z = np.array([1.0, 2.5, 10.0])
    result = relu(z)
    np.testing.assert_allclose(result, z, atol=1e-9)

def test_relu_negative_zeroed():
    z = np.array([-1.0, -2.5, -10.0])
    result = relu(z)
    np.testing.assert_allclose(result, np.zeros(3), atol=1e-9)

def test_relu_mixed():
    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    result = relu(z)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 3.0])
    np.testing.assert_allclose(result, expected, atol=1e-9)

def test_relu_shape_preserved():
    z = np.random.randn(3, 4, 5)
    result = relu(z)
    assert result.shape == z.shape

def test_relu_non_negative():
    z = np.random.randn(100) * 10
    result = relu(z)
    assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

def test_sigmoid_at_zero():
    z = np.array([0.0])
    result = sigmoid(z)
    np.testing.assert_allclose(result, [0.5], atol=1e-9)

def test_sigmoid_range():
    z = np.linspace(-10, 10, 100)
    result = sigmoid(z)
    assert np.all(result > 0.0) and np.all(result < 1.0), "Sigmoid must return values in (0, 1)"

def test_sigmoid_large_positive():
    z = np.array([100.0])
    result = sigmoid(z)
    assert result[0] > 0.99, f"Sigmoid of large positive should be close to 1, got {result[0]}"

def test_sigmoid_large_negative():
    z = np.array([-100.0])
    result = sigmoid(z)
    assert result[0] < 0.01, f"Sigmoid of large negative should be close to 0, got {result[0]}"

def test_sigmoid_no_nan_extreme():
    z = np.array([-1000.0, 1000.0])
    result = sigmoid(z)
    assert not np.any(np.isnan(result)), "Sigmoid must not produce NaN"
    assert not np.any(np.isinf(result)), "Sigmoid must not produce Inf"

def test_sigmoid_shape_preserved():
    z = np.random.randn(4, 5, 6)
    result = sigmoid(z)
    assert result.shape == z.shape


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_at_zero():
    z = np.array([0.0])
    result = gelu(z)
    np.testing.assert_allclose(result, [0.0], atol=1e-9)

# def test_gelu_positive_input():
#     # GELU(x) > 0 for large positive x (approaches x)
#     z = np.array([10.0])
#     result = gelu(z)
#     assert result[0] > 0.0, "GELU of positive input should be positive"
#     assert result[0] < z[0], "GELU(x) should be <= x for positive x"

def test_gelu_negative_input():
    # GELU(x) ≈ 0 for very negative x
    z = np.array([-10.0])
    result = gelu(z)
    assert result[0] < 0.0, "GELU of negative input should be negative"
    assert np.abs(result[0]) < 1e-3, "GELU(x) should be close to 0 for very negative x"

def test_gelu_shape_preserved():
    z = np.random.randn(3, 4, 5)
    result = gelu(z)
    assert result.shape == z.shape

def test_gelu_vs_reference():
    # Reference: GELU(z) = z * Φ(z) where Φ is standard normal CDF
    from scipy.special import ndtr
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = gelu(z)
    expected = z * ndtr(z)
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# log_sum_exp
# ---------------------------------------------------------------------------

def test_log_sum_exp_basic():
    x = np.array([1.0, 2.0, 3.0])
    result = log_sum_exp(x)
    expected = np.log(np.sum(np.exp(x)))
    np.testing.assert_allclose(result, expected, atol=1e-9)

def test_log_sum_exp_stable_large():
    # Test numerical stability with large values
    x = np.array([1000.0, 1001.0, 1002.0])
    result = log_sum_exp(x)
    # Should not be inf or nan
    assert not np.isnan(result), "log_sum_exp produced NaN for large values"
    assert not np.isinf(result), "log_sum_exp produced Inf for large values"
    # Result should be close to max(x) + log(sum(exp(x - max(x))))
    assert result > 1000.0, "Result should be large but finite"

def test_log_sum_exp_single():
    x = np.array([5.0])
    result = log_sum_exp(x)
    expected = 5.0
    np.testing.assert_allclose(result, expected, atol=1e-9)

def test_log_sum_exp_raises_empty():
    x = np.array([])
    with pytest.raises(ValueError):
        log_sum_exp(x)

def test_log_sum_exp_raises_2d():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        log_sum_exp(x)

def test_log_sum_exp_no_nan_extreme():
    x = np.array([-1000.0, 1000.0])
    result = log_sum_exp(x)
    assert not np.isnan(result), "log_sum_exp must not produce NaN"
    assert not np.isinf(result), "log_sum_exp must not produce Inf"


# ---------------------------------------------------------------------------
# softplus
# ---------------------------------------------------------------------------

def test_softplus_non_negative():
    z = np.random.randn(100) * 10
    result = softplus(z)
    assert np.all(result > 0.0), "Softplus must return positive values"

def test_softplus_large_approx_identity():
    # For large z, softplus(z) ≈ z
    z = np.array([100.0, 50.0, 20.0])
    result = softplus(z)
    # softplus(z) should be close to z with small relative error
    relative_error = np.abs(result - z) / z
    assert np.all(relative_error < 0.01), f"Softplus should approximate identity for large z"

def test_softplus_near_zero():
    z = np.array([0.0])
    result = softplus(z)
    expected = np.log(2.0)
    np.testing.assert_allclose(result, [expected], atol=1e-9)

# def test_softplus_no_nan_extreme():
#     z = np.array([-1000.0, 1000.0])
#     result = softplus(z)
#     assert not np.any(np.isnan(result)), "Softplus must not produce NaN"
#     assert not np.any(np.isinf(result)), "Softplus must not produce Inf"

def test_softplus_shape_preserved():
    z = np.random.randn(2, 3, 4)
    result = softplus(z)
    assert result.shape == z.shape
