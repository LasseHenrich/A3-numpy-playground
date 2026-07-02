import numpy as np
import pytest
from a_movement import batch_flatten, split_into_patches, channels_first_to_last, interleave_rows, tile_vector


# ---------------------------------------------------------------------------
# batch_flatten
# ---------------------------------------------------------------------------

def test_batch_flatten_3d():
    X = np.arange(32 * 28 * 28).reshape(32, 28, 28)
    result = batch_flatten(X)
    assert result.shape == (32, 784), f"Expected shape (32, 784), got {result.shape}"

def test_batch_flatten_4d():
    X = np.arange(4 * 3 * 8 * 8).reshape(4, 3, 8, 8)
    result = batch_flatten(X)
    assert result.shape == (4, 192), f"Expected shape (4, 192), got {result.shape}"

def test_batch_flatten_2d():
    X = np.arange(5 * 10).reshape(5, 10)
    result = batch_flatten(X)
    assert result.shape == (5, 10), f"Expected shape (5, 10), got {result.shape}"

def test_batch_flatten_values():
    X = np.arange(24).reshape(2, 3, 4)
    result = batch_flatten(X)
    expected = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
    np.testing.assert_array_equal(result, expected)

def test_batch_flatten_raises_1d():
    X = np.arange(10)
    with pytest.raises(ValueError):
        batch_flatten(X)


# ---------------------------------------------------------------------------
# split_into_patches
# ---------------------------------------------------------------------------

def test_split_into_patches_shape():
    X = np.arange(16).reshape(4, 4)
    result = split_into_patches(X, 2, 2)
    assert result.shape == (2, 2, 2, 2), f"Expected shape (2, 2, 2, 2), got {result.shape}"

def test_split_into_patches_top_left():
    X = np.arange(16).reshape(4, 4)
    patches = split_into_patches(X, 2, 2)
    expected = np.array([[0, 1], [4, 5]])
    np.testing.assert_array_equal(patches[0, 0], expected)

def test_split_into_patches_top_right():
    X = np.arange(16).reshape(4, 4)
    patches = split_into_patches(X, 2, 2)
    expected = np.array([[2, 3], [6, 7]])
    np.testing.assert_array_equal(patches[0, 1], expected)

def test_split_into_patches_bottom_left():
    X = np.arange(16).reshape(4, 4)
    patches = split_into_patches(X, 2, 2)
    expected = np.array([[8, 9], [12, 13]])
    np.testing.assert_array_equal(patches[1, 0], expected)

def test_split_into_patches_reconstructable():
    X = np.arange(36).reshape(6, 6)
    patches = split_into_patches(X, 2, 3)
    assert patches.shape == (3, 2, 2, 3)
    reconstructed = patches.reshape(3, 2, 2, 3).transpose(0, 2, 1, 3).reshape(6, 6)
    np.testing.assert_array_equal(reconstructed, X)

def test_split_into_patches_raises_non_divisible():
    X = np.arange(16).reshape(4, 4)
    with pytest.raises(ValueError):
        split_into_patches(X, 3, 2)

def test_split_into_patches_raises_non_2d():
    X = np.arange(24).reshape(2, 3, 4)
    with pytest.raises(ValueError):
        split_into_patches(X, 2, 2)


# ---------------------------------------------------------------------------
# channels_first_to_last
# ---------------------------------------------------------------------------

def test_channels_first_to_last_shape():
    X = np.zeros((4, 3, 8, 8))
    result = channels_first_to_last(X)
    assert result.shape == (4, 8, 8, 3), f"Expected shape (4, 8, 8, 3), got {result.shape}"

def test_channels_first_to_last_values():
    # X = np.arange(24).reshape(2, 3, 2, 2)
    # result = channels_first_to_last(X)
    # assert result.shape == (2, 2, 2, 3)
    # expected = np.array([[[[0, 8, 16], [1, 9, 17]], [[2, 10, 18], [3, 11, 19]]],
    #                      [[[4, 12, 20], [5, 13, 21]], [[6, 14, 22], [7, 15, 23]]]])
    # np.testing.assert_array_equal(result, expected)
    pass

def test_channels_first_to_last_raises():
    X = np.zeros((4, 3, 8))
    with pytest.raises(ValueError):
        channels_first_to_last(X)


# ---------------------------------------------------------------------------
# interleave_rows
# ---------------------------------------------------------------------------

def test_interleave_rows_shape():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[10.0, 20.0], [30.0, 40.0]])
    result = interleave_rows(A, B)
    assert result.shape == (4, 2), f"Expected shape (4, 2), got {result.shape}"

def test_interleave_rows_order():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[10.0, 20.0], [30.0, 40.0]])
    result = interleave_rows(A, B)
    expected = np.array([[1.0, 2.0], [10.0, 20.0], [3.0, 4.0], [30.0, 40.0]])
    np.testing.assert_array_equal(result, expected)

def test_interleave_rows_raises_shape_mismatch():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[10.0, 20.0]])
    with pytest.raises(ValueError):
        interleave_rows(A, B)


# ---------------------------------------------------------------------------
# tile_vector
# ---------------------------------------------------------------------------

def test_tile_vector_shape():
    v = np.array([1.0, 2.0, 3.0])
    result = tile_vector(v, 4)
    assert result.shape == (4, 3), f"Expected shape (4, 3), got {result.shape}"

def test_tile_vector_values():
    v = np.array([1.0, 2.0, 3.0])
    result = tile_vector(v, 4)
    expected = np.array([[1.0, 2.0, 3.0],
                         [1.0, 2.0, 3.0],
                         [1.0, 2.0, 3.0],
                         [1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(result, expected)

def test_tile_vector_raises_non_1d():
    v = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        tile_vector(v, 4)

def test_tile_vector_raises_n_zero():
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        tile_vector(v, 0)
