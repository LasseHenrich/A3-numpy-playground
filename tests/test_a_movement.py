import numpy as np
import pytest
from a_movement import batch_flatten, split_into_patches, channels_first_to_last, interleave_rows, tile_vector


# ---------------------------------------------------------------------------
# batch_flatten
# ---------------------------------------------------------------------------

def test_batch_flatten_3d():
    pass

def test_batch_flatten_4d():
    pass

def test_batch_flatten_2d():
    pass

def test_batch_flatten_values():
    pass

def test_batch_flatten_raises_1d():
    pass


# ---------------------------------------------------------------------------
# split_into_patches
# ---------------------------------------------------------------------------

def test_split_into_patches_shape():
    pass

def test_split_into_patches_top_left():
    pass

def test_split_into_patches_top_right():
    pass

def test_split_into_patches_bottom_left():
    pass

def test_split_into_patches_reconstructable():
    pass

def test_split_into_patches_raises_non_divisible():
    pass

def test_split_into_patches_raises_non_2d():
    pass


# ---------------------------------------------------------------------------
# channels_first_to_last
# ---------------------------------------------------------------------------

def test_channels_first_to_last_shape():
    pass

def test_channels_first_to_last_values():
    pass

def test_channels_first_to_last_raises():
    pass


# ---------------------------------------------------------------------------
# interleave_rows
# ---------------------------------------------------------------------------

def test_interleave_rows_shape():
    pass

def test_interleave_rows_order():
    pass

def test_interleave_rows_raises_shape_mismatch():
    pass


# ---------------------------------------------------------------------------
# tile_vector
# ---------------------------------------------------------------------------

def test_tile_vector_shape():
    pass

def test_tile_vector_values():
    pass

def test_tile_vector_raises_non_1d():
    pass

def test_tile_vector_raises_n_zero():
    pass
