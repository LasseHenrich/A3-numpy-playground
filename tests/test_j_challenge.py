import numpy as np
import pytest
from j_challenge import rank_rows, label_mean, sliding_window_mean_var


# ---------------------------------------------------------------------------
# rank_rows
# ---------------------------------------------------------------------------

def test_rank_rows_known():
    pass

def test_rank_rows_shape():
    pass

def test_rank_rows_values_are_permutation():
    pass

def test_rank_rows_sorted_input():
    pass

def test_rank_rows_reversed_input():
    pass

def test_rank_rows_single_column():
    pass

def test_rank_rows_raises_non_2d():
    pass


# ---------------------------------------------------------------------------
# label_mean
# ---------------------------------------------------------------------------

def test_label_mean_known():
    pass

def test_label_mean_shape():
    pass

def test_label_mean_single_class():
    pass

def test_label_mean_matches_manual():
    pass

def test_label_mean_raises_non_2d_X():
    pass

def test_label_mean_raises_non_1d_labels():
    pass

def test_label_mean_raises_length_mismatch():
    pass

def test_label_mean_raises_out_of_range_label():
    pass


# ---------------------------------------------------------------------------
# sliding_window_mean_var
# ---------------------------------------------------------------------------

def test_sliding_window_mean_var_shape():
    pass

def test_sliding_window_mean_known_means():
    pass

def test_sliding_window_mean_var_matches_numpy():
    pass

def test_sliding_window_var_constant_signal():
    pass

def test_sliding_window_full_window():
    pass

def test_sliding_window_non_negative_variance():
    pass

def test_sliding_window_raises_not_1d():
    pass

def test_sliding_window_raises_w_too_small():
    pass

def test_sliding_window_raises_w_too_large():
    pass
