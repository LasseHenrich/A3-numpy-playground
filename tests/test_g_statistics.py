import numpy as np
import pytest
from g_statistics import sample_mean, sample_variance, pearson_correlation, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# sample_mean
# ---------------------------------------------------------------------------

def test_sample_mean_basic():
    pass

def test_sample_mean_single():
    pass

def test_sample_mean_matches_numpy():
    pass

def test_sample_mean_raises_empty():
    pass


# ---------------------------------------------------------------------------
# sample_variance
# ---------------------------------------------------------------------------

def test_sample_variance_known():
    pass

def test_sample_variance_two_elements():
    pass

def test_sample_variance_raises_single():
    pass

def test_sample_variance_non_negative():
    pass


# ---------------------------------------------------------------------------
# pearson_correlation
# ---------------------------------------------------------------------------

def test_pearson_perfect_positive():
    pass

def test_pearson_perfect_negative():
    pass

def test_pearson_range():
    pass

def test_pearson_raises_length_mismatch():
    pass

def test_pearson_raises_zero_variance():
    pass


# ---------------------------------------------------------------------------
# mean_absolute_error
# ---------------------------------------------------------------------------

def test_mae_perfect():
    pass

def test_mae_known():
    pass

def test_mae_non_negative():
    pass

def test_mae_raises_empty():
    pass

def test_mae_raises_shape_mismatch():
    pass


# ---------------------------------------------------------------------------
# r2_score
# ---------------------------------------------------------------------------

def test_r2_perfect():
    pass

def test_r2_baseline_mean_predictor():
    pass

def test_r2_raises_constant_ytrue():
    pass

def test_r2_raises_empty():
    pass
