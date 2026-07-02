import numpy as np
import pytest
from i_vectorisation import npdistance, npnearest, npbatch, pydistance, pynearest, pybatch


# ---------------------------------------------------------------------------
# npdistance
# ---------------------------------------------------------------------------

def test_npdistance_matches_python():
    pass

def test_npdistance_zero_for_equal():
    pass

def test_npdistance_known_value():
    pass

def test_npdistance_non_negative():
    pass

def test_npdistance_raises_not_1d():
    pass

def test_npdistance_raises_length_mismatch():
    pass


# ---------------------------------------------------------------------------
# npnearest
# ---------------------------------------------------------------------------

def test_npnearest_matches_python():
    pass

def test_npnearest_returns_closest():
    pass

def test_npnearest_exact_match():
    pass

def test_npnearest_raises_shape():
    pass


# ---------------------------------------------------------------------------
# npbatch
# ---------------------------------------------------------------------------

def test_npbatch_matches_python_k1():
    pass

def test_npbatch_matches_python_k3():
    pass

def test_npbatch_output_shape():
    pass

def test_npbatch_single_query():
    pass

def test_npbatch_majority_vote():
    pass

def test_npbatch_raises_k_out_of_range():
    pass

def test_npbatch_raises_shape_mismatch():
    pass
