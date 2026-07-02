"""
Formula translation — descriptive statistics.

Implement each function directly from its mathematical definition using
NumPy array operations.  Do NOT use np.mean, np.var, np.std, np.cov,
np.corrcoef, or scipy.stats — the point is to practise translating
formulae into reduce + elementwise expressions.

Formulae:

  Sample mean:
      μ = (1/n) · Σ_i x_i

  Sample variance (Bessel-corrected, ddof=1):
      s² = (1/(n−1)) · Σ_i (x_i − μ)²

  Pearson correlation:
      r(x,y) = cov(x,y) / (s_x · s_y)
      cov(x,y) = (1/(n−1)) · Σ_i (x_i − μ_x)(y_i − μ_y)

  Mean Absolute Error:
      MAE = (1/n) · Σ_i |ŷ_i − y_i|

  R² score:
      R² = 1 − SS_res / SS_tot
      SS_res = Σ_i (y_i − ŷ_i)²
      SS_tot = Σ_i (y_i − ȳ)²

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np


def sample_mean(x: np.ndarray) -> float:
    """
    Computes the arithmetic mean.  Do NOT use np.mean.

    Arguments:
        x -- np.ndarray of shape (n,), n >= 1

    Returns:
        float

    Raises:
        ValueError -- if x is empty
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def sample_variance(x: np.ndarray) -> float:
    """
    Computes the Bessel-corrected sample variance (ddof=1).
    Do NOT use np.var or np.std.

    Arguments:
        x -- np.ndarray of shape (n,), n >= 2

    Returns:
        float >= 0

    Raises:
        ValueError -- if x has fewer than 2 elements
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient.
    Do NOT use np.corrcoef or np.cov.

    Arguments:
        x -- np.ndarray of shape (n,), n >= 2
        y -- np.ndarray of shape (n,)

    Returns:
        float in [−1, 1]

    Raises:
        ValueError -- if x and y have different lengths or fewer than 2 elements
        ValueError -- if either x or y has zero variance
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error.

    Arguments:
        y_true -- np.ndarray of shape (n,)
        y_pred -- np.ndarray of shape (n,)

    Returns:
        float >= 0

    Raises:
        ValueError -- if arrays have different lengths or are empty
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the R² coefficient of determination.

    Arguments:
        y_true -- np.ndarray of shape (n,)
        y_pred -- np.ndarray of shape (n,)

    Returns:
        float (1.0 is perfect; can be negative for bad models)

    Raises:
        ValueError -- if arrays have different lengths or are empty
        ValueError -- if y_true is constant (SS_tot == 0)
    """

    # --- IMPLEMENT SOLUTION HERE ---
    pass


if __name__ == "__main__":
    print("=== Statistics ===\n")

    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    print(f"x = {x}")
    print(f"sample_mean:     {sample_mean(x):.4f}  (expected 5.0)")
    print(f"sample_variance: {sample_variance(x):.4f}  (expected 4.5714)")

    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = 3 * y + 1
    print(f"\npearson_correlation (perfectly linear): {pearson_correlation(x2, y):.4f}  (expected 1.0)")

    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5,  0.0, 2.0, 8.0])
    print(f"\nMAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R²:   {r2_score(y_true, y_pred):.4f}")
