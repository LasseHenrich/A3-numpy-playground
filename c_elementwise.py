"""
Elementwise ops — applying a function to every element independently.

Elementwise ops are O(n): they read each element once and write once.
No shape changes, no axis reasoning.  The skill here is in translating
a mathematical formula into a numerically stable NumPy expression.

Numerical stability matters whenever you compute exp, log, or divisions
that could produce inf or NaN.  The standard trick is to use an equivalent
algebraic form that avoids large intermediate values.

Key identities:
  log(sum_i exp(x_i)) = c + log(sum_i exp(x_i - c))  for any constant c
                        (choosing c = max(x) keeps exp arguments <= 0)

  sigmoid(z) = 1 / (1 + exp(-z))
             = exp(z) / (1 + exp(z))    (use this form when z < 0)

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np
from scipy.special import erf, ndtr


def relu(Z: np.ndarray) -> np.ndarray:
    """
    Applies the Rectified Linear Unit element-wise: ReLU(z) = max(0, z).

    Arguments:
        Z -- np.ndarray of any shape

    Returns:
        np.ndarray of the same shape, all values >= 0
    """

    return np.maximum(0, Z)


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid function element-wise: σ(z) = 1 / (1 + exp(-z)).

    Must be numerically stable: no NaN or Inf for any finite input,
    including large magnitudes such as z = ±1000.

    Hint: split on the sign of z and use the algebraically equivalent form
          exp(z) / (1 + exp(z)) for negative inputs.

    Arguments:
        Z -- np.ndarray of any shape

    Returns:
        np.ndarray of the same shape, values in (0, 1)
    """

    return np.where(Z > 0, 1 / (1 + np.exp(-Z)), np.exp(Z) / (1 + np.exp(Z)))


def gelu(Z: np.ndarray) -> np.ndarray:
    """
    Applies the Gaussian Error Linear Unit element-wise.

    Exact definition (Hendrycks & Gimpel, 2016):
        GELU(z) = z * Φ(z)
    where Φ is the CDF of the standard normal distribution.

    You may use scipy.special.ndtr or express Φ in terms of the error
    function: Φ(z) = 0.5 * (1 + erf(z / sqrt(2))).
    Use math.sqrt or np.sqrt — do not hardcode the constant.

    Arguments:
        Z -- np.ndarray of any shape

    Returns:
        np.ndarray of the same shape
    """

    return Z * ndtr(Z)


def log_sum_exp(x: np.ndarray) -> float:
    """
    Computes log(sum_i exp(x_i)) in a numerically stable way.

    Naive computation overflows for large x_i.  Use the identity:
        log(sum_i exp(x_i)) = c + log(sum_i exp(x_i - c))
    with c = max(x).

    Arguments:
        x -- np.ndarray of shape (n,), n >= 1

    Returns:
        float

    Raises:
        ValueError -- if x is empty or not 1-D
    """

    c = max(x)
    return c + np.log(np.sum(np.exp(x - c)))


def softplus(Z: np.ndarray) -> np.ndarray:
    """
    Applies the softplus function element-wise:
        softplus(z) = log(1 + exp(z))

    This is a smooth approximation to ReLU.  It must be numerically stable:
    for large z, log(1 + exp(z)) ≈ z (use this approximation when z > 20
    to avoid overflow).

    Arguments:
        Z -- np.ndarray of any shape

    Returns:
        np.ndarray of the same shape, all values > 0
    """

    return np.where(Z > 20, np.log(1 + np.exp(Z)), np.log(1 + np.exp(Z)))


if __name__ == "__main__":
    import math

    print("=== Elementwise Ops ===\n")

    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print("z:       ", z)
    print("relu:    ", relu(z))
    print("sigmoid: ", sigmoid(z))
    print("gelu:    ", gelu(z))
    print("softplus:", softplus(z))

    print("\nStability check (z = ±1000):")
    extreme = np.array([-1000.0, 1000.0])
    print("sigmoid: ", sigmoid(extreme), "  (should be [~0, ~1], no NaN)")
    print("softplus:", softplus(extreme), "  (should be [~0, 1000], no NaN)")

    x = np.array([1000.0, 1001.0, 1002.0])
    print("\nlog_sum_exp([1000, 1001, 1002]):", log_sum_exp(x))
    print("expected ≈", 1002 + math.log(math.exp(-2) + math.exp(-1) + 1))
