# Assignment 3 — NumPy Playground

Ten puzzle files that build your NumPy fluency from the ground up.
Each puzzle asks you to implement a set of functions using the
operation category named in the file. Work through them in order —
difficulty increases as the index letter increases.

## Puzzles

| File | Topic |
|------|-------|
| `a_movement.py` | Reshape, ravel, and axis permutation |
| `b_transpose.py` | Transpose and matrix products |
| `c_elementwise.py` | Activation functions and numerical stability |
| `d_reduce.py` | Axis reductions without `np.mean`/`np.var`/`np.std` |
| `e_broadcasting.py` | Broadcasting rules and common patterns |
| `f_advanced_broadcasting.py` | Axis insertion with `np.newaxis` |
| `g_statistics.py` | Descriptive statistics from their definitions |
| `h_geometry.py` | Distances: Euclidean, cosine, Haversine, k-NN |
| `i_vectorisation.py` | Replacing Python loops with NumPy expressions |
| `j_challenge.py` | Open-ended problems combining all three categories |

The three primitive categories referred to throughout are:
**movement** (reshape/transpose), **elementwise** (per-element ops),
and **reduce** (axis collapse).

## Setup

Requires [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/).

```
make install
```

To activate the environment in your shell:

```bash
source .venv/bin/activate
```

## Running tests

```bash
make test_all          # run every test
make test_a            # run tests for one puzzle (replace a with any letter)
```

## Submitting

Run the grader and upload the resulting file to the course webpage.

```bash
make submit_all        # grade and write submission.json for all puzzles
make submit_a          # grade and write submission.json for one puzzle
```
