"""Shared LP-encoding primitives for piecewise-linear penalty methodologies.

These helpers are methodology-agnostic and used by:
- ``dispatch_collath`` / ``dispatch_stacked_collath`` (Collath 2023 throughput-
  per-window penalty)
- ``dispatch_stacked_rainflow`` (Shi-Xu 2018 rainflow-cycle penalty, planned)
- any future piecewise-linear in-objective formulation calibrated against
  the Note 3 Wang+Naumann kernel.

The shared piece is the upper-envelope encoding: given breakpoints
``x_brk`` and corresponding values ``y_brk`` of a (preferably convex)
function, an LP variable ``f`` constrained by ``f >= a_i x + b_i`` for
all per-segment slopes/intercepts ``(a_i, b_i)`` is exactly the
piecewise-linear function on convex pieces and an over-estimate (always
≥ true value) on concave pieces — which keeps the LP feasible even if
the underlying calibration is not strictly convex.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def segment_lines(
    x_brk: np.ndarray, y_brk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slope and intercept of each segment between consecutive breakpoints.

    Suitable for upper-envelope LP encoding ``f >= a_i × x + b_i``.

    Args:
        x_brk: (n,) breakpoints in input variable (monotonic increasing).
        y_brk: (n,) function values at breakpoints.

    Returns:
        (a, b): each (n-1,) array — slopes and intercepts.
    """
    a = np.diff(y_brk) / np.diff(x_brk)
    b = y_brk[:-1] - a * x_brk[:-1]
    return a, b


def check_convexity(
    values: np.ndarray,
    breakpoints: np.ndarray | None = None,
    tol: float = 1e-10,
) -> dict:
    """Second-difference + slope-ratio convexity inspector for piecewise data.

    Args:
        values: (n,) function values at breakpoints.
        breakpoints: (n,) optional — if provided, slopes computed against
            actual spacing; otherwise unit spacing assumed.
        tol: numerical tolerance for "Δ² ≥ 0" test (accounts for rounding).

    Returns:
        dict with keys:
            ``convex`` (bool): all Δ² ≥ -tol
            ``second_diffs`` (np.ndarray, shape (n-2,))
            ``slope_ratio`` (float): max|slope| / min|slope| over non-zero
                slopes; ratio of 1.0 means perfectly linear, larger means
                strongly convex (or steep). 1.11× was the eve_lf280k cyclic
                value behind the L7-Collath null result.
            ``tol`` (float): tolerance applied
    """
    if breakpoints is not None:
        slopes = np.diff(values) / np.diff(breakpoints)
    else:
        slopes = np.diff(values)
    second_diffs = np.diff(slopes)
    convex = bool(np.all(second_diffs >= -tol))
    abs_slopes = np.abs(slopes)
    abs_slopes_pos = abs_slopes[abs_slopes > 0]
    if len(abs_slopes_pos) >= 2:
        slope_ratio = float(abs_slopes_pos.max() / abs_slopes_pos.min())
    else:
        slope_ratio = 1.0
    return {
        "convex": convex,
        "second_diffs": second_diffs,
        "slope_ratio": slope_ratio,
        "tol": tol,
    }


def load_piecewise_coefficients(path: Path | str) -> dict[str, np.ndarray]:
    """Generic loader for ``.npz`` files of breakpoints/values arrays.

    Returns a plain dict of all keys present in the archive; methodology-
    specific callers unpack the schema they expect (e.g. Collath uses
    ``soc_breakpoints``/``cal_fade``/``throughput_breakpoints``/``cyc_fade``;
    Rainflow uses ``dod_breakpoints``/``cycle_fade``).
    """
    # allow_pickle=True so calibration metadata (object arrays) round-trips.
    # Pickle is safe here — these `.npz` files are written by our own
    # calibration scripts under version control.
    data = np.load(Path(path), allow_pickle=True)
    return {key: data[key] for key in data.files}
