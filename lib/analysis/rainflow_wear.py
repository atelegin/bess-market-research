"""
Rainflow-cycle wear cost for the Note 4 trader-aging-aware methodology
sidebar (Shi-Xu et al., ACC 2018, arXiv:1703.07968).

This module is the **Rainflow sidebar** counterpart of
``physics_wear_lookup.py``. It exposes a 2-pass wear function whose
second-pass scalar EUR/MWh comes from rainflow-extracted cycles of
the observed SoC trajectory, evaluated against a piecewise-linear
per-cycle aging function ``f(DoD)`` calibrated against the Note 3
Wang+Naumann eve_lf280k kernel at ``kernel_scale = 0.66`` (Note 4
manufacturer-anchor calibration).

How it differs from :func:`physics_wear_from_duty` (the L6 form):

L6 evaluates the kernel at the **mean** DoD / C-rate / SoC of the day.
Rainflow extracts the **distribution** of cycle depths and evaluates a
convex per-cycle aging function. For traders who do mostly shallow
cycles plus one deep cycle per day, the two forms diverge: L6 splits
fade across all throughput uniformly, Rainflow attributes most fade to
the deep cycle. This is the empirically testable difference for the
Note 4 sidebar — methodology channel (b) in the [ROADMAP] taxonomy.

Implementation note on Shi-Xu:

Shi-Xu et al. proved the **rainflow cycle cost is convex** in the SoC
trajectory when the per-cycle aging function ``f(DoD)`` is convex,
hence amenable to subgradient optimization. They did not claim a
closed-form LP encoding of rainflow itself (rainflow extraction is
combinatorial — full LP-internal encoding requires either MILP or
an iterative subgradient method). The pragmatic translation we use
here is the standard 2-pass approach: solve a baseline LP (warm-start
wear), extract rainflow cycles on the observed SoC trace, evaluate
the convex per-cycle penalty, convert to a scalar EUR/MWh, and re-
solve. This is methodologically equivalent to what L6 does but with
**rainflow-cycle-cost** as the second-pass formula instead of the
duty-mean kernel call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lib.models.dispatch_stacked import StackedDispatchResult


# ---------------------------------------------------------------------------
# Rainflow extraction (ASTM E1049-85 four-point algorithm, single-pass).
# ---------------------------------------------------------------------------


def extract_rainflow_cycles(
    trajectory: np.ndarray,
) -> list[tuple[float, float]]:
    """Extract rainflow cycles from a 1-D signal (e.g. SoC fraction over time).

    Implements the four-point ASTM E1049-85 method on the sequence of
    local extrema (turning points). Returns a list of cycle ranges with
    the convention used by the Note 3 kernel:

    Returns:
        List of ``(range, mean)`` tuples, where:
            ``range`` = full peak-to-peak amplitude of the cycle (DoD,
                i.e. for SoC traces ranging in [0, 1] this is in [0, 1]).
            ``mean`` = (peak + trough) / 2 — useful for SoC-stress
                weighting if the kernel cares about cycle midpoint.

    Open half-cycles (residue) are returned at the end as well; the
    rainflow algorithm guarantees that every reversal is accounted for
    exactly once between full cycles and residue half-cycles.

    Notes:
        - Range is the full swing (max - min). One full rainflow cycle
          corresponds to one charge + one discharge of that depth, i.e.
          2× DoD of throughput. Some literature counts each half as a
          separate "half-cycle" and divides count by 2 — we keep the
          full-cycle convention because the per-cycle aging function
          ``f(DoD)`` is calibrated per **complete** cycle.
    """
    # Reduce to alternating extrema (turning points only)
    arr = np.asarray(trajectory, dtype=float).ravel()
    if len(arr) < 2:
        return []
    # Drop consecutive duplicates first
    keep = np.concatenate([[True], np.diff(arr) != 0.0])
    arr = arr[keep]
    if len(arr) < 2:
        return []
    # Keep only turning points (sign change in delta)
    if len(arr) >= 3:
        deltas = np.diff(arr)
        sign = np.sign(deltas)
        # turning point at i if sign[i-1] != sign[i]; first and last
        # endpoints are always turning points by definition
        keep = np.zeros(len(arr), dtype=bool)
        keep[0] = True
        keep[-1] = True
        for i in range(1, len(arr) - 1):
            if sign[i - 1] != sign[i]:
                keep[i] = True
        arr = arr[keep]
    # Four-point rainflow on the turning-point series
    stack: list[float] = []
    cycles: list[tuple[float, float]] = []
    for x in arr:
        stack.append(float(x))
        while len(stack) >= 4:
            x1, x2, x3, x4 = stack[-4], stack[-3], stack[-2], stack[-1]
            r1 = abs(x2 - x1)
            r2 = abs(x3 - x2)
            r3 = abs(x4 - x3)
            # If the middle range r2 is bounded by both neighbours, x2-x3
            # is a closed full cycle; remove and continue.
            if r2 <= r1 and r2 <= r3:
                rng = abs(x3 - x2)
                mean = 0.5 * (x2 + x3)
                cycles.append((rng, mean))
                # Remove x2, x3 from the stack
                del stack[-3:-1]
            else:
                break
    # Residue: pair adjacent points as half-cycles (count as 0.5 each).
    # The Note 3 kernel doesn't distinguish half from full, so we report
    # them with their range and mean; the per-cycle function will be
    # evaluated and the contribution simply scales linearly. To match
    # convention, we keep them as full cycles with the full range.
    for i in range(len(stack) - 1):
        rng = abs(stack[i + 1] - stack[i])
        mean = 0.5 * (stack[i] + stack[i + 1])
        cycles.append((rng, mean))
    return cycles


# ---------------------------------------------------------------------------
# Piecewise-linear evaluation of the convex per-cycle fade function f(DoD).
# ---------------------------------------------------------------------------


def evaluate_piecewise(
    x: float,
    breakpoints: np.ndarray,
    values: np.ndarray,
) -> float:
    """Linear interpolation of a piecewise-linear function at scalar x.

    Args:
        x: Query point.
        breakpoints: (n,) monotonic increasing breakpoints.
        values: (n,) function values at breakpoints.

    Returns:
        Linearly interpolated value; clamped to the endpoints outside
        the breakpoint range.
    """
    return float(np.interp(x, breakpoints, values, left=values[0], right=values[-1]))


# ---------------------------------------------------------------------------
# Public wear function — drop-in replacement for physics_wear_from_duty.
# ---------------------------------------------------------------------------


def rainflow_wear_from_duty(
    day_result: "StackedDispatchResult",
    energy_mwh: float,
    soh_current: float,
    rainflow_coeffs: dict,
    capex_eur_per_mwh: float = 100_000.0,
    warranty_floor: float = 0.80,
    epsilon: float = 0.005,
    max_wear_eur_per_mwh: float = 500.0,
    age_accel_slope: float = 2.5,
) -> float:
    """Scalar EUR/MWh throughput cost from rainflow cycles of the observed dispatch.

    Pipeline:

    1. Extract rainflow cycles ``(DoD, mean_SoC)`` from ``day_result.soc``.
       SoC is normalised to fraction in [0, 1].
    2. For each cycle, evaluate the convex per-cycle aging function
       ``f(DoD)`` at the cycle DoD (linear interpolation between the
       calibrated breakpoints in ``rainflow_coeffs``).
    3. Sum to get total cyclic fade for the day.
    4. Apply age acceleration (matches L6: ``1 + 2.5 × (1 − SoH)``).
    5. Convert to per-MWh-throughput: divide by observed throughput
       (``2 × FEC × energy_mwh``).
    6. Monetise: multiply by ``CAPEX / headroom(SoH)``.
    7. Cap at ``max_wear_eur_per_mwh``.

    Returns ``0.0`` on zero-FEC days (no cycling observed) — calendar
    fade is captured separately in the SoH update loop.

    ``rainflow_coeffs`` schema: dict with keys
        ``dod_breakpoints`` — (n,) array, DoD support points (e.g. 0..1)
        ``cycle_fade``     — (n,) array, fractional SoH loss per single
            cycle of the corresponding DoD, calibrated at fixed mean-
            SoC, C-rate and temperature with kernel_scale=0.66
            (Note 4 manufacturer anchor).
    """
    fec_day = float(
        getattr(day_result, "full_equivalent_cycles", None)
        or getattr(day_result, "fec", 0.0) or 0.0
    )
    if fec_day < 1e-6 or energy_mwh <= 0:
        return 0.0
    soc_trace = np.asarray(day_result.soc, dtype=float).ravel()
    if soc_trace.size < 2:
        return 0.0
    soc_frac = np.clip(soc_trace / energy_mwh, 0.0, 1.0)
    cycles = extract_rainflow_cycles(soc_frac)
    if not cycles:
        return 0.0

    dod_brk = np.asarray(rainflow_coeffs["dod_breakpoints"], dtype=float)
    fade_brk = np.asarray(rainflow_coeffs["cycle_fade"], dtype=float)

    total_fade = 0.0
    for dod, _mean in cycles:
        total_fade += evaluate_piecewise(dod, dod_brk, fade_brk)

    # Age-acceleration multiplier (matches L6 convention)
    age_accel = 1.0 + age_accel_slope * max(0.0, 1.0 - soh_current)
    total_fade *= age_accel

    throughput_mwh = 2.0 * fec_day * energy_mwh
    if throughput_mwh < 1e-6:
        return 0.0
    fade_per_mwh = total_fade / throughput_mwh
    headroom = max(soh_current - warranty_floor, epsilon)
    wear = fade_per_mwh * capex_eur_per_mwh / headroom
    return float(min(wear, max_wear_eur_per_mwh))
