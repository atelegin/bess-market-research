"""
Production-grade LFP/graphite degradation model (Note 3).

Two-channel SoH(t):

    Qloss_cyc = B_cyc · exp(−Ea_cyc / R·T) · [FEC · DoD · C_nom]^z_cyc
              × f_dod_extra(DoD)                              # Wang 2011

    Qloss_cal = Σ_bucket (h_b / 8760) · k_cal(SoC_b) · t^β_cal
              × exp(−Ea_cal / R·T)                            # Naumann 2018

    eps_cell  ~ N(0, k_cyc · k_cyc_cov)                       # Severson 2019

    SoH(t)    = 1 − Qloss_cyc − Qloss_cal − eps_cell

Duty is represented as a ``DutyCycle`` with **first-class SoC distribution** —
the integration over SoC buckets is how the model captures the "high-SoC
dwell hurts" mechanism that separates FCR-style operation from arbitrage at
equal FEC. Limitations of the bucketing approach are documented inline and in
the Note 3 plan §2.

This is the production module. Simple closed-form parity lives in
``degradation.project_capacity_simple`` and is enforced by
``test_simple_detailed_parity.py`` per Note 3 §4.5.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from lib.models.degradation import CellPreset, ChemistryFamily, PRESETS


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

R_GAS = 8.617333262e-5  # eV / K (Boltzmann-eV form — matches Ea in eV)
T_REF_K = 298.15         # 25 °C reference

# Naumann-form continuous SoC dependence:
#   k_cal(SoC) = a + b · u + c · u³,   u = max(SoC − 0.5, 0)
#
# Coefficients solved so evaluation at bucket midpoints {0.25, 0.65, 0.90}
# reproduces the legacy 3-bucket averages {low=0.60, mid=1.00, high=1.60}
# exactly — all existing calibrations (Naumann 17-point fit, SimSES parity,
# each preset's test anchor) remain numerically invariant. The continuous
# form only diverges from buckets when caller supplies a finer SoC histogram
# or a non-midpoint SoC value, which is how future rest-SoC-sensitive work
# (Note 3b dispatch optimiser, fleet telemetry ingest) gets native resolution
# instead of three-step aliasing.
_SOC_LABEL_MIDPOINT: Dict[str, float] = {"low": 0.25, "mid": 0.65, "high": 0.90}
_DEFAULT_SOC_WEIGHTS: Dict[str, float] = {"low": 0.60, "mid": 1.00, "high": 1.60}


def _soc_weights_to_coeffs(weights: Dict[str, float]) -> Tuple[float, float, float]:
    """Solve (a, b, c) so ``_k_cal_of_soc`` at {0.25, 0.65, 0.90} reproduces
    legacy bucket weights {low, mid, high}. Lets per-preset overrides keep
    their bucket-weight API while running through the continuous evaluator.
    """
    low = float(weights.get("low", 0.60))
    mid = float(weights.get("mid", 1.00))
    high = float(weights.get("high", 1.60))
    a = low  # k_cal(0.25) = a, since u=0 below SoC=0.5
    # 0.15·b + 0.003375·c = mid − a   (u = 0.15 at SoC=0.65)
    # 0.40·b + 0.064·c    = high − a  (u = 0.40 at SoC=0.90)
    M = np.array([[0.15, 0.003375], [0.40, 0.064]])
    y = np.array([mid - a, high - a])
    b, c = np.linalg.solve(M, y)
    return (float(a), float(b), float(c))


_DEFAULT_SOC_COEFFS: Tuple[float, float, float] = _soc_weights_to_coeffs(_DEFAULT_SOC_WEIGHTS)


def _k_cal_of_soc(
    soc: float,
    coeffs: Tuple[float, float, float] = _DEFAULT_SOC_COEFFS,
) -> float:
    """Naumann-form calendar multiplier: ``k_cal(SoC) = a + b·u + c·u³`` where
    ``u = max(SoC − 0.5, 0)``. Flat below SoC=0.5 (no calendar acceleration
    below the thermodynamic midpoint), rises monotonically above it.
    """
    a, b, c = coeffs
    u = max(float(soc) - 0.5, 0.0)
    return a + b * u + c * (u ** 3)


def _label_to_soc(label: str) -> float:
    """Resolve a SoC-bucket label to a representative SoC fraction.

    Legacy labels {low, mid, high} → {0.25, 0.65, 0.90}. Any label castable
    to float is taken as the SoC value itself — lets callers pass finer
    histograms (e.g. via ``from_timeseries`` with custom ``bucket_edges``)
    and have the continuous evaluator respond natively.
    """
    if label in _SOC_LABEL_MIDPOINT:
        return _SOC_LABEL_MIDPOINT[label]
    try:
        return float(label)
    except ValueError:
        return 0.65  # fallback = legacy mid


# ────────────────────────────────────────────────────────────────────────────
# Duty cycle (first-class SoC distribution)
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DutyCycle:
    """Yearly operating profile, bucketed by SoC for calendar integration.

    ``soc_bucket_hours`` maps a bucket label ("low" | "mid" | "high", or a
    finer 10%-grid) to hours/year spent in that bucket. The dict is
    *independent* of ``mean_dod`` and ``fec_per_year`` — those drive the cycle
    channel, the SoC dict drives the calendar channel.

    Bucketing loses transition dynamics (see Note 3 §2 limitations). For
    stationary LFP this is acceptable; for future NMC work the timeseries
    factory is retained.
    """

    fec_per_year: float
    mean_dod: float
    soc_bucket_hours: Dict[str, float]
    mean_crate: float
    mean_temp_C: float
    soc_timeseries: Optional[np.ndarray] = None  # 8760 SoC (fraction) if provided

    @classmethod
    def from_mean(
        cls,
        fec_per_year: float,
        mean_dod: float,
        mean_soc: float = 0.55,
        mean_crate: float = 0.50,
        mean_temp_C: float = 25.0,
    ) -> "DutyCycle":
        """3-bucket SoC dwell layout, continuous in ``mean_soc``.

        Bucket hour allocations are linearly interpolated between four anchor
        ``mean_soc`` values — 0.30, 0.55, 0.75, 0.90 — where the allocation
        matches a trapezoidal SoC distribution centred on that mean. Legacy
        calibrations (Note 1 fleet baseline at 0.55, Trina anchors, SimSES
        parity scenarios) land on anchors and are numerically invariant; duties
        between anchors (0.65, 0.70, 0.80, …) get smoothly interpolated hour
        shares instead of the four-step branch jumps the previous version had.
        """
        s = float(np.clip(mean_soc, 0.0, 1.0))
        # (anchor mean_soc, {low, mid, high} hours). Below 0.30 and above 0.90
        # the allocation is clamped to the boundary anchor — matches the old
        # branch semantics there (anything ``< 0.50`` and anything ``>= 0.85``
        # got the same allocation regardless of exact value). Between anchors
        # the allocation is linearly interpolated instead of branch-stepped,
        # so mean_soc=0.65 vs 0.70 vs 0.80 differ smoothly in calendar load
        # — the real operator knob in Note 3b dispatch work.
        anchors = [
            (0.30, (6000.0, 2760.0, 0.0)),
            (0.55, (2000.0, 6000.0, 760.0)),
            (0.75, (500.0, 3760.0, 4500.0)),
            (0.90, (0.0, 2000.0, 6760.0)),
        ]
        if s <= anchors[0][0]:
            lo, mi, hi = anchors[0][1]
        elif s >= anchors[-1][0]:
            lo, mi, hi = anchors[-1][1]
        else:
            for i in range(1, len(anchors)):
                m_hi, alloc_hi = anchors[i]
                if s <= m_hi:
                    m_lo, alloc_lo = anchors[i - 1]
                    t = (s - m_lo) / (m_hi - m_lo)
                    lo = (1 - t) * alloc_lo[0] + t * alloc_hi[0]
                    mi = (1 - t) * alloc_lo[1] + t * alloc_hi[1]
                    hi = (1 - t) * alloc_lo[2] + t * alloc_hi[2]
                    break
        hours = {"low": float(lo), "mid": float(mi), "high": float(hi)}
        return cls(
            fec_per_year=fec_per_year,
            mean_dod=mean_dod,
            soc_bucket_hours=hours,
            mean_crate=mean_crate,
            mean_temp_C=mean_temp_C,
        )

    @classmethod
    def from_timeseries(
        cls,
        soc: np.ndarray,
        fec_per_year: float,
        mean_dod: float,
        mean_crate: float = 0.50,
        mean_temp_C: float = 25.0,
        bucket_edges: tuple = (0.0, 0.5, 0.8, 1.0),
    ) -> "DutyCycle":
        """Aggregate 8760-h SoC trace into buckets. Finer than 10% is just noise."""
        if len(soc) < 1:
            raise ValueError("soc timeseries must not be empty")
        hours_per_sample = 8760.0 / len(soc)
        hours = {}
        labels = ["low", "mid", "high"] if len(bucket_edges) == 4 else [
            f"b{i}" for i in range(len(bucket_edges) - 1)
        ]
        for i in range(len(bucket_edges) - 1):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            mask = (soc >= lo) & (soc < hi) if i < len(bucket_edges) - 2 else (soc >= lo) & (soc <= hi)
            hours[labels[i]] = float(mask.sum() * hours_per_sample)
        return cls(
            fec_per_year=fec_per_year,
            mean_dod=mean_dod,
            soc_bucket_hours=hours,
            mean_crate=mean_crate,
            mean_temp_C=mean_temp_C,
            soc_timeseries=np.asarray(soc, dtype=float),
        )

    @classmethod
    def constant_soc_band(
        cls,
        soc_low: float,
        soc_high: float,
        fec_per_year: float,
        mean_crate: float = 0.50,
        mean_temp_C: float = 25.0,
    ) -> "DutyCycle":
        """Band-bounded SoC (e.g. FCR narrow ±2% around 50%)."""
        mean_dod = max(soc_high - soc_low, 0.0)
        mean_soc = 0.5 * (soc_low + soc_high)
        return cls.from_mean(
            fec_per_year=fec_per_year,
            mean_dod=mean_dod,
            mean_soc=mean_soc,
            mean_crate=mean_crate,
            mean_temp_C=mean_temp_C,
        )

    @property
    def total_hours(self) -> float:
        return float(sum(self.soc_bucket_hours.values()))


# ────────────────────────────────────────────────────────────────────────────
# Chemistry aging kernel — abstract + LFP/graphite implementation
# ────────────────────────────────────────────────────────────────────────────


class ChemistryAgingKernel:
    """Abstract interface. Concrete kernels implement Wang/Naumann or future NMC."""

    family: ChemistryFamily

    def cycle_loss(
        self,
        fec: float,
        dod: float,
        c_rate: float,
        temp_C: float,
        preset: CellPreset,
    ) -> float:
        raise NotImplementedError

    def calendar_loss(
        self,
        elapsed_years: float,
        soc_bucket_hours: Dict[str, float],
        temp_C: float,
        preset: CellPreset,
    ) -> float:
        raise NotImplementedError


def _arrhenius(temp_C: float, Ea_eV: float) -> float:
    """Arrhenius ratio referenced at 25 °C."""
    T_K = float(temp_C) + 273.15
    return float(np.exp(-Ea_eV / R_GAS * (1.0 / T_K - 1.0 / T_REF_K)))


def _f_dod_extra(dod: float) -> float:
    """Empirical super-linear DoD multiplier (Xu 2018 / himax)."""
    return float((max(dod, 1e-3) / 0.80) ** 0.5)


class LFPGraphiteWangNaumann(ChemistryAgingKernel):
    """Wang 2011 cycle × Naumann 2018 calendar kernel for LFP/graphite."""

    family = ChemistryFamily.LFP_GRAPHITE

    def cycle_loss(
        self,
        fec: float,
        dod: float,
        c_rate: float,
        temp_C: float,
        preset: CellPreset,
    ) -> float:
        if fec <= 0:
            return 0.0
        # Stationary-BESS C-rate branch: separate Wang FEC·DoD stress from the
        # C-rate factor so per-preset ``c_rate_exponent`` can bend it without
        # distorting the FEC term. At ``c_rate_exponent=1.0`` and a 0.5C anchor
        # the old ``stress = FEC·DoD·C_eff`` behaviour is recovered (k_cyc
        # absorbs the anchor's C^z scale).
        c_eff = max(c_rate, 1e-3)
        lo, hi = preset.c_rate_range
        if c_rate > hi:
            warnings.warn(
                f"C-rate {c_rate:.2f} exceeds preset range {hi:.2f}; cycle loss is conservative overestimate.",
                stacklevel=2,
            )
        arr = _arrhenius(temp_C, preset.Ea_cyc_eV)
        stress = fec * max(dod, 1e-6)
        c_factor = c_eff ** (preset.c_rate_exponent * preset.z_cyc)
        return float(
            preset.k_cyc * arr * (stress ** preset.z_cyc) * c_factor * _f_dod_extra(dod)
        )

    def calendar_loss(
        self,
        elapsed_years: float,
        soc_bucket_hours: Dict[str, float],
        temp_C: float,
        preset: CellPreset,
    ) -> float:
        if elapsed_years <= 0:
            return 0.0
        total_h = sum(soc_bucket_hours.values())
        if total_h <= 0:
            return 0.0
        arr = _arrhenius(temp_C, preset.Ea_cal_eV)
        t = max(elapsed_years, 0.0)
        # Continuous Naumann-form SoC dependence. Per-preset bucket-weight
        # overrides (e.g. Trina's steeper high/low ratio from its 40%/100% SoC
        # anchors) are converted to continuous coefficients that reproduce the
        # bucket values at label midpoints — so bucket-API callers are
        # numerically invariant, while timeseries or finer-histogram callers
        # get native SoC resolution.
        if preset.calendar_soc_weights:
            coeffs = _soc_weights_to_coeffs(preset.calendar_soc_weights)
        else:
            coeffs = _DEFAULT_SOC_COEFFS
        bucket_weight = 0.0
        for label, hours in soc_bucket_hours.items():
            frac = hours / total_h
            w = _k_cal_of_soc(_label_to_soc(label), coeffs)
            bucket_weight += frac * w
        return float(preset.k_cal * bucket_weight * (t ** preset.beta_cal) * arr)


_DEFAULT_KERNEL = LFPGraphiteWangNaumann()


# ────────────────────────────────────────────────────────────────────────────
# Public model API
# ────────────────────────────────────────────────────────────────────────────


def _validate_duty_in_range(duty: DutyCycle, preset: CellPreset) -> None:
    t_lo, t_hi = preset.temp_range_C
    if not (t_lo <= duty.mean_temp_C <= t_hi):
        warnings.warn(
            f"duty.mean_temp_C={duty.mean_temp_C} outside preset range [{t_lo}, {t_hi}]; "
            "extrapolating with edge value — treat result as illustrative.",
            stacklevel=2,
        )


def cell_soh_detailed(
    duty: DutyCycle,
    years: float,
    preset: CellPreset,
    n_mc: int = 1,
    rng: Optional[np.random.Generator] = None,
    kernel: Optional[ChemistryAgingKernel] = None,
) -> Union[float, np.ndarray]:
    """SoH at ``years`` for a single cell under ``duty``.

    ``n_mc=1`` returns a scalar (median cell, no variation).
    ``n_mc>1`` returns an array of SoH samples over Monte Carlo cells.
    """
    _validate_duty_in_range(duty, preset)
    kernel = kernel or _DEFAULT_KERNEL

    q_cyc = kernel.cycle_loss(
        fec=float(duty.fec_per_year) * years,
        dod=duty.mean_dod,
        c_rate=duty.mean_crate,
        temp_C=duty.mean_temp_C,
        preset=preset,
    )
    q_cal = kernel.calendar_loss(
        elapsed_years=years,
        soc_bucket_hours=duty.soc_bucket_hours,
        temp_C=duty.mean_temp_C,
        preset=preset,
    )

    soh_base = 1.0 - q_cyc - q_cal
    if n_mc <= 1:
        return float(max(soh_base, 0.0))

    rng = rng if rng is not None else np.random.default_rng(0)
    # Noise scales with actually-accumulated loss in each channel. The old
    # formulation tied sigma only to cycle stress (``k_cyc · CoV · FEC^z``),
    # which meant a cell stored at high SoC with zero FEC had zero Monte
    # Carlo spread — wrong whenever calendar dominates (FCR-narrow duty,
    # storage-heavy cases, post-retirement estimates). Channels assumed
    # independent; noise sources combine in quadrature.
    sigma_cyc = preset.k_cyc_cov * abs(q_cyc)
    sigma_cal = preset.k_cal_cov * abs(q_cal)
    sigma = float(np.sqrt(sigma_cyc ** 2 + sigma_cal ** 2))
    eps = rng.normal(0.0, sigma, size=n_mc)
    return np.clip(soh_base - eps, 0.0, 1.0)


def project_capacity_detailed(
    duty: DutyCycle,
    years: float,
    preset: CellPreset,
    n_mc: int = 200,
    return_kind: str = "pack",
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Capacity fraction with cell-to-cell variation.

    ``return_kind``:
      - ``"pack"``   — 10th percentile (weakest-decile-drives-pack convention).
      - ``"median"`` — 50th percentile.
      - ``"distribution"`` — returns 3-tuple ``(p10, p50, p90)``.
    """
    if n_mc <= 1:
        scalar = cell_soh_detailed(duty=duty, years=years, preset=preset, n_mc=1, rng=rng)
        if return_kind == "distribution":
            return (scalar, scalar, scalar)  # type: ignore[return-value]
        return float(scalar)  # type: ignore[arg-type]

    samples = cell_soh_detailed(duty=duty, years=years, preset=preset, n_mc=n_mc, rng=rng)
    samples = np.asarray(samples)
    p10, p50, p90 = np.percentile(samples, [10, 50, 90])
    if return_kind == "pack":
        return float(p10)
    if return_kind == "median":
        return float(p50)
    if return_kind == "distribution":
        return (float(p10), float(p50), float(p90))  # type: ignore[return-value]
    raise ValueError(f"unknown return_kind={return_kind!r}")


def lifecycle_value_detailed(
    year1_revenue: float,
    duty: DutyCycle,
    preset: CellPreset,
    years: int,
    discount_rate: float = 0.08,
    annual_market_decline: float = 0.0,
    n_mc: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Lifetime revenue/NPV profile using detailed SoH(t).

    Uses the pack-level SoH (10th percentile) for revenue attenuation,
    matching how a BESS owner would actually see capacity.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    records = []
    cum_rev = 0.0
    cum_disc = 0.0
    for year in range(1, years + 1):
        elapsed = year - 1
        cap = project_capacity_detailed(
            duty=duty,
            years=max(elapsed, 1e-6),
            preset=preset,
            n_mc=n_mc,
            return_kind="pack",
            rng=rng,
        )
        retired = bool(elapsed > 0 and cap <= preset.eol_capacity_fraction)
        market_factor = (1 + annual_market_decline) ** elapsed
        annual = 0.0 if retired else float(year1_revenue * cap * market_factor)
        disc = 0.0 if retired else float(annual / (1 + discount_rate) ** elapsed)
        cum_rev += annual
        cum_disc += disc
        records.append(
            {
                "year": year,
                "capacity_fraction_start": cap,
                "annual_revenue_eur_per_mw": annual,
                "discounted_revenue_eur_per_mw": disc,
                "cumulative_revenue_eur_per_mw": cum_rev,
                "cumulative_discounted_revenue_eur_per_mw": cum_disc,
                "retired": retired,
            }
        )
    return pd.DataFrame(records)
