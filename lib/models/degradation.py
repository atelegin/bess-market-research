"""
Battery degradation model — calendar and cycle ageing (simple closed-form).

This module is the **fast approximation** of the production degradation model.
It preserves the legacy closed-form SoH(t) that Notes 1, 2, and Best-Days rely
on, while adding a `CellPreset` layer so each duty/chemistry combination has
a single identity that travels across the library.

For the production Wang 2011 + Naumann 2018 model with Monte Carlo cell-to-cell
variation see ``lib.models.degradation_detailed``. The two are kept in parity
at baseline duty per Note 3 §4.5 (``test_simple_detailed_parity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Chemistry taxonomy
# ────────────────────────────────────────────────────────────────────────────


class ChemistryFamily(Enum):
    """Cathode/anode chemistry family. LFP/graphite only in this release.

    NMC/LTO/NCA are placeholders — calibration (e.g. Schmalstieg 2014 for NMC)
    is tracked in ROADMAP and will plug in through ``ChemistryAgingKernel``
    in ``degradation_detailed.py`` without breaking this API.
    """

    LFP_GRAPHITE = "lfp_graphite"
    NMC_GRAPHITE = "nmc_graphite"   # reserved, not calibrated
    LTO = "lto"                     # reserved, not calibrated
    NCA = "nca"                     # reserved, not calibrated


@dataclass(frozen=True)
class TestAnchor:
    """Lab anchor: constant T, single DoD, controlled C-rate, fresh cells.

    Used to calibrate ``k_cyc``. This is the physics anchor, not a warranty.
    """

    cycles: float
    dod: float
    c_rate: float
    temp_C: float
    retention: float          # capacity fraction at ``cycles`` (e.g. 0.80)
    source: str


@dataclass(frozen=True)
class WarrantyAnchor:
    """OEM system warranty: field-realistic, margin-loaded.

    Always weaker than ``TestAnchor`` — the gap captures field T-swings,
    partial cycles, cell-to-cell spread, and OEM commercial margin. Used
    in tornado as a conservative bound and as a SoH(t) sanity check.
    """

    cycles: float
    dod: float
    retention: float
    source: str


@dataclass(frozen=True)
class CellPreset:
    """A single cell/pack identity with provenance and aging parameters.

    Fields are intentionally verbose so one ``CellPreset`` carries every
    number the detailed model needs — no side lookups by name downstream.

    ``calibration_status`` is the honest tag of how well this preset is
    pinned to real data — downstream code and notes should cite
    ``multi_anchor`` presets with more confidence than ``single_anchor_*``
    ones. See module docstring for the per-preset table.
    """

    name: str
    manufacturer: str
    chemistry: ChemistryFamily
    source_url: str
    # Wang 2011 cycle term
    k_cyc: float                  # calibrated per preset
    alpha_cyc: float              # DoD super-linear exponent (empirical)
    z_cyc: float                  # FEC exponent (Wang)
    Ea_cyc_eV: float              # activation energy, cycle channel
    c_rate_exponent: float        # C-rate stress exponent (1.0 = Wang linear)
    # Self-heating: T_cell = T_amb + coeff · C_rate². Default 0.0 — callers
    # pass cell-internal temperature directly. Set >0 (typical prismatic LFP
    # ~2.0 °C/C² at 2C → +8 °C rise) when the caller is passing ambient and
    # the duty has meaningful C-rate. Only applied to the cycle channel —
    # active-cycling fraction of wall-clock year is small (<20% even for
    # arbitrage) so calendar integration uses the storage temperature.
    self_heating_coeff_C_per_C2: float
    # Naumann 2018 calendar term
    k_cal: float                  # base calendar pre-factor
    beta_cal: float               # time exponent (Naumann)
    Ea_cal_eV: float              # activation energy, calendar channel
    # Cell-to-cell spread (Severson 2019 cycling default).
    # ``k_cyc_cov`` applies to the cycle channel; ``k_cal_cov`` to calendar.
    # Separable so future calibration work (e.g. Lam 2024 K2 cell-to-cell
    # spread) can tighten one without touching the other. Default both to
    # Severson's 8% — the cleanest single published LFP number — pending a
    # matched calendar-aging spread study.
    k_cyc_cov: float
    k_cal_cov: float
    # EOL convention
    eol_capacity_fraction: float
    # Ranges where the model is considered valid
    temp_range_C: tuple
    c_rate_range: tuple
    # Anchors
    test_anchor: TestAnchor
    warranty_anchor: Optional[WarrantyAnchor]
    # Legacy simple-model parameters (fit to detailed baseline per §4.5)
    cal_budget: float             # calendar capacity budget (fraction)
    cycle_budget: float           # cycle capacity budget (fraction)
    ref_fec: float                # reference FEC for cycle term
    cal_life_years: float         # calendar life at anchor
    # Honest calibration tag — one of:
    #   "multi_anchor"            — ≥2 independent datasheet anchors
    #   "single_anchor_datasheet" — one lab/datasheet anchor
    #   "single_anchor_marketing" — one brochure/marketing anchor only
    #   "synthetic"               — internal fleet-average, not a real cell
    calibration_status: str = "single_anchor_datasheet"
    # Per-preset calendar SoC weights (overrides shared Naumann defaults).
    # ``None`` → use shared ``_soc_bucket_k_cal`` from degradation_detailed.
    calendar_soc_weights: Optional[Dict[str, float]] = None
    notes: str = ""


# ────────────────────────────────────────────────────────────────────────────
# Public presets — four LFP cells + internal baseline
# ────────────────────────────────────────────────────────────────────────────
#
# Physics defaults (shared across LFP/graphite presets):
#   z_cyc  = 0.55   (Wang 2011 fit)
#   α      = 0.50   (Xu 2018 / empirical DoD super-linear extra)
#   β_cal  = 0.50   (Naumann 2018 sqrt-of-time)
#   Ea_cyc = 0.30 eV
#   Ea_cal = 0.55 eV
#   CoV    = 0.08   (Severson 2019)
# ``k_cyc`` is fit per preset to ``test_anchor`` at baseline (see §4.3).
# The simple-model budgets (cal_budget, cycle_budget) are fit to the detailed
# baseline trajectory (see §4.5 — enforced by ``test_simple_detailed_parity``).

_LFP_DEFAULTS = dict(
    chemistry=ChemistryFamily.LFP_GRAPHITE,
    # alpha_cyc here is the simple-path FEC exponent; kept at 1.0 so the simple
    # closed-form matches the detailed model's linear-in-FEC cycle loss at
    # baseline duty (parity 0pp, see ``test_simple_detailed_parity``).
    alpha_cyc=1.00,
    # z_cyc = 1.0 (linear-in-FEC cycle loss) for stationary LFP/graphite duty.
    # Wang, Liu, Kloess 2011 fit z≈0.55 on A123 ANR26650 LFP cells, but that
    # exponent produces a concave-in-FEC trajectory that front-loads ~25% of
    # lifetime cycle loss into year 1 — inconsistent with stationary BESS field
    # data at moderate C-rate. Naumann 2020 LFP cycle-aging refinement and
    # Sarasketa-Zabala 2014 (LFP, EVE/GS Yuasa cells) both support a near-linear
    # FEC dependence when temperature and C-rate are separated out, which is
    # what this kernel does. We keep the "Wang 2011" label for the DoD super-
    # linear multiplier and the Arrhenius form, and use Naumann/Sarasketa for
    # the FEC exponent.
    z_cyc=1.00,
    Ea_cyc_eV=0.30,
    c_rate_exponent=1.00,   # Wang linear-C default; overridden for multi-C-anchor presets
    self_heating_coeff_C_per_C2=0.00,  # off by default; callers pass cell-internal T
    beta_cal=0.50,
    Ea_cal_eV=0.55,
    k_cyc_cov=0.08,
    k_cal_cov=0.08,
    eol_capacity_fraction=0.70,
    temp_range_C=(15.0, 60.0),
    c_rate_range=(0.0, 2.0),
)


PRESETS: Dict[str, CellPreset] = {
    "eve_lf280k": CellPreset(
        name="eve_lf280k",
        manufacturer="EVE Energy",
        source_url=(
            "https://www.battery-germany.de/wp-content/uploads/2022/02/"
            "LF280K-280Ah-Product-Specification-Version-B.pdf"
        ),
        # Fit to 25°C anchor with literature-grounded k_cal (see Trina validation).
        # The datasheet 45°C/2500 cycle point is a combined (cycle+accelerated-calendar)
        # endurance test; our two-channel separation diverges ~13pp there, documented
        # as an honest known limitation — not a fit target.
        k_cyc=5.0520e-5,
        k_cal=0.0318,
        test_anchor=TestAnchor(
            cycles=6000, dod=1.00, c_rate=0.5, temp_C=25.0, retention=0.80,
            source="EVE LF280K spec v.B — 25°C 0.5C/0.5C endurance curve",
        ),
        warranty_anchor=None,
        cal_budget=0.1366,
        cycle_budget=0.1516,
        ref_fec=6000.0,
        cal_life_years=20.0,
        calibration_status="single_anchor_datasheet",
        notes=(
            "Calibrated to 25°C anchor only. Datasheet also shows 2500 cls @ 45°C to 80% — "
            "model predicts ~66% there, 14pp below the datasheet point; treated as "
            "informational, not a fit target, because two-channel separation cannot cleanly "
            "resolve combined cycle+accelerated-calendar endurance tests. k_cal borrowed from "
            "Trina's calendar-validated value as literature-grounded LFP proxy."
        ),
        **_LFP_DEFAULTS,
    ),
    "catl_enerc_plus_306ah": CellPreset(
        name="catl_enerc_plus_306ah",
        manufacturer="CATL",
        source_url=(
            "https://www.catl.com/en/uploads/1/file/public/202303/"
            "20230315092000_ahw9vpn63j.pdf"
        ),
        k_cyc=6.6354e-5,
        k_cal=0.042,
        test_anchor=TestAnchor(
            cycles=7000, dod=1.00, c_rate=0.5, temp_C=25.0, retention=0.70,
            source="CATL EnerC+ brochure (6000-8000 cls to 70%, no test-condition footer)",
        ),
        warranty_anchor=WarrantyAnchor(
            cycles=7300, dod=0.90, retention=0.70,
            source="CATL BESS brochure 2023 (20-yr system warranty, typical)",
        ),
        cal_budget=0.1805,
        cycle_budget=0.2322,
        ref_fec=7000.0,
        cal_life_years=20.0,
        calibration_status="single_anchor_marketing",
        notes=(
            "Only a brochure-level cycle claim (no explicit T / DoD / C-rate footer) and a "
            "system-warranty number. No calendar test data. Use retention numbers from this "
            "preset with noticeably less confidence than eve_lf280k or trina_elementa_280ah."
        ),
        **_LFP_DEFAULTS,
    ),
    "byd_mc_cube_t": CellPreset(
        name="byd_mc_cube_t",
        manufacturer="BYD",
        source_url="https://www.bydenergy.com/en/productDetails/Utility-Scale/MC_Cube-T_BESS",
        k_cyc=2.2940e-5,
        k_cal=0.048,
        test_anchor=TestAnchor(
            cycles=12000, dod=1.00, c_rate=0.5, temp_C=25.0, retention=0.80,
            source="BYD MC Cube-T product page (marketing claim, no test footer)",
        ),
        warranty_anchor=WarrantyAnchor(
            cycles=8000, dod=0.90, retention=0.70,
            source="MC Cube-T datasheet retention curve (commercial)",
        ),
        cal_budget=0.2062,
        cycle_budget=0.1376,
        ref_fec=12000.0,
        cal_life_years=20.0,
        calibration_status="single_anchor_marketing",
        notes=(
            "Single marketing anchor (12000 cls / 80% SoH). No calendar data. Warranty "
            "curve used as conservative tornado bound only. Treat retention numbers from "
            "this preset as the most optimistic of the four LFP presets — the marketing "
            "anchor assumes near-ideal conditions."
        ),
        **_LFP_DEFAULTS,
    ),
    "trina_elementa_280ah": CellPreset(
        name="trina_elementa_280ah",
        manufacturer="Trina Storage",
        source_url="https://www.trinasolar.com/en-apac/storage/elementa",
        # Multi-anchor calibration:
        #   cycle:    10000 cls @ 25°C/0.5P/1.0 DoD → 70% SoH
        #   calendar: 100% SoC / 25°C / 10000 days → 70% retention
        #             40%  SoC / 25°C / 2 years    → ≥98% retention
        # Calendar SoC weights fitted to the high/low storage anchor ratio (~4.0),
        # steeper than Naumann 2018 defaults (~2.67).
        k_cyc=4.8297e-5,
        k_cal=0.0318,
        test_anchor=TestAnchor(
            cycles=10000, dod=1.00, c_rate=0.5, temp_C=25.0, retention=0.70,
            source="Trina Elementa 280Ah cell spec (internal AES Technology Research Institute report, 2025.04)",
        ),
        warranty_anchor=WarrantyAnchor(
            cycles=7300, dod=0.90, retention=0.70,
            source="Elementa warranty curve (LTSA-typical)",
        ),
        cal_budget=0.1342,
        cycle_budget=0.2415,
        ref_fec=10000.0,
        cal_life_years=20.0,
        calibration_status="multi_anchor",
        calendar_soc_weights={"low": 0.45, "mid": 1.00, "high": 1.80},
        notes=(
            "Best-calibrated preset: cycle + two calendar (40% and 100% SoC) anchors all "
            "reproduce to ±0.5pp. Strübbel LTSA project uses Trina cells — this preset is "
            "the basis for that retrofit. Public source URL points to Trina marketing; "
            "the multi-point anchor set comes from an internal AES reference shared with "
            "BayWa r.e., not publicly quoted."
        ),
        **_LFP_DEFAULTS,
    ),
    "baseline_fleet": CellPreset(
        name="baseline_fleet",
        manufacturer="Synthetic (fleet-average)",
        source_url="internal://lib.config.fleet_degradation_factor (legacy)",
        # Fit so that 2 c/d, DoD 0.80, mean SoC 0.55, 25 °C, 0.5C reproduces
        # the legacy ~1.8%/yr effective fade (±0.2pp through year 20).
        k_cyc=5.2042e-5,
        k_cal=0.050,
        test_anchor=TestAnchor(
            cycles=7300, dod=0.80, c_rate=0.5, temp_C=25.0, retention=0.80,
            source="Internal calibration to legacy ANNUAL_FADE_RATE=0.018",
        ),
        warranty_anchor=None,
        cal_budget=0.2148,
        cycle_budget=0.1520,
        ref_fec=7300.0,
        cal_life_years=20.0,
        calibration_status="synthetic",
        notes=(
            "Synthetic anchor — fleet-average surrogate for pre-preset fleet projections "
            "in Note 1. Not a real cell — fitted to legacy linear-fade shape. Keeps Note 1 "
            "numbers stable to ±0.2pp vs the retired ANNUAL_FADE_RATE constant."
        ),
        **_LFP_DEFAULTS,
    ),
}


# ────────────────────────────────────────────────────────────────────────────
# Legacy API (kept verbatim for Notes 1 / 2 / Best-Days / Strübbel)
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DegradationAssumptions:
    reference_cycle_life_fec: float = 7300.0
    calendar_life_years: float = 20.0
    eol_capacity_fraction: float = 0.60
    calendar_capacity_loss_at_reference: float = 0.18
    dod_reference: float = 1.0
    dod_exponent: float = 1.30
    cycle_fade_exponent: float = 1.25

    @property
    def reference_warranty_fec_per_year(self) -> float:
        return self.reference_cycle_life_fec / self.calendar_life_years


DEFAULT_DEGRADATION_ASSUMPTIONS = DegradationAssumptions()


def compute_annual_degradation(
    cycles_per_year: float,
    avg_dod: float,
    base_cycle_degradation: float = 0.003,
    dod_exponent: float = 1.5,
    calendar_aging: float = 0.008,
) -> float:
    dod_factor = (max(avg_dod, 1e-6) / 0.8) ** dod_exponent
    cycle_deg = cycles_per_year * (base_cycle_degradation / 100) * dod_factor
    return calendar_aging + cycle_deg


def equivalent_stress_fec_per_year(
    dispatch_frame: pd.DataFrame,
    assumptions: DegradationAssumptions = DEFAULT_DEGRADATION_ASSUMPTIONS,
) -> float:
    if dispatch_frame.empty:
        return 0.0
    return float(dispatch_frame["full_equivalent_cycles"].fillna(0.0).sum())


def project_capacity_fraction(
    elapsed_years: float,
    annual_stress_fec: float,
    assumptions: DegradationAssumptions = DEFAULT_DEGRADATION_ASSUMPTIONS,
) -> float:
    if elapsed_years <= 0:
        return 1.0
    cycle_loss_budget = max(
        1.0 - assumptions.eol_capacity_fraction - assumptions.calendar_capacity_loss_at_reference,
        0.0,
    )
    calendar_loss = assumptions.calendar_capacity_loss_at_reference * np.sqrt(
        max(elapsed_years / max(assumptions.calendar_life_years, 1e-9), 0.0)
    )
    cycle_ratio = max(
        (annual_stress_fec * elapsed_years) / max(assumptions.reference_cycle_life_fec, 1e-9),
        0.0,
    )
    cycle_loss = cycle_loss_budget * (cycle_ratio ** assumptions.cycle_fade_exponent)
    return float(max(1.0 - calendar_loss - cycle_loss, 0.0))


def estimate_years_to_eol(
    dispatch_frame: pd.DataFrame,
    assumptions: DegradationAssumptions = DEFAULT_DEGRADATION_ASSUMPTIONS,
    max_years: float = 40.0,
    step_years: float = 1.0 / 12.0,
) -> float:
    annual_stress_fec = equivalent_stress_fec_per_year(dispatch_frame, assumptions=assumptions)
    elapsed_years = 0.0
    while elapsed_years <= max_years:
        capacity_fraction = project_capacity_fraction(
            elapsed_years=elapsed_years,
            annual_stress_fec=annual_stress_fec,
            assumptions=assumptions,
        )
        if capacity_fraction <= assumptions.eol_capacity_fraction:
            return float(elapsed_years)
        elapsed_years += step_years
    return float(max_years)


def lifecycle_value_profile(
    year1_revenue: float,
    dispatch_frame: pd.DataFrame,
    years: int,
    discount_rate: float = 0.0,
    annual_market_decline: float = 0.0,
    assumptions: DegradationAssumptions = DEFAULT_DEGRADATION_ASSUMPTIONS,
) -> pd.DataFrame:
    annual_stress_fec = equivalent_stress_fec_per_year(dispatch_frame, assumptions=assumptions)
    records = []
    cumulative_revenue = 0.0
    cumulative_discounted = 0.0
    for year in range(1, years + 1):
        elapsed_years = year - 1
        capacity_fraction = project_capacity_fraction(
            elapsed_years=elapsed_years,
            annual_stress_fec=annual_stress_fec,
            assumptions=assumptions,
        )
        retired = bool(
            elapsed_years > 0 and capacity_fraction <= assumptions.eol_capacity_fraction
        )
        market_factor = (1 + annual_market_decline) ** elapsed_years
        annual_revenue = (
            0.0 if retired else float(year1_revenue * capacity_fraction * market_factor)
        )
        discounted_revenue = (
            0.0 if retired else float(annual_revenue / (1 + discount_rate) ** elapsed_years)
        )
        cumulative_revenue += annual_revenue
        cumulative_discounted += discounted_revenue
        records.append(
            {
                "year": year,
                "capacity_fraction_start": capacity_fraction,
                "annual_revenue_eur_per_mw": annual_revenue,
                "discounted_revenue_eur_per_mw": discounted_revenue,
                "cumulative_revenue_eur_per_mw": cumulative_revenue,
                "cumulative_discounted_revenue_eur_per_mw": cumulative_discounted,
                "retired": retired,
            }
        )
    return pd.DataFrame(records)


def summarize_dispatch_degradation(dispatch_frame: pd.DataFrame) -> dict[str, float]:
    cycles_per_year = float(dispatch_frame["cycles"].sum())
    if cycles_per_year > 0:
        avg_dod = float(np.average(dispatch_frame["avg_dod"], weights=dispatch_frame["cycles"]))
    else:
        avg_dod = 0.0
    annual_degradation = compute_annual_degradation(
        cycles_per_year=cycles_per_year, avg_dod=avg_dod
    )
    return {
        "cycles_per_year": cycles_per_year,
        "full_equivalent_cycles_per_year": (
            float(dispatch_frame["full_equivalent_cycles"].sum())
            if "full_equivalent_cycles" in dispatch_frame
            else 0.0
        ),
        "stress_fec_per_year": equivalent_stress_fec_per_year(dispatch_frame),
        "avg_dod": avg_dod,
        "annual_degradation": annual_degradation,
        "years_to_eol": estimate_years_to_eol(dispatch_frame),
    }


def capacity_trajectory(annual_degradation: float, years: int) -> np.ndarray:
    return np.array([(1 - annual_degradation) ** year for year in range(years + 1)])


def cumulative_revenue_profile(
    year1_revenue: float,
    annual_degradation: float,
    years: int,
) -> pd.DataFrame:
    capacity = capacity_trajectory(annual_degradation=annual_degradation, years=years)
    annual_revenue = year1_revenue * capacity[:-1]
    return pd.DataFrame(
        {
            "year": np.arange(1, years + 1),
            "annual_revenue_eur_per_mw": annual_revenue,
            "cumulative_revenue_eur_per_mw": np.cumsum(annual_revenue),
            "capacity_fraction": capacity[:-1],
        }
    )


# ────────────────────────────────────────────────────────────────────────────
# Preset-aware simple model (§4.1)
# ────────────────────────────────────────────────────────────────────────────


def project_capacity_simple(
    fec_per_year: float,
    dod: float,
    elapsed_years: float,
    preset: CellPreset,
) -> float:
    """Closed-form SoH(t) for a preset + duty summary.

    ``SoH = 1 − cal_budget · sqrt(t / cal_life)
             − cycle_budget · (FEC·DoD / (ref_FEC · dod_ref))^α``

    Cheap and deterministic. Fits Notes 1/2/Best-Days use cases. For duty with
    SoC-distribution, temperature, or cell-to-cell variation use
    ``degradation_detailed.project_capacity_detailed``.
    """
    if elapsed_years <= 0:
        return 1.0
    cal_loss = preset.cal_budget * np.sqrt(
        max(elapsed_years / max(preset.cal_life_years, 1e-9), 0.0)
    )
    dod_ref = preset.test_anchor.dod
    fec_total = fec_per_year * elapsed_years * (max(dod, 1e-6) / max(dod_ref, 1e-6))
    cycle_ratio = fec_total / max(preset.ref_fec, 1e-9)
    cycle_loss = preset.cycle_budget * (cycle_ratio ** preset.alpha_cyc)
    return float(max(1.0 - cal_loss - cycle_loss, 0.0))


# ── Baseline fleet duty (used by projection.py through fleet_average_capacity) ──
# (2 c/d, DoD 0.80, mean SoC 0.55, 25 °C, 0.5C) — the legacy anchor.
DEFAULT_FLEET_DUTY = {
    "fec_per_year": 730.0,
    "dod": 0.80,
}


def _fleet_cohort_capacity(
    age_years: float,
    preset: CellPreset,
    duty: Optional[dict] = None,
) -> float:
    """Effective capacity of a single cohort aged ``age_years``.

    For the ``baseline_fleet`` preset this uses the legacy linear-fade +
    one-shot-augmentation shape so Note 1 fleet numbers are preserved to
    ±0.2pp. For any other preset it defers to the closed-form preset model
    (``project_capacity_simple``) — those presets have their own test anchors
    and are not supposed to match the legacy synthetic fleet curve.

    The inline augmentation restore here is *not* the augmentation framework —
    that lives in a future deep-dive (see Note 3 §2, Note 4).
    """
    duty = duty or DEFAULT_FLEET_DUTY
    fec = float(duty["fec_per_year"])
    dod = float(duty["dod"])

    if preset.name == "baseline_fleet":
        fade_rate = 0.018
        aug_year = 8500.0 / max(fec, 1.0)
        if age_years < aug_year:
            cap = 1.0 - fade_rate * max(age_years, 0.0)
        else:
            cap = 0.92 - fade_rate * (age_years - aug_year)
        return float(max(cap, 0.50))

    return project_capacity_simple(
        fec_per_year=fec,
        dod=dod,
        elapsed_years=age_years,
        preset=preset,
    )


def fleet_average_capacity(
    year: int,
    buildout: Dict[int, float],
    preset: CellPreset = None,
    duty: Optional[dict] = None,
) -> float:
    """MW-weighted fleet-average capacity factor for ``year``.

    Replaces the retired ``lib.config.fleet_degradation_factor``. Same
    cohort-weighted math, but capacity per cohort comes from a ``CellPreset``
    (default ``baseline_fleet``, calibrated to legacy ±0.2pp) rather than the
    hard-coded ``ANNUAL_FADE_RATE`` constant.
    """
    if preset is None:
        preset = PRESETS["baseline_fleet"]

    sorted_years = sorted(buildout.keys())
    total_mw = 0.0
    weighted_cap = 0.0

    for i, vy in enumerate(sorted_years):
        if vy > year:
            break
        prev_gw = buildout[sorted_years[i - 1]] if i > 0 else 0.0
        delta_gw = max(buildout[vy] - prev_gw, 0.0)
        if delta_gw <= 0:
            continue
        age = year - vy
        cap = _fleet_cohort_capacity(age_years=age, preset=preset, duty=duty)
        total_mw += delta_gw
        weighted_cap += delta_gw * cap

    if total_mw <= 0:
        return 1.0
    return float(weighted_cap / total_mw)
