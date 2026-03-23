"""
Shared configuration — battery defaults, fleet buildout, market constants.

All assumptions documented with provenance (source comments inline).
"""

from dataclasses import dataclass, field
from typing import Dict

# ── Battery defaults ─────────────────────────────────────────
@dataclass
class BatteryConfig:
    duration_h: float = 2.0          # hours
    rte: float = 0.85                # round-trip efficiency
    max_cycles_per_day: int = 2
    dod: float = 1.0                 # depth of discharge

    @property
    def eta_charge(self) -> float:
        return self.rte ** 0.5

    @property
    def eta_discharge(self) -> float:
        return self.rte ** 0.5

    @property
    def e_max(self) -> float:
        """MWh per MW rated (= duration × DoD)."""
        return self.duration_h * self.dod


# ── API endpoints ────────────────────────────────────────────
ENERGY_CHARTS_BASE = "https://api.energy-charts.info"
ENERGY_CHARTS_PRICE = ENERGY_CHARTS_BASE + "/price"
# params: bzn=DE-LU, start=ISO, end=ISO

NETZTRANSPARENZ_DATA_API = "https://ds.netztransparenz.de/api/v1/"
# OAuth2 client credentials; free registration at api-portal.netztransparenz.de

REGELLEISTUNG_DATACENTER = "https://www.regelleistung.net/apps/datacenter/tenders/"


# ── BESS fleet buildout (GW) ────────────────────────────────
# Source: MaStR for 2024-2025, NEP/BNetzA grid development plan for 2026+
# Adjustable via UI slider for 2040 target
#
# The model tracks power (GW), not energy capacity (GWh).
# Cannibalisation is driven by GW competing for the same hourly spreads
# and MW slots in ancillary tenders. Duration affects revenue per MW
# separately (via duration_h parameter in dispatch and ancillary models).
DEFAULT_BESS_BUILDOUT: Dict[int, float] = {
    2020: 0.5,  2021: 0.7,  2022: 1.0,  2023: 1.5,
    2024: 2.4,  2025: 3.5,  2026: 5.0,  2027: 7.0,
    2028: 10.0, 2029: 13.0, 2030: 17.0, 2031: 20.0,
    2032: 23.0, 2033: 26.0, 2034: 28.0, 2035: 30.0,
    2036: 32.0, 2037: 34.0, 2038: 36.0, 2039: 38.0,
    2040: 40.0,
}

# ── RE generation (TWh) ─────────────────────────────────────
# Source: Energy-Charts historical, EEG 2023 §4 targets
DEFAULT_RE_TWH: Dict[int, float] = {
    2020: 233, 2021: 234, 2022: 244, 2023: 260, 2024: 280,
    2025: 290,
}
# Forward: linear interpolation 280 (2026) → 695 (2040)
RE_2026 = 280.0
RE_2040 = 695.0

# ── Demand (TWh) ─────────────────────────────────────────────
# Source: dena Leitstudie / Agora Energiewende (~600 → 1000+ TWh)
DEMAND_2026 = 600.0
DEMAND_2040 = 1020.0

# ── Ancillary market depth ───────────────────────────────────
# Source: regelleistung.net / SO GL
FCR_DEMAND_MW = 613           # Denmark West/Germany LFC block (2026)
# Source: ENTSO-E "Demand of each LFC-block towards the regional FCR
# Cooperation market for the year 2026", published 01/12/2025
# https://www.entsoe.eu/network_codes/eb/fcr/
AFRR_DEPTH_MW = 4000          # combined pos+neg addressable by BESS
ANCILLARY_COMBINED_GW = 4.5   # FCR ~0.6 GW (regelleistung.net) + aFRR ~4 GW (SO GL)

# ── Industry projection overlay (for comparison) ─────────────
# Proprietary; not used in model logic, only for optional chart overlay.
MODO_V2 = {
    2026: {"da": 55, "id": 50, "fcr": 8,  "afrr_cap": 115, "afrr_energy": 12},
    2027: {"da": 55, "id": 50, "fcr": 6,  "afrr_cap": 60,  "afrr_energy": 7},
    2028: {"da": 55, "id": 50, "fcr": 4,  "afrr_cap": 33,  "afrr_energy": 5},
    2029: {"da": 52, "id": 48, "fcr": 3,  "afrr_cap": 17,  "afrr_energy": 4},
    2030: {"da": 48, "id": 43, "fcr": 2,  "afrr_cap": 8,   "afrr_energy": 3},
    2031: {"da": 45, "id": 40, "fcr": 1,  "afrr_cap": 5,   "afrr_energy": 2},
    2032: {"da": 43, "id": 38, "fcr": 1,  "afrr_cap": 3,   "afrr_energy": 2},
    2033: {"da": 42, "id": 37, "fcr": 1,  "afrr_cap": 2,   "afrr_energy": 1},
    2034: {"da": 41, "id": 36, "fcr": 1,  "afrr_cap": 2,   "afrr_energy": 1},
    2035: {"da": 40, "id": 36, "fcr": 0,  "afrr_cap": 1,   "afrr_energy": 1},
    2036: {"da": 40, "id": 35, "fcr": 0,  "afrr_cap": 1,   "afrr_energy": 1},
    2037: {"da": 39, "id": 35, "fcr": 0,  "afrr_cap": 1,   "afrr_energy": 1},
    2038: {"da": 39, "id": 35, "fcr": 0,  "afrr_cap": 1,   "afrr_energy": 1},
    2039: {"da": 40, "id": 36, "fcr": 0,  "afrr_cap": 1,   "afrr_energy": 1},
    2040: {"da": 42, "id": 38, "fcr": 0,  "afrr_cap": 1,   "afrr_energy": 1},
}

# ── Degradation model (cohort-based) ──────────────────────
# Assumptions: 2 cycles/day, 2-hour battery, including
# degradation and augmentation after ~10k cycles
# Each year's new BESS capacity is a separate cohort with its own age.
# Fleet-average degradation = capacity-weighted mean across all cohorts.
CYCLES_PER_YEAR = 730          # 2 cycles/day × 365
AUGMENTATION_AT_CYCLES = 8500  # augmentation threshold (~year 11.6)
ANNUAL_FADE_RATE = 0.018       # ~1.8%/yr effective capacity loss per cohort
AUGMENTATION_RESTORE = 0.92    # augmentation restores to 92% of nameplate


def _cohort_capacity(age_years: float) -> float:
    """Effective capacity of a single cohort aged `age_years`."""
    capacity = 1.0 - ANNUAL_FADE_RATE * age_years
    aug_year = AUGMENTATION_AT_CYCLES / CYCLES_PER_YEAR
    if age_years >= aug_year:
        capacity = AUGMENTATION_RESTORE - ANNUAL_FADE_RATE * (age_years - aug_year)
    return max(capacity, 0.50)


def fleet_degradation_factor(
    year: int,
    buildout: 'Dict[int, float] | None' = None,
) -> float:
    """
    Fleet-average degradation factor for `year`.

    Each vintage (year of commissioning) is a cohort whose MW = delta
    of cumulative buildout. The fleet-average capacity is the
    MW-weighted mean of all cohort capacities.

    Returns value in (0, 1].
    """
    if buildout is None:
        buildout = DEFAULT_BESS_BUILDOUT

    # Build sorted list of (vintage_year, installed_gw)
    sorted_years = sorted(buildout.keys())

    total_mw = 0.0
    weighted_cap = 0.0

    for i, vy in enumerate(sorted_years):
        if vy > year:
            break
        # Delta MW added in vintage year
        prev_gw = buildout[sorted_years[i - 1]] if i > 0 else 0.0
        delta_gw = max(buildout[vy] - prev_gw, 0.0)
        if delta_gw <= 0:
            continue

        age = year - vy
        cap = _cohort_capacity(age)
        total_mw += delta_gw
        weighted_cap += delta_gw * cap

    if total_mw <= 0:
        return 1.0
    return weighted_cap / total_mw


# ── Gas price (TTF) ─────────────────────────────────────────
# Source: ICE TTF front-year, EEX forwards
# Gas sets marginal cost for peaker hours → drives peak DA prices → wider spreads
TTF_2026 = 35.0    # €/MWh — 2026 TTF forward (approx.)
TTF_2040 = 30.0    # €/MWh — long-term consensus (decarbonisation pull)
# Elasticity: fitted on DE DA spreads 2020-2025, R²=0.97
# When gas doubles, DA spread increases ~40%
GAS_SPREAD_ELASTICITY = 0.405

# ── Solar PV fleet (GW) ────────────────────────────────────
# Source: MaStR / Fraunhofer ISE for historical, EEG targets for forward
# More PV → deeper duck curve → wider midday-evening spreads (positive for BESS)
PV_GW_2026 = 100.0   # ~95 GW end-2025 + additions
PV_GW_2040 = 300.0   # EEG target: 215 GW by 2030, ~300 GW by 2040
# Fitted on DE+UK panel 2023-2025
PV_SPREAD_SENSITIVITY = 33.0  # kEUR/MW BESS revenue per 100 GW PV above baseline
# Fitted on DE+UK+ES+IT panel (PV range 16-100 GW). ES data (25-35 GW, high
# PV/demand ratio) provides strong cross-sectional identification.
