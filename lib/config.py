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


# ── Degradation model ─────────────────────────────────────
# The cohort-weighted fleet-average lives in
# ``lib.models.degradation.fleet_average_capacity`` (CellPreset-driven).
# Legacy constants (CYCLES_PER_YEAR, AUGMENTATION_AT_CYCLES, ANNUAL_FADE_RATE,
# AUGMENTATION_RESTORE, _cohort_capacity, fleet_degradation_factor) retired
# in Note 3; ``baseline_fleet`` preset reproduces them to ±0.2pp.


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
