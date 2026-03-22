"""
One-zone energy system LP optimizer.

Minimizes total annual system cost (annualized CAPEX + OPEX + fuel)
subject to hourly energy balance for a full year (8760 hours).

Technologies: Solar PV, Wind, Nuclear, GreenFirm (H2 peaker), BESS (4h).

Uses scipy.optimize.linprog — no extra dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import linprog


# ── Technology parameters ────────────────────────────────────


@dataclass
class TechParams:
    """Economic and physical parameters for a generation/storage technology."""
    capex_eur_per_kw: float
    lifetime_years: int
    wacc: float
    fixed_opex_eur_per_kw_yr: float = 0.0
    var_opex_eur_per_mwh: float = 0.0
    fuel_cost_eur_per_mwh: float = 0.0
    capacity_factor: float | None = None  # for RE; None for dispatchable
    min_stable_load_frac: float = 0.0  # for nuclear
    efficiency_charge: float = 1.0  # for BESS
    efficiency_discharge: float = 1.0  # for BESS
    energy_capex_eur_per_kwh: float = 0.0  # for BESS
    duration_hours: int = 0  # for BESS
    max_build_gw: float | None = None  # build constraint

    @property
    def annuity_factor(self) -> float:
        if self.wacc == 0:
            return 1.0 / self.lifetime_years
        r = self.wacc
        n = self.lifetime_years
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    @property
    def annual_capex_eur_per_kw(self) -> float:
        return self.capex_eur_per_kw * self.annuity_factor

    @property
    def annual_energy_capex_eur_per_kwh(self) -> float:
        return self.energy_capex_eur_per_kwh * self.annuity_factor


# ── Default assumptions (2050 projections) ───────────────────


def default_tech_params() -> dict[str, TechParams]:
    return {
        "solar": TechParams(
            capex_eur_per_kw=300,
            lifetime_years=30,
            wacc=0.05,
            fixed_opex_eur_per_kw_yr=7,
            capacity_factor=0.11,
        ),
        "wind": TechParams(
            capex_eur_per_kw=1000,
            lifetime_years=25,
            wacc=0.05,
            fixed_opex_eur_per_kw_yr=25,
            capacity_factor=0.28,
        ),
        "nuclear": TechParams(
            capex_eur_per_kw=7000,
            lifetime_years=60,
            wacc=0.08,
            fixed_opex_eur_per_kw_yr=90,
            var_opex_eur_per_mwh=5,
            fuel_cost_eur_per_mwh=5,
            capacity_factor=0.80,
            min_stable_load_frac=0.50,
        ),
        "green_firm": TechParams(
            capex_eur_per_kw=700,
            lifetime_years=30,
            wacc=0.05,
            fixed_opex_eur_per_kw_yr=15,
            var_opex_eur_per_mwh=3,
            fuel_cost_eur_per_mwh=80,
        ),
        "bess": TechParams(
            capex_eur_per_kw=100,
            lifetime_years=20,
            wacc=0.05,
            efficiency_charge=0.94,
            efficiency_discharge=0.90,
            energy_capex_eur_per_kwh=80,
            duration_hours=4,
        ),
    }


# ── Demand and RE profile generation ─────────────────────────


def generate_hourly_demand(total_twh: float = 1000.0, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic hourly demand profile for Germany (8760 hours).

    Shape: winter-peaking with daily cycle (morning + evening peaks).
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(8760)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    # Seasonal: winter higher, summer lower
    seasonal = 1.0 + 0.20 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # Daily: morning and evening peaks
    daily_profile = np.array([
        0.75, 0.70, 0.68, 0.68, 0.70, 0.78,
        0.90, 1.00, 1.08, 1.10, 1.10, 1.08,
        1.05, 1.02, 1.00, 1.00, 1.05, 1.15,
        1.18, 1.12, 1.02, 0.92, 0.85, 0.78,
    ])
    daily = daily_profile[hour_of_day]

    # Small random noise
    noise = 1.0 + rng.normal(0, 0.03, 8760)

    profile = seasonal * daily * noise
    # Scale to target annual total
    target_mwh = total_twh * 1e6
    profile = profile / profile.sum() * target_mwh
    return profile


def generate_solar_profile(seed: int = 42) -> np.ndarray:
    """Synthetic hourly solar capacity factor profile for Germany."""
    rng = np.random.default_rng(seed)
    hours = np.arange(8760)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    # Day length effect: more sun hours in summer
    # Solar noon centered around hour 12
    solar_elevation = np.maximum(
        0, np.sin(np.pi * (hour_of_day - 6) / 12)
    ) * np.where((hour_of_day >= 6) & (hour_of_day <= 20), 1, 0)

    # Seasonal: summer much stronger
    seasonal = 0.6 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    seasonal = np.maximum(seasonal, 0.2)

    # Day length: longer days in summer
    sunrise = 8.0 - 3.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    sunset = 16.0 + 3.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daylight = np.where(
        (hour_of_day >= sunrise[hours]) & (hour_of_day <= sunset[hours]),
        1.0, 0.0,
    )

    cf = solar_elevation * seasonal * daylight
    # Cloud cover randomness
    cloud = rng.uniform(0.3, 1.0, 8760)
    cf = cf * cloud

    # Normalize to target annual CF ~11%
    if cf.sum() > 0:
        cf = cf / cf.max()
        current_cf = cf.mean()
        if current_cf > 0:
            cf = cf * (0.11 / current_cf)
    cf = np.clip(cf, 0, 1)
    return cf


def generate_wind_profile(seed: int = 43) -> np.ndarray:
    """Synthetic hourly wind capacity factor profile for Germany."""
    rng = np.random.default_rng(seed)
    hours = np.arange(8760)
    day_of_year = hours // 24

    # Seasonal: windier in winter
    seasonal = 0.30 + 0.12 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # Multi-day weather patterns (autocorrelated)
    weather = np.zeros(8760)
    state = 0.0
    for i in range(8760):
        state = 0.98 * state + rng.normal(0, 0.05)
        weather[i] = state
    weather = (weather - weather.min()) / (weather.max() - weather.min())

    cf = seasonal * (0.4 + 0.6 * weather)
    # Small hourly noise
    noise = rng.normal(0, 0.03, 8760)
    cf = cf + noise

    # Normalize to target annual CF ~28%
    cf = np.clip(cf, 0.02, 0.95)
    current_cf = cf.mean()
    if current_cf > 0:
        cf = cf * (0.28 / current_cf)
    cf = np.clip(cf, 0, 1)
    return cf


# ── LP Optimizer ─────────────────────────────────────────────


@dataclass
class ScenarioConfig:
    """Which technologies are available in a scenario."""
    name: str
    allow_solar: bool = True
    allow_wind: bool = True
    allow_nuclear: bool = True
    allow_green_firm: bool = True
    allow_bess: bool = True


@dataclass
class OptimizationResult:
    """Results from the LP optimization."""
    scenario: str
    status: str
    total_cost_eur: float
    lcoe_eur_per_mwh: float
    capacities_gw: dict[str, float]
    generation_twh: dict[str, float]
    bess_power_gw: float
    bess_energy_gwh: float
    curtailment_twh: float
    unserved_twh: float
    hourly_dispatch: pd.DataFrame


VOLL = 10_000.0  # Value of lost load, EUR/MWh


def optimize_energy_system(
    demand: np.ndarray,
    solar_cf: np.ndarray,
    wind_cf: np.ndarray,
    tech: dict[str, TechParams] | None = None,
    scenario: ScenarioConfig | None = None,
) -> OptimizationResult:
    """
    Solve the one-zone energy system LP.

    Decision variables (all in MW or MWh):
    - cap_solar, cap_wind, cap_nuclear, cap_firm: installed capacity (MW)
    - cap_bess_power (MW), cap_bess_energy (MWh)
    - gen_solar[t], gen_wind[t], gen_nuclear[t], gen_firm[t]: generation each hour
    - bess_charge[t], bess_discharge[t]: BESS flows each hour
    - soc[t]: state of charge each hour
    - curtail[t]: curtailed energy each hour
    - unserved[t]: unserved energy each hour
    """
    if tech is None:
        tech = default_tech_params()
    if scenario is None:
        scenario = ScenarioConfig(name="default")

    T = len(demand)
    assert T == 8760, f"Expected 8760 hours, got {T}"

    # ── Build variable index map ─────────────────────────────
    idx = {}
    n = 0

    def add_var(name: str, count: int = 1) -> int:
        nonlocal n
        start = n
        idx[name] = (start, start + count)
        n += count
        return start

    # Capacity variables (scalars, in MW)
    i_cap_solar = add_var("cap_solar")
    i_cap_wind = add_var("cap_wind")
    i_cap_nuclear = add_var("cap_nuclear")
    i_cap_firm = add_var("cap_firm")
    i_cap_bess_pw = add_var("cap_bess_power")
    i_cap_bess_en = add_var("cap_bess_energy")

    # Hourly variables
    i_gen_solar = add_var("gen_solar", T)
    i_gen_wind = add_var("gen_wind", T)
    i_gen_nuclear = add_var("gen_nuclear", T)
    i_gen_firm = add_var("gen_firm", T)
    i_bess_ch = add_var("bess_charge", T)
    i_bess_dis = add_var("bess_discharge", T)
    i_soc = add_var("soc", T)
    i_curtail = add_var("curtail", T)
    i_unserved = add_var("unserved", T)

    num_vars = n

    # ── Objective: minimize total annual cost ─────────────────
    c = np.zeros(num_vars)

    # Solar
    s = tech["solar"]
    annual_cost_solar = s.annual_capex_eur_per_kw + s.fixed_opex_eur_per_kw_yr
    c[i_cap_solar] = annual_cost_solar * 1000  # per MW

    # Wind
    w = tech["wind"]
    annual_cost_wind = w.annual_capex_eur_per_kw + w.fixed_opex_eur_per_kw_yr
    c[i_cap_wind] = annual_cost_wind * 1000

    # Nuclear
    nu = tech["nuclear"]
    annual_cost_nuclear = nu.annual_capex_eur_per_kw + nu.fixed_opex_eur_per_kw_yr
    c[i_cap_nuclear] = annual_cost_nuclear * 1000
    # Variable + fuel cost per MWh of generation
    for t in range(T):
        c[i_gen_nuclear + t] = nu.var_opex_eur_per_mwh + nu.fuel_cost_eur_per_mwh

    # GreenFirm
    gf = tech["green_firm"]
    annual_cost_firm = gf.annual_capex_eur_per_kw + gf.fixed_opex_eur_per_kw_yr
    c[i_cap_firm] = annual_cost_firm * 1000
    for t in range(T):
        c[i_gen_firm + t] = gf.var_opex_eur_per_mwh + gf.fuel_cost_eur_per_mwh

    # BESS
    b = tech["bess"]
    annual_cost_bess_pw = b.annual_capex_eur_per_kw
    annual_cost_bess_en = b.annual_energy_capex_eur_per_kwh
    c[i_cap_bess_pw] = annual_cost_bess_pw * 1000  # per MW
    c[i_cap_bess_en] = annual_cost_bess_en * 1000  # per MWh

    # Unserved energy penalty
    for t in range(T):
        c[i_unserved + t] = VOLL

    # ── Bounds ────────────────────────────────────────────────
    bounds = [(0, None)] * num_vars

    # Disable technologies not in scenario
    if not scenario.allow_solar:
        bounds[i_cap_solar] = (0, 0)
        for t in range(T):
            bounds[i_gen_solar + t] = (0, 0)
    if not scenario.allow_wind:
        bounds[i_cap_wind] = (0, 0)
        for t in range(T):
            bounds[i_gen_wind + t] = (0, 0)
    if not scenario.allow_nuclear:
        bounds[i_cap_nuclear] = (0, 0)
        for t in range(T):
            bounds[i_gen_nuclear + t] = (0, 0)
    if not scenario.allow_green_firm:
        bounds[i_cap_firm] = (0, 0)
        for t in range(T):
            bounds[i_gen_firm + t] = (0, 0)
    if not scenario.allow_bess:
        bounds[i_cap_bess_pw] = (0, 0)
        bounds[i_cap_bess_en] = (0, 0)
        for t in range(T):
            bounds[i_bess_ch + t] = (0, 0)
            bounds[i_bess_dis + t] = (0, 0)
            bounds[i_soc + t] = (0, 0)

    # ── Inequality constraints (A_ub @ x <= b_ub) ────────────
    # We'll collect rows as sparse (row_idx, col_idx, value) triples
    A_rows = []
    A_cols = []
    A_vals = []
    b_ub_list = []
    row = 0

    def add_ub_constraint(coeffs: dict[int, float], rhs: float):
        nonlocal row
        for col, val in coeffs.items():
            A_rows.append(row)
            A_cols.append(col)
            A_vals.append(val)
        b_ub_list.append(rhs)
        row += 1

    for t in range(T):
        # gen_solar[t] <= cap_solar * solar_cf[t]
        add_ub_constraint({i_gen_solar + t: 1, i_cap_solar: -solar_cf[t]}, 0)

        # gen_wind[t] <= cap_wind * wind_cf[t]
        add_ub_constraint({i_gen_wind + t: 1, i_cap_wind: -wind_cf[t]}, 0)

        # gen_nuclear[t] <= cap_nuclear * nuclear_cf
        add_ub_constraint(
            {i_gen_nuclear + t: 1, i_cap_nuclear: -nu.capacity_factor}, 0
        )

        # gen_nuclear[t] >= cap_nuclear * min_stable_load (as <= constraint)
        # -gen_nuclear[t] + cap_nuclear * min_stable <= 0
        if scenario.allow_nuclear and nu.min_stable_load_frac > 0:
            add_ub_constraint(
                {i_gen_nuclear + t: -1, i_cap_nuclear: nu.min_stable_load_frac}, 0
            )

        # gen_firm[t] <= cap_firm
        add_ub_constraint({i_gen_firm + t: 1, i_cap_firm: -1}, 0)

        # bess_charge[t] <= cap_bess_power
        add_ub_constraint({i_bess_ch + t: 1, i_cap_bess_pw: -1}, 0)

        # bess_discharge[t] <= cap_bess_power
        add_ub_constraint({i_bess_dis + t: 1, i_cap_bess_pw: -1}, 0)

        # soc[t] <= cap_bess_energy
        add_ub_constraint({i_soc + t: 1, i_cap_bess_en: -1}, 0)

    num_ub = row

    # ── Equality constraints (A_eq @ x = b_eq) ───────────────
    Aeq_rows = []
    Aeq_cols = []
    Aeq_vals = []
    b_eq_list = []
    eq_row = 0

    def add_eq_constraint(coeffs: dict[int, float], rhs: float):
        nonlocal eq_row
        for col, val in coeffs.items():
            Aeq_rows.append(eq_row)
            Aeq_cols.append(col)
            Aeq_vals.append(val)
        b_eq_list.append(rhs)
        eq_row += 1

    eta_ch = b.efficiency_charge
    eta_dis = b.efficiency_discharge

    for t in range(T):
        # Energy balance:
        # gen_solar + gen_wind + gen_nuclear + gen_firm
        # + bess_discharge - bess_charge - curtail + unserved = demand[t]
        add_eq_constraint(
            {
                i_gen_solar + t: 1,
                i_gen_wind + t: 1,
                i_gen_nuclear + t: 1,
                i_gen_firm + t: 1,
                i_bess_dis + t: 1,
                i_bess_ch + t: -1,
                i_curtail + t: -1,
                i_unserved + t: 1,
            },
            demand[t],
        )

        # SoC dynamics:
        # soc[t] = soc[t-1] + eta_ch * charge[t] - discharge[t] / eta_dis
        # => soc[t] - eta_ch * charge[t] + discharge[t] / eta_dis = soc[t-1]
        if t == 0:
            # soc[0] = 0 + eta_ch * charge[0] - discharge[0] / eta_dis
            add_eq_constraint(
                {
                    i_soc + t: 1,
                    i_bess_ch + t: -eta_ch,
                    i_bess_dis + t: 1 / eta_dis,
                },
                0,
            )
        else:
            # soc[t] - soc[t-1] - eta_ch * charge[t] + discharge[t] / eta_dis = 0
            add_eq_constraint(
                {
                    i_soc + t: 1,
                    i_soc + t - 1: -1,
                    i_bess_ch + t: -eta_ch,
                    i_bess_dis + t: 1 / eta_dis,
                },
                0,
            )

    # ── Assemble sparse matrices ─────────────────────────────
    from scipy.sparse import coo_matrix

    A_ub = coo_matrix(
        (A_vals, (A_rows, A_cols)),
        shape=(num_ub, num_vars),
    ).tocsr()
    b_ub = np.array(b_ub_list)

    A_eq = coo_matrix(
        (Aeq_vals, (Aeq_rows, Aeq_cols)),
        shape=(eq_row, num_vars),
    ).tocsr()
    b_eq = np.array(b_eq_list)

    # ── Solve ────────────────────────────────────────────────
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 300},
    )

    if not result.success:
        return OptimizationResult(
            scenario=scenario.name,
            status=result.message,
            total_cost_eur=0,
            lcoe_eur_per_mwh=0,
            capacities_gw={},
            generation_twh={},
            bess_power_gw=0,
            bess_energy_gwh=0,
            curtailment_twh=0,
            unserved_twh=0,
            hourly_dispatch=pd.DataFrame(),
        )

    x = result.x

    # ── Extract results ──────────────────────────────────────
    cap_solar_mw = x[i_cap_solar]
    cap_wind_mw = x[i_cap_wind]
    cap_nuclear_mw = x[i_cap_nuclear]
    cap_firm_mw = x[i_cap_firm]
    cap_bess_pw_mw = x[i_cap_bess_pw]
    cap_bess_en_mwh = x[i_cap_bess_en]

    gen_solar = x[i_gen_solar: i_gen_solar + T]
    gen_wind = x[i_gen_wind: i_gen_wind + T]
    gen_nuclear = x[i_gen_nuclear: i_gen_nuclear + T]
    gen_firm = x[i_gen_firm: i_gen_firm + T]
    bess_charge = x[i_bess_ch: i_bess_ch + T]
    bess_discharge = x[i_bess_dis: i_bess_dis + T]
    soc = x[i_soc: i_soc + T]
    curtail = x[i_curtail: i_curtail + T]
    unserved = x[i_unserved: i_unserved + T]

    total_demand_mwh = demand.sum()

    dispatch_df = pd.DataFrame({
        "hour": np.arange(T),
        "demand_mw": demand,
        "solar_mw": gen_solar,
        "wind_mw": gen_wind,
        "nuclear_mw": gen_nuclear,
        "firm_mw": gen_firm,
        "bess_charge_mw": bess_charge,
        "bess_discharge_mw": bess_discharge,
        "soc_mwh": soc,
        "curtailment_mw": curtail,
        "unserved_mw": unserved,
    })

    return OptimizationResult(
        scenario=scenario.name,
        status="optimal",
        total_cost_eur=result.fun,
        lcoe_eur_per_mwh=result.fun / total_demand_mwh,
        capacities_gw={
            "solar": cap_solar_mw / 1000,
            "wind": cap_wind_mw / 1000,
            "nuclear": cap_nuclear_mw / 1000,
            "green_firm": cap_firm_mw / 1000,
        },
        generation_twh={
            "solar": gen_solar.sum() / 1e6,
            "wind": gen_wind.sum() / 1e6,
            "nuclear": gen_nuclear.sum() / 1e6,
            "green_firm": gen_firm.sum() / 1e6,
        },
        bess_power_gw=cap_bess_pw_mw / 1000,
        bess_energy_gwh=cap_bess_en_mwh / 1000,
        curtailment_twh=curtail.sum() / 1e6,
        unserved_twh=unserved.sum() / 1e6,
        hourly_dispatch=dispatch_df,
    )


# ── Convenience runner ───────────────────────────────────────


def run_scenarios(
    scenarios: list[ScenarioConfig] | None = None,
    tech: dict[str, TechParams] | None = None,
    total_demand_twh: float = 1000.0,
    seed: int = 42,
) -> dict[str, OptimizationResult]:
    """Run multiple scenarios with shared demand/RE profiles."""
    if tech is None:
        tech = default_tech_params()
    if scenarios is None:
        scenarios = [
            ScenarioConfig(
                name="renewables",
                allow_solar=True,
                allow_wind=True,
                allow_nuclear=False,
                allow_green_firm=True,
                allow_bess=True,
            ),
            ScenarioConfig(
                name="nuclear",
                allow_solar=False,
                allow_wind=False,
                allow_nuclear=True,
                allow_green_firm=True,
                allow_bess=True,
            ),
        ]

    demand = generate_hourly_demand(total_demand_twh, seed=seed)
    solar_cf = generate_solar_profile(seed=seed)
    wind_cf = generate_wind_profile(seed=seed + 1)

    results = {}
    for sc in scenarios:
        results[sc.name] = optimize_energy_system(
            demand=demand,
            solar_cf=solar_cf,
            wind_cf=wind_cf,
            tech=tech,
            scenario=sc,
        )
    return results


def article_tech_params() -> dict[str, TechParams]:
    """Original article assumptions for comparison."""
    return {
        "solar": TechParams(
            capex_eur_per_kw=500,
            lifetime_years=25,
            wacc=0.06,
            fixed_opex_eur_per_kw_yr=10,
            capacity_factor=0.11,
        ),
        "wind": TechParams(
            capex_eur_per_kw=1100,
            lifetime_years=25,
            wacc=0.06,
            fixed_opex_eur_per_kw_yr=25,
            capacity_factor=0.28,
        ),
        "nuclear": TechParams(
            capex_eur_per_kw=7000,
            lifetime_years=60,
            wacc=0.06,
            fixed_opex_eur_per_kw_yr=90,
            var_opex_eur_per_mwh=5,
            fuel_cost_eur_per_mwh=5,
            capacity_factor=0.80,
            min_stable_load_frac=0.50,
        ),
        "green_firm": TechParams(
            capex_eur_per_kw=800,
            lifetime_years=30,
            wacc=0.06,
            fixed_opex_eur_per_kw_yr=15,
            var_opex_eur_per_mwh=3,
            fuel_cost_eur_per_mwh=120,
        ),
        "bess": TechParams(
            capex_eur_per_kw=200,
            lifetime_years=20,
            wacc=0.06,
            efficiency_charge=0.90,
            efficiency_discharge=0.90,
            energy_capex_eur_per_kwh=200,
            duration_hours=4,
        ),
    }


def _print_result(name: str, r: OptimizationResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"Scenario: {name} — Status: {r.status}")
    print(f"System LCOE: {r.lcoe_eur_per_mwh:.1f} EUR/MWh")
    print(f"Total cost: {r.total_cost_eur / 1e9:.1f} B EUR/yr")
    print("Capacities (GW):")
    for tech_name, gw in r.capacities_gw.items():
        if gw > 0.01:
            print(f"  {tech_name}: {gw:.1f}")
    print(f"  BESS power: {r.bess_power_gw:.1f} GW / {r.bess_energy_gwh:.1f} GWh")
    print("Generation (TWh):")
    for tech_name, twh in r.generation_twh.items():
        if twh > 0.1:
            print(f"  {tech_name}: {twh:.1f}")
    print(f"Curtailment: {r.curtailment_twh:.1f} TWh")
    print(f"Unserved: {r.unserved_twh:.2f} TWh")


if __name__ == "__main__":
    scenarios = [
        ScenarioConfig(
            name="renewables",
            allow_solar=True, allow_wind=True,
            allow_nuclear=False, allow_green_firm=True, allow_bess=True,
        ),
        ScenarioConfig(
            name="nuclear_only",
            allow_solar=False, allow_wind=False,
            allow_nuclear=True, allow_green_firm=True, allow_bess=True,
        ),
        ScenarioConfig(
            name="mixed",
            allow_solar=True, allow_wind=True,
            allow_nuclear=True, allow_green_firm=True, allow_bess=True,
        ),
    ]

    print("=" * 60)
    print("CORRECTED ASSUMPTIONS (2050 projections)")
    print("=" * 60)
    results = run_scenarios(scenarios=scenarios)
    for name, r in results.items():
        _print_result(name, r)

    print("\n\n")
    print("=" * 60)
    print("ARTICLE ASSUMPTIONS (for comparison)")
    print("=" * 60)
    article_results = run_scenarios(
        scenarios=scenarios,
        tech=article_tech_params(),
        total_demand_twh=1487.0,
    )
    for name, r in article_results.items():
        _print_result(name, r)
