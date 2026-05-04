"""Legacy-parity + preset determinism for the simple degradation model."""
from __future__ import annotations

import pytest

from lib.config import DEFAULT_BESS_BUILDOUT
from lib.models.degradation.simple import (
    DEFAULT_DEGRADATION_ASSUMPTIONS,
    PRESETS,
    compute_annual_degradation,
    fleet_average_capacity,
    project_capacity_fraction,
    project_capacity_simple,
)


# Legacy-retired closed-form (kept only here for parity assertion)
_CYCLES_PER_YEAR = 730
_AUGMENTATION_AT_CYCLES = 8500
_ANNUAL_FADE_RATE = 0.018
_AUGMENTATION_RESTORE = 0.92


def _legacy_cohort(age: float) -> float:
    cap = 1.0 - _ANNUAL_FADE_RATE * age
    aug_year = _AUGMENTATION_AT_CYCLES / _CYCLES_PER_YEAR
    if age >= aug_year:
        cap = _AUGMENTATION_RESTORE - _ANNUAL_FADE_RATE * (age - aug_year)
    return max(cap, 0.50)


def _legacy_fleet(year: int, buildout: dict) -> float:
    sy = sorted(buildout.keys())
    tot = 0.0
    w = 0.0
    for i, vy in enumerate(sy):
        if vy > year:
            break
        prev = buildout[sy[i - 1]] if i > 0 else 0.0
        d = max(buildout[vy] - prev, 0.0)
        if d <= 0:
            continue
        age = year - vy
        tot += d
        w += d * _legacy_cohort(age)
    return w / tot if tot > 0 else 1.0


@pytest.mark.parametrize("year", [2026, 2030, 2035, 2040])
def test_baseline_fleet_parity_with_legacy(year: int):
    proj = {y: v for y, v in DEFAULT_BESS_BUILDOUT.items() if y >= 2026}
    new = fleet_average_capacity(year, proj, PRESETS["baseline_fleet"])
    old = _legacy_fleet(year, proj)
    assert abs(new - old) <= 0.002, f"{year}: {new=}, {old=}"


def test_preset_registry_has_minimum_fields():
    for name, p in PRESETS.items():
        assert p.source_url and p.manufacturer, name
        assert p.test_anchor is not None
        # Warranty anchor is optional for cell-level presets.


def test_project_capacity_simple_monotone_in_years():
    preset = PRESETS["baseline_fleet"]
    prev = 1.0
    for t in [0.5, 1, 5, 10, 20]:
        cap = project_capacity_simple(730, 0.80, t, preset)
        assert cap <= prev + 1e-9
        prev = cap


def test_project_capacity_fraction_legacy_api_still_works():
    # Sanity: legacy scalar API unchanged.
    assert project_capacity_fraction(5.0, 730, DEFAULT_DEGRADATION_ASSUMPTIONS) < 1.0


def test_compute_annual_degradation_monotone_in_dod():
    a = compute_annual_degradation(730, 0.6)
    b = compute_annual_degradation(730, 1.0)
    assert b > a
