"""Multi-anchor calibration tests — Trina calendar + EVE informational bound.

The core ``test_anchor`` retention is verified in test_degradation_detailed.py.
This module adds tests for:
  * Trina's two calendar anchors (40% and 100% SoC storage)
  * EVE's informational 45°C endurance bound (loose — documented divergence)
  * ``calibration_status`` tagging integrity
"""
from __future__ import annotations

import numpy as np

from lib.models.degradation.simple import PRESETS
from lib.models.degradation.detailed import DutyCycle, cell_soh_detailed, LFPGraphiteWangNaumann


def test_trina_calendar_40pct_soc_anchor():
    """Trina 40% SoC / 25°C / 2 years storage → ≥98% retention."""
    preset = PRESETS["trina_elementa_280ah"]
    kernel = LFPGraphiteWangNaumann()
    q_cal = kernel.calendar_loss(
        elapsed_years=2.0,
        soc_bucket_hours={"low": 8760.0, "mid": 0.0, "high": 0.0},
        temp_C=25.0,
        preset=preset,
    )
    retention = 1.0 - q_cal
    assert retention >= 0.98 - 0.005, f"Trina 40% SoC 2yr retention {retention:.3f} below 0.98"


def test_trina_calendar_100pct_soc_anchor():
    """Trina 100% SoC / 25°C / ~27.4 years → ~70% retention (10000 days to EOL)."""
    preset = PRESETS["trina_elementa_280ah"]
    kernel = LFPGraphiteWangNaumann()
    q_cal = kernel.calendar_loss(
        elapsed_years=27.4,
        soc_bucket_hours={"low": 0.0, "mid": 0.0, "high": 8760.0},
        temp_C=25.0,
        preset=preset,
    )
    retention = 1.0 - q_cal
    assert abs(retention - 0.70) <= 0.02, f"Trina 100% SoC 27.4yr retention {retention:.3f} ≠ 0.70 ±2pp"


def test_trina_calendar_soc_ratio_steeper_than_naumann_default():
    """Trina's fitted SoC weights imply a steeper high/low ratio than Naumann default."""
    preset = PRESETS["trina_elementa_280ah"]
    assert preset.calendar_soc_weights is not None
    w = preset.calendar_soc_weights
    ratio = w["high"] / w["low"]
    # Naumann 2018 default ratio (1.60/0.60) = 2.67. Trina fitted ~4.0.
    assert ratio > 3.5, f"Trina high/low weight ratio {ratio:.2f} not steeper than Naumann 2.67"


def test_eve_informational_45c_bound():
    """EVE 2500 FEC @ 45°C/0.5C is informational — model is expected to diverge, bound at 20pp."""
    preset = PRESETS["eve_lf280k"]
    duty = DutyCycle.from_mean(fec_per_year=2500, mean_dod=1.0, mean_soc=0.55, mean_crate=0.5, mean_temp_C=45.0)
    soh = cell_soh_detailed(duty, years=1.0, preset=preset, n_mc=1)
    # Datasheet says 80% at this point; model predicts ~66% (14pp off) — documented in preset.notes.
    assert abs(soh - 0.80) <= 0.20, f"EVE 45°C divergence {abs(soh-0.80):.3f} exceeds documented 20pp bound"


def test_calibration_status_tags_present_and_honest():
    """Each preset carries a calibration_status from the documented vocabulary."""
    allowed = {
        "multi_anchor",
        "multi_anchor_partial_private",
        "single_anchor_datasheet",
        "single_anchor_marketing",
        "synthetic",
    }
    for name, p in PRESETS.items():
        assert p.calibration_status in allowed, (name, p.calibration_status)
    # Specific assertions: our honest tagging
    assert PRESETS["trina_elementa_280ah"].calibration_status == "multi_anchor_partial_private"
    assert PRESETS["eve_lf280k"].calibration_status == "single_anchor_datasheet"
    assert PRESETS["catl_enerc_plus_306ah"].calibration_status == "single_anchor_marketing"
    assert PRESETS["byd_mc_cube_t"].calibration_status == "single_anchor_marketing"
    assert PRESETS["baseline_fleet"].calibration_status == "synthetic"


def test_c_rate_exponent_default_is_one():
    """All current LFP presets use c_rate_exponent=1 (Wang linear). No preset fits away from default yet."""
    for name, p in PRESETS.items():
        assert p.c_rate_exponent == 1.0, (name, p.c_rate_exponent)
