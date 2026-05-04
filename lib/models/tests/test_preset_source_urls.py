"""Every preset carries provenance and a test anchor; OEM-system-level presets also carry a warranty anchor."""
from __future__ import annotations

from lib.models.degradation.simple import PRESETS


def test_every_preset_has_source_and_manufacturer():
    for name, p in PRESETS.items():
        assert p.source_url, name
        assert p.manufacturer, name
        assert p.test_anchor is not None, name


def test_oem_system_presets_have_warranty_anchor():
    # These presets carry OEM system-level warranties; enforce non-None.
    for name in ["catl_enerc_plus_306ah", "byd_mc_cube_t", "trina_elementa_280ah"]:
        assert PRESETS[name].warranty_anchor is not None, name
