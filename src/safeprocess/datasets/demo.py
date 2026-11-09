"""Runtime-generated correlated process observations for bounded CPU checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import base_manifest, chronological_segments, write_recording

FEATURES = ["mixer0.tank_B201.level", "mixer0.sensor_continuous_pressure_tank_B201.p", "bottling0.tank_B401.level", "bottling0.sensor_continuous_volumeFlowRate.V_flow"]
OWNERS = ["mixer0", "mixer0", "bottling0", "bottling0"]
CONTEXT = [f"{owner}.state_{state}.active" for owner in ("mixer0", "bottling0") for state in ("filling", "emptying")]


def _frame(rows: int, offset: int, rng: np.random.Generator) -> pd.DataFrame:
    time = np.arange(offset, offset + rows, dtype=float)
    phase = (time.astype(int) // 600) % 2
    downstream_phase = ((time.astype(int) + 120) // 600) % 2
    latent = 0.45 + 0.22 * np.sin(2 * np.pi * time / 1200) + 0.025 * np.sin(time / 37)
    delayed = 0.45 + 0.22 * np.sin(2 * np.pi * (time - 80) / 1200)
    frame = pd.DataFrame({
        FEATURES[0]: latent + rng.normal(0, 0.003, rows),
        FEATURES[1]: 2 + 1.7 * latent + 0.08 * phase + rng.normal(0, 0.006, rows),
        FEATURES[2]: 0.3 + 0.8 * delayed + rng.normal(0, 0.003, rows),
        FEATURES[3]: 0.8 + 0.9 * delayed + 0.08 * downstream_phase + rng.normal(0, 0.004, rows),
        CONTEXT[0]: (phase == 0).astype(float),
        CONTEXT[1]: (phase == 1).astype(float),
        CONTEXT[2]: (downstream_phase == 0).astype(float),
        CONTEXT[3]: (downstream_phase == 1).astype(float),
    })
    frame.insert(0, "source_row", np.arange(rows, dtype=np.int64))
    return frame


def describe(out: Path) -> dict:
    rng = np.random.default_rng(20261106)
    manifest = base_manifest("demo", "synthetic", FEATURES, OWNERS, CONTEXT)
    manifest["channel_descriptions"] = [{"name": name, "owner": owner, "observable_property": "synthetic-sensor-output", "unit_uri": None, "evidence": "Runtime synthetic generator"} for name, owner in zip(FEATURES, OWNERS)]
    manifest["context_bindings"] = {owner: [name for name in CONTEXT if name.startswith(owner + ".")] for owner in sorted(set(OWNERS))}
    manifest.update({"context_unit": "hai_demo", "components": ["mixer0", "bottling0"], "connections": [["mixer0", "bottling0"]], "sample_period_seconds": 1.0, "synthetic": True, "generator": "correlated-two-component-v1", "generator_seed": 20261106, "split_fractions": [0.60, 0.15, 0.10, 0.15]})
    frame = _frame(24000, 0, rng)
    manifest["recordings"].append(write_recording(out, "demo-normal", frame, None, normal=True, segments=chronological_segments(len(frame)), scenario="synthetic-normal", source_group="demo-normal"))
    categories = ("supported_native", "ambiguous_native", "negative_native")
    for index, category in enumerate(categories):
        native = _frame(1800, 21000 + index * 2000, rng)
        if category == "supported_native":
            native.loc[600:, FEATURES[0]] += 0.13
            native.loc[600:, FEATURES[2]] += 0.12
        elif category == "ambiguous_native":
            native.loc[600:, FEATURES[1]] += 0.03
        ident = f"demo-native-{index + 1}"
        manifest["recordings"].append(write_recording(out, ident, native, None, normal=False, segments=[{"start": 0, "end": len(native), "split": "native"}], scenario=f"synthetic-{category}", source_group=ident, native_category=category, response_qualification="synthetic_only", native_onset_row=600, native_role="assessment", synthetic=True, qualification_note="Synthetic generator category, not an empirical process-fault qualification"))
    return manifest
