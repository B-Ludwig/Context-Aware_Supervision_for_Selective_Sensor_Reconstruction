"""HAI-CPPS continuous simulation CSVs with per-scenario setup descriptions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    base_manifest,
    channel_metadata,
    chronological_segments,
    dataset_root,
    digest,
    numeric_frame,
    write_recording,
)


def classify_column(name: str) -> str:
    """Source-specific roles; fault proxies and commands are never AE inputs."""
    lower = name.lower()
    if ".state_" in lower and lower.endswith(".active"):
        return "operating_context"
    if any(x in lower for x in ("clogging_valve.opening", "leaking_valve.opening")) or lower.endswith(".pump_n_in"):
        return "parameter"
    if "sensor_discrete" in lower or "sensor_continuous_pressure" in lower or "sensor_continuous_volumeflowrate" in lower:
        return "sensor"
    if ((".tank_" in lower or ".distill." in lower) and lower.endswith(".level")) or lower.endswith(".port.t"):
        return "sensor"
    if lower.endswith((".opening", ".n_in")):
        return "actuator"
    return "process"


def sensor_description(name: str, owner: str) -> dict:
    lower = name.lower()
    property_name, unit = "sensor-output", None
    if "sensor_continuous_pressure" in lower:
        property_name, unit = "pressure", "PA"
    elif "sensor_continuous_volumeflowrate" in lower:
        property_name, unit = ("volume-flow-rate", "M3-PER-SEC") if lower.endswith(".v_flow") else ("mass-flow-rate", "KiloGM-PER-SEC")
    elif lower.endswith(".level"):
        property_name, unit = "liquid-level", "M"
    elif lower.endswith(".port.t"):
        property_name, unit = "temperature", "K"
    return {"name": name, "owner": owner, "observable_property": property_name, "unit_uri": f"http://qudt.org/vocab/unit/{unit}" if unit else None, "evidence": "HAI-CPPS released simulation channel definitions and sim_setup.json"}


def root_path(data_root: Path) -> Path:
    root = dataset_root(data_root, ("HAI", "HAI-CPPS", "hai_cpps"))
    return root / "Datasets" if (root / "Datasets").is_dir() else root


def discover(dataset: str, data_root: Path) -> tuple[Path, list[Path]]:
    match = re.fullmatch(r"hai_ds(10|[1-9])", dataset)
    if not match:
        raise ValueError("HAI configuration must be hai_ds1 through hai_ds10")
    number = int(match.group(1))
    root = root_path(data_root)
    files = []
    for collection in ("continuous-normal", "continuous-anomalies"):
        for path in sorted((root / collection).glob("*/*.csv")):
            found = re.match(r"ds(\d+)(?:_|$)", path.parent.name)
            if found and int(found.group(1)) == number:
                files.append(path)
    normal = [p for p in files if p.parent.parent.name == "continuous-normal"]
    if len(normal) != 1:
        raise FileNotFoundError(f"Expected one continuous normal CSV for {dataset}, found {len(normal)} under {root}")
    for path in files:
        if not (path.parent / "sim_setup.json").is_file():
            raise FileNotFoundError(f"Missing sim_setup.json for {path.name}")
    return root, files


def _source_metadata(path: Path) -> tuple[dict, list[str], list[list[str]], float]:
    setup = json.loads((path.parent / "sim_setup.json").read_text())
    modules = setup["model"]["modules"]
    components = sorted(modules)
    edges = setup["model"].get("edges", {})
    pairs = sorted({tuple(x.split(".", 1)[0] for x in edge) for edge in edges.values()})
    sim = setup.get("sim_setup", {})
    interval_count = sim.get("numberOfIntervals", 0)
    duration = sim.get("stopTime", 0) - sim.get("startTime", 0)
    period = float(duration / interval_count) if interval_count and duration > 0 else 1.0
    return setup, components, [list(x) for x in pairs], period


def preflight(dataset: str, data_root: Path) -> dict:
    root, sources = discover(dataset, data_root)
    expected = [x["name"] for x in channel_metadata(dataset)]
    records = []
    for source in sources:
        columns = list(pd.read_csv(source, nrows=0).columns)
        missing = sorted(set(expected) - set(columns))
        if missing:
            raise ValueError(f"{source.name} is missing model sensors: {missing}")
        context = [c for c in columns if classify_column(c) == "operating_context"]
        if not context:
            raise ValueError(f"No observable operating-state columns in {source.name}")
        _source_metadata(source)
        records.append({"file": source.relative_to(root).as_posix(), "normal": source.parent.parent.name == "continuous-normal", "context_channels": len(context)})
    return {"dataset": dataset, "ok": True, "features": len(expected), "source_root": str(root), "recordings": records}


def describe(dataset: str, data_root: Path, out: Path) -> dict:
    root, sources = discover(dataset, data_root)
    mapping = channel_metadata(dataset)
    features = [x["name"] for x in mapping]
    owners = [x["owner"] for x in mapping]
    normal_source = next(p for p in sources if p.parent.parent.name == "continuous-normal")
    columns = list(pd.read_csv(normal_source, nrows=0).columns)
    context = sorted(c for c in columns if classify_column(c) == "operating_context")
    if not context:
        raise ValueError("HAI requires explicit observable state channels")
    if any(classify_column(c) != "sensor" for c in features):
        raise ValueError("The authored HAI feature order includes a non-sensor")
    _, components, connections, period = _source_metadata(normal_source)
    manifest = base_manifest(dataset, "hai-cpps", features, owners, context)
    manifest.update({"context_unit": dataset, "components": components, "connections": connections, "sample_period_seconds": period, "split_fractions": [0.60, 0.15, 0.10, 0.15], "channel_descriptions": [sensor_description(name, owner) for name, owner in zip(features, owners)], "context_bindings": {owner: [col for col in context if col.startswith(owner + ".")] for owner in sorted(set(owners))}})
    for source in sources:
        raw = pd.read_csv(source)
        frame = numeric_frame(raw, features + context)
        frame.insert(0, "source_row", np.arange(len(frame), dtype=np.int64))
        normal = source.parent.parent.name == "continuous-normal"
        scenario = source.parent.name
        setup, source_components, source_connections, source_period = _source_metadata(source)
        if source_components != components or source_connections != connections or source_period != period:
            raise ValueError(f"Structural metadata differs in scenario {scenario}")
        ident = f"{dataset}-{'normal' if normal else scenario}"
        extra = {"scenario": scenario, "source_group": f"hai-continuous-{scenario}", "setup_sha256": digest(source.parent / "sim_setup.json"), "nominal_sample_period_seconds": period}
        if normal:
            segments = chronological_segments(len(frame))
        else:
            segments = [{"start": 0, "end": len(frame), "split": "native"}]
            native_role = "decision_fit" if scenario.endswith("pump50") else "risk_calibration" if scenario.endswith("pump75") else "assessment"
            extra.update({"native_category": "locally_unverified", "response_qualification": "locally_unverified", "native_role": native_role, "native_onset_row": int(round(2500 / period)), "source_native_category": scenario.split("_anom_", 1)[-1], "perturbation_metadata": {name: value.get("faults", {}) for name, value in setup["model"]["modules"].items() if value.get("faults")}})
        manifest["recordings"].append(write_recording(out, ident, frame, source, normal=normal, segments=segments, source_root=root, **extra))
    # Byte-identical native exports are one source, even when names imply
    # different perturbations. A cross-role alias cannot become fit evidence.
    groups = {}
    for recording in manifest["recordings"]:
        if not recording["normal"]:
            groups.setdefault(recording["source_sha256"], []).append(recording)
    for checksum, aliases in groups.items():
        if len({row["native_role"] for row in aliases}) > 1:
            for row in aliases:
                row["native_role"] = "assessment"
                row["source_group"] = f"hai-source-{checksum}"
                row["source_aliases"] = sorted(x["scenario"] for x in aliases)
                row["native_role_reason"] = "Byte-identical source aliases had conflicting development roles"
    manifest["native_role_recipe"] = "pump50 scenario suffix: decision_fit; pump75: risk_calibration; other scenarios: assessment. Conflicting byte-identical aliases remain assessment."
    return manifest
