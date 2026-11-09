"""SWaT 2015 physical observations, retained sensors and attack annotations.

Physical Parquet equivalents and the provider's XLSX workbooks are accepted.
The network capture is neither used nor required. Normal_v0 and Normal_v1 are
different exports of the same recording and are never independent sources.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from openpyxl import load_workbook

from .common import base_manifest, channel_metadata, chronological_segments, contiguous_runs, dataset_root, digest, metadata, numeric_frame, write_recording

CONTEXT = ["MV301", "MV302", "P301", "P302", "P401", "P402"]


def canonical_column(value) -> str:
    text = re.sub(r"\s+", "", "" if value is None else str(value))
    if text.lower() in {"timestamp", "date"}:
        return "Timestamp"
    if text.lower() in {"normal/attack", "normalattack", "label"}:
        return "Label"
    return text.replace("-", "")


def parse_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)
    values = series.astype("string").str.strip()
    parsed = pd.to_datetime(values, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    if parsed.isna().any():
        parsed.loc[parsed.isna()] = pd.to_datetime(values[parsed.isna()], dayfirst=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("SWaT physical timestamps could not be parsed")
    return parsed


def root_path(data_root: Path) -> Path:
    return dataset_root(data_root, ("SWaT", "swat"))


def source_files(data_root: Path) -> tuple[Path, dict[str, Path]]:
    root = root_path(data_root)
    folders = [root, root / "cache/physical", root / "Physical", root / "raw/Physical", root / "raw"]
    found = {}
    # Prefer supplied Parquet equivalents. Filename matching is explicit so
    # unrelated network files cannot silently become physical observations.
    for stem in ("SWaT_Dataset_Normal_v1", "SWaT_Dataset_Normal_v0", "SWaT_Dataset_Attack_v0"):
        for suffix in (".parquet", ".xlsx", ".xlsm"):
            for folder in folders:
                candidate = folder / (stem + suffix)
                if candidate.is_file():
                    found[stem] = candidate
                    break
            if stem in found:
                break
    if "SWaT_Dataset_Normal_v1" not in found and "SWaT_Dataset_Normal_v0" not in found:
        raise FileNotFoundError(f"Acquire the SWaT physical normal workbook (Normal_v1 preferred) under {root}")
    return root, found


def _workbook_header(path: Path) -> tuple[list[str], int]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        for number, row in enumerate(workbook.active.iter_rows(values_only=True), start=1):
            columns = [canonical_column(x) for x in row]
            if "Timestamp" in columns and "FIT101" in columns:
                columns = [x or f"unused_{i}" for i, x in enumerate(columns)]
                if len(columns) != len(set(columns)):
                    raise ValueError("Duplicate normalized physical-workbook column")
                return columns, number
            if number >= 10:
                break
    finally:
        workbook.close()
    raise ValueError(f"Could not locate the SWaT physical header in {path.name}")


def read_physical(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        frame.columns = [canonical_column(c) for c in frame]
    else:
        columns, header_row = _workbook_header(path)
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = (row for row in workbook.active.iter_rows(min_row=header_row + 1, values_only=True) if any(v is not None for v in row))
            frame = pd.DataFrame.from_records(rows, columns=columns)
        finally:
            workbook.close()
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Duplicate SWaT physical columns")
    frame["Timestamp"] = parse_timestamp(frame["Timestamp"])
    if "Label" in frame:
        frame["Label"] = frame["Label"].astype("string").str.strip().str.lower()
        if not frame["Label"].isin(["normal", "attack"]).all():
            raise ValueError("Unknown SWaT physical annotation")
    return frame


def load_attack_metadata(path: Path) -> list[dict]:
    """Load provider metadata or a portable CSV transcription, as annotations."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = pd.read_csv(path)
        if not {"attack_number", "start", "end"} <= set(rows):
            raise ValueError("Attack CSV needs attack_number, start and end columns")
        # Drop missing cells and normalize values for JSON without assigning a
        # physical-response qualification from narrative fields.
        return [{key: value.item() if isinstance(value, np.generic) else value for key, value in row.items() if pd.notna(value)} for row in rows.to_dict("records")]
    workbook = load_workbook(path, read_only=True, data_only=True)
    output = []
    try:
        iterator = workbook.active.iter_rows(values_only=True)
        columns = [str(value).strip() for value in next(iterator)]
        for values in iterator:
            if not values or not isinstance(values[0], int) or not 1 <= values[0] <= 41:
                continue
            row = dict(zip(columns, values))
            start = pd.Timestamp(row["Start Time"])
            corrected = row["Attack #"] >= 37 and start.year == 2015 and start.month == 1
            if corrected:
                start = start.replace(year=2016)
            end_value = row.get("End Time")
            end = None
            if end_value is not None:
                if isinstance(end_value, dt.time):
                    end = pd.Timestamp.combine(start.date(), end_value)
                    if end < start:
                        end += pd.Timedelta(days=1)
                else:
                    end = pd.Timestamp(end_value)
            output.append({"attack_number": int(row["Attack #"]), "start": start.isoformat(), "end": end.isoformat() if end is not None else None, "attack_point": str(row.get("Attack Point", "")).strip(), "attack": str(row.get("Attack", "")).strip(), "expected_impact_or_intent": str(row.get("Expected Impact or attacker intent", "")).strip(), "unexpected_outcome": str(row.get("Unexpected Outcome", "")).strip(), "metadata_year_corrected_to_2016": bool(corrected)})
    finally:
        workbook.close()
    return output


def _attack_metadata_path(root: Path) -> Path | None:
    for rel in ("List_of_attacks_Final.xlsx", "raw/List_of_attacks_Final.xlsx", "attack_metadata.csv", "physical_attack_metadata.csv", "analysis/physical_attack_metadata.csv"):
        if (root / rel).is_file():
            return root / rel
    return None


def preflight(data_root: Path) -> dict:
    root, sources = source_files(data_root)
    features = [x["name"] for x in channel_metadata("swat")]
    records = []
    for stem, source in sources.items():
        if source.suffix == ".parquet":
            info = pq.ParquetFile(source)
            columns = [canonical_column(x) for x in info.schema.names]
            count = info.metadata.num_rows
        else:
            columns, _ = _workbook_header(source)
            count = None
        missing = set(features + CONTEXT + ["Timestamp"]) - set(columns)
        if missing:
            raise ValueError(f"{source.name} is missing {sorted(missing)}")
        records.append({"file": source.relative_to(root).as_posix(), "rows": count, "source_identity": stem})
    annotation = _attack_metadata_path(root)
    return {"dataset": "swat", "ok": True, "features": len(features), "source_root": str(root), "recordings": records, "attack_metadata": annotation.relative_to(root).as_posix() if annotation else None, "normal_source_alias_warning": "Normal_v0 and Normal_v1 are versions of one recording", "startup_trim_v1_rows": 19800}


def describe(data_root: Path, out: Path) -> dict:
    root, sources = source_files(data_root)
    mapping = channel_metadata("swat")
    features, owners = [x["name"] for x in mapping], [x["owner"] for x in mapping]
    topology = metadata("swat_topology.yaml")["plants"][0]
    manifest = base_manifest("swat", "swat-physical", features, owners, CONTEXT)
    engineering = metadata("swat_channel_metadata.yaml")["channels"]
    manifest["channel_descriptions"] = [{"name": name, "owner": owner, "observable_property": engineering[name]["observable_property"], "description": engineering[name]["description"], "evidence": engineering[name]["evidence"], "unit_uri": engineering[name].get("raw_unit_uri")} for name, owner in zip(features, owners)]
    manifest["context_bindings"] = {owner: list(CONTEXT) for owner in sorted(set(owners))}
    manifest["context_binding_note"] = "The declared six actuator/state signals are observable operating-history sources, separate from AE sensor inputs. This binding does not assert causal fault localization."
    manifest.update({"context_unit": "swat", "components": [topology["id"]] + [x["id"] for x in topology["components"]], "connections": [[x["source"], x["target"]] for x in topology["system_connections"]], "sample_period_seconds": 1.0, "split_fractions": [0.55, 0.15, 0.15, 0.15], "source_aliases": [{"source_group": "swat-normal-2015", "aliases": ["SWaT_Dataset_Normal_v0", "SWaT_Dataset_Normal_v1"], "reason": "Versions of the same source recording, with 1800 startup rows removed in v1"}]})
    normal_stem = "SWaT_Dataset_Normal_v1" if "SWaT_Dataset_Normal_v1" in sources else "SWaT_Dataset_Normal_v0"
    source = sources[normal_stem]
    raw = read_physical(source)
    if "Label" in raw and not raw["Label"].eq("normal").all():
        raise ValueError("SWaT normal acquisition contains attack annotations")
    trim = 19800 if normal_stem.endswith("v1") else 21600
    prepared = numeric_frame(raw, features + CONTEXT)
    prepared.insert(0, "source_row", np.arange(len(raw), dtype=np.int64))
    prepared.insert(1, "timestamp", raw["Timestamp"].to_numpy())
    segments = chronological_segments(len(raw), (0.55, 0.15, 0.15, 0.15), start=trim)
    manifest["recordings"].append(write_recording(out, "swat-normal", prepared, source, normal=True, segments=segments, source_root=root, source_identity=normal_stem, source_group="swat-normal-2015", startup_trim_rows=trim, scenario="normal-physical-2015"))
    if "SWaT_Dataset_Attack_v0" in sources:
        source = sources["SWaT_Dataset_Attack_v0"]
        raw = read_physical(source)
        if "Label" not in raw:
            raise ValueError("SWaT attack acquisition requires its Normal/Attack annotation")
        annotations_path = _attack_metadata_path(root)
        annotations = load_attack_metadata(annotations_path) if annotations_path else []
        prepared = numeric_frame(raw, features + CONTEXT)
        prepared.insert(0, "source_row", np.arange(len(raw), dtype=np.int64))
        prepared.insert(1, "timestamp", raw["Timestamp"].to_numpy())
        segments, events = [], []
        for run, (start, end) in enumerate(contiguous_runs(raw["Label"].eq("attack").to_numpy()), start=1):
            numbers = []
            for row in annotations:
                meta_start = pd.Timestamp(row["start"])
                meta_end = pd.Timestamp(row.get("end")) if row.get("end") else meta_start
                if meta_start <= raw["Timestamp"].iloc[end - 1] and meta_end >= raw["Timestamp"].iloc[start]:
                    numbers.append(int(row["attack_number"]))
            native_role = ("decision_fit", "risk_calibration", "assessment")[(run - 1) % 3]
            segments.append({"start": start, "end": end, "split": "native", "label_run_index": run, "attack_numbers": numbers, "native_role": native_role})
            events.append({"start": start, "end": end, "label_run_index": run, "attack_number": numbers[0] if len(numbers) == 1 else None, "attack_numbers": numbers, "native_role": native_role, "native_category": "locally_unverified", "response_qualification": "locally_unverified"})
        manifest["recordings"].append(write_recording(out, "swat-native", prepared, source, normal=False, segments=segments, source_root=root, source_identity="SWaT_Dataset_Attack_v0", source_group="swat-attack-2015", scenario="attack-physical-2015", native_category="locally_unverified", response_qualification="locally_unverified", native_role="assessment", events=events, attack_metadata=annotations, attack_metadata_sha256=digest(annotations_path) if annotations_path else None))
        manifest["native_role_recipe"] = "Chronological disjoint attack-label episodes cycle decision_fit, risk_calibration and assessment. Segment roles override the recording-level assessment default."
    return manifest
