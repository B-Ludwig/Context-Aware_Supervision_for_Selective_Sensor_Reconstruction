"""SmA FourTank CSV preparation with whole-batch partition membership."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import base_manifest, channel_metadata, dataset_root, digest, metadata, numeric_frame, write_recording

LABEL = "DeviationID ValueY"
STEP = "CuStepNo ValueY"
ACTIVE_STEPS = (1, 7, 8, 3)
CSV = "SmA-Four-Tank-Batch-Process_V2.csv"


def source_path(data_root: Path) -> tuple[Path, Path]:
    root = dataset_root(data_root, ("FourTank", "four_tank", "four-tank"))
    for source in (root / CSV, root / "raw" / CSV):
        if source.is_file():
            return root, source
    raise FileNotFoundError(f"Acquire {CSV} and place it under {root} or its raw/ directory")


def segment_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[dict, pd.DataFrame]]]:
    """Keep batch, label and timestamp barriers. Exclude undocumented code zero."""
    frame = frame.copy().reset_index(drop=True)
    required = {"timestamp", LABEL, STEP}
    if not required <= set(frame):
        raise ValueError(f"Missing FourTank boundary columns: {sorted(required - set(frame))}")
    times = pd.to_datetime(frame["timestamp"], errors="raise")
    dt = times.diff().dt.total_seconds()
    if times.isna().any() or (dt.dropna() <= 0).any():
        raise ValueError("FourTank timestamps must increase in acquisition order")
    for name in (LABEL, STEP):
        values = pd.to_numeric(frame[name], errors="raise")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"Nonintegral FourTank state: {name}")
        frame[name] = values.astype(int)
    if not frame[LABEL].isin(range(11)).all() or not frame[STEP].isin((*ACTIVE_STEPS, 2)).all():
        raise ValueError("Unknown FourTank deviation or controller-step code")
    starts = frame[STEP].eq(1) & (frame[STEP].shift().ne(1) | dt.ne(1))
    batch = starts.cumsum()
    eligible = frame[LABEL].between(1, 10) & frame[STEP].isin(ACTIVE_STEPS) & batch.gt(0)
    audit = pd.DataFrame({"source_row": frame.index, "batch_id": batch, "deviation_id": frame[LABEL], "eligible": eligible, "exclusion": np.where(frame[LABEL].eq(0), "unknown_deviation_0", np.where(frame[STEP].eq(2), "transition_step_2", np.where(batch.eq(0), "no_batch_start", "")))})
    complete = {}
    for ident, rows in frame[eligible].groupby(batch[eligible]):
        sequence = rows[STEP][rows[STEP].ne(rows[STEP].shift())].tolist()
        labels = rows[LABEL].unique().tolist()
        complete[int(ident)] = sequence == list(ACTIVE_STEPS) and len(labels) == 1
    barriers = batch.ne(batch.shift()) | dt.ne(1) | frame[LABEL].ne(frame[LABEL].shift()) | eligible.ne(eligible.shift())
    groups = barriers.cumsum()
    segments = []
    for _, rows in frame[eligible].groupby(groups[eligible]):
        ident = int(batch.loc[rows.index[0]])
        label = int(rows[LABEL].iloc[0])
        if label == 1 and not complete[ident]:
            audit.loc[rows.index, ["eligible", "exclusion"]] = [False, "incomplete_normal_batch"]
            continue
        info = {"batch_id": ident, "deviation_id": label, "source_start_row": int(rows.index[0]), "source_end_row": int(rows.index[-1] + 1), "complete_batch": complete[ident]}
        rows = rows.copy()
        rows["source_row"] = rows.index
        segments.append((info, rows.reset_index(drop=True)))
    return audit, segments


def assign_partitions(segments: list[tuple[dict, pd.DataFrame]]) -> dict[int, str]:
    batches = sorted({info["batch_id"] for info, _ in segments if info["deviation_id"] == 1})
    if len(batches) < 4:
        raise ValueError("At least four complete normal FourTank batches are required")
    raw = np.array((0.60, 0.15, 0.10, 0.15)) * len(batches)
    sizes = np.maximum(1, np.floor(raw).astype(int))
    while sizes.sum() < len(batches):
        sizes[int(np.argmax(raw - sizes))] += 1
    while sizes.sum() > len(batches):
        candidates = np.flatnonzero(sizes > 1)
        sizes[candidates[np.argmax((sizes - raw)[candidates])]] -= 1
    names = ("train", "validation", "calibration", "test_normal")
    return dict(zip(batches, np.repeat(names, sizes).tolist()))


def preflight(data_root: Path) -> dict:
    root, source = source_path(data_root)
    columns = list(pd.read_csv(source, sep=";", nrows=0).columns)
    features = [x["name"] for x in channel_metadata("four_tank")]
    missing = set(features + [STEP, LABEL, "timestamp"]) - set(columns)
    if missing:
        raise ValueError(f"FourTank CSV schema is missing {sorted(missing)}")
    # Only the three boundary columns are loaded, no model execution or cache
    # creation. This checks the documented 21/5/4/5 normal-batch membership.
    boundary = pd.read_csv(source, sep=";", usecols=["timestamp", STEP, LABEL])
    _, segments = segment_frame(boundary)
    partitions = assign_partitions(segments)
    counts = {name: list(partitions.values()).count(name) for name in ("train", "validation", "calibration", "test_normal")}
    return {"dataset": "four_tank", "ok": True, "features": len(features), "source_root": str(root), "file": source.relative_to(root).as_posix(), "normal_batches": len(partitions), "normal_batch_split": counts, "matches_paper_batch_counts": list(counts.values()) == [21, 5, 4, 5]}


def describe(data_root: Path, out: Path) -> dict:
    root, source = source_path(data_root)
    engineering = metadata("four_tank_metadata.yaml")
    mapping = channel_metadata("four_tank")
    features = [x["name"] for x in mapping]
    owners = [x["owner"] for x in mapping]
    frame = pd.read_csv(source, sep=";")
    audit, segments = segment_frame(frame)
    partitions = assign_partitions(segments)
    if len(partitions) != 35:
        raise ValueError("The paper FourTank recipe requires 35 complete normal batches (21/5/4/5)")
    manifest = base_manifest("four_tank", "four-tank", features, owners, [STEP])
    descriptions = {row["name"]: row for row in engineering["channels"]}
    manifest["channel_descriptions"] = [{"name": name, "owner": owner, "observable_property": descriptions[name]["observable_property"], "description": descriptions[name]["description"], "evidence": descriptions[name]["evidence"], "unit_uri": None} for name, owner in zip(features, owners)]
    manifest["context_bindings"] = {owner: [STEP] for owner in sorted(set(owners))}
    manifest.update({"context_unit": "four_tank", "components": [x["name"] for x in engineering["components"]], "connections": [[x["source"], x["target"]] for x in engineering["connections"]], "sample_period_seconds": 1.0, "normal_batch_split": {name: list(partitions.values()).count(name) for name in ("train", "validation", "calibration", "test_normal")}, "excluded_rows": {str(key): int(value) for key, value in audit.loc[~audit.eligible, "exclusion"].value_counts().items()}, "split_boundary_purge": 0, "batch_boundary_rule": "No windows cross batch starts, deviation changes, timestamp gaps or excluded controller steps"})
    source_hash = digest(source)
    fault_names = {x["deviation_id"]: x["name"] for x in engineering["faults"]}
    native_membership = {}
    for deviation in sorted(fault_names):
        batches = sorted({info["batch_id"] for info, _ in segments if info["deviation_id"] == deviation})
        for index, batch_id in enumerate(batches):
            native_membership[batch_id] = ("decision_fit", "risk_calibration", "assessment")[index % 3]
    manifest["native_role_recipe"] = "Within each documented deviation, chronological whole batches cycle decision_fit, risk_calibration and assessment. Every segment of a batch retains its batch role."
    for info, rows in segments:
        normal = info["deviation_id"] == 1
        ident = f"four-tank-batch-{info['batch_id']:04d}-row-{info['source_start_row']:06d}"
        prepared = numeric_frame(rows, features + [STEP])
        prepared.insert(0, "source_row", rows["source_row"].to_numpy())
        prepared.insert(1, "timestamp", pd.to_datetime(rows["timestamp"]).to_numpy())
        split = partitions[info["batch_id"]] if normal else "native"
        extra = {**info, "scenario": ident, "source_group": f"four-tank-batch-{info['batch_id']:04d}", "source_file": source.relative_to(root).as_posix(), "source_sha256": source_hash}
        if not normal:
            extra.update({"native_category": "locally_unverified", "source_native_category": fault_names[info["deviation_id"]], "response_qualification": "locally_unverified", "native_role": native_membership[info["batch_id"]]})
        manifest["recordings"].append(write_recording(out, ident, prepared, None, normal=normal, segments=[{"start": 0, "end": len(rows), "split": split}], **extra))
    return manifest
