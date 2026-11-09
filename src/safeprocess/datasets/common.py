"""Shared deterministic data preparation and identity helpers."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

SPLITS = ("train", "validation", "calibration", "test_normal")
WINDOW = 60
PURGE = WINDOW - 1


def resource(name: str) -> Path:
    return Path(str(files("safeprocess").joinpath("resources", name)))


def metadata(name: str) -> dict:
    text = resource(name).read_text(encoding="utf-8")
    return json.loads(text) if name.endswith(".json") else yaml.safe_load(text)


def channel_metadata(dataset: str) -> list[dict]:
    return metadata("paper_channels.json")[dataset]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dataset_root(root: Path, names: Iterable[str]) -> Path:
    """Accept either the dataset itself or a parent containing named datasets."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Recording directory does not exist: {root}")
    names = tuple(names)
    children = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for name in names:
        if name.lower() in children:
            return children[name.lower()]
    return root


def chronological_segments(
    rows: int,
    fractions: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
    *,
    start: int = 0,
    purge: int = PURGE,
) -> list[dict]:
    """Use half-open source coordinates with a 59-row gap after each boundary.

    Boundaries are calculated before purging. This prevents a length-60 training
    window from sharing samples with any later partition. No concatenation of
    segments is permitted before window generation.
    """
    if rows <= start or len(fractions) != 4 or not np.isclose(sum(fractions), 1):
        raise ValueError("Invalid chronological partition specification")
    boundaries = [start] + [
        start + int((rows - start) * f) for f in np.cumsum(fractions)[:-1]
    ] + [rows]
    answer = []
    for index, split in enumerate(SPLITS):
        lo = boundaries[index] + (purge if index else 0)
        hi = boundaries[index + 1]
        if hi - lo < WINDOW:
            raise ValueError(f"Partition {split} has fewer than {WINDOW} rows after purging")
        answer.append({"start": lo, "end": hi, "split": split})
    return answer


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required sensor/context columns: {missing}")
    out = frame.loc[:, columns].copy()
    for name in columns:
        out[name] = pd.to_numeric(out[name], errors="raise").astype("float64")
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Sensor/context inputs contain missing or nonfinite values")
    return out


def write_recording(
    out: Path,
    ident: str,
    frame: pd.DataFrame,
    source: Path | None,
    *,
    normal: bool,
    segments: list[dict],
    source_root: Path | None = None,
    **extra,
) -> dict:
    directory = out / "recordings"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{ident}.parquet"
    frame.to_parquet(target, index=False, compression="zstd")
    row = {
        "id": ident,
        "path": target.relative_to(out).as_posix(),
        "normal": bool(normal),
        "rows": len(frame),
        "sha256": digest(target),
        "segments": segments,
        **extra,
    }
    if source is not None:
        # Source provenance is relative to the acquisition root, never required
        # to read this newly written, self-contained recording.
        row["source_file"] = source.relative_to(source_root).as_posix() if source_root else source.name
        row["source_sha256"] = digest(source)
    return row


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(boundaries[::2].tolist(), boundaries[1::2].tolist()))


def base_manifest(dataset: str, kind: str, features: list[str], owners: list[str], context: list[str]) -> dict:
    return {
        "schema_version": "safeprocess-dataset-v1",
        "dataset": dataset,
        "kind": kind,
        "features": features,
        "context_columns": context,
        "owners": owners,
        "connections": [],
        "recordings": [],
        "window_length": WINDOW,
        "window_stride": 10,
        "split_boundary_purge": PURGE,
        "sensor_encoding": "identity",
        "annotation_policy": "Annotations are evaluator metadata, never model or context inputs.",
    }
