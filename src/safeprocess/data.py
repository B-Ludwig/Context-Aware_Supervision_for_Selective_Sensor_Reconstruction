"""Public local-data and portable acquisition-manifest interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import demo as demo_adapter
from .datasets import four_tank, hai, swat
from .datasets.common import digest
from .ontology import validate_manifest, write_description

PAPER_DATASETS = tuple(f"hai_ds{i}" for i in range(1, 11)) + ("four_tank", "swat")


class DatasetManifest(dict):
    """Dictionary with an in-memory path hint that is not serialized into JSON."""

    def __init__(self, value: dict, base_dir: Path):
        super().__init__(value)
        self.base_dir = Path(base_dir).resolve()


def canonical_dataset(dataset: str) -> str:
    dataset = str(dataset).lower().replace("-", "_")
    aliases = {"fourtank": "four_tank", "hai_demo": "demo", "swat_physical": "swat"}
    dataset = aliases.get(dataset, dataset)
    if dataset.startswith("ds") and dataset[2:].isdigit():
        dataset = "hai_" + dataset
    if dataset not in (*PAPER_DATASETS, "demo"):
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from demo or {', '.join(PAPER_DATASETS)}")
    return dataset


def describe_dataset(dataset: str, data_root: Path, out: Path, *, demo: bool = False) -> dict:
    """Prepare data/ontology descriptions without fitting or executing any model.

    ``data_root`` may be an acquisition root containing HAI/FourTank/SWaT
    subdirectories or a dataset's own root. Output recordings and RDF use only
    paths relative to ``out``. The demo creates its input signals at runtime.
    """
    dataset = "demo" if demo else canonical_dataset(dataset)
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if dataset == "demo":
        value = demo_adapter.describe(out)
    elif dataset.startswith("hai_ds"):
        value = hai.describe(dataset, Path(data_root), out)
    elif dataset == "four_tank":
        value = four_tank.describe(Path(data_root), out)
    else:
        value = swat.describe(Path(data_root), out)
    validation = write_description(value, out)
    value["ontology"] = {"path": "ontology.ttl", "sha256": validation["ontology_sha256"], "schema_version": "CPSWatch-1.0.1", "schema_sha256": validation["schema_sha256"], "validation": "ontology-validation.json"}
    path = out / "manifest.json"
    temporary = out / "manifest.json.part"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return DatasetManifest(value, out)


def load_manifest(path: Path | str | dict) -> DatasetManifest:
    if isinstance(path, DatasetManifest):
        return path
    if isinstance(path, dict):
        hint = path.get("_base_path")
        if hint is None:
            raise ValueError("A plain manifest dictionary needs an in-memory _base_path hint. Prefer load_manifest(path) or describe_dataset(...).")
        value = {key: item for key, item in path.items() if key != "_base_path"}
        validate_manifest(value)
        return DatasetManifest(value, Path(hint))
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(value)
    return DatasetManifest(value, path.parent)


def recording_path(manifest: DatasetManifest, recording: dict) -> Path:
    relative = Path(recording["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Recording path is not portable")
    target = (manifest.base_dir / relative).resolve()
    if not target.is_relative_to(manifest.base_dir):
        raise ValueError("Recording path escapes its manifest directory")
    return target


def read_recording(manifest_path_or_dict: Path | str | dict, recording_id: str, *, verify_hash: bool = False) -> pd.DataFrame:
    manifest = load_manifest(manifest_path_or_dict)
    matches = [row for row in manifest["recordings"] if row["id"] == recording_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one recording {recording_id!r}, found {len(matches)}")
    row = matches[0]
    path = recording_path(manifest, row)
    if verify_hash and digest(path) != row["sha256"]:
        raise ValueError(f"Recording integrity check failed for {recording_id}")
    frame = pd.read_parquet(path)
    if len(frame) != row["rows"]:
        raise ValueError(f"Recording row count changed for {recording_id}")
    required = manifest["features"] + manifest["context_columns"]
    if set(required) - set(frame):
        raise ValueError(f"Recording schema changed for {recording_id}")
    return frame


def training_segments(manifest_path: Path | str | dict, split: str = "train") -> list[np.ndarray]:
    """Return separate ordered sensor arrays. Context/annotations stay outside AE input."""
    if split not in {"train", "validation", "calibration", "test_normal"}:
        raise ValueError("training_segments accepts normal partitions only")
    manifest = load_manifest(manifest_path)
    arrays = []
    for recording in manifest["recordings"]:
        segments = [s for s in recording["segments"] if s["split"] == split]
        if not segments:
            continue
        if not recording["normal"]:
            raise ValueError("A native recording cannot enter a normal partition")
        frame = read_recording(manifest, recording["id"])
        for segment in segments:
            values = frame.iloc[segment["start"]:segment["end"]].loc[:, manifest["features"]].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite values in {recording['id']}/{split}")
            arrays.append(np.ascontiguousarray(values))
    if not arrays:
        raise ValueError(f"No normal segments for split {split}")
    return arrays


def preflight_dataset(dataset: str, data_root: Path) -> dict:
    """Check acquisition schemas and boundaries, without conversion or inference."""
    dataset = canonical_dataset(dataset)
    if dataset == "demo":
        return {"dataset": "demo", "ok": True, "features": 4, "source": "generated at runtime"}
    if dataset.startswith("hai_ds"):
        return hai.preflight(dataset, Path(data_root))
    if dataset == "four_tank":
        return four_tank.preflight(Path(data_root))
    return swat.preflight(Path(data_root))
