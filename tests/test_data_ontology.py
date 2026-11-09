"""Data isolation and topology checks without model execution or private inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook
from rdflib import Graph, RDF

from safeprocess.data import describe_dataset, load_manifest, preflight_dataset, read_recording, training_segments
from safeprocess.datasets.common import channel_metadata, chronological_segments
from safeprocess.datasets.four_tank import LABEL, STEP, assign_partitions, segment_frame
from safeprocess.datasets.hai import classify_column
from safeprocess.datasets.swat import CONTEXT, canonical_column, load_attack_metadata, read_physical
from safeprocess.ontology import CPSWATCH, connected_peak_components, induced_connections, validate_manifest


def test_demo_sensor_split_isolation_and_portability(tmp_path):
    manifest = describe_dataset("demo", tmp_path, tmp_path / "original")
    assert manifest["context_unit"].startswith("hai")
    assert len(manifest["features"]) == 4
    assert set(manifest["features"]).isdisjoint(manifest["context_columns"])
    assert "base_dir" not in json.loads((tmp_path / "original/manifest.json").read_text())
    segments = manifest["recordings"][0]["segments"]
    for previous, current in zip(segments, segments[1:]):
        assert current["start"] - previous["end"] == 59
    normal = read_recording(manifest, "demo-normal", verify_hash=True)
    for owner in set(manifest["owners"]):
        states = [x for x in manifest["context_columns"] if x.startswith(owner + ".")]
        assert np.all(normal[states].sum(axis=1).to_numpy() == 1)
    train = training_segments(manifest, "train")
    expected = normal.iloc[segments[0]["start"]:segments[0]["end"]][manifest["features"]].to_numpy(np.float32)
    np.testing.assert_array_equal(train[0], expected)
    assert train[0].shape == (14400, 4)
    assert len([r for r in manifest["recordings"] if not r["normal"]]) == 3
    assert all(r.get("synthetic") for r in manifest["recordings"] if not r["normal"])
    shutil.copytree(tmp_path / "original", tmp_path / "moved")
    shutil.rmtree(tmp_path / "original")
    relocated = load_manifest(tmp_path / "moved")
    np.testing.assert_array_equal(training_segments(relocated)[0], train[0])
    graph = Graph().parse(tmp_path / "moved/ontology.ttl")
    assert len(list(graph.subjects(RDF.type, CPSWATCH.MonitoringSensor))) == 4


def test_peak_graph_never_uses_hierarchy_or_intermediate_owners():
    connections = [["a", "middle"], ["middle", "b"], ["b", "c"]]
    assert induced_connections({"a", "b"}, connections) == []
    assert not connected_peak_components({"a", "b"}, connections)
    assert connected_peak_components({"b", "c"}, connections)
    assert connected_peak_components({"a", "middle", "b"}, connections)
    assert not connected_peak_components({"a"}, connections)
    # Owning a common parent is not declared as a system connection.
    assert not connected_peak_components({"a", "b"}, [])


def test_manifest_rejects_annotation_features_native_training_and_path_escape(tmp_path):
    manifest = describe_dataset("demo", tmp_path, tmp_path / "demo")
    bad = json.loads(json.dumps(manifest))
    bad["features"][0] = "Label"
    with pytest.raises(ValueError, match="annotations"):
        validate_manifest(bad)
    bad = json.loads(json.dumps(manifest))
    bad["recordings"][1]["segments"][0]["split"] = "train"
    with pytest.raises(ValueError, match="Fault recordings"):
        validate_manifest(bad)
    bad = json.loads(json.dumps(manifest))
    bad["recordings"][0]["path"] = "../outside.parquet"
    with pytest.raises(ValueError, match="relative"):
        validate_manifest(bad)


def test_chronological_partitions_reject_too_short_sources():
    with pytest.raises(ValueError, match="fewer"):
        chronological_segments(300)


def test_hai_roles_exclude_fault_proxies_and_keep_observable_states():
    assert classify_column("mixer0.tank_B201.level") == "sensor"
    assert classify_column("mixer0.sensor_continuous_volumeFlowRate.V_flow") == "sensor"
    assert classify_column("mixer0.state_filling.active") == "operating_context"
    assert classify_column("mixer0.clogging_valve.opening") == "parameter"
    assert classify_column("mixer0.leaking_valve.opening") == "parameter"
    assert classify_column("mixer0.pump_P101.n_in") == "actuator"
    assert classify_column("distill0.heater_distill.port.Q_flow") == "process"


def _four_tank_fixture():
    steps = ([1] * 70 + [7] * 70 + [8] * 70 + [3] * 70) * 5
    labels = [1] * (280 * 4) + [0] * 280
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=len(steps), freq="s"), STEP: steps, LABEL: labels})


def test_four_tank_boundaries_and_zero_label_quarantine():
    frame = _four_tank_fixture()
    # A gap in the filling step must split the source rather than close it.
    frame.loc[100:, "timestamp"] += pd.Timedelta(seconds=3)
    audit, segments = segment_frame(frame)
    assert not audit.loc[audit.deviation_id.eq(0), "eligible"].any()
    assert all(info["deviation_id"] == 1 for info, _ in segments)
    assert len(segments) == 5
    assert not any(info["source_start_row"] < 100 < info["source_end_row"] for info, _ in segments)
    membership = assign_partitions(segments)
    assert len(membership) == 4
    for info, _ in segments:
        assert info["batch_id"] in membership
    archived_shape = [({"batch_id": i, "deviation_id": 1}, pd.DataFrame()) for i in range(35)]
    counts = list(assign_partitions(archived_shape).values())
    assert [counts.count(name) for name in ("train", "validation", "calibration", "test_normal")] == [21, 5, 4, 5]


def test_swat_workbook_header_normalization_and_attack_table(tmp_path):
    book = Workbook()
    sheet = book.active
    sheet.append(["SWaT physical data"])
    sheet.append([" Timestamp ", " FIT 101 ", "Normal/Attack"])
    sheet.append(["28/12/2015 10:00:00 AM", 2.5, "Normal"])
    path = tmp_path / "SWaT_Dataset_Normal_v1.xlsx"
    book.save(path)
    frame = read_physical(path)
    assert frame.columns.tolist() == ["Timestamp", "FIT101", "Label"]
    assert frame["Timestamp"].iloc[0] == pd.Timestamp("2015-12-28 10:00:00")
    assert canonical_column("DPIT-301") == "DPIT301"
    book = Workbook()
    sheet = book.active
    sheet.append(["Attack #", "Start Time", "End Time", "Attack Point", "Attack"])
    sheet.append([37, pd.Timestamp("2015-01-01 23:59:00").to_pydatetime(), pd.Timestamp("2015-01-01 00:01:00").time(), "P-101", "Stop pump"])
    path = tmp_path / "List_of_attacks_Final.xlsx"
    book.save(path)
    attack = load_attack_metadata(path)[0]
    assert attack["metadata_year_corrected_to_2016"]
    assert attack["start"].startswith("2016-01-01")
    assert attack["end"].startswith("2016-01-02")


def test_swat_retains_eleven_sensor_order_and_excludes_annotations(tmp_path):
    root = tmp_path / "SWaT/cache/physical"
    root.mkdir(parents=True)
    features = [x["name"] for x in channel_metadata("swat")]
    count = 24000
    values = np.arange(count, dtype=float)
    frame = pd.DataFrame({name: 1 + values / (count + i) for i, name in enumerate(features)})
    for name in CONTEXT:
        frame[name] = 1.0
    frame["Timestamp"] = pd.date_range("2015-12-22", periods=count, freq="s")
    frame["Label"] = "Normal"
    frame.to_parquet(root / "SWaT_Dataset_Normal_v1.parquet", index=False)
    frame.to_parquet(root / "SWaT_Dataset_Normal_v0.parquet", index=False)
    native = frame.iloc[:600].copy()
    native.loc[native.index >= 300, "Label"] = "Attack"
    native.to_parquet(root / "SWaT_Dataset_Attack_v0.parquet", index=False)
    check = preflight_dataset("swat", tmp_path)
    assert check["ok"] and check["features"] == 11
    manifest = describe_dataset("swat", tmp_path, tmp_path / "out")
    assert manifest["features"] == features
    assert manifest["recordings"][0]["segments"][0]["start"] == 19800
    assert len([x for x in manifest["recordings"] if x["normal"]]) == 1
    assert manifest["source_aliases"][0]["source_group"] == "swat-normal-2015"
    assert "Label" not in read_recording(manifest, "swat-native")
    assert manifest["recordings"][1]["response_qualification"] == "locally_unverified"
    assert manifest["recordings"][1]["segments"] == [{"start": 300, "end": 600, "split": "native", "label_run_index": 1, "attack_numbers": [], "native_role": "decision_fit"}]
